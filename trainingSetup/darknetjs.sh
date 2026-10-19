#!/bin/bash
# if the script fails - the below line helps terminate the script as soon as an error occurs
set -euo pipefail
# set -x

TMPDIR=""
SHUTDOWN=0

log_exit() {
  if [ -n "$TMPDIR" -a -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
  fi
  local code=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') Script exited with code $code"
  if [ -f $HOME/.openrc.sh -a ! -f $HOME/.noshutdown -a "$SHUTDOWN" = "1" ]; then
      source $HOME/.openrc.sh
      openstack server shelve `cat /run/cloud-init/.instance-id`
  fi
}

# This line makes sure log_exit is called no matter how the script exits (e.g., exit 1, error, or reaching the end)
trap log_exit EXIT

rclone() {
  $RCLONE_BIN --config $RCLONE_CONF "$@"
}

download() {
  rm -f "$2"
  case "$1" in 
    https://drive.google.com/*) gdown --fuzzy -q -O "$2" "$1";;
    http*) wget --no-check-certificate -q -O "$2" "$1";;
    *) rclone copyto "$1" "$2";;
  esac
  if [ ! -s "$2" ]; then
    echo "Download $1 failed..." 1>&2
    exit 1;
  fi
}

download_weights() {
  case "$1" in
    yolov3-darknet53)
      # darknet53.conv.74
      download "https://drive.google.com/uc?id=1A2tUanRGnlFkK7clccpVLiGFnEeQ1Jc2" "darknet53.conv.74";;
    yolov3) 
      # yolov3.conv.81
      download "https://drive.google.com/uc?id=1BLNPV1_1wBCFewX17UOJWOYBwQS8_nMV" "yolov3.conv.81";;
    yolov3-tiny) 
      # yolov3-tiny.conv.15
      download "https://drive.google.com/uc?id=1iSTibH4ZRLsw3VcijZY1r41Ia-CH_2MK" "yolov3-tiny.conv.15";;
    yolov4)  
      # yolov4.conv.137
      download "https://drive.google.com/uc?id=1OCRGWiUznDoJ4QNIBqBYcvfbtFW-Hhu8" "yolov4.conv.137";;
    yolov4-tiny)  
      # yolov4-tiny.conv.29
      download "https://drive.google.com/uc?id=1c6iGzCr3jlC4YX34QaFLg2ZbpLPq5bgR" "yolov4-tiny.conv.29";;
    yolov7)
      # yolov7.conv.133  
      download "https://drive.google.com/uc?id=1k3yEw3mnhFooFAWRdZDuVHibPD1RVwDr" "yolov7.conv.133";;
    yolov7-tiny)  
      # yolov7-tiny.conv.89
      download "https://drive.google.com/uc?id=13vQDJM0AD6lyo1x9AJxmNNbbQA9tb1I_" "yolov7-tiny.conv.89";;
    *)
      echo "Bad YOLO config $1..." 1>&2
      exit 1;;
  esac
  echo *.conv.*
}

upload() {
  rclone copyto "$1" "$2"
}

exists() {
  rclone ls "$1" 1>/dev/null 2>&1
}

backbone_layers() {
  case "$1" in
    yolov3-darknet53) echo 74;;
    yolov3)           echo 81;;
    yolov3-tiny)      echo 15;;
    yolov4)           echo 137;;
    yolov4-tiny)      echo 29;;
    yolov7)           echo 133;;
    yolov7-tiny)      echo 89;;
    *) echo "";;
  esac
}

RESULTS=""
NAME=""
CLEAN="0"
IOU=""
CONF=""
CONFIG="yolov3-darknet53"
WEIGHTS=""
BASE_LABELS_FILE=""
SPLIT="0.8"
SHUTDOWN=""
FULL_WEIGHTS=""
OLD_CLASS_COUNT=0
BASELINE_DIR=""
CLEAR_FLAG=""

while [ "$#" -gt 0 ]; do
    case $1 in
        --image_folder)
            RESULTS="$2"
            shift 2
            ;;
        --job_name)
            NAME="$2"
            shift 2
            ;;
	--config)
            CONFIG="$2"
	    shift 2
	    ;;
  --weights)
        WEIGHTS="$2"
        shift 2
            ;;
        --base_labels)
            BASE_LABELS_FILE="$2"
            shift 2
            ;;
	--split)
            SPLIT="$2"
	    shift 2
	    ;;
        --iou)
            IOU="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --noshutdown)
            SHUTDOWN=0
            shift
            ;;
	--clean)
            CLEAN=1
            shift
            ;;
        -h|--help)
            echo "Usage: ./darknetjs.sh --image_folder <location> --job_name <name> \[optional parameters\] \[YOLO config parameters\]"
            echo ""
            echo "Required:"
            echo "  --image_folder   Writable Google drive folder for images and results"
            echo "  --job_name       Job name for this training run"
            echo ""
            echo "Optional:"
            echo "  --noshutdown     Do not shelve the instance when done."
            echo "  --clean          Remove local and remote job folders"
            echo ""
            echo "Darknet command-line parameters (optional):"
            echo "  --config         YOLO config. Default: yolov3-darknet53."
            echo "                   If using for fine-tuning, ensure that the config used for fine-tuning is the same that was used by the pretrained yolo model being used"
            echo "  --weights        For fine-tuning. Pretrained model weights (via rclone path, Google Drive URL, or http(s)) needs to be provided."
            echo "                   If omitted, the default pretrained darknet weights for --config is downloaded."
            echo "                   When pretrained weights are provided, training starts with Darknet -clear (fine-tune, not resume training)."
            echo "  --base_labels    For fine-tuning. Path to labels file from the previous training run (via rclone path, Google Drive URL, or local file) is needed"
            echo "                   if additional class labels need to trained (which are not present in the base model)"
            echo "                   If omitted, it is assumed that the class labels are unchanged when compared with the base model classes."
            echo "  --iou            IoU for mAP evaluation. Default: 0.5."
            echo "  --conf           Confidence threshold for mAP evaluation. Default: 0.25."
            echo "  --split          Proportion of images to use for training. Default: 0.8."
            echo ""
            echo "YOLO config (optional, must at the end of arguments list):"
            echo "  --max_batches    Number of iterations. Default: max(#classes*2000,6000)."
            echo "  --nms_kind       Non-maximal suppression algorithm. One of default, greedynms, diounms, cornernms."
            echo "  --beta_nms       Non-maximal suppression threshold for greedynms. Default: 0.6."
            echo "  --batch          Batch size. Default: 64."
            echo "  --subdivisions   Subdivisions. Default: 16."
            echo "  --height         Input image height. Default: 416."
            echo "  --width          Input image width. Default: 416."
            echo "  --learning_rate  Learning rate. Default: 0.001."
            echo ""
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

if [ "$SHUTDOWN" = "" ]; then
     SHUTDOWN=1
fi
if [ "$SHUTDOWN" -eq 1 ]; then
  touch $HOME/.noshutdown
fi
if [ "$CONF" != "" ]; then
    CONF="-thresh $CONF"
fi
if [ "$IOU" != "" ]; then
    IOU="-iou_thresh $IOU"
fi

WEIGHTS_ARGS=()
if [ -n "$WEIGHTS" ]; then
  WEIGHTS_ARGS=(--weights "$WEIGHTS")
fi

BASE_LABELS_ARGS=()
if [ -n "$BASE_LABELS_FILE" ]; then
  BASE_LABELS_ARGS=(--base_labels "$BASE_LABELS_FILE")
fi

BASE="$PWD"
SCRIPTS="$BASE/scripts"
RCLONE_CONF="$BASE/.rclone.conf"
RCLONE_BIN="$BASE/rclone/rclone"

# Validate required args
if [ -z "$RESULTS" ]; then
    echo "Error: --image_folder is required"
    exit 1
fi
if [ -z "$NAME" ]; then
    echo "Error: --job_name is required"
    exit 1
fi
if [ -d "$NAME" ]; then
    if [ "$CLEAN" -eq 1 ]; then
	rm -rf "$NAME"
    else
        echo "Error: --job_name has already been used locally."
        exit 1
    fi
fi
if [ ! -f "$RCLONE_CONF" ]; then
    echo "Error: "$RCLONE_CONF" not found." 
    exit 1
fi
if exists "$RESULTS/$NAME" ; then
    if [ "$CLEAN" -eq 1 ]; then
	rclone purge "$RESULTS/$NAME"
	if exists "$RESULTS/$NAME" ; then
	    echo "Error: --job_name $NAME is still in $RESULTS."
	    exit 1
        fi
    else
	echo "Error: --job_name $NAME has already been used in $RESULTS."
	exit 1
    fi
fi

echo "Image folder: $RESULTS"
echo "Job name: $NAME"
# echo "Darknet parameters:" --config "$CONFIG" "$IOU" "$CONF" --split "$SPLIT"
echo "Darknet parameters:" --config "$CONFIG" "${WEIGHTS_ARGS[@]}" "${BASE_LABELS_ARGS[@]}" "$IOU" "$CONF" --split "$SPLIT"

echo "Training parameters:" "$@"

EXP="$NAME"
EXPROOT="$BASE/$NAME"
DRIVEROOT="$RESULTS/$NAME"

# saving config files that YOLO requires to train locally as well on the drive.
# We will access the local copy for quick use
YOLO_DATA="data"   # folder where training data is unzipped
YOLO_WEIGHTS="weights"
YOLO_INIT_WEIGHTS="${CONFIG}_${EXP}.weights"
YOLO_CONFIG="${CONFIG}_${EXP}.cfg"
TRAIN_CONFIG="train.data"
TRAINING_FILE="train.txt"
VALIDATION_FILE="valid.txt"
PARAMS_LOG="params-log.txt"
TRAIN_LOG="train-log.txt"
RCLONE_LOG="rclone-log.txt"

mkdir -p "$EXPROOT"
cd "$EXPROOT"

touch "$PARAMS_LOG"
echo "Image folder: $RESULTS" >> "$PARAMS_LOG"
echo "Job name: $NAME"  >> "$PARAMS_LOG"
# echo "Darknet parameters:" --config "$CONFIG" "$IOU" "$CONF" --split "$SPLIT" >> "$PARAMS_LOG"
echo "Darknet parameters:" --config "$CONFIG" "${WEIGHTS_ARGS[@]}" "${BASE_LABELS_ARGS[@]}" "$IOU" "$CONF" --split "$SPLIT" >> "$PARAMS_LOG"
echo "Training parameters:" "$@" >> "$PARAMS_LOG"

echo "INFO: Download $RESULTS/images.zip from Google Drive..."
download "$RESULTS/images.zip" "images.zip"
mkdir -p "$YOLO_DATA"
unzip -qq -j "images.zip" -d "$YOLO_DATA"

# Move classes.txt out of YOLO_DATA
mv "$YOLO_DATA/classes.txt" .
cp "classes.txt" "classes.names"
cp "classes.txt" "${CONFIG}_${EXP}.labels"

upload "$EXPROOT/classes.txt" "$DRIVEROOT/classes.txt"
upload "$EXPROOT/${CONFIG}_${EXP}.labels" "$DRIVEROOT/${CONFIG}_${EXP}.labels"

if [ -f "$YOLO_DATA/model.ini" ]; then
    mv "$YOLO_DATA/model.ini" .
    cp "model.ini" "${CONFIG}_${EXP}.model"
    upload "$EXPROOT/${CONFIG}_${EXP}.model" "$DRIVEROOT/${CONFIG}_${EXP}.model"
fi

# YOLO_CLASSES - Count the number of non-empty lines in classes.txt
# This tells us how many classes are defined (ignoring any blank lines)
YOLO_CLASSES=$(grep -v '^\s*$' "classes.txt" | wc -l)
echo "INFO: Total number of classes: $YOLO_CLASSES"
echo "INFO: Total number of classes: $YOLO_CLASSES" >> "$PARAMS_LOG"

YOLO_FILTERS=$(( (YOLO_CLASSES + 5) * 3 ))
TRAIN_CONFIG="train.data"
cat >"$TRAIN_CONFIG" <<EOF
classes = $YOLO_CLASSES
train = $TRAINING_FILE
valid = $VALIDATION_FILE
names = classes.names
backup = $YOLO_WEIGHTS
EOF

# need to create training and validation sets
python3 $SCRIPTS/split_data.py --image_dir $YOLO_DATA --train_txt $TRAINING_FILE --val_txt $VALIDATION_FILE --split_ratio "$SPLIT"

if [ -n "$WEIGHTS" ]; then
  BASELINE_DIR="baseline_model"
  mkdir -p "$BASELINE_DIR"
  SRC_BASE=$(basename "$WEIGHTS")
  SRC_BASE="${SRC_BASE%%\?*}"
  case "$SRC_BASE" in
    ""|uc|view|open|file|d) SRC_BASE="${CONFIG}_source.weights" ;;
  esac
  case "$SRC_BASE" in
    *.weights|*.conv.*) ;;
    *) SRC_BASE="${SRC_BASE}.weights" ;;
  esac
  FULL_WEIGHTS="${BASELINE_DIR}/${SRC_BASE}"
  download "$WEIGHTS" "$FULL_WEIGHTS"
  upload "$FULL_WEIGHTS" "$DRIVEROOT/$FULL_WEIGHTS"

  # If --base_labels (i.e labels that the base weights file was trained for) file is provided, 
  # the compararison between the old vs new class's will be done to decide if darknet partial is needed.
  # If not provided, assume classes are unchanged and proceed with full weights.
  if [ -n "$BASE_LABELS_FILE" ]; then
    # Keep the same filename that was passed (basename only) alongside the weights copy.
    OLD_LABELS_LOCAL="${BASELINE_DIR}/$(basename "$BASE_LABELS_FILE")"
    download "$BASE_LABELS_FILE" "$OLD_LABELS_LOCAL"
    upload "$OLD_LABELS_LOCAL" "$DRIVEROOT/$OLD_LABELS_LOCAL"
    OLD_CLASS_COUNT=$(grep -v '^\s*$' "$OLD_LABELS_LOCAL" | wc -l)
    echo "INFO: Old class count: $OLD_CLASS_COUNT, new class count: $YOLO_CLASSES"
    echo "INFO: Old class count: $OLD_CLASS_COUNT, new class count: $YOLO_CLASSES" >> "$PARAMS_LOG"

    # Validate: at least one common label exists, and every common label is at the same line number in both files
    COMMON_COUNT=0
    MISMATCH=0
    while IFS= read -r new_label; do
      [ -z "$new_label" ] && continue
      if grep -qxF "$new_label" "$OLD_LABELS_LOCAL" 2>/dev/null; then
        COMMON_COUNT=$((COMMON_COUNT + 1))
        OLD_LINE=$(grep -nxF "$new_label" "$OLD_LABELS_LOCAL" 2>/dev/null | head -1 | cut -d: -f1) || OLD_LINE=""
        NEW_LINE=$(grep -nxF "$new_label" "classes.txt" 2>/dev/null | head -1 | cut -d: -f1) || NEW_LINE=""
        if [ "$OLD_LINE" != "$NEW_LINE" ]; then
          echo "WARNING: Label '$new_label' is on line $OLD_LINE in base labels but line $NEW_LINE in new labels."
          MISMATCH=1
        fi
      fi
    done < "classes.txt"

    if [ "$COMMON_COUNT" -eq 0 ]; then
      echo "ERROR: No common labels found between base labels and new classes.txt. Fine-tuning requires at least one shared class. Stopping."
      exit 1
    fi
    if [ "$MISMATCH" -eq 1 ]; then
      echo "ERROR: Common label(s) are at different line positions in base and new labels. The new labels file must preserve the original positions for shared classes. Stopping."
      exit 1
    fi
    echo "INFO: Label compatibility check passed — $COMMON_COUNT common label(s) at matching line positions."
    echo "INFO: Label compatibility check passed — $COMMON_COUNT common label(s) at matching positions" >> "$PARAMS_LOG"
  fi

  YOLO_INIT_WEIGHTS="$FULL_WEIGHTS"
  CLEAR_FLAG="-clear"
  echo "INFO: Fine-tuning from $WEIGHTS (config $CONFIG)"
  echo "INFO: Fine-tuning from $WEIGHTS (config $CONFIG)" >> "$PARAMS_LOG"
  echo "INFO: Baseline weights snapshot: $FULL_WEIGHTS"
  echo "INFO: Baseline weights snapshot: $FULL_WEIGHTS" >> "$PARAMS_LOG"
  if [ -n "$BASE_LABELS_FILE" ]; then
    echo "INFO: Base labels from $BASE_LABELS_FILE"
    echo "INFO: Base labels from $BASE_LABELS_FILE" >> "$PARAMS_LOG"
    echo "INFO: Baseline labels snapshot: $OLD_LABELS_LOCAL"
    echo "INFO: Baseline labels snapshot: $OLD_LABELS_LOCAL" >> "$PARAMS_LOG"
  fi
else
  YOLO_INIT_WEIGHTS=$(download_weights "${CONFIG}")
fi

DARKNET_DIR="$BASE/darknet"
# get the original yolo config everytime (-f flag ensures this behaviour)
if [ "${CONFIG}" = "yolov3-darknet53" ]; then
  cp -f $DARKNET_DIR/cfg/yolov3.cfg "$YOLO_CONFIG"
else
  cp -f $DARKNET_DIR/cfg/${CONFIG}.cfg "$YOLO_CONFIG"
fi

python3 $SCRIPTS/update_yolo_cfg.py --yolo_config $YOLO_CONFIG --classes $YOLO_CLASSES "$@"

# copy/overwrite yolo config file to drive
upload "$YOLO_CONFIG" "$DRIVEROOT/$YOLO_CONFIG"

echo "INFO: Starting training..."

rm -rf "$YOLO_WEIGHTS"
mkdir -p "$YOLO_WEIGHTS"

LAST_WEIGHTS_FILE="$YOLO_WEIGHTS/${CONFIG}_${EXP}_last.weights"
FINAL_WEIGHTS_FILE="$YOLO_WEIGHTS/${CONFIG}_${EXP}_final.weights"
BEST_WEIGHTS_FILE="$YOLO_WEIGHTS/${CONFIG}_${EXP}_best.weights"

# DARKNET="sudo docker run --rm --gpus all -v .:/src sherensberk/darknet:2204.550.1241-devel darknet"
sudo docker pull glyomics/darknet:latest
DARKNET="sudo docker run --rm --gpus all -v .:/src glyomics/darknet darknet"

# If class count has increased, extract backbone-only weights via darknet partial so the
# detection head (sized for old classes) is not loaded — it would corrupt the binary read.
if [ -n "$WEIGHTS" ] && [ "$OLD_CLASS_COUNT" -gt 0 ] && [ "$YOLO_CLASSES" -ne "$OLD_CLASS_COUNT" ]; then
  if [ "$YOLO_CLASSES" -lt "$OLD_CLASS_COUNT" ]; then
    echo "ERROR: New class count ($YOLO_CLASSES) is less than old ($OLD_CLASS_COUNT)."
    exit 1
  fi
  N=$(backbone_layers "$CONFIG")
  if [ -z "$N" ]; then
    echo "ERROR: Unknown backbone depth for config '$CONFIG' — cannot run darknet partial automatically."
    exit 1
  fi
  echo "INFO: Class count changed ($OLD_CLASS_COUNT --> $YOLO_CLASSES): extracting backbone (${N} conv layers) via darknet partial..."
  echo "INFO: Class count changed ($OLD_CLASS_COUNT --> $YOLO_CLASSES): darknet partial ($N layers)" >> "$PARAMS_LOG"
  BACKBONE_WEIGHTS="${BASELINE_DIR}/backbone_${CONFIG}.conv.${N}"
  $DARKNET partial "$YOLO_CONFIG" "$FULL_WEIGHTS" "$BACKBONE_WEIGHTS" "$N"
  upload "$BACKBONE_WEIGHTS" "$DRIVEROOT/$BACKBONE_WEIGHTS"
  YOLO_INIT_WEIGHTS="$BACKBONE_WEIGHTS"
  CLEAR_FLAG=""
  echo "INFO: Using backbone weights for training — detection head will be randomly initialized"
  echo "INFO: Using backbone weights (detection head randomly initialized)" >> "$PARAMS_LOG"
fi

# echo darknet train "$TRAIN_CONFIG" "$YOLO_CONFIG" "$YOLO_INIT_WEIGHTS" -dont_show -map -random -nocolour $CONF $IOU
echo darknet train "$TRAIN_CONFIG" "$YOLO_CONFIG" "$YOLO_INIT_WEIGHTS" -dont_show -map -random -nocolour $CLEAR_FLAG $CONF $IOU
# nohup $DARKNET detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" "$YOLO_INIT_WEIGHTS" -dont_show -map -random -nocolour $CONF $IOU </dev/null >$TRAIN_LOG 2>&1 &
nohup $DARKNET detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" "$YOLO_INIT_WEIGHTS" -dont_show -map -random -nocolour $CLEAR_FLAG $CONF $IOU </dev/null >$TRAIN_LOG 2>&1 &

if [ "$SHUTDOWN" -eq 1 ]; then
  rm -f $HOME/.noshutdown
fi

# Get the PID of the training process
TRAIN_PID=$!

TMPDIR=$(mktemp -d)
touch "$RCLONE_LOG"

upload_files() {
  for FILE in "$@"; do
    if [ -f "$FILE" ]; then
      BASENAME=$(basename "$FILE")
      if [ ! -f "$TMPDIR/$BASENAME" -o "$FILE" -nt "$TMPDIR/$BASENAME" ]; then
        echo "INFO: Uploading $BASENAME to drive..."  >>"$RCLONE_LOG" 2>&1
        # Make a temp copy to avoid errors from writing in progress
        cp -f "$FILE" "$TMPDIR/$BASENAME"
      fi
    fi
  done
  rclone copy --verbose \
          "$TMPDIR" "$DRIVEROOT" >>"$RCLONE_LOG" 2>&1 
}


echo ">> Monitoring weights and uploading to Drive..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do

  upload_files $YOLO_WEIGHTS/yolo*.weights *-log.txt chart*.png ${EXPROOT}.log
  sleep 60

done

# clear $TMPDIR to make sure we get the last version of everything!
rm -f $TMPDIR/*

echo "INFO: Uploading last weights to Drive..."
upload_files $YOLO_WEIGHTS/yolo*.weights *-log.txt chart*.png ${EXPROOT}.log

if [ -f "$LAST_WEIGHTS_FILE" ]; then
  echo "SUCCESS: Training complete. Final sync of weights to Drive done..."
else
  echo "WARNING: Training completed but final weights not found at $LAST_WEIGHTS_FILE"
fi


    

