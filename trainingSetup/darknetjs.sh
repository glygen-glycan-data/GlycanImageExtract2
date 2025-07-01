#!/bin/bash
# if the script fails - the below line helps terminate the script as soon as an error occurs
set -euo pipefail
# set -x

TMPDIR=""

log_exit() {
  if [ -n "$TMPDIR" -a -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
  fi
  local code=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') Script exited with code $code"
}

# This line makes sure log_exit is called no matter how the script exits (e.g., exit 1, error, or reaching the end)
trap log_exit EXIT

rclone() {
  $RCLONE_BIN --config $RCLONE_CONF "$@"
}

download() {
  rm -f "$2"
  case "$1" in 
    https://drive.google.com/*) gdown -q -O "$2" "$1";;
    http*) wget --no-check-certificate -q -O "$2" "$1";;
    *) rclone copyto "$1" "$2";;
  esac
  if [ ! -s "$2" ]; then
    echo "Download $1 failed..." 1>&2
    exit 1;
  fi
}

upload() {
  rclone copyto "$1" "$2"
}

exists() {
  rclone ls "$1" 1>/dev/null 2>&1
}

RESULTS=""
NAME=""
CLEAN="0"

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
	--clean)
            CLEAN=1
            shift
            ;;
        -h|--help)
            echo "Usage: ./darknetjs.sh --image_folder <location> --job_name <name> \[optional parameters\]"
            echo ""
            echo "Required:"
            echo "  --image_folder   Writable Google drive folder for images and results"
            echo "  --job_name       Job name for this training run"
            echo ""
            echo "Optional:"
            echo "  --clean          Remove local and remote job folders"
            echo "  --batch          Batch size for YOLO config"
            echo "  --subdivisions   Subdivisions for YOLO config"
            echo "  --height         Input image height"
            echo "  --width          Input image width"
            echo "  --learning_rate  Learning rate for YOLO"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

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
echo "Training parameters:" "$@"

EXP="$NAME"
EXPROOT="$BASE/$NAME"
DRIVEROOT="$RESULTS/$NAME"

# saving config files that YOLO requires to train locally as well on the drive.
# We will access the local copy for quick use
YOLO_DATA="data"   # folder where training data is unzipped
YOLO_WEIGHTS="weights"
YOLO_CONFIG="yolov3_${EXP}.cfg"
TRAIN_CONFIG="train.data"
TRAINING_FILE="train.txt"
VALIDATION_FILE="valid.txt"
TRAIN_LOG="train-log.txt"
RCLONE_LOG="rclone-log.txt"

mkdir -p "$EXPROOT"
cd "$EXPROOT"

echo "INFO: Download $RESULTS/images.zip from Google Drive..."
download "$RESULTS/images.zip" "images.zip"
mkdir -p "$YOLO_DATA"
unzip -qq -j "images.zip" -d "$YOLO_DATA"

# Move classes.txt out of YOLO_DATA
mv "$YOLO_DATA/classes.txt" .
cp "classes.txt" "classes.names"
cp "classes.txt" "yolov3_${EXP}.labels"

upload "$EXPROOT/classes.txt" "$DRIVEROOT/classes.txt"
upload "$EXPROOT/yolov3_${EXP}.labels" "$DRIVEROOT/yolov3_${EXP}.labels"

# YOLO_CLASSES - Count the number of non-empty lines in classes.txt
# This tells us how many classes are defined (ignoring any blank lines)
YOLO_CLASSES=$(grep -v '^\s*$' "classes.txt" | wc -l)
echo "INFO: Total number of classes: $YOLO_CLASSES"

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
python3 $SCRIPTS/split_data.py --image_dir $YOLO_DATA --train_txt $TRAINING_FILE --val_txt $VALIDATION_FILE --split_ratio 0.8


# get darknet initial weights
if [ ! -f ./darknet53.conv.74 ]; then
  echo "Downloading darknet53.conv.74..."
  download "https://drive.google.com/uc?id=16i5nt2np-4cVw9NlQVL_UrYDBo5zLcgm" darknet53.conv.74
else
  echo "darknet53.conv.74 already exists. Skipping download."
fi

DARKNET_DIR="$BASE/darknet"
# get the original yolo config everytime (-f flag ensures this behaviour)
cp -f $DARKNET_DIR/cfg/yolov3.cfg "$YOLO_CONFIG"

python3 $SCRIPTS/update_yolo_cfg.py --yolo_config $YOLO_CONFIG --classes $YOLO_CLASSES "$@"

# copy/overwrite yolo config file to drive
upload "$YOLO_CONFIG" "$DRIVEROOT/$YOLO_CONFIG"

echo "INFO: Starting training..."

rm -rf "$YOLO_WEIGHTS"
mkdir -p "$YOLO_WEIGHTS"

LAST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_last.weights"
FINAL_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_final.weights"
BEST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_best.weights"

DARKNET="sudo docker run --gpus all -v .:/src sherensberk/darknet:2204.550.1241-devel darknet"

$DARKNET detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" ./darknet53.conv.74 -dont_show -map -nocolour </dev/null >$TRAIN_LOG 2>&1 &

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
  rclone copy --update --verbose \
          "$TMPDIR" "$DRIVEROOT" >>"$RCLONE_LOG" 2>&1 
}


echo ">> Monitoring weights and uploading to Drive..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do

  upload_files $YOLO_WEIGHTS/yolo*.weights *-log.txt chart*.png
  sleep 60

done

if [ -f "$LAST_WEIGHTS_FILE" ]; then

  sleep 5

  echo "INFO: Uploading last weights to Drive..."
  upload_files $YOLO_WEIGHTS/yolo*.weights *.log chart*.png
  
  echo "SUCCESS: Training complete. Final sync of weights to Drive done..."

else

  echo "WARNING: Training completed but final weights not found at $LAST_WEIGHTS_FILE"

fi


    

