#!/bin/bash
# if the script fails - the below line helps terminate the script as soon as an error occurs
set -euo pipefail
set -x

log_exit() {
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
    https://drive.google.com/*) gdown "$1" "$2";;
    http*) wget --no-check-certificate -O "$2" "$1";;
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

IMAGES=""
RESULTS=""
RCLONE=""
NAME=""
PY_ARGS=()

echo "Starting..." | tee -a "$LOGFILE"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --image_zipfile)
            IMAGES="$2"
            shift 2
            ;;
        --result_folder)
            RESULTS="$2"
            shift 2
            ;;
        --job_name)
            NAME="$2"
            shift 2
            ;;
        --rclone_config)
            RCLONE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./darknet.sh --drive_folder_name NAME \[other options passed to Python\]"
            echo ""
            echo "Required:"
            echo "  --image_zipfile       URL/rclone location of images.zip"
            echo "  --result_folder       rclone location for results/weights"
            echo "  --job_name            job name for this training run"
	    echo "  --rclone_config       URL location of rclone configuration \(optional\)"
            echo ""
            echo "Optional Python args \(passed to update_yolo_cfg.py\):"
            echo "  --batch               Batch size for YOLO config"
            echo "  --subdivisions        Subdivisions for YOLO config"
            echo "  --height              Input image height"
            echo "  --width               Input image width"
            echo "  --learning_rate       Learning rate for YOLO"
            exit 0
            ;;
        --*)
            PY_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            # If you want to handle positional args differently, do it here
            shift
            ;;
    esac
done

SCRIPTS="$HOME/scripts"
SCRIPTURL="https://raw.githubusercontent.com/glygen-glycan-data/GlycanImageExtract2/refs/heads/dev/trainingSetup/"
RCLONE_CONF="$HOME/.rclone.conf"
RCLONE_DIR="$HOME/rclone"
RCLONE_BIN="$HOME/rclone/rclone"
RCLONE_VERSION="v1.69.1"
RCLONE_FULLVER="rclone-${RCLONE_VERSION}-linux-amd64"

if [ -n "$RCLONE" -a ! -f "$RCLONE_CONF" ]; then
    download "$RCLONE" "$RCLONE_CONF"
fi
if [ ! -d "$RCLONE_DIR" ]; then
    download "https://downloads.rclone.org/${RCLONE_VERSION}/${RCLONE_FULLVER}.zip" "${RCLONE_FULLVER}.zip"
    unzip "${RCLONE_FULLVER}.zip"
    mv -f "${RCLONE_FULLVER}" "${RCLONE_DIR}"
    rm -f "${RCLONE_FULLVER}.zip"
fi
if [ ! -d "$SCRIPTS" ]; then
  mkdir -p "$SCRIPTS"
  for f in darknet.sh update_yolo_cfg.py split_data.py; do
    download "$SCRIPTURL/$f" "$SCRIPTS/$f"
  done
fi

# Validate required args
if [ -z "$IMAGES" ]; then
    echo "Error: --image_zipfile is required" | tee -a "$LOGFILE"
    exit 1
fi
if [ -z "$RESULTS" ]; then
    echo "Error: --result_folder is required" | tee -a "$LOGFILE"
    exit 1
fi
if [ -z "$NAME" ]; then
    echo "Error: --job_name is required" | tee -a "$LOGFILE"
    exit 1
fi
if [ -d "$NAME" ]; then
    echo "Error: --job_name has already been used locally." | tee -a "$LOGFILE"
    exit 1
fi
if exists "$RESULTS/$NAME" ; then
    echo "Error: --job_name has already been used in $RESULTS." | tee -a "$LOGFILE"
    exit 1
fi
if [ ! -f "$RCLONE_CONF" ]; then
    echo "Error: --rclone_config is required" | tee -a "$LOGFILE"
    exit 1
fi

echo "Image zipfile location: $IMAGES"
echo "Result folder: $RESULTS"
echo "Job name: $NAME"
echo "Other args for Python script: ${PY_ARGS[*]}"

EXP="$NAME"
EXPROOT="$HOME/$NAME"
DRIVEROOT="$RESULTS/$NAME"

# saving config files that YOLO requires to train locally as well on the drive.
# We will access the local copy for quick use
YOLO_DATA="data"   # folder where training data is unzipped
YOLO_WEIGHTS="weights"
DRIVE_WEIGHTS="$DRIVEROOT/weights"
YOLO_CONFIG="yolov3_${EXP}.cfg"
TRAIN_CONFIG="train.data"
TRAINING_FILE="train.txt"
VALIDATION_FILE="valid.txt"
TRAIN_LOG="train.log"
RCLONE_LOG="rclone.log"

mkdir -p "$EXPROOT"
cd "$EXPROOT"

echo "INFO: Download $IMAGES from Google Drive..." | tee -a "$LOGFILE"
download "$IMAGES" "images.zip"
mkdir -p "$YOLO_DATA"
unzip -j "images.zip" -d "$YOLO_DATA"

# Move classes.txt out of YOLO_DATA
mv "$YOLO_DATA/classes.txt" .
cp "classes.txt" "classes.names"
cp "classes.txt" "yolov3_${EXP}.labels"

upload "$EXPROOT/classes.txt" "$DRIVEROOT/classes.txt"
upload "$EXPROOT/yolov3_${EXP}.labels" "$DRIVEROOT/yolov3_${EXP}.labels"

# YOLO_CLASSES - Count the number of non-empty lines in classes.txt
# This tells us how many classes are defined (ignoring any blank lines)
YOLO_CLASSES=$(grep -v '^\s*$' "classes.txt" | wc -l)
echo "INFO: Total number of classes: $YOLO_CLASSES" | tee -a "$LOGFILE"

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
python3 $HOME/split_data.py --image_dir $YOLO_DATA --train_txt $TRAINING_FILE --val_txt $VALIDATION_FILE --split_ratio 0.8

DARKNET_DIR="$HOME/darknet"
if [ ! -d "$DARKNET_DIR" ]; then
    echo ">> Cloning darknet..."
    wget -O $HOME/darknet.zip https://github.com/AlexeyAB/darknet/archive/6f3ba4422e5719a0fed1ff45045ebaa0c236d582.zip
    unzip $HOME/darknet.zip -d $HOME
    rm -f $HOME/darknet.zip

    # Rename the extracted folder to 'darknet'
    mv $HOME/darknet-* $DARKNET_DIR
else
    echo ">> Darknet already exists. Skipping clone."
fi

# get darknet initial weights
if [ ! -f ./darknet53.conv.74 ]; then
  echo "Downloading darknet53.conv.74..."
  gdown "https://drive.google.com/uc?id=16i5nt2np-4cVw9NlQVL_UrYDBo5zLcgm"
else
  echo "darknet53.conv.74 already exists. Skipping download."
fi

# get the original yolo config everytime (-f flag ensures this behaviour)
cp -f $HOME/darknet/cfg/yolov3.cfg "$YOLO_CONFIG"

python3 $HOME/update_yolo_cfg.py  --yolo_config $YOLO_CONFIG --classes $YOLO_CLASSES "${PY_ARGS[@]}"

# copy/overwrite yolo config file to drive
upload "$YOLO_CONFIG" "$DRIVEROOT/$YOLO_CONFIG"

echo "INFO: Starting training..." | tee -a "$LOGFILE"

# ensures that your drive has a weights folder - if it doesnt already exist
rclone mkdir "$DRIVE_WEIGHTS"
LAST_WEIGHTS_FILE="yolov3_${EXP}_last.weights"

rm -rf "$YOLO_WEIGHTS"
mkdir -p "$YOLO_WEIGHTS"

# Check if last_weights file already exists on drive - if true we will download 
# it to our local folder and use it to train...
# Else create a new folder on the drive to save weights from the fresh training...
if ! rclone lsf "$DRIVE_WEIGHTS" | grep -q "^${LAST_WEIGHTS_FILE}$"; then
    echo "INFO: Your drive doesnt contain any past weights files, hence new ones will be created" | tee -a "$LOGFILE"
    rclone mkdir "$DRIVE_WEIGHTS"
else
    echo "INFO: Found weights file: $LAST_WEIGHTS_FILE. Copying to local folder: $YOLO_WEIGHTS" | tee -a "$LOGFILE"
    rclone copy "$DRIVE_WEIGHTS/$LAST_WEIGHTS_FILE" "$YOLO_WEIGHTS"
fi

LAST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_last.weights"
FINAL_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_final.weights"
BEST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_best.weights"

DARKNET="sudo docker run -it --gpus all -v .:/src sherensberk/darknet:2204.550.1241-devel darknet"

# Start training (& - run in background)
# if last_weights file exists continue training else start new training
if [ -f "$LAST_WEIGHTS_FILE" ]; then
  echo ">> Using previous weights for training..." | tee -a "$LOGFILE"
  sleep 5
  $DARKNET detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" "$LAST_WEIGHTS_FILE" -dont_show -map | tee -a "$LOGFILE" &
else
  echo ">> Starting fresh training..." | tee -a "$LOGFILE"
  sleep 5
  $DARKNET detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" ./darknet53.conv.74 -dont_show -map | tee -a "$LOGFILE" &
fi


# Get the PID of the training process
TRAIN_PID=$!


TRAINING_LOG="$PWD/training_log.txt"
CHART_PNG="$EXPROOT/chart.png"

echo ">> Monitoring weights and uploading to Drive..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  for FILE in "$LAST_WEIGHTS_FILE" "$FINAL_WEIGHTS_FILE" "$BEST_WEIGHTS_FILE" "$TRAINING_LOG" "$CHART_PNG"; do
    if [ -f "$FILE" ]; then
      BASENAME=$(basename "$FILE")
      echo "INFO: Found $BASENAME. Uploading to Drive..." | tee -a "$LOGFILE"

      # Make a temp copy to avoid errors from writing in progress
      TMP_COPY="/tmp/rclone_upload_temp"
      cp "$FILE" "$TMP_COPY"

      # Upload using the original filename by specifying the full destination path
      rclone copyto \
        --update --verbose --progress \
        --log-file="$LOGFILE" \
        "$TMP_COPY" "$DRIVE_WEIGHTS/$BASENAME"
      rm -f "$TMP_COPY"
    else
      echo "INFO: $FILE not found yet..." | tee -a "$LOGFILE"
    fi
  done
  sleep 60
done

# LAST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_last.weights"
# FINAL_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_final.weights"
# BEST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_best.weights"


# After training is complete - upload the last set of updated files for safety
if [ -f "$LAST_WEIGHTS_FILE" ]; then
  sleep 5
  echo "INFO: Uploading last weights to Drive..." | tee -a "$LOGFILE"
  
  for FILE in "$LAST_WEIGHTS_FILE" "$FINAL_WEIGHTS_FILE" "$BEST_WEIGHTS_FILE" "$TRAINING_LOG" "$CHART_PNG"; do
    if [ -f "$FILE" ]; then
      BASENAME=$(basename "$FILE")
      TMP_COPY="/tmp/rclone_upload_temp"
      cp "$FILE" "$TMP_COPY"
      
      rclone copyto \
        --update --verbose --progress \
        --log-file="$LOGFILE" \
        "$TMP_COPY" "$DRIVE_WEIGHTS/$BASENAME"
      
      rm -f "$TMP_COPY"
    else
      echo "INFO: $FILE not found, skipping upload..." | tee -a "$LOGFILE"
    fi
  done
  
  echo "SUCCESS: Training complete. Final sync of weights to Drive done..." | tee -a "$LOGFILE"
else
  echo "WARNING: Training completed but final weights not found at $LAST_WEIGHTS_FILE" | tee -a "$LOGFILE"
fi


    

