#!/bin/bash
# if the script fails - the below line helps terminate the script as soon as an error occurs
set -euo pipefail

# create log file if it doesnt exist - if it exists then empty the file if past logs exist
LOGFILE="$PWD/training_log.txt"
: > "$LOGFILE"

log_exit() {
  local code=$?
  # appends the log statement to a file as well outputs it on the terminal
  echo "$(date '+%Y-%m-%d %H:%M:%S') Script exited with code $code" | tee -a "$LOGFILE"
}

# This line makes sure log_exit is called no matter how the script exits (e.g., exit 1, error, or reaching the end)
trap log_exit EXIT

PY_ARGS=()
EXP=""

echo "Starting..." | tee -a "$LOGFILE"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --drive_folder_name)
            EXP="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./darknet.sh --drive_folder_name NAME [other options passed to Python]"
            echo ""
            echo "Required:"
            echo "  --drive_folder_name   Name of folder in Google Drive for training assets"
            echo ""
            echo "Optional Python args (passed to update_yolo_cfg.py):"
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

echo "Drive folder name: $EXP"
echo "Other args for Python script: ${PY_ARGS[*]}"

# Validate required arg
if [[ -z "$EXP" ]]; then
    echo "Error: --drive_folder_name is required" | tee -a "$LOGFILE"
    exit 1
fi


# Get the GPU's compute capability using nvidia-smi
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)

# Check if we found valid CUDA compute capability
if [ -z "$CUDA_ARCH" ]; then
    echo "Error: Unable to determine GPU compute capability. Exiting." | tee -a "$LOGFILE"
    exit 1
fi


echo -e "\n>> Starting YOLO Training Pipeline for: $EXP\n"


ROOT_DIR="$PWD"
EXPROOT="$ROOT_DIR/$EXP"
DRIVEROOT="my_drive:YOLO/$EXP"

# saving config files that YOLO requires to train locally as well on the drive.
# We will access the local copy for quick use
YOLO_DATA="$EXPROOT/data"   # folder where training data is unzipped
YOLO_WEIGHTS="$EXPROOT/weights"
DRIVE_WEIGHTS="$DRIVEROOT/weights"
CONFIG_FILENAME="yolov3_${EXP}.cfg"
YOLO_CONFIG="$EXPROOT/$CONFIG_FILENAME"
TRAIN_CONFIG="$EXPROOT/train.cfg"
TRAINING_FILE="$EXPROOT/train.txt"
VALIDATION_FILE="$EXPROOT/valid.txt"


mkdir -p "$EXPROOT"

echo "INFO: Downloading $EXP folder from Google Drive..." | tee -a "$LOGFILE"
echo ">> Current contents in the drive:"
./rclone/rclone --config ./rclone.conf ls "$DRIVEROOT"



# Check if the images.zip file exists on the drive before unzipping
if ! ./rclone/rclone --config ./rclone.conf lsf "$DRIVEROOT" | grep -q '^images.zip$'; then
  echo "Error: images.zip not found in $DRIVEROOT" | tee -a "$LOGFILE"
  exit 1
fi

if [ -d "$YOLO_DATA" ]; then
  echo "INFO: Unzipped training data already exists at $YOLO_DATA" | tee -a "$LOGFILE" 
  echo "INFO: Will delete the existing training data. Load a new copy of images.zip from drive and do a fresh unzip to avoid corrupted data..." | tee -a "$LOGFILE" 
  rm -rf "$YOLO_DATA"
fi

# get a new copy of images.zip from drive and unzip it locally every single time,
# to ensure that we do not use corrupted data (if present)
./rclone/rclone --config ./rclone.conf copy "$DRIVEROOT" "$EXPROOT" 

# Recreate a local folder and unzip images from drive
mkdir -p "$YOLO_DATA"
unzip -j "$EXPROOT/images.zip" -d "$YOLO_DATA"

# Move classes.txt out of YOLO_DATA
mv "$YOLO_DATA/classes.txt" "$EXPROOT"
cp "$EXPROOT/classes.txt" "$EXPROOT/classes.names"
cp "$EXPROOT/classes.txt" "$EXPROOT/yolov3_${EXP}.labels"

./rclone/rclone --config ./rclone.conf copy "$EXPROOT/classes.txt" "$DRIVEROOT"
./rclone/rclone --config ./rclone.conf copy "$EXPROOT/yolov3_${EXP}.labels" "$DRIVEROOT"


# YOLO_CLASSES - Count the number of non-empty lines in classes.txt
# This tells us how many classes are defined (ignoring any blank lines)
YOLO_CLASSES=$(grep -v '^\s*$' "$EXPROOT/classes.txt" | wc -l)
echo "INFO: Total number of classes: $YOLO_CLASSES" | tee -a "$LOGFILE"

# YOLO_CLASSES=$(grep -cv '^\s*$' "$EXPROOT/classes.txt")
YOLO_FILTERS=$(( (YOLO_CLASSES + 5) * 3 ))
echo "INFO: YOLO FILTERS: $YOLO_FILTERS" | tee -a "$LOGFILE"

TRAIN_CONFIG="$EXPROOT/train.cfg"
# echo -e "classes = $YOLO_CLASSES\ntrain = $EXPROOT/train.txt\nvalid = $EXPROOT/test.txt\nnames = $EXPROOT/classes.txt\nbackup = $YOLO_RUN" > "$TRAIN_CONFIG"
cat <<EOF > "$TRAIN_CONFIG"
classes = $YOLO_CLASSES
train = $TRAINING_FILE
valid = $VALIDATION_FILE
names = $EXPROOT/classes.names
backup = $YOLO_WEIGHTS
EOF

# create train.txt - which contains paths of all png images
# ls "$YOLO_DATA"/*.png > "$EXPROOT/train.txt"

# need to create training and validation sets
python3 split_data.py --image_dir $YOLO_DATA --train_txt $TRAINING_FILE --val_txt $VALIDATION_FILE --split_ratio 0.8


DARKNET_DIR="./darknet"
if [ ! -d "$DARKNET_DIR" ]; then
    echo ">> Cloning darknet..."
#     # git clone https://github.com/EdwardsLabProjects/darknet
    wget -O darknet.zip https://github.com/AlexeyAB/darknet/archive/6f3ba4422e5719a0fed1ff45045ebaa0c236d582.zip
    unzip darknet.zip
    rm darknet.zip

    # Rename the extracted folder to 'darknet'
    mv darknet-* $DARKNET_DIR
else
    echo ">> Darknet already exists. Skipping clone."
fi

# # Dynamically generate the ARCH flags based on the compute capability
ARCH_SETTING="ARCH= -gencode arch=compute_${CUDA_ARCH//./},code=[sm_${CUDA_ARCH//./},compute_${CUDA_ARCH//./}]"

# ARCH for logging purposes
echo "INFO: Detected GPU compute capability: $CUDA_ARCH" | tee -a "$LOGFILE"
echo "Setting ARCH to: $ARCH_SETTING"


# commands for editing the Makefile to enable GPU support and 
# set the appropriate compute architecture for compiling Darknet.
cd darknet
make clean
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i "s/ARCH= -gencode arch=compute_[0-9]*,code=\[sm_[0-9]*,compute_[0-9]*\]/$ARCH_SETTING/" Makefile
make

# get darknet initial weights
if [ ! -f ./darknet53.conv.74 ]; then
  echo "Downloading darknet53.conv.74..."
  # wget is not working for large files - sol: use gdown
  # wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1JHq95mL5RjuNscfqw5m4RpkdkYJ6Qdhz" -O ./darknet53.conv.74
  gdown "https://drive.google.com/uc?id=16i5nt2np-4cVw9NlQVL_UrYDBo5zLcgm"
else
  echo "darknet53.conv.74 already exists. Skipping download."
fi

# get the original yolo config everytime (-f flag ensures this behaviour)
cp -f cfg/yolov3.cfg "$YOLO_CONFIG"

# Apply changes to the yolo config file
# TO DO: change lines numbers - so than any yolo version can be used
# sed -i 's/batch=1/batch=64/' "$YOLO_CONFIG"
# sed -i 's/subdivisions=1/subdivisions=16/' "$YOLO_CONFIG"
# sed -i 's/max_batches = 500200/max_batches = 4000/' "$YOLO_CONFIG"
# # update classes
# sed -i "610 s@classes=[0-9]\+@classes=$YOLO_CLASSES@" "$YOLO_CONFIG"
# sed -i "696 s@classes=[0-9]\+@classes=$YOLO_CLASSES@" "$YOLO_CONFIG"
# sed -i "783 s@classes=[0-9]\+@classes=$YOLO_CLASSES@" "$YOLO_CONFIG"
# # update filters
# sed -i "603 s@filters=[0-9]\+@filters=$YOLO_FILTERS@" "$YOLO_CONFIG"
# sed -i "689 s@filters=[0-9]\+@filters=$YOLO_FILTERS@" "$YOLO_CONFIG"
# sed -i "776 s@filters=[0-9]\+@filters=$YOLO_FILTERS@" "$YOLO_CONFIG"

python3 ../update_yolo_cfg.py  --yolo_config $YOLO_CONFIG --classes $YOLO_CLASSES "${PY_ARGS[@]}"


cd ..

# copy/overwrite yolo config file to drive
./rclone/rclone --config ./rclone.conf copy "$YOLO_CONFIG" "$DRIVEROOT"


echo "INFO: Starting training..." | tee -a "$LOGFILE"

# ensures that your drive has a weights folder - if it doesnt already exist
./rclone/rclone --config ./rclone.conf mkdir "$DRIVE_WEIGHTS"
LAST_WEIGHTS_FILE="yolov3_${EXP}_last.weights"

rm -rf "$YOLO_WEIGHTS"
mkdir -p "$YOLO_WEIGHTS"

# Check if last_weights file already exists on drive - if true we will download 
# it to our local folder and use it to train...
# Else create a new folder on the drive to save weights from the fresh training...
if ! ./rclone/rclone --config ./rclone.conf lsf "$DRIVE_WEIGHTS" | grep -q "^${LAST_WEIGHTS_FILE}$"; then
    echo "INFO: Your drive doesnt contain any past weights files, hence new ones will be created" | tee -a "$LOGFILE"
    ./rclone/rclone --config ./rclone.conf mkdir "$DRIVE_WEIGHTS"
else
    echo "INFO: Found weights file: $LAST_WEIGHTS_FILE. Copying to local folder: $YOLO_WEIGHTS" | tee -a "$LOGFILE"
    ./rclone/rclone --config ./rclone.conf copy "$DRIVE_WEIGHTS/$LAST_WEIGHTS_FILE" "$YOLO_WEIGHTS"
fi


# if WEIGHTS_FOLDER exists on drive check if $YOLO_WEIGHTS/yolov3_${EXP}_last.weights exists
# and if true - download it to your local weights folder - then you can continue training over it
# else create a new WEIGHTS_FOLDER on your drive

# LAST_WEIGHTS_FILE="$YOLO_WEIGHTS/$LAST_WEIGHTS_FILE"


LAST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_last.weights"
FINAL_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_final.weights"
BEST_WEIGHTS_FILE="$YOLO_WEIGHTS/yolov3_${EXP}_best.weights"

# Start training (& - run in background)
# if last_weights file exists continue training else start new training
if [ -f "$LAST_WEIGHTS_FILE" ]; then
  echo ">> Using previous weights for training..." | tee -a "$LOGFILE"
  sleep 5
  ./darknet/darknet detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" "$LAST_WEIGHTS_FILE" -dont_show -map | tee -a "$LOGFILE" &
else
  echo ">> Starting fresh training..." | tee -a "$LOGFILE"
  sleep 5
  # mkdir -p "$YOLO_RUN"
  ls ./darknet
  # ./darknet/darknet detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" ./darknet/darknet53.conv.74 -dont_show &
  ./darknet/darknet detector train "$TRAIN_CONFIG" "$YOLO_CONFIG" ./darknet/darknet53.conv.74 -dont_show -map | tee -a "$LOGFILE" &

fi


# Get the PID of the training process
TRAIN_PID=$!


TRAINING_LOG="$PWD/training_log.txt"
CHART_PNG="$PWD/chart.png"

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
      ./rclone/rclone --config ./rclone.conf copyto \
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
      
      ./rclone/rclone --config ./rclone.conf copyto \
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


    

