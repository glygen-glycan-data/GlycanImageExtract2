#!/bin/bash
# if the script fails - the below line helps terminate the script as soon as an error occurs
set -euo pipefail
# set -x

SCRIPTURL="https://raw.githubusercontent.com/glygen-glycan-data/GlycanImageExtract2/refs/heads/dev/trainingSetup/"
DARKNET_SRC="https://github.com/AlexeyAB/darknet/archive/6f3ba4422e5719a0fed1ff45045ebaa0c236d582.zip"
RCLONE_VERSION="v1.69.1"

log_exit() {
  local code=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') Script exited with code $code"
}

# This line makes sure log_exit is called no matter how the script exits (e.g., exit 1, error, or reaching the end)
trap log_exit EXIT

download() {
  rm -f "$2"
  if [ -f "$1" ]; then
    cp -r "$1" "$2"
  else
    wget --no-check-certificate -O "$2" "$1";
    if [ ! -s "$2" ]; then
      echo "Download $1 failed..." 1>&2
      exit 1;
    fi
  fi
}

RCLONE=""
CLEAN="0"

while [ "$#" -gt 0 ]; do
    case $1 in
        --rclone_config)
            RCLONE="$2"
            shift 2
            ;;
        --clean)
            CLEAN="1"
            shift 1
            ;;
        -h|--help)
            echo "Usage: ./bootstrap.sh [ options ]"
            echo ""
            echo "Arguments:"
	    echo "  --rclone_config   File or URL of rclone config."
	    echo "  --clean           Remove existing downloads."
            exit 0
            ;;
        *)
            # If you want to handle positional args differently, do it here
            break
            ;;
    esac
done

BASE="$PWD"
SCRIPTS="$BASE/scripts"
RCLONE_CONF="$BASE/.rclone.conf"
RCLONE_DIR="$BASE/rclone"
RCLONE_BIN="$BASE/rclone/rclone"
RCLONE_VERSION="v1.69.1"
RCLONE_FULLVER="rclone-${RCLONE_VERSION}-linux-amd64"
DARKNET_DIR="$BASE/darknet"

if [ "$CLEAN" -eq 1 ]; then
  rm -rf "$RCLONE_CONF" "$RCLONE_DIR" "$SCRIPTS" "$DARKNET_DIR"
fi

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
  for f in darknet.sh darknetjs.sh update_yolo_cfg.py split_data.py bootstrap.sh; do
    download "$SCRIPTURL/$f" "$SCRIPTS/$f"
    case $f in
      *.sh) chmod +x $f ;;
    esac
  done
fi
if [ ! -d "$DARKNET_DIR" ]; then
    echo ">> Cloning darknet..."
    wget -O $BASE/darknet.zip "$DARKNET_SRC"
    unzip $BASE/darknet.zip -d $BASE
    rm -f $BASE/darknet.zip

    # Rename the extracted folder to 'darknet'
    mv $BASE/darknet-* $DARKNET_DIR
else
    echo ">> Darknet already exists. Skipping download."
fi

python3 -m pip install --user gdown

# Validate required args
if [ ! -f "$RCLONE_CONF" ]; then
    echo "Error: --rclone_config is required"
    exit 1
fi

