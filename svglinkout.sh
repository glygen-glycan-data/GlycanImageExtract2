# Get the current working directory
CURRENT_DIR=$(pwd)
OUTPUT_FOLDER="$CURRENT_DIR/glycan_images"  # Specify your desired output folder

VENV_PYTHON="$CURRENT_DIR/.venv/bin/python"

# Create the output folder if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Generate random SVG images
PYTHON_FILE="$CURRENT_DIR/randimgs.py"

$VENV_PYTHON $PYTHON_FILE 10 svg $OUTPUT_FOLDER
