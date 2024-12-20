# User needs to specify a folder_name - where all SVG images will get populated

if ["$1" = ""]; then
    echo "Usage: generate_svg.sh <outputdir>"
    exit 1;
fi

if [-d "$1"]; then
    echo "Directory $1 already exists" 1>&2
    exit 1;
fi

mkdir "$1"
cd "$1"


VENV_PYTHON="/home/nmathias/GlycanImageExtract2/.venv/bin/python"

# Generate random SVG images (ensure the script exists in ../scripts)
$VENV_PYTHON ../scripts/randimgs.py 100 svg


