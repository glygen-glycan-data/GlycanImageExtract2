#!/bin/sh

GD="$1"
DIR="$2"
if [ "$DIR" != "" ]; then
    mkdir -p "$DIR"
    DIR="$DIR/"
fi
NAME=`echo "$1" | sed 's/^.*\///'`
BASE=`echo "$1" | sed 's/\/[^/]*$//' | sed 's/^.*\///'`
BEST=`rclone ls "$GD" | fgrep _best.weights | awk '{print $2}'`
if [ "$BEST" != "" ]; then
    echo rclone copyto "$GD/$BEST" "$DIR$NAME.weights"
    rclone copyto "$GD/$BEST" "$DIR$NAME.weights"
fi
CFG=`rclone ls "$GD" | fgrep .cfg | awk '{print $2}'`
if [ "$CFG" != "" ]; then
    echo rclone copyto "$GD/$CFG" "$DIR$NAME.cfg"
    rclone copyto "$GD/$CFG" "$DIR$NAME.cfg"
fi
LAB=`rclone ls "$GD" | fgrep .labels | awk '{print $2}'`
if [ "$LAB" != "" ]; then
    echo rclone copyto "$GD/$LAB" "$DIR$NAME.labels"
    rclone copyto "$GD/$LAB" "$DIR$NAME.labels"
fi
MOD=`rclone ls "$GD" | fgrep .model | awk '{print $2}'`
if [ "$MOD" != "" ]; then
    echo rclone copyto "$GD/$MOD" "$DIR$NAME.model"
    rclone copyto "$GD/$MOD" "$DIR$NAME.model"
fi
cat <<EOF
[Finder:]
# $BASE
weights=$DIR$NAME.weights
config=$DIR$NAME.cfg
class=
EOF
