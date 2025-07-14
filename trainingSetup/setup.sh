#!/bin/sh

IP="$1"

if [ ! -f "rclone.conf" ]; then
  echo "Required file rclone.conf missing." 1>&2
  exit 1
fi
if [ ! -f "openrc.sh" ]; then
  echo "Required file openrc.sh missing." 1>&2
  exit 1
fi
if [ ! -n "$IP" ]; then
  echo "Required IP address missing." 1>&2
  exit 1
fi

scp rclone.conf exouser@${IP}:.rclone.conf
scp openrc.sh exouser@${IP}:.openrc.sh
cat bootstrap.sh | ssh exouser@${IP} /bin/sh
