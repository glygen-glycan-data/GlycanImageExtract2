#!/bin/sh
set -x
DATE=`date +%F`
MD5=`md5sum install.sh | awk '{print $1}'`
docker build --tag glyomics/darknet:$DATE --tag glyomics/darknet:latest --build-arg build_date=$DATE --build-arg install_md5=$MD5 .
docker push glyomics/darknet:$DATE
docker push glyomics/darknet:latest
