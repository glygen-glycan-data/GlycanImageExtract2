#!/bin/sh
DATE=`date +%F`
docker build --tag glyomics/darknet:$DATE --tag glyomics/darknet:latest --build-arg build_date=$DATE .
docker push glyomics/darknet:$DATE
docker push glyomics/darknet:latest
