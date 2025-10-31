#!/bin/bash
cd /src/darknet
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 package
dpkg -i *.deb
