#!/bin/bash

if [ ! -d "$HOME/menoh/lib" ]; then
    git clone https://github.com/pfnet-research/menoh.git
    cd menoh && mkdir -p build && cd build
    if [ "$TRAVIS_OS_NAME" == "osx" ]; then
      cmake ..
    else
      CC=gcc-7 CXX=g++-7 cmake -DMKLDNN_INCLUDE_DIR="$HOME/mkl-dnn/include" \
      -DMKLDNN_LIBRARY="$HOME/mkl-dnn/lib/libmkldnn.so" \
      -DCMAKE_INSTALL_PREFIX="$HOME/menoh" \
      ..
    fi
    make
    make install
    cd ../../
else
    echo "Using cached directory."
fi
