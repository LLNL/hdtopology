#!/bin/bash
set -e -x

# Install dependent library
yum install -y ann-devel

# Install cmake
yum install -y wget
wget http://www.cmake.org/files/v3.12/cmake-3.12.1.tar.gz
tar -xvzf cmake-3.12.1.tar.gz
cd cmake-3.12.1/
./configure
make
sudo make install

mkdir /io/build
cd /io/build
cmake ..
make
make install
