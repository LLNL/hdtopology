#!/bin/bash
set -e -x

# Install dependent library
yum install -y ann-devel

wget http://www.cmake.org/files/v3.12/cmake-3.12.1.tar.gz
tar -xvzf cmake-3.12.1.tar.gz
cd cmake-3.12.1/
./configure
make
sudo make install
