#!/bin/bash
rm -f gemu.tgz
qmake
make distclean
qmake -config lib
cov-build --dir cov-int make
tar czvf gemu.tgz cov-int/
rm -rf cov-int
