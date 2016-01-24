#!/bin/bash
rm -f gemu.tgz
make distclean
cov-build --dir cov-int make lib
tar czvf gemu.tgz cov-int/
rm -rf cov-int
