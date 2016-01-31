#!/bin/bash
nvcc -ptx --gpu-architecture=sm_20 -O0 $1 -o $1.ptx
if [[ $? -eq 0 ]]; then
	cat $1.ptx | awk '{print "\"" $0 "\\n\""}'
	rm -f $1.ptx
fi
