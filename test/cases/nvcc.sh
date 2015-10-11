#!/bin/bash
nvcc -ptx $1 -o $1.ptx
if [[ $? -eq 0 ]]; then
	cat $1.ptx | awk '{print "\"" $0 "\""}'
	rm -f $1.ptx
fi
