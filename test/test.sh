#!/bin/bash
if [[ ! -f test ]]; then
	make
fi
if [[ ! -f cuda_lib/libcuda.so.1 ]]; then
	make lib
fi
./test
if [[ $? -ne 0 ]]; then
	exit
fi
cd 'cases'
index=$((0))
ls | grep cpp | while read -r fname; do
	rm 1.out &>/dev/null
	rm 2.out &>/dev/null
	LD_PRELOAD=
	export LD_PRELOAD
	base_name=$(echo "$fname" | sed s/.cpp//)
	echo "$index - $base_name ..."
	index=$(($index+1))
	g++ -std=c++11 $fname -o $base_name -lcuda -pthread
	./$base_name > 1.out
	if [[ $? -ne 0 ]]; then
		exit;
	fi
	LD_PRELOAD=../cuda_lib/libcuda.so.1
	export LD_PRELOAD
	./$base_name > 2.out
	diff 1.out 2.out &>/dev/null
	if [[ $? -ne 0 ]]; then
		echo " failed !"
	else
		rm 1.out &>/dev/null
		rm 2.out &>/dev/null
	fi
	rm $base_name &>/dev/null
done
