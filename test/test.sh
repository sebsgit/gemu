#!/bin/bash
qmake -config lib && make
qmake && make
if [[ $# -eq 0 ]]; then
	./test
	if [[ $? -ne 0 ]]; then
		exit
	fi
fi
cd 'cases'
index=$((0))
CASES="."
if [[ $# -eq 1 ]]; then
	CASES=$@
fi
ls | grep cpp | grep $CASES | while read -r fname; do
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
	LD_PRELOAD=../cuda_lib/libcuda.so.1.0.0
	export LD_PRELOAD
	./$base_name > 2.out
	diff 1.out 2.out &>/dev/null
	diff_result=$?
	if [[ $base_name == "verify" ]]; then
		if [[ $diff_result -ne 1 ]]; then
			echo "gemu library not used!"
			exit
		fi
		diff_result=0
	fi
	if [[ $diff_result -ne 0 ]]; then
		echo " failed !"
		exit
	else
		rm 1.out &>/dev/null
		rm 2.out &>/dev/null
	fi
	rm $base_name &>/dev/null
done
