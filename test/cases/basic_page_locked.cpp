#include "test_base.h"

int main(){
	init_test();
	int * value;
	cu_assert(cuMemAllocHost((void **)&value, sizeof(int) * 3));
	*value = 10;
	*(value + 1) = 15;
	*(value + 2) = 20;
	std::cout << *value << *(value+1) << (*value+2) << '\n';
	cu_assert(cuMemFreeHost(value));
	return 0;
}
