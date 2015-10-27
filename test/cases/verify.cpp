#include "test_base.h"

int main(){
	init_test();
	CUdevice devId = 0;
	cu_assert(cuDeviceGet(&devId, 0));
	char name[512];
	cu_assert(cuDeviceGetName(name,512,devId));
	std::cout << name << '\n';
	return 0;
}
