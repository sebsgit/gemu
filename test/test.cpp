extern void test_ptx();
extern void test_cuda();

#include "../arch/Device.h"

int main(){
	test_ptx();
	test_cuda();
	return 0;
}
