extern void test_ptx();
extern void test_cuda();

#include "CudaApiLoader.h"
#include <iostream>

using namespace gemu;

int main(){
	test_ptx();
	test_cuda();

    try {

        CudaApiLoader::init("C:/Windows/System32/nvcuda.dll");
        cuInit__zz(0);
        CudaApiLoader::cleanup();

    } catch (const std::exception& exc) {
        std::cout << exc.what() << '\n';
    }

	return 0;
}
