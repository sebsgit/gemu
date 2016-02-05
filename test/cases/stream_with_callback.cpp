#include "test_base.h"

void 
#ifdef __WIN32
__attribute__((__stdcall__)) 
#endif
callback_1(CUstream hStream, CUresult status, void* userData){
	std::cout << "\ncallback 1: " << *(int*)userData << '\n';
}

int main(){
    init_test();
    CUstream stream;
    int userData = 23;
    std::cout << cuStreamCreate(&stream,CU_STREAM_DEFAULT) << '\n';
    std::cout << cuStreamAddCallback(stream, callback_1, &userData, 0);
    std::cout << cuStreamSynchronize(stream) << '\n';
    std::cout << cuStreamDestroy(stream) << '\n';
    return 0;
}
