#include "test_base.h"

int main(){
    init_test();
    CUstream stream;
    std::cout << cuStreamCreate(&stream,CU_STREAM_DEFAULT) << '\n';
    std::cout << cuStreamSynchronize(stream) << '\n';
    std::cout << cuStreamDestroy(stream) << '\n';
    std::cout << cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING) << '\n';
    unsigned int flags;
    std::cout << cuStreamGetFlags(stream, &flags) << '\n';
    std::cout << flags << '\n';
    std::cout << cuStreamDestroy(stream) << '\n';
    return 0;
}
