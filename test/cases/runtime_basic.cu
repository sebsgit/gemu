#include <iostream>
#include <string>
#include <cuda.h>

using namespace std;

__global__ void kernel(int * out){
	*out = 5;
}

int main(){
	int * result;
	cudaMalloc(&result, sizeof(*result));
	kernel<<< dim3(1,1,1), dim3(1,1,1) >>> (result);
	int hostResult=0;
	cudaMemcpy(&hostResult, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(result);
	std::cout << hostResult << '\n';
	return 0;
}
