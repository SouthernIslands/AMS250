/**
*Base on https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
*/
#include <stdio.h>
#include <cuda_runtime.h>

// print device properties
void showDevice(const cudaDeviceProp &prop)
{
    printf("Device Name : %s\n", prop.name);
    printf("Major revision number:     %d\n",  devProp.major);
    printf("Minor revision number:     %d\n",  devProp.minor);
    printf("Number of Stream MultiProcessor : %d.\n", prop.multiProcessorCount);
    printf("Memory Clock Rate (KHz) : %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits) : %d\n",prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("Total Global Memory : %d.\n", prop.totalGlobalMem);
    printf("Shared Memory Per Block : %d.\n", prop.sharedMemPerBlock);
    printf("Registers Per Block : %d.\n", prop.regsPerBlock);
    printf("Warp Size : %d.\n", prop.warpSize);
    printf("Max Threads Per Block : %d.\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total Const Memory : %d.\n", prop.totalConstMem);
    printf("Clock Rate : %d.\n", prop.clockRate);
    printf("Texture Alignment : %d.\n", prop.textureAlignment);
    printf("Device Overlap : %d.\n", prop.deviceOverlap);

}

bool initCUDA()
{
    int count;

    printf("CUDA Device Query...\n");
    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    printf("You now have %d CUDA devices.\n",count);

    // find the device >= 1.X
    int i;
    for (i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                showDevice(prop);
                break;
            }
        }
    }

    // if can't find the device
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    // set cuda device
    cudaSetDevice(i);

    return true;
}

int main(int argc, char const *argv[])
{
    if (initCUDA()) {
        printf("CUDA initialized.\n");
    }

    return 0;
}
