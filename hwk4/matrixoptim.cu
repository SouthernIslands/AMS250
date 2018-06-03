/**
* Base on example codes of CUDA Documentation
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
**/

#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A

 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{//index of blockrow and blockcol
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    //one dimension memory layout
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    if(row < BLOCK_SIZE && col < BLOCK_SIZE ){
        for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

            // Get sub-matrix Asub of A
            Matrix Asub = GetSubMatrix(A, blockRow, m);

            // Get sub-matrix Bsub of B
            Matrix Bsub = GetSubMatrix(B, m, blockCol);

            // Shared memory used to store Asub and Bsub respectively
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load Asub and Bsub from device memory to shared memory
            // Each thread loads one element of each sub-matrix
            As[row][col] = GetElement(Asub, row, col);
            Bs[row][col] = GetElement(Bsub, row, col);

            // Synchronize to make sure the sub-matrices are loaded
            // before starting the computation
            __syncthreads();
            // Multiply Asub and Bsub together
            for (int e = 0; e < BLOCK_SIZE; ++e)
                Cvalue += As[row][e] * Bs[e][col];

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write Csub to device memory
        // Each thread writes one element
        SetElement(Csub, row, col, Cvalue);
    }
}

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size_a = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size_a);
    cudaMemcpy(d_A.elements, A.elements, size_a, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; 
    d_B.height = B.height;
    size_t size_b = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size_b);
    cudaMemcpy(d_B.elements, B.elements, size_b, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; 
    d_C.height = C.height;
    size_t size_c = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size_c);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1)/ dimBlock.y);
    // dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size_c, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main(int argc, char **argv){
    //Initialize matrixs
    Matrix A;
    A.width = 32;
    A.height = 32;
    A.elements = (float*) malloc(A.width*A.height*sizeof(float));
    for(int i = 0 ; i < A.width*A.height ; i++ ){
        A.elements[i] = float(i%A.width);
    }
  
    Matrix B;
    B.width = 32;
    B.height = 32;
    B.elements = (float*) malloc(B.width*B.height*sizeof(float));
    for(int i = 0 ; i < B.width*B.height ; i++ ){
        B.elements[i] = float(i%B.width);
    }
  
    for(int i = 0 ; i < A.width*A.height ; i++ ){
          printf("%.1\t", A.elements[i]);
          if((i % A.width ) == A.width - 1){printf("\n");}
    }
    for(int i = 0 ; i < B.width*B.height ; i++ ){
          printf("%.1f\t", B.elements[i]);
          if((i % B.width ) == B.width - 1){printf("\n");}
    }
    printf("========================================================\n");
  
  
    Matrix C;
    C.width = 32;
    C.height = 32;
    C.elements = (float*) malloc(C.width*C.height*sizeof(float));
    for(int i = 0 ; i < C.width*C.height ; i++ ){
        C.elements[i] = float(i);
    }
  
    MatMul(A, B, C);
  
    for(int i = 0 ; i < C.width*C.height ; i++ ){
          printf("%.1f\t", C.elements[i]);
          if((i % C.width ) == C.width - 1){printf("\n");}
    }
  
    return 0;
  }
