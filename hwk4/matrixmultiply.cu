#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

__global__
void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size_a = A.width * A.height * sizeof(float);
    cudaMalloc((void **)&d_A.elements, size_a);
    cudaMemcpy(d_A.elements, A.elements, size_a, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size_t size_b = B.width * B.height * sizeof(float);
    cudaMalloc((void **)&d_B.elements, size_b);
    cudaMemcpy(d_B.elements, B.elements, size_b, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size_t size_c = C.width * C.height * sizeof(float);
    cudaMalloc((void **)&d_C.elements, size_c);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1)/ dimBlock.y);
  //  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
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
  A.width = 3;
  A.height = 5;
  A.elements = (float*) malloc(A.width*A.height*sizeof(float));
  for(int i = 0 ; i < A.width*A.height ; i++ ){
      A.elements[i] = float(i);
  }

  Matrix B;
  B.width = 5;
  B.height = 3;
  B.elements = (float*) malloc(B.width*B.height*sizeof(float));
  for(int i = 0 ; i < B.width*B.height ; i++ ){
      B.elements[i] = float(i);
  }

  for(int i = 0 ; i < A.width*A.height ; i++ ){
        printf("%f\t", A.elements[i]);
        if((i % A.width ) == A.width - 1){printf("\n");}
  }
  for(int i = 0 ; i < B.width*B.height ; i++ ){
        printf("%f\t", B.elements[i]);
        if((i % B.width ) == B.width - 1){printf("\n");}
  }
  printf("=========================================\n");


  Matrix C;
  C.width = 5;
  C.height = 5;
  C.elements = (float*) malloc(C.width*C.height*sizeof(float));
  for(int i = 0 ; i < C.width*C.height ; i++ ){
      C.elements[i] = float(i);
  }

  MatMul(A, B, C);

  for(int i = 0 ; i < C.width*C.height ; i++ ){
        printf("%5f\t", C.elements[i]);
        if((i % C.width ) == C.width - 1){printf("\n");}
  }

  return 0;
}
