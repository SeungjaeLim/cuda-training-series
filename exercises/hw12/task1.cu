#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

// Thread block size
#define BLOCK_SIZE 32

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e){
            Cvalue += As[row][e] * Bs[e][col];}

    }
    SetElement(Csub, row, col, Cvalue);
}

void MatMulHost(const Matrix A, const Matrix B, Matrix C) {
    for (int i = 0; i < A.height; ++i) {
        for (int j = 0; j < B.width; ++j) {
            float sum = 0;
            for (int k = 0; k < A.width; ++k) {
                sum += A.elements[i * A.stride + k] * B.elements[k * B.stride + j];
            }
            C.elements[i * C.stride + j] = sum;
        }
    }
}

int main() {
    const int num_m = 3;
    const int side_dim = 128;
    Matrix *m = new Matrix[num_m];
    std::mt19937 rng(time(0));
    std::uniform_int_distribution<int> dist(1, side_dim);

    for (int i = 0; i < num_m; i++) {
        m[i].width = m[i].height = m[i].stride = side_dim;
        m[i].elements = new float[side_dim * side_dim];
        if (i < 2)
            for (int j = 0; j < side_dim * side_dim; j++)
                m[i].elements[j] = static_cast<float>(dist(rng));
    }

    MatMul(m[0], m[1], m[2]);
    Matrix host_C;
    host_C.width = host_C.height = host_C.stride = side_dim;
    host_C.elements = new float[side_dim * side_dim];
    MatMulHost(m[0], m[1], host_C);

    bool match = true;
    for (int i = 0; i < side_dim * side_dim; i++) {
        if (fabs(m[2].elements[i] - host_C.elements[i]) > 1e-4) {
            std::cout << "Mismatch at index: " << i
                      << " expected: " << host_C.elements[i]
                      << " got: " << m[2].elements[i] << std::endl;
            match = false;
            break;
        }
    }
    if (match) std::cout << "Success!" << std::endl;

    for (int i = 0; i < num_m; i++)
        delete[] m[i].elements;
    delete[] m;
    delete[] host_C.elements;

    return 0;
}
