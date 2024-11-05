#include <stdio.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

// Error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define N 500000

// Simple short kernels
__global__
void kernel_a(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

__global__
void kernel_c(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

int main() {

    cudaStream_t stream1;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream1);

    // Set up host data and initialize
    float* h_x = (float*) malloc(N * sizeof(float));
    float* h_y = (float*) malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_x[i] = float(i);
        h_y[i] = float(i);
    }

    // Print out the first 25 values of h_y
    for (int i = 0; i < 25; ++i) {
        printf("%2.0f ", h_y[i]);
    }
    printf("\n");

    // Set up device data
    float* d_x;
    float* d_y;
    float d_a = 5.0;
    cudaMalloc((void**) &d_x, N * sizeof(float));
    cudaMalloc((void**) &d_y, N * sizeof(float));
    cudaCheckErrors("cudaMalloc failed");

    cublasSetVector(N, sizeof(h_x[0]), h_x, 1, d_x, 1);
    cublasSetVector(N, sizeof(h_y[0]), h_y, 1, d_y, 1);
    cudaCheckErrors("cublasSetVector failed");

    // Set up graph
    cudaGraph_t graph;
    cudaGraph_t libraryGraph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t kernelNode1, kernelNode2, libraryNode;

    cudaKernelNodeParams kernelNode1Params {0};
    cudaKernelNodeParams kernelNode2Params {0};

    cudaGraphCreate(&graph, 0);
    cudaCheckErrors("cudaGraphCreate failure");

    // kernel_a and kernel_c use same args
    void *kernelArgs[2] = {(void *)&d_x, (void *)&d_y};

    int threads = 512;
    int blocks = (N + (threads - 1) / threads);

    // Adding 1st node, kernel_a
    kernelNode1Params.func = (void *)kernel_a;
    kernelNode1Params.gridDim = dim3(blocks, 1, 1);
    kernelNode1Params.blockDim = dim3(threads, 1, 1);
    kernelNode1Params.sharedMemBytes = 0;
    kernelNode1Params.kernelParams = (void **)kernelArgs;
    kernelNode1Params.extra = NULL;

    cudaGraphAddKernelNode(&kernelNode1, graph, NULL, 0, &kernelNode1Params);
    cudaCheckErrors("Adding kernelNode1 failed");

    nodeDependencies.push_back(kernelNode1);

    // Timing cublasSaxpy
    auto start = high_resolution_clock::now();

    // Adding 2nd node, libraryNode, with kernelNode1 as dependency
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    cudaCheckErrors("Stream capture begin failure");

    // Library call
    cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);
    cudaCheckErrors("cublasSaxpy failure");

    cudaStreamEndCapture(stream1, &libraryGraph);
    cudaCheckErrors("Stream capture end failure");

    auto end = high_resolution_clock::now();
    duration<double> saxpy_time = duration_cast<duration<double>>(end - start);
    std::cout << "cublasSaxpy Execution Time: " << saxpy_time.count() << " s" << std::endl;

    cudaGraphAddChildGraphNode(&libraryNode, graph, nodeDependencies.data(), nodeDependencies.size(), libraryGraph);
    cudaCheckErrors("Adding libraryNode failed");

    nodeDependencies.clear();
    nodeDependencies.push_back(libraryNode);

    // Adding 3rd node, kernel_c
    kernelNode2Params.func = (void *)kernel_c;
    kernelNode2Params.gridDim = dim3(blocks, 1, 1);
    kernelNode2Params.blockDim = dim3(threads, 1, 1);
    kernelNode2Params.sharedMemBytes = 0;
    kernelNode2Params.kernelParams = (void **)kernelArgs;
    kernelNode2Params.extra = NULL;

    cudaGraphAddKernelNode(&kernelNode2, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNode2Params);
    cudaCheckErrors("Adding kernelNode2 failed");

    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode2);

    cudaGraphNode_t *nodes = NULL;
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nodes, &numNodes);
    cudaCheckErrors("Graph get nodes failed");
    printf("Number of the nodes in the graph = %zu\n", numNodes);

    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaCheckErrors("Graph instantiation failed");

    // Timing the graph launch
    start = high_resolution_clock::now();

    // Launch the graph instance 100 times
    for (int i = 0; i < 100; ++i) {
        cudaGraphLaunch(instance, stream1);
        cudaStreamSynchronize(stream1);
    }
    cudaCheckErrors("Graph launch failed");

    end = high_resolution_clock::now();
    duration<double> graph_launch_time = duration_cast<duration<double>>(end - start);
    std::cout << "Graph Launch Execution Time (100 launches): " << graph_launch_time.count() << " s" << std::endl;

    cudaDeviceSynchronize();

    // Copy memory back to host
    cudaMemcpy(h_y, d_y, N, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Finishing memcpy failed");

    cudaDeviceSynchronize();

    // Print out the first 25 values of h_y
    for (int i = 0; i < 25; ++i) {
        printf("%2.0f ", h_y[i]);
    }
    printf("\n");

    // Clean up
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream1);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(libraryGraph);

    return 0;
}
