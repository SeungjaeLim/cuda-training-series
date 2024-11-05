#include <stdio.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>

using namespace std::chrono;  

// error checking macro
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
void kernel_a(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];
}

__global__
void kernel_b(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];
}

__global__
void kernel_c(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];
}

__global__
void kernel_d(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];
}

int main(){

    // Set up and create events
    cudaEvent_t event1;
    cudaEvent_t event2;

    cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
    cudaCheckErrors("Event1 creation failed");
    cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
    cudaCheckErrors("Event2 creation failed");

    // Set up and create streams
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i){
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Set up and initialize host data
    float* h_x = (float*) malloc(N * sizeof(float));
    float* h_y = (float*) malloc(N * sizeof(float));
    float* out_y = (float*) malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i){
        h_x[i] = (float)i;
        h_y[i] = (float)i;
    }

    // Set up device data
    float* d_x;
    float* d_y;
    cudaMalloc((void**) &d_x, N * sizeof(float));
    cudaMalloc((void**) &d_y, N * sizeof(float));
    cudaCheckErrors("cudaMalloc failed");
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy to device failed");

    // Graph execution setup
    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    int threads = 512;
    int blocks = (N + (threads - 1)) / threads;

    // Timing without graph
    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel a failed");

        cudaEventRecord(event1, streams[0]);
        cudaCheckErrors("Event1 record failed");

        cudaStreamWaitEvent(streams[0], event1);
        kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel b failed");

        cudaStreamWaitEvent(streams[1], event1);
        kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);
        cudaCheckErrors("Kernel c failed");

        cudaEventRecord(event2, streams[0]);
        cudaEventRecord(event2, streams[1]);
        cudaCheckErrors("Event2 record failed");

        cudaStreamWaitEvent(streams[0], event2);
        cudaStreamWaitEvent(streams[1], event2);
        kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel d failed");
    }

    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();
    duration<double> no_graph_time = duration_cast<duration<double>>(t2 - t1);
    std::cout << "No Graph Time: " << no_graph_time.count() << " s" << std::endl;

    cudaMemcpy(out_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy from device to host failed");

    std::cout << "First 10 elements of out_y (no graph):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << out_y[i] << " ";
    }
    std::cout << std::endl;

    // Reset d_y before graph execution
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Reset d_y before graph execution failed");

    // Create the graph for the first execution
    t1 = high_resolution_clock::now();
    if (!graphCreated) {
        cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);
        cudaCheckErrors("Stream begin capture failed");

        kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel a during capture failed");

        cudaEventRecord(event1, streams[0]);
        cudaCheckErrors("Event1 record failed");

        kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel b during capture failed");

        cudaStreamWaitEvent(streams[1], event1);
        cudaCheckErrors("Stream wait for event1 failed");

        kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);
        cudaCheckErrors("Kernel c during capture failed");

        cudaEventRecord(event2, streams[1]);
        cudaCheckErrors("Event2 record failed");

        cudaStreamWaitEvent(streams[0], event2);
        cudaCheckErrors("Stream wait for event2 failed");

        kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel d during capture failed");

        cudaStreamEndCapture(streams[0], &graph);
        cudaCheckErrors("Stream end capture failed");

        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        cudaCheckErrors("Graph instantiation failed");

        graphCreated = true;
    }

    // Timing with graph
    t1 = high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i){
        cudaGraphLaunch(instance, streams[0]);
        cudaCheckErrors("Graph launch failed");
    }
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    duration<double> graph_time = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Graph Execution Time: " << graph_time.count() << " s" << std::endl;
    
    cudaMemcpy(out_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Finishing memcpy failed");

    std::cout << "First 10 elements of out_y (with graph):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << out_y[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaGraphDestroy(graph);
    cudaFree(d_x);
    cudaFree(d_y);
    for (int i = 0; i < num_streams; ++i){
        cudaStreamDestroy(streams[i]);
    }
    free(h_x);
    free(h_y);
    free(out_y);

    return 0;
}
