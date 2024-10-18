#include <cooperative_groups.h>
#include <stdio.h>
#include <cstdlib>  // for rand()
#include <ctime>   // for time()
using namespace cooperative_groups;

const int nTPB = 256;

__device__ int reduce(thread_group g, int *x, int val) { 
  int lane = g.thread_rank();
  for (int i = g.size() / 2; i > 0; i /= 2) {
    x[lane] = val;       
    g.sync();
    if (lane < i) val += x[lane + i];  
    g.sync();
  }
  if (g.thread_rank() == 0) {
    return val;
  }
  return 0;
}

__global__ void my_reduce_kernel(int *data, int *total_sum, int *g1_counter, int *g2_counter, int *g3_counter) {
  __shared__ int sdata[nTPB];

  // Task 1a: create a proper thread block group below
  auto g1 = this_thread_block();
  size_t gindex = g1.group_index().x * nTPB + g1.thread_index().x;

  // Task 1b: create a proper 32-thread tile below, using group g1 created above
  auto g2 = tiled_partition(g1, 32);

  // Task 1c: create a proper 16-thread tile below, using group g2 created above
  auto g3 = tiled_partition(g2, 16);

  // g1 sum reduction
  int sdata_offset = (g1.thread_index().x / g1.size()) * g1.size();
  atomicAdd(&total_sum[0], reduce(g1, sdata + sdata_offset, data[gindex]));
  g1.sync();
  
  // g1 counter
  if (g1.thread_rank() == 0) atomicAdd(g1_counter, 1);

  // g2 sum reduction
  sdata_offset = (g1.thread_index().x / g2.size()) * g2.size();
  atomicAdd(&total_sum[1], reduce(g2, sdata + sdata_offset, data[gindex]));
  g2.sync();
  
  // g2 counter
  if (g2.thread_rank() == 0) atomicAdd(g2_counter, 1);

  // g3 sum reduction
  sdata_offset = (g1.thread_index().x / g3.size()) * g3.size();
  atomicAdd(&total_sum[2], reduce(g3, sdata + sdata_offset, data[gindex]));
  g3.sync();
  
  // g3 counter
  if (g3.thread_rank() == 0) atomicAdd(g3_counter, 1);
}

int main() {
  int *data, *total_sum, *g1_counter, *g2_counter, *g3_counter;
  cudaMallocManaged(&data, nTPB * sizeof(int));
  cudaMallocManaged(&total_sum, 3 * sizeof(int));
  cudaMallocManaged(&g1_counter, sizeof(int));
  cudaMallocManaged(&g2_counter, sizeof(int));
  cudaMallocManaged(&g3_counter, sizeof(int));

  // Initialize random data and counters
  std::srand(std::time(0));
  int host_sum = 0;
  for (int i = 0; i < nTPB; i++) {
    data[i] = std::rand() % 10;  // Random number between 0 and 9
    host_sum += data[i];         // Host sum for validation
  }
  
  // Initialize total_sum and counters
  for (int i = 0; i < 3; i++) total_sum[i] = 0;
  *g1_counter = 0;
  *g2_counter = 0;
  *g3_counter = 0;

  // Launch kernel
  my_reduce_kernel<<<1, nTPB>>>(data, total_sum, g1_counter, g2_counter, g3_counter);
  cudaError_t err = cudaDeviceSynchronize();

  // Output total sums and counters
  printf("Host sum: %d\n", host_sum);
  printf("Total sum 1 (g1): %d\n", total_sum[0]);
  printf("Total sum 2 (g2): %d\n", total_sum[1]);
  printf("Total sum 3 (g3): %d\n", total_sum[2]);

  // Output group counters
  printf("g1 executed %d times\n", *g1_counter);
  printf("g2 executed %d times\n", *g2_counter);
  printf("g3 executed %d times\n", *g3_counter);

  // Check if the results are correct
  bool pass = true;

  // Check if the sums are correct
  if (total_sum[0] != host_sum) {
    printf("Fail: g1 sum does not match host sum\n");
    pass = false;
  } 

  if (total_sum[1] != host_sum) {
    printf("Fail: g2 sum does not match host sum\n");
    pass = false;
  } 

  if (total_sum[2] != host_sum) {
    printf("Fail: g3 sum does not match host sum\n");
    pass = false;
  } 

  // Check if the group counters are correct
  if (*g1_counter != 1) {
    printf("Fail: g1 did not execute exactly 1 time\n");
    pass = false;
  } 

  if (*g2_counter != 8) {
    printf("Fail: g2 did not execute 8 times\n");
    pass = false;
  } 

  if (*g3_counter != 16) {
    printf("Fail: g3 did not execute 16 times\n");
    pass = false;
  } 

  if (pass) {
    printf("All tests passed!\n");
  } else {
    printf("Some tests failed!\n");
  }

  // Error handling
  if (err != cudaSuccess) 
    printf("cuda error: %s\n", cudaGetErrorString(err));

  // Free memory
  cudaFree(data);
  cudaFree(total_sum);
  cudaFree(g1_counter);
  cudaFree(g2_counter);
  cudaFree(g3_counter);

  return 0;
}
