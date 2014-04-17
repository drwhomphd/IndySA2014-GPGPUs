#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>

float *hellocuda_gpu_interface (float *, float *, int);
__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats);

int main(int argc, char **argv) {

  if(argc < 2) {
    printf("hellomemoryxfer [number of floats]\n");
    exit(-1);
  }

  printf("This example will fail if you do not have a Kepler-based GPU or better.\n");

  // We'll use the first argument of the input for the size of our floats.
  int count = atoi(argv[1]);

  printf("Count: %d\n", count);

  float *x, *y;
  cudaMallocManaged(&x, count * sizeof(float));
  cudaMallocManaged(&y, count * sizeof(float));

  // We're going to generate count random numbers
  FILE *rng = fopen ("/dev/urandom", "r");
  fread(x, sizeof(float), count, rng);
  fread(y, sizeof(float), count, rng);
  
  // We use these for benchmark timing
  cudaEvent_t gpuStart, gpuStop;
  float runtime = 0;
  cudaEventCreate(&gpuStart);
  cudaEventCreate(&gpuStop);

  cudaEventRecord(gpuStart,0); // Start event recording
  float *res = hellocuda_gpu_interface(x, y, count);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);

  printf("Managed Elapsed Time: %f\n", runtime);

  cudaFree(res);
  cudaFree(x);
  cudaFree(y);

  return 0;
  
}

float *hellocuda_gpu_interface (float *x, float *y, int num_floats) {

  float *ans_host;
  cudaMallocManaged(&ans_host, sizeof(float) * num_floats);

  // Calculate the block size
  int threads_per_block = 192; // 1 SMX Per Block
  int num_blocks = (num_floats + (threads_per_block - (num_floats % threads_per_block))) / threads_per_block;

  // Execute our kernel
  hellocuda_kernel<<< num_blocks, threads_per_block >>>(x, y, ans_host, num_floats);

  return ans_host;
}

__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats) {

  register const uint32_t full_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (full_thread_id < num_floats) {
    ans[full_thread_id] = x[full_thread_id] * y[full_thread_id];
  }

}

