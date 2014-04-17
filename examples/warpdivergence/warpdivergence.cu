#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>

float *hellocuda_gpu_interface (float *, float *, int);
__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats);
__global__ void divergentcuda_kernel(float *x, float *y, float *ans, int num_floats);

int main(int argc, char **argv) {

  if(argc < 2) {
    printf("warpdivergence [number of floats]\n");
    exit(-1);
  }

  // We'll use the first argument of the input for the size of our floats.
  int count = atoi(argv[1]);

  printf("Count: %d\n", count);
  
  float *x = (float *) malloc(count * sizeof(float));
  float *y = (float *) malloc(count * sizeof(float));

  // We're going to generate count random numbers
  FILE *rng = fopen ("/dev/urandom", "r");
  fread(x, sizeof(float), count, rng);
  fread(y, sizeof(float), count, rng);
  
  float *res = hellocuda_gpu_interface(x, y, count);
  
  if(count < 100) {
    for(int i = 0; i < count; i++) {
      printf("%f ? %f = %f\n", x[i], y[i], res[i]);
    }
  }

  free(res);

  return 0;
  
}

float *hellocuda_gpu_interface (float *x, float *y, int num_floats) {
  
  // We use these for benchmark timing
  cudaEvent_t gpuStart, gpuStop;
  float runtime = 0;
  cudaEventCreate(&gpuStart);
  cudaEventCreate(&gpuStop);

  float *x_device;
  float *y_device;

  float *ans_device;
  float *ans_host = (float *) malloc (sizeof(float) * num_floats);

  // Allocate our device memory
  cudaMalloc( (void **) &x_device, num_floats * sizeof(float) );
  cudaMalloc( (void **) &y_device, num_floats * sizeof(float) );
  cudaMalloc( (void **) &ans_device, num_floats * sizeof(float) );

  // Transfer our buffer to the device
  cudaMemcpy(x_device, x, sizeof(float) * num_floats, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * num_floats, cudaMemcpyHostToDevice);

  // Calculate the block size
  int threads_per_block = 128; // 1 SMX Per Block
  int num_blocks = (num_floats + (threads_per_block - (num_floats % threads_per_block))) / threads_per_block;

  cudaEventRecord(gpuStart,0); // Start event recording
  // Execute our kernel
  hellocuda_kernel<<< num_blocks, threads_per_block >>>(x_device, y_device, ans_device, num_floats);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);
  printf("Regular Elapsed Time: %f\n", runtime);
  
  cudaEventRecord(gpuStart,0); // Start event recording
  // Execute our kernel
  divergentcuda_kernel<<< num_blocks, threads_per_block >>>(x_device, y_device, ans_device, num_floats);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);
  printf("Divergent Elapsed Time: %f\n", runtime);


  // Transfer our answer off the device
  cudaMemcpy(ans_host, ans_device, sizeof(float) * num_floats, cudaMemcpyDeviceToHost);

  cudaFree(x_device);
  cudaFree(y_device);
  cudaFree(ans_device);

  return ans_host;
}

__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats) {

  register const uint32_t full_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (full_thread_id < num_floats) {
    ans[full_thread_id] = x[full_thread_id] * y[full_thread_id];
  }

}

/* Right now this kernel is actually FASTER by a factor of 4(!) than the hellocuda kernel */
__global__ void divergentcuda_kernel(float *x, float *y, float *ans, int num_floats) {
  
  register const uint32_t full_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (full_thread_id < num_floats) {
      ans[full_thread_id] = x[full_thread_id] * y[full_thread_id];

      if(isinf(ans[full_thread_id]) || isnan(ans[full_thread_id]))
        ans[full_thread_id] = x[full_thread_id] / y[full_thread_id]; 
      else
        ans[full_thread_id] = x[full_thread_id] + y[full_thread_id];

  }

}

