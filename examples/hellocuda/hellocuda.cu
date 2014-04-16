#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>

float *hellocuda_gpu_interface (float *, float *, int);
__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats);

int main(char **argv, int argc) {

  float x[5] = { 1.1, 2.0, 4.2, 9.2, 4.3 };
  float y[5] = { 3.9, 9.1, 7.1, 5.5, 2.2 };

  float *res = hellocuda_gpu_interface(x, y, 5);

  for(int i = 0; i < 5; i++) {
    printf("%f * %f = %f\n", x[i], y[i], res[i]);
  }

  free(res);

  return 0;
  
}

float *hellocuda_gpu_interface (float *x, float *y, int num_floats) {

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
  int threads_per_block = 192; // 1 SMX Per Block
  int num_blocks = (num_floats + (threads_per_block - (num_floats % threads_per_block))) / threads_per_block;

  // Execute our kernel
  hellocuda_kernel<<< num_blocks, threads_per_block >>>(x_device, y_device, ans_device, num_floats);

  // Transfer our answer off the device
  cudaMemcpy(ans_host, ans_device, sizeof(float) * num_floats, cudaMemcpyDeviceToHost);

  return ans_host;
}

__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats) {

  register const uint32_t full_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (full_thread_id < num_floats) {
    ans[full_thread_id] = x[full_thread_id] * y[full_thread_id];
  }

}

