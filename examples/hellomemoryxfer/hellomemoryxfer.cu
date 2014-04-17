#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>

float *hellocuda_gpu_interface (float *, float *, int);
float *hellocuda_mapped_interface (float *, float *, int);
float *hellocuda_pinned_interface (float *, float *, int); 
__global__ void hellocuda_kernel(float *x, float *y, float *ans, int num_floats);

int main(int argc, char **argv) {

  if(argc < 2) {
    printf("hellomemoryxfer [number of floats]\n");
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

  printf("Regular Elapsed Time: %f\n", runtime);
  free(res);
  
  cudaEventRecord(gpuStart,0); // Start event recording
  res = hellocuda_pinned_interface(x, y, count);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);
  
  printf("Pinned Elapsed Time: %f\n", runtime);
  free(res);
  
  cudaEventRecord(gpuStart,0); // Start event recording
  res = hellocuda_mapped_interface(x, y, count);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);
  
  printf("Mapped Elapsed Time: %f\n", runtime);

  free(res);

  return 0;
  
}

float *hellocuda_mapped_interface (float *x, float *y, int num_floats) {

  float *x_device;
  float *y_device;

  float *ans_device;
  float *ans_host = (float *) malloc (sizeof(float) * num_floats);

  cudaHostRegister( x, num_floats * sizeof(float), cudaHostRegisterMapped);
  cudaHostRegister( y, num_floats * sizeof(float), cudaHostRegisterMapped);
  cudaHostRegister( ans_host, num_floats * sizeof(float), cudaHostRegisterMapped);

  cudaHostGetDevicePointer ( (void **) &x_device, x, NULL);
  cudaHostGetDevicePointer ( (void **) &y_device, y, NULL);
  cudaHostGetDevicePointer ( (void **) &ans_device, ans_host, NULL);

  // Calculate the block size
  int threads_per_block = 192; // 1 SMX Per Block
  int num_blocks = (num_floats + (threads_per_block - (num_floats % threads_per_block))) / threads_per_block;

  // Execute our kernel
  hellocuda_kernel<<< num_blocks, threads_per_block >>>(x_device, y_device, ans_device, num_floats);

  cudaHostUnregister(x);
  cudaHostUnregister(y);
  cudaHostUnregister(ans_host);

  cudaFree(x_device);
  cudaFree(y_device);
  cudaFree(ans_device);

  return ans_host;
}

float *hellocuda_pinned_interface (float *x, float *y, int num_floats) {

  float *x_device;
  float *y_device;

  float *ans_device;
  float *ans_host = (float *) malloc (sizeof(float) * num_floats);

  // Allocate our device memory
  cudaMalloc( (void **) &x_device, num_floats * sizeof(float) );
  cudaMalloc( (void **) &y_device, num_floats * sizeof(float) );
  cudaMalloc( (void **) &ans_device, num_floats * sizeof(float) );

  cudaHostRegister( x, num_floats * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister( y, num_floats * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister( ans_host, num_floats * sizeof(float), cudaHostRegisterDefault);

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

  cudaHostUnregister(x);
  cudaHostUnregister(y);
  cudaHostUnregister(ans_host);

  cudaFree(x_device);
  cudaFree(y_device);
  cudaFree(ans_device);

  return ans_host;
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

