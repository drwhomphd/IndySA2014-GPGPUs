#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <omp.h>

#include <cuda.h>

double *hellocuda_gpu_interface (double *, double *, unsigned long long);
double *hellocuda_cpu_interface (double *, double *, unsigned long long);
double *hellocuda_parallel_interface (double *, double *, unsigned long long);
__global__ void hellocuda_kernel(double *x, double *y, double *ans, unsigned long long num_doubles);

int main(int argc, char **argv) {

  if(argc < 2) {
    printf("hellomemoryxfer [number of doubles]\n");
    exit(-1);
  }

  // We'll use the first argument of the input for the size of our doubles.
  unsigned long long count = atoll(argv[1]);

  printf("Count: %llu\n", count);

  double *x = (double *) malloc(count * sizeof(double));
  double *y = (double *) malloc(count * sizeof(double));

  // We're going to generate count random numbers
  FILE *rng = fopen ("/dev/urandom", "r");

  fread(x, sizeof(double), count, rng);
  fread(y, sizeof(double), count, rng);
  
  // We use these for benchmark timing
  cudaEvent_t gpuStart, gpuStop;
  float runtime = 0;
  cudaEventCreate(&gpuStart);
  cudaEventCreate(&gpuStop);

  cudaEventRecord(gpuStart,0); // Start event recording
  double *res = hellocuda_gpu_interface(x, y, count);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);

  printf("GPU Elapsed Time: %fms\n", runtime);
  free(res);
  
  cudaEventRecord(gpuStart,0); // Start event recording
  res = hellocuda_cpu_interface(x, y, count);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);
  
  printf("CPU Elapsed Time: %fms\n", runtime);
  free(res);
  
  cudaEventRecord(gpuStart,0); // Start event recording
  res = hellocuda_parallel_interface(x, y, count);
  cudaEventRecord(gpuStop,0); //  Stop event recording
  cudaEventSynchronize(gpuStop);
  cudaEventElapsedTime( &runtime, gpuStart, gpuStop);
  
  printf("Parallel CPU Elapsed Time: %fms\n", runtime);

  free(x);
  free(y);
  free(res);

  return 0;
  
}

double *hellocuda_cpu_interface (double *x, double *y, unsigned long long num_doubles) {

  double *ans_host = (double *) malloc (sizeof(double) * num_doubles);

  for(unsigned long long i = 0; i < num_doubles; i++) {
    ans_host[i] = sqrt(x[i]) * sqrt(y[i]);
    double t1 = x[i] - y[i];
    double t2 = x[i] / y[i];
    ans_host[i] /= (cos(t1) * cos(t2));
  }


  return ans_host;
}

double *hellocuda_parallel_interface (double *x, double *y, unsigned long long num_doubles) {
  
  double *ans_host = (double *) malloc (sizeof(double) * num_doubles);

  #pragma omp parallel for
  for(unsigned long long i = 0; i < num_doubles; i++) {
    ans_host[i] = sqrt(x[i]) * sqrt(y[i]);
    double t1 = x[i] - y[i];
    double t2 = x[i] / y[i];
    ans_host[i] /= (cos(t1) * cos(t2));
  }

  return ans_host;

}

double *hellocuda_gpu_interface (double *x, double *y, unsigned long long num_doubles) {

  double *x_device;
  double *y_device;

  double *ans_device;
  double *ans_host = (double *) malloc (sizeof(double) * num_doubles);

  cudaHostRegister( x, num_doubles * sizeof(double), cudaHostRegisterMapped);
  cudaHostRegister( y, num_doubles * sizeof(double), cudaHostRegisterMapped);
  cudaHostRegister( ans_host, num_doubles * sizeof(double), cudaHostRegisterMapped);

  cudaHostGetDevicePointer ( (void **) &x_device, x, NULL);
  cudaHostGetDevicePointer ( (void **) &y_device, y, NULL);
  cudaHostGetDevicePointer ( (void **) &ans_device, ans_host, NULL);

  // Calculate the block size
  int threads_per_block = 128; // 1 SMX Per Block
  int num_blocks = (num_doubles + (threads_per_block - (num_doubles % threads_per_block))) / threads_per_block;

  // Execute our kernel
  hellocuda_kernel<<< num_blocks, threads_per_block >>>(x_device, y_device, ans_device, num_doubles);

  cudaHostUnregister(x);
  cudaHostUnregister(y);
  cudaHostUnregister(ans_host);

  cudaFree(x_device);
  cudaFree(y_device);
  cudaFree(ans_device);

  return ans_host;
}

__global__ void hellocuda_kernel(double *x, double *y, double *ans, unsigned long long num_doubles) {

  register const uint32_t full_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (full_thread_id < num_doubles) {
    ans[full_thread_id] = sqrt(x[full_thread_id]) * sqrt(y[full_thread_id]);
    double t1 = x[full_thread_id] - y[full_thread_id];
    double t2 = x[full_thread_id] / y[full_thread_id];
    ans[full_thread_id] /= (cos(t1) * cos(t2));
  }

}

