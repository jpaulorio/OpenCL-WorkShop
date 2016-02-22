#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef MATRIX_WIDTH
#define MATRIX_WIDTH 128
#endif

#ifndef MATRIX_HEIGHT
#define MATRIX_HEIGHT 128
#endif

int main(int argc, char* argv[]) {
  clock_t start, end;
  double cpu_time_used;
    
  float *in, *out;
  int matrixWidth = MATRIX_WIDTH;
  long matrixSize = MATRIX_WIDTH * MATRIX_HEIGHT;

  start = clock();

  cl_int status;
  cl_uint numPlatforms = 0;
  cl_platform_id *platforms = NULL;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id)); 
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);

  cl_uint numDevices = 0;
  cl_device_id *devices = NULL;
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id)); 
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

  cl_context context = NULL;
  context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);

  #define MAX_SOURCE_SIZE (0x100000)

  FILE *fp;

  const char fileName[] = "matrices-sum.cl";

  size_t source_size;

  char *source_str;

  fp = fopen(fileName, "r");

  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n"); exit(1);
  }

  source_str = (char *)malloc(MAX_SOURCE_SIZE); source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); fclose(fp);
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t *)&source_size, &status);
  status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

  if (status != CL_SUCCESS)
  {
    printf("clBuildProgram() Error %d: Failed to build program!\n", status);
    return 0;
  }

  cl_kernel kernel = NULL;
  kernel = clCreateKernel(program, "cl_matrices_sum", &status);

  if (status != CL_SUCCESS)
  {
    printf("clCreateKernel() Error %d: Failed to create kernel!\n", status);
    return 0;
  }

  float *matrixA, *matrixB, *matrixC;

  matrixA = (float*)malloc(matrixSize * sizeof(float));
  matrixB = (float*)malloc(matrixSize * sizeof(float));
  matrixC = (float*)malloc(matrixSize * sizeof(float));

  int i;
  for (i = 0; i < matrixSize; i++)
  {
    matrixA[i] = (float)i;
    matrixB[i] = (float)i * 2;    
  }

  cl_mem bufferMatrixA;
  cl_mem bufferMatrixB;
  cl_mem bufferMatrixC;

  bufferMatrixA = clCreateBuffer(context, CL_MEM_READ_ONLY, matrixSize * sizeof(float), NULL, &status);

  if (status != CL_SUCCESS)
  {
    printf("clCreateBuffer() Error %d: Failed to create bufferMatrixA!\n", status);
    return 0;
  }

  bufferMatrixB = clCreateBuffer(context, CL_MEM_READ_ONLY, matrixSize * sizeof(float), NULL, &status);

  if (status != CL_SUCCESS)
  {
    printf("clCreateBuffer() Error %d: Failed to create bufferMatrixB!\n", status);
    return 0;
  }

  bufferMatrixC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize * sizeof(float), NULL, &status);

  if (status != CL_SUCCESS)
  {
    printf("clCreateBuffer() Error %d: Failed to create bufferMatrixC!\n", status);
    return 0;
  }

  status = clEnqueueWriteBuffer(cmdQueue, bufferMatrixA, CL_TRUE, 0, matrixSize * sizeof(float), matrixA, 0, NULL, NULL);
  if (status != CL_SUCCESS)
  {
    printf("clEnqueueWriteBuffer() Error %d: Failed to write bufferMatrixA!\n", status);
    return 0;
  }

  status = clEnqueueWriteBuffer(cmdQueue, bufferMatrixB, CL_TRUE, 0, matrixSize * sizeof(float), matrixB, 0, NULL, NULL);
  if (status != CL_SUCCESS)
  {
    printf("clEnqueueWriteBuffer() Error %d: Failed to write bufferMatrixB!\n", status);
    return 0;
  }

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferMatrixA); 
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferMatrixB); 
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferMatrixC); 
  status |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&matrixWidth); 
 
  if (status != CL_SUCCESS)
  {
    printf("clSetKernelArg() Error %d: Failed to set kernel args!\n", status);
    return 0;
  }

  size_t globalWorkSize[2];
  globalWorkSize[0] = MATRIX_WIDTH;
  globalWorkSize[1] = MATRIX_HEIGHT;

  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if (status != CL_SUCCESS)
  {
    printf("clEnqueueNDRangeKernel() Error %d: Failed to execute kernel!\n", status);
    return 0;
  }

  status = clEnqueueReadBuffer(cmdQueue, bufferMatrixC, CL_TRUE, 0, matrixSize * sizeof(float), matrixC, 0, NULL, NULL);
  if (status != CL_SUCCESS)
  {
    printf("clEnqueueReadBuffer() Error %d: Failed to read buffer!\n", status);
    return 0;
  }

  for (i = 0; i < MATRIX_WIDTH * MATRIX_HEIGHT; i++)
  {
    printf("%.2f + %.2f = %.2f\n", matrixA[i], matrixB[i], matrixC[i]);
  }

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmdQueue);
  clReleaseMemObject(bufferMatrixA);
  clReleaseMemObject(bufferMatrixB);
  clReleaseMemObject(bufferMatrixC);
  clReleaseContext(context);
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(platforms);
  free(devices);

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; 
  printf("Execution time is: %0.3f ms \n", cpu_time_used * 1000);
}