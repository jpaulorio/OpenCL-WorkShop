__kernel
void cl_matrices_sum(__global float* A, __global float* B, __global float* C,  int matrixWidth)
{
     //will range from 0 to 127
     int i = get_global_id(0);
     //will range from 0 to 127
     int j = get_global_id(1);
     float value=0;
     
     C[i + matrixWidth * j] = A[i + matrixWidth * j] + B[i + matrixWidth * j];
 }
