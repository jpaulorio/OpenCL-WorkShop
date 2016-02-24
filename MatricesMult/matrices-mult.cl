__kernel
void cl_matrices_mult(__global float* A, __global float* B, __global float* C,  int widthA, int widthB)
{
    //will range from 0 to 127
    int i = get_global_id(0);
    //will range from 0 to 127
    int j = get_global_id(1);

    // value stores the element that is computed by the thread
   	float value = 0;
   	for (int k = 0; k < widthA; ++k)
	{
	  value = value + A[k + j * widthA] * B[k * widthB + i];
	}
 
	C[i + widthA * j] = value;   	
 }
