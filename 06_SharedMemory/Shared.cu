#include <iostream>


__global__ void FindMax(int *d)
{
  __shared__ int s[9];
  
  s[threadIdx.x + 3*threadIdx.y] = 0;

  for (int i=0; i < 4; i++)
  {
    for (int j=0; j<4;j++)
    {
      int index = i+(4*threadIdx.x) + (j+(4*threadIdx.y))*12+144*blockIdx.y;
      if (d[index] > s[threadIdx.x + 3*threadIdx.y]){
        s[threadIdx.x + 3*threadIdx.y] = d[index];
      }

    }
  }

  __syncthreads();
  int max = 0;
  for (int n=0; n<9; n++)
  {
    if (s[n] > max){
      max = s[n];
    }
  }

  for (int i=0; i < 4; i++)
  {
    for (int j=0; j<4;j++)
    {
      int index = i+(4*threadIdx.x) + (j+(4*threadIdx.y))*12+144*blockIdx.y;
      d[index] = max;
    }
  }

}

int main(void)
{
  /*
  Shared memory allows r/w to an array that is accessible to all threads in a block.
  This example shows how a threads each access their 4x4 sub blocks, find the largest number and store it in the shared memory array
  Then the largest value is written to all array elements via the threads in the block.
  */

  //A 12 x 24 array that will be separated into 2 blocks.
  //Each block will have 3x3 thread dimensions.
  //Each thread will search a 4x4 square for the largest value.

  int a[288] = {
  40,	44,	93,	3,	50,	73,	58,	47,	29,	8,	92,	47,
  54,	52,	20,	49,	27,	72,	2,	92,	37,	56,	98,	34,
  5,	52,	21,	51,	96,	89,	12,	44,	70,	7,	98,	72,
  25,	38,	82,	27,	70,	13,	18,	18,	68,	54,	95,	40,
  49,	43,	11,	85,	45,	69,	23,	13,	20,	71,	3,	13,
  61,	44,	10,	87,	78,	36,	46,	87,	19,	3,	86,	38,
  95,	56,	67,	62,	12,	83,	79,	71,	51,	59,	36,	42,
  89,	77,	64,	33,	1,	51,	20,	72,	2,	78,	30,	04,
  70,	05,	87,	84,	76,	72,	85,	82,	46,	81,	83,	01,
  30,	42,	14,	36,	73,	1,	71,	28,	73,	57,	88,	3,
  73,	19,	5,	69,	43,	30,	70,	35,	94,	60,	66,	54,
  86,	89,	8,	63,	32,	50,	9,	51,	27,	74,	43,	44,
  90,	91,	42,	90,	64,	82,	80,	64,	60,	75,	78,	56,
  68,	11,	29,	61,	76,	2,	61,	86,	27,	54,	72,	14,
  58,	44,	18,	80,	00,	18,	85,	82,	11,	94,	66,	3,
  67,	21,	83,	13,	90,	40,	0,	39,	45,	30,	49,	79,
  98,	00,	19,	77,	34,	17,	85,	10,	15,	34,	69,	46,
  22,	94,	69,	21,	59,	59,	92,	44,	30,	33,	99,	19,
  30,	1,	89,	44,	29,	4,	75,	11,	26,	64,	89,	55,
  10,	35,	05,	92,	23,	4,	91,	46,	0,	53,	70,	86,
  98,	24,	15,	76,	9,	77,	32,	8,	27,	93,	75,	18,
  55,	22,	30,	76,	13,	22,	98,	31,	25,	10,	18,	76,
  89,	41,	5,	15,	62,	2,	93,	53,	29,	9,	99,	56,
  13,	98,	76,	88,	36,	84,	89,	24,	60,	14,	00,	69,
  };
  
  int *dev_a;
  int a_result[288];
  cudaMalloc(&dev_a, 288 * sizeof(int)); 
  cudaMemcpy(dev_a, a, 288*sizeof(int), cudaMemcpyHostToDevice);


  dim3 grids(1,2,1);
  dim3 threads(3,3,1);

  FindMax<<<grids,threads>>>(dev_a);

  cudaMemcpy(a_result, dev_a, 288*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Max value of the first block" << a_result[0] << "\n";
  std::cout << "Max value of the second block" << a_result[145] << "\n";
  
}