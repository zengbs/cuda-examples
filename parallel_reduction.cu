#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>


#include "macros.h"



unsigned long Time = 0;



__global__ void ReduceNeighbored( int *g_array, int *g_output, int arrayLength ){

   int tid = blockIdx.x*blockDim.x + threadIdx.x;

   if (tid >= arrayLength) return;

   int *idata = g_array + blockIdx.x*blockDim.x;

   for ( int offset = 1; offset < blockDim.x; offset *= 2 ){

      if (threadIdx.x % ( 2*offset ) == 0){
         idata[threadIdx.x] += idata[threadIdx.x+offset];
      }
      __syncthreads();
   }

   if (threadIdx.x == 0) g_output[blockIdx.x] = g_array[blockIdx.x*blockDim.x];

}


__global__ void ReduceNeighboredLess( int *g_array, int *g_output, int arrayLength ){

   int tid = blockIdx.x*blockDim.x + threadIdx.x;

   if (tid >= arrayLength) return;

   int *idata = g_array + blockIdx.x*blockDim.x;

   for ( int offset = 1; offset < blockDim.x; offset *= 2 ){

      if ( threadIdx.x < blockDim.x/2/offset ){
         idata[threadIdx.x*offset*2] += idata[threadIdx.x*offset*2 + offset];
      }
      __syncthreads();
   }

   if (threadIdx.x == 0) g_output[blockIdx.x] = g_array[blockIdx.x*blockDim.x];

}


// warp divergence only occurs in the last five rounds
// when the number of active threads is less than a warp size (32)

__global__ void ReduceInterleaved( int *g_array, int *g_output, int arrayLength ){

   int tid = blockIdx.x*blockDim.x + threadIdx.x;

   if (tid >= arrayLength) return;

   int *idata = g_array + blockIdx.x*blockDim.x;

   for ( int offset = blockDim.x/2; offset > 0; offset /= 2 ){

      if (threadIdx.x < offset){
         idata[threadIdx.x] += idata[threadIdx.x+offset];
      }
      __syncthreads();
   }

   if (threadIdx.x == 0) g_output[blockIdx.x] = g_array[blockIdx.x*blockDim.x];

   // The line below shows "1024 4 4 4 4 4 4 4" since the blocks with block IDs greater than 1
   // will access g_array[] before the 0th block
   //if (threadIdx.x == 0) g_output[blockIdx.x] = g_array[threadIdx.x];

}


__global__ void ReduceUnrolling2( int *g_array, int *g_output, int arrayLength ){

   int *idata = g_array + 2*blockIdx.x*blockDim.x;

   if ( threadIdx.x + blockDim.x < arrayLength ){
      int a0 = idata[threadIdx.x           ];
      int a1 = idata[threadIdx.x+blockDim.x];
      idata[threadIdx.x] = a0 + a1;
   }

   __syncthreads();

   for ( int offset = blockDim.x/2; offset > 0; offset /= 2 ){

      if (threadIdx.x < offset){
         idata[threadIdx.x] += idata[threadIdx.x+offset];
      }
      __syncthreads();
   }

   if (threadIdx.x == 0) g_output[blockIdx.x] = idata[0];
}


__global__ void ReduceUnrolling3( int *g_array, int *g_output, int arrayLength ){

   int *idata = g_array + 3*blockIdx.x*blockDim.x;

   if ( threadIdx.x + 2*blockDim.x < arrayLength ){
      int a0 = idata[threadIdx.x             ];
      int a1 = idata[threadIdx.x+  blockDim.x];
      int a2 = idata[threadIdx.x+2*blockDim.x];
      idata[threadIdx.x] = a0 + a1 + a2;
   }

   __syncthreads();


   for ( int offset = blockDim.x/2; offset > 0; offset /= 2 ){

      if (threadIdx.x < offset){
         idata[threadIdx.x] += idata[threadIdx.x+offset];
      }

      __syncthreads();
   }

   if (threadIdx.x == 0) g_output[blockIdx.x] = idata[0];
}

int sum( int *array, int arrayLength ){

   int sum = 0;

   for ( int i=0; i<arrayLength; i++ ){
      sum += array[i];
   }

   return sum;
}



int main(){

   int arrayLength = 128*28;

   int blockSize = 128;

   dim3 block ( blockSize, 1 );
   dim3 grid ( ( arrayLength + block.x - 1 ) / block.x, 1 );


   int *h_array, *g_array, *g_output, *h_output = NULL;

   h_array  = (int*)calloc(arrayLength, sizeof(int));
   h_output = (int*)calloc(     grid.x, sizeof(int));

   int max = +100;
   int min =  +10;
   srand(time(NULL));

   for (int i=0; i<arrayLength; i++) h_array[i] = min + (rand() % (max - min + 1));

   CHECK(  cudaMalloc( &g_array, arrayLength*sizeof(int) )  );
   CHECK(  cudaMalloc( &g_output,     grid.x*sizeof(int) )  );

   CHECK( cudaMemcpy( g_array      , h_array      , arrayLength*sizeof(int), cudaMemcpyHostToDevice ) );


   BlockReduction4<<< grid.x/2, block, 0, 0 >>>
   ( g_array, g_output, arrayLength );

   CHECK( cudaMemcpy( h_output, g_output, grid.x*sizeof(int), cudaMemcpyDeviceToHost ) );


   // sum up partial sum from thread blocks
   int g_sum = 0;

   for (int i=0; i<grid.x; i++) g_sum += h_output[i];


   if ( g_sum == sum(h_array, arrayLength) ) printf("Pass!\n");
   else                                      printf("Fail!\n");



   return 0;
}
