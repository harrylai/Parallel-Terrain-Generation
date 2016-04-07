/************************************************************
*	ECE408 Parallel Programming - Final Project				*
*															*
*	Topic: Terrain Generation								*
*	Members: Lai,Haoming; Ma,Yunhan; Wang,Bangqi			*
*															*
************************************************************/

/* 
* Terrain Generation:
* Algorithmn: Diamond Square Algorithmn.
* Version:
*			0. Serial version: 1 * square loop + 4 * diamond loop;
*			1. Parallel version: 1 * sdsfsdfsdf + 4 * diamond kernel; 
*			2. Less Kernel Version: 1 * square kernal + 1 * simple diamond kernel (1 thread => 4 vertex);
*			3. Smarter Kernel Version: 1 * sqaure kernel + 1 * smart diamond kernel (1 thread => 1 vertex);
*			4. One Kernel Version: 1 * square_diamond kernel combined; (based on version 2)
*			5. Kernel Device Version: 1 * kernel + 1 * square device + 1 * diamond device;
*			6. Less Threads Version: 1 * kernel + 1 * square device + 1 * diamond device (only active threads we need);
*			7. Shared Memory Version: 1 * kernel + 1 * square device + 1 * diamond device (use share memory);
*
*			8. 2D Smarter Kernel Versio: 1 * sqaure kernel + 1 * smart diamond kernel (1 thread => 1 vertex);
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>

/* Set the parameter */
/* Choose the version to use */
//#define VERSION 0
/* Set the length of each edge. please put power of 2 */
#define SIZE 512
/* Set number of array */
#define N (SIZE+1)*(SIZE+1)
/* Set the roughness for terrain */
#define ROUGHNESS 10
/* Set the height for each corner */
#define CORNER 0

/* main function for different version */
int version_0 (void);
int version_1 (void);
int version_2 (void);
int version_3 (bool print, int block_size);
int version_4 (bool print);
int version_5 (void);
int version_6 (void);
int version_7 (void);
int version_8 (bool print, int block_size);
int version_9 (bool print, int block_size);

/* main function */
int main (void){
	int VERSION;
	int p;
	int block_size; 

	bool print= false;
	printf("what version do you want: ");
	scanf("%d", &VERSION);
	printf("print? (0/1): ");
	scanf("%d", &p);
	printf("please define block_size(max = 32): ");
	scanf("%d", &block_size);
	if(p)
		print = true; 

	switch(VERSION){
		/* test version 0 */
		case 0:
			version_0();
			break;
		case 1:
		/* test version 1 */
			version_1();
			break;
		case 2:
		/* test version 2 */
			version_2();
			break;
		case 3:
		/* test version 3 */
			version_3(print, block_size);
			break;
		case 4:
		/* test version 4 */
			version_4(print);
			break;
		case 5:
		/* test version 5 */
			version_5();
			break;
		case 6:
		/* test version 5 */
			version_6();
			break;
		case 7:
		/* test version 5 */
			version_7();
			break;
		case 8:
		/* test version 5 */
			version_8(print, block_size);
			break;
		case 9:
		/* test version 5 */
			version_9(print, block_size);
			break;
		default:
		/* test version 0 */
			version_0();
			return 0;
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 0.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 0.0: 
*			0.0 Serial version: 1 * square loop + 4 * loop;  
*			
*/

/* host code for serial version */
int version_0 (void) {
	clock_t start, end;
	double runTime;
	float heightMap[SIZE+1][SIZE+1];
	for(int i=0; i<SIZE; i++){
		for(int j=0; j<SIZE; j++){
			heightMap[i][j] = 0.0;
		}
	}

	//initial the first four points
	heightMap[0][0] = 0; 
	heightMap[SIZE][0] = 0; 
	heightMap[0][SIZE] = 0; 
	heightMap[SIZE][SIZE] = 0;  

	start = clock();
	int stride = SIZE;
	while(stride>=2){
		for(int i = 0; i<(SIZE/stride); i++){
			for(int j = 0; j<(SIZE/stride); j++){
				int leftbottom_x = i* stride;
				int leftbottom_y = j* stride;
				float average =  heightMap[leftbottom_x][leftbottom_y] + heightMap[leftbottom_x + stride][leftbottom_y] + heightMap[leftbottom_x][leftbottom_y+stride] + heightMap[leftbottom_x + stride][leftbottom_y +stride];
				average = average /4 ;
				heightMap[leftbottom_x + stride/2][leftbottom_y + stride/2]= average + rand() %10 - 5; 


				heightMap[leftbottom_x + stride/2 ][leftbottom_y] = (average + heightMap[leftbottom_x][leftbottom_y]  + heightMap[leftbottom_x + stride][leftbottom_y] ) /3 + rand() %10 -5; 
				heightMap[leftbottom_x][leftbottom_y + stride/2] = (average + heightMap[leftbottom_x][leftbottom_y]  + heightMap[leftbottom_x][leftbottom_y + stride] ) /3 + rand() %10 -5 ; 
				heightMap[leftbottom_x + stride][leftbottom_y+ stride/2] = (average + heightMap[leftbottom_x + stride ][leftbottom_y]  + heightMap[leftbottom_x + stride][leftbottom_y + stride] ) /3 +rand() %10-5; 
				heightMap[leftbottom_x+ stride/2][leftbottom_y+ stride] = (average + heightMap[leftbottom_x][leftbottom_y + stride]  + heightMap[leftbottom_x + stride][leftbottom_y + stride] ) /3 +rand() %10-5; 
			}


		}
		printf("%d \n", stride);
		stride = stride/2;
	}

 	for (int i=0; i<=SIZE; i++){
 		for(int j=0; j<=SIZE; j++){
	 		printf("%d: x = %d, y = %d; hm = %f\n", i*j, i, j, heightMap[i][j]);
		}
	}

	end = clock();
 	runTime = (double)(end - start)/CLOCKS_PER_SEC;

	printf("Run time for Version_0: %f\n", runTime);
	printf("Version 0\n");
	return 0;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 1.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 1.0: 
*			1.0 Parallel version: 1 * square kernel + 4 * diamond kernel;  
*			This parallel function parallelize the serial code directly. it change the one square loop to
*			one square kernel and change four diamond loop to four different diamond kernel.	1
*/

/* square kernel to calculate the middle point */
__global__ void Square_1(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, nj, mi, mj;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		nj = j + rect;
		mi = i + half;
		mj = j + half;

		/* set check value */
		check1[idx] = mi;
		check2[idx] = mj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);
		rng[idx] = localState;

	    /* set height map */
		hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		__syncthreads();
  	}
}

/* diamond kernel 1_1 to calcualte middle bottom point */
__global__ void Diamond_1_1(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, mi, j;
		int pmi_b, pmj_b;
		float hm_b;
		int num_b;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		mi = i + half;

		/* find 4 diamond vertex */
		/* bottom vertex */
		pmi_b = mi;
		pmj_b = j;

		/* set the value */
		/* bottom height */
		hm_b = 0;
		num_b = 3;
		if (pmj_b - half >= 0){
			hm_b += hm[pmi_b + (pmj_b-half)*(SIZE+1)];
			num_b = 4;
		}
		hm_b += hm[pmi_b + (pmj_b+half)*(SIZE+1)];
		hm_b += hm[(pmi_b-half) + pmj_b*(SIZE+1)];
		hm_b += hm[(pmi_b+half) + pmj_b*(SIZE+1)];

		/* set check value */
		// check1[idx] = hm_l;
		// check2[idx] = hm_l;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand1 = v1 + (v2 - v1) * curand_uniform(&localState);

	    /* set height map */
		hm[pmi_b + pmj_b*(SIZE+1)] = hm_b/num_b + rand1;
		// hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		rng[idx] = localState;
		__syncthreads();     
  	}
  }

/* diamond kernel 1_2 to calcualte left point */
__global__ void Diamond_1_2(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, j, mj;
		int pmi_l, pmj_l;
		float hm_l;
		int num_l;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		mj = j + half;

		/* find 4 diamond vertex */
		/* left vertex */
		pmi_l = i;
		pmj_l = mj;

		/* set the value */
		/* left height */
		hm_l = 0;
		num_l = 3;
		if (pmi_l - half >= 0){
			hm_l += hm[(pmi_l-half) + pmj_l*(SIZE+1)];
			num_l = 4;
		}
		hm_l += hm[(pmi_l+half) + pmj_l*(SIZE+1)];
		hm_l += hm[pmi_l + (pmj_l-half)*(SIZE+1)];
		hm_l += hm[pmi_l + (pmj_l+half)*(SIZE+1)];

		/* set check value */
		// check1[idx] = hm_l;
		// check2[idx] = hm_l;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand2 = v1 + (v2 - v1) * curand_uniform(&localState);

	    /* set height map */
      	hm[pmi_l + pmj_l*(SIZE+1)] = hm_l/num_l + rand2;

		// hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		rng[idx] = localState;
		__syncthreads();     
  	}
}

/* diamond kernel 1_3 to calcualte right point */
__global__ void Diamond_1_3(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, mj;
		int pmi_r, pmj_r;
		float hm_r;
		int num_r;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		mj = j + half;

		/* find 4 diamond vertex */
		/* right vertex */
		pmi_r = ni;
		pmj_r = mj;

		/* set the value */
		/* right height */
		hm_r = 0;
		num_r = 3;
		if (pmi_r + half <= SIZE){
			hm_r += hm[(pmi_r+half) + pmj_r*(SIZE+1)];
			num_r = 4;
		}
		hm_r += hm[(pmi_r-half) + pmj_r*(SIZE+1)];
		hm_r += hm[pmi_r + (pmj_r-half)*(SIZE+1)];
		hm_r += hm[pmi_r + (pmj_r+half)*(SIZE+1)];

		/* set check value */
		// check1[idx] = hm_l;
		// check2[idx] = hm_l;

		/* get height for  */

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand3 = v1 + (v2 - v1) * curand_uniform(&localState);

	    /* set height map */
      	hm[pmi_r + pmj_r*(SIZE+1)] = hm_r/num_r + rand3;
		// hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		rng[idx] = localState;
		__syncthreads();     
  	}
}

/* diamond kernel 1_4 to calcualte middle top point */
__global__ void Diamond_1_4(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, j, mi, nj;
		int pmi_t, pmj_t;
		float hm_t;
		int num_t;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		nj = j + rect;
		mi = i + half;

		/* find 4 diamond vertex */
		/* top vertex */
		pmi_t = mi;
		pmj_t = nj;

		/* set the value */
		/* top height */
		hm_t = 0;
		num_t = 3;
		if (pmj_t + half <= SIZE){
			hm_t += hm[pmi_t + (pmj_t+half)*(SIZE+1)];
			num_t = 4;
		}
		hm_t += hm[pmi_t + (pmj_t-half)*(SIZE+1)];
		hm_t += hm[(pmi_t-half) + pmj_t*(SIZE+1)];
		hm_t += hm[(pmi_t+half) + pmj_t*(SIZE+1)];

		/* set check value */
		// check1[idx] = hm_l;
		// check2[idx] = hm_l;

		/* get height for  */

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand4 = v1 + (v2 - v1) * curand_uniform(&localState);

	    /* set height map */
      	hm[pmi_t + pmj_t*(SIZE+1)] = hm_t/num_t + rand4;  
		// hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		rng[idx] = localState;
		__syncthreads();     
  	}
}

/* host code for version 1.0 */
int version_1 (void) {
	printf("Version 1: square kernel + 4 diamond kernel\n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
 	for (int i=SIZE; i>1; i=i/2){
		Square_1<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_1_1<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_1_2<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_1_3<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_1_4<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	for (int i=0; i<N; i++){
	  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
	}

	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_1: %f\n", runTime);
	return EXIT_SUCCESS;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 2.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 2.0:
*			2.0 Less Kernel Version: 1 * square kernal + 1 * simple diamond kernel (1 thread => 4 vertex); 
*			This kernel combine the four diamond kernel to one single kernel. However, each thread in diamond
*			kernel needs to calculate four vertex.
*/

/* combined diamond kernel to calculate the four point in each thread */
__global__ void Diamond_2(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, nj, mi, mj;
		int pmi_b, pmj_b, pmi_l, pmj_l, pmi_r, pmj_r, pmi_t, pmj_t;
		float hm_b, hm_l, hm_r, hm_t;
		int num_b, num_l, num_r, num_t;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		nj = j + rect;
		mi = i + half;
		mj = j + half;

		/* find 4 diamond vertex */
		/* bottom vertex */
		pmi_b = mi;
		pmj_b = j;
		/* left vertex */
		pmi_l = i;
		pmj_l = mj;
		/* right vertex */
		pmi_r = ni;
		pmj_r = mj;
		/* top vertex */
		pmi_t = mi;
		pmj_t = nj;

		/* set the value */
		/* bottom height */
		hm_b = 0;
		num_b = 3;
		if (pmj_b - half >= 0){
			hm_b += hm[pmi_b + (pmj_b-half)*(SIZE+1)];
			num_b = 4;
		}
		hm_b += hm[pmi_b + (pmj_b+half)*(SIZE+1)];
		hm_b += hm[(pmi_b-half) + pmj_b*(SIZE+1)];
		hm_b += hm[(pmi_b+half) + pmj_b*(SIZE+1)];

		/* left height */
		hm_l = 0;
		num_l = 3;
		if (pmi_l - half >= 0){
			hm_l += hm[(pmi_l-half) + pmj_l*(SIZE+1)];
			num_l = 4;
		}
		hm_l += hm[(pmi_l+half) + pmj_l*(SIZE+1)];
		hm_l += hm[pmi_l + (pmj_l-half)*(SIZE+1)];
		hm_l += hm[pmi_l + (pmj_l+half)*(SIZE+1)];

		/* right height */
		hm_r = 0;
		num_r = 3;
		if (pmi_r + half <= SIZE){
			hm_r += hm[(pmi_r+half) + pmj_r*(SIZE+1)];
			num_r = 4;
		}
		hm_r += hm[(pmi_r-half) + pmj_r*(SIZE+1)];
		hm_r += hm[pmi_r + (pmj_r-half)*(SIZE+1)];
		hm_r += hm[pmi_r + (pmj_r+half)*(SIZE+1)];

		/* top height */
		hm_t = 0;
		num_t = 3;
		if (pmj_t + half <= SIZE){
			hm_t += hm[pmi_t + (pmj_t+half)*(SIZE+1)];
			num_t = 4;
		}
		hm_t += hm[pmi_t + (pmj_t-half)*(SIZE+1)];
		hm_t += hm[(pmi_t-half) + pmj_t*(SIZE+1)];
		hm_t += hm[(pmi_t+half) + pmj_t*(SIZE+1)];

		/* set check value */
		check1[idx] = hm_l;
		check2[idx] = hm_l;

		/* get height for  */

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand1 = v1 + (v2 - v1) * curand_uniform(&localState);
	    float rand2 = v1 + (v2 - v1) * curand_uniform(&localState);
	    float rand3 = v1 + (v2 - v1) * curand_uniform(&localState);
	    float rand4 = v1 + (v2 - v1) * curand_uniform(&localState);

	    /* set height map */
		hm[pmi_b + pmj_b*(SIZE+1)] = hm_b/num_b + rand1;
      	hm[pmi_l + pmj_l*(SIZE+1)] = hm_l/num_l + rand2;
      	hm[pmi_r + pmj_r*(SIZE+1)] = hm_r/num_r + rand3;
      	hm[pmi_t + pmj_t*(SIZE+1)] = hm_t/num_t + rand4;  
		// hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		rng[idx] = localState;
		__syncthreads();     
  	}
}


/* the host code for version 2: 1 square kernel + 1 stupid diamond kernel */
int version_2 (void) {
	printf("Version 2: square kernel + stupid diamond kernel\n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
 	for (int i=SIZE; i>1; i=i/2){
		Square_1<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_2<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	for (int i=0; i<N; i++){
	  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
	}

	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_2: %0.20f\n", runTime);
	return EXIT_SUCCESS;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 3.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 3.0:
*			3.0 Smarter Kernel Version: 1 * sqaure kernel + 1 * smart diamond kernel (1 thread => 1 vertex);
*			This version reconstruct the diamond kernel to use different threads for different vertx. Each 
*			thread in diamond kernel only need to calculate one vertex.
*/

/* smart diamond kernel calculate the diamond vertex and each thread only calculate one vertex */
__global__ void Diamond_3(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
		/* initialize vairable */
		int half = rect/2;
		int i, j;
		int pmi, pmj;
		float hm_p;
		int num_p;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;

		/* Calculate the diamond vertex use idx */
		int tid = idx/(squareInRow*squareInRow)%4;
		pmi = i + (1 - tid%2)*half + tid/2*half;
		pmj = j + tid%2*half + tid/2*half;

		/* Set the value */
		hm_p = 0;
		num_p = 0;
		if (pmi - half >= 0){
			hm_p += hm[(pmi-half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmi + half <= SIZE){
			hm_p += hm[(pmi+half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmj - half >= 0){
			hm_p += hm[pmi + (pmj-half)*(SIZE+1)];
			num_p++;
		}
		if (pmj + half <= SIZE){
			hm_p += hm[pmi + (pmj+half)*(SIZE+1)];
			num_p++;
		}

		/* set check value */
		check1[idx] = pmi;
		check2[idx] = pmj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);

		/* get height for  */
		hm[pmi + pmj*(SIZE+1)] = hm_p/num_p + rand;
		rng[idx] = localState;
		__syncthreads();    
  	}
}

/* the host code for version 3: 1 square kernel + 1 smart diamond kernel */
int version_3 (bool print, int block_size) {
	printf("Version 3: square kernel + smart diamond kernel\n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	int size = block_size * block_size;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
 	for (int i=SIZE; i>1; i=i/2){
		Square_1<<<ceil((float)N/size),size>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_3<<<ceil((float)N/size),size>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	if(print){
		for (int i=0; i<N; i++){
		  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
		}
	}
	// printf("\n");	
	// for (int i=0; i<N; i++){
	//   printf("%d: pmi = %f, pmj = %f\n", i, check1[i], check2[i]);
	// }

	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_3: %0.20f\n", runTime);
	return EXIT_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 4.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 4.0:
*			4.0 Less Kernel Version: 1 * square kernal + 1 * simple diamond kernel (1 thread => 4 vertex); 
*			This kernel combine the four diamond kernel to one single kernel. However, each thread in diamond
*			kernel needs to calculate four vertex.
*/
/* combined diamond kernel to calculate the four point in each thread */
__global__ void Square_Diamond_4(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx < N){
  		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, nj, mi, mj;
		int pmi_b, pmj_b, pmi_l, pmj_l, pmi_r, pmj_r, pmi_t, pmj_t;
		float hm_b, hm_l, hm_r, hm_t;
		int num_b, num_l, num_r, num_t;
		int squareInRow = SIZE/rect;

  		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		nj = j + rect;
		mi = i + half;
		mj = j + half;

		/* set check value */
		check1[idx] = mi;
		check2[idx] = mj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);
		rng[idx] = localState;

	    /* set height map */
		hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		__syncthreads();

		/* find 4 diamond vertex */
		/* bottom vertex */
		pmi_b = mi;
		pmj_b = j;
		/* left vertex */
		pmi_l = i;
		pmj_l = mj;
		/* right vertex */
		pmi_r = ni;
		pmj_r = mj;
		/* top vertex */
		pmi_t = mi;
		pmj_t = nj;

		/* set the value */
		/* bottom height */
		hm_b = 0;
		num_b = 3;
		if (pmj_b - half >= 0){
			hm_b += hm[pmi_b + (pmj_b-half)*(SIZE+1)];
			num_b = 4;
		}
		hm_b += hm[pmi_b + (pmj_b+half)*(SIZE+1)];
		hm_b += hm[(pmi_b-half) + pmj_b*(SIZE+1)];
		hm_b += hm[(pmi_b+half) + pmj_b*(SIZE+1)];

		/* left height */
		hm_l = 0;
		num_l = 3;
		if (pmi_l - half >= 0){
			hm_l += hm[(pmi_l-half) + pmj_l*(SIZE+1)];
			num_l = 4;
		}
		hm_l += hm[(pmi_l+half) + pmj_l*(SIZE+1)];
		hm_l += hm[pmi_l + (pmj_l-half)*(SIZE+1)];
		hm_l += hm[pmi_l + (pmj_l+half)*(SIZE+1)];

		/* right height */
		hm_r = 0;
		num_r = 3;
		if (pmi_r + half <= SIZE){
			hm_r += hm[(pmi_r+half) + pmj_r*(SIZE+1)];
			num_r = 4;
		}
		hm_r += hm[(pmi_r-half) + pmj_r*(SIZE+1)];
		hm_r += hm[pmi_r + (pmj_r-half)*(SIZE+1)];
		hm_r += hm[pmi_r + (pmj_r+half)*(SIZE+1)];

		/* top height */
		hm_t = 0;
		num_t = 3;
		if (pmj_t + half <= SIZE){
			hm_t += hm[pmi_t + (pmj_t+half)*(SIZE+1)];
			num_t = 4;
		}
		hm_t += hm[pmi_t + (pmj_t-half)*(SIZE+1)];
		hm_t += hm[(pmi_t-half) + pmj_t*(SIZE+1)];
		hm_t += hm[(pmi_t+half) + pmj_t*(SIZE+1)];

		/* set check value */
		check1[idx] = hm_l;
		check2[idx] = hm_l;

		/* get height for  */

		/* set random generator */
	    float rand1 = v1 + (v2 - v1) * curand_uniform(&localState);
	    float rand2 = v1 + (v2 - v1) * curand_uniform(&localState);
	    float rand3 = v1 + (v2 - v1) * curand_uniform(&localState);
	    float rand4 = v1 + (v2 - v1) * curand_uniform(&localState);

	    /* set height map */
		hm[pmi_b + pmj_b*(SIZE+1)] = hm_b/num_b + rand1;
      	hm[pmi_l + pmj_l*(SIZE+1)] = hm_l/num_l + rand2;
      	hm[pmi_r + pmj_r*(SIZE+1)] = hm_r/num_r + rand3;
      	hm[pmi_t + pmj_t*(SIZE+1)] = hm_t/num_t + rand4;  
		// hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		rng[idx] = localState;
		__syncthreads();     
  	}
}


/* the host code for version 2: 1 square kernel + 1 stupid diamond kernel */
int version_4 (bool print) {
	printf("Version 2: square kernel + stupid diamond kernel\n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
 	for (int i=SIZE; i>1; i=i/2){
		Square_Diamond_4<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	if(print){
		for (int i=0; i<N; i++){
		  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
		}
	}
	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_4: %0.20f\n", runTime);
	return EXIT_SUCCESS;
}

int version_5 (void) {
	printf("5\n");
	return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 6.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 6.0:
*			6. Less Threads Version: 1 * kernel + 1 * square device + 1 * diamond device (only active threads we need);
*			This kernel combine the four diamond kernel to one single kernel. However, each thread in diamond
*			kernel needs to calculate four vertex.
*/

/* square kernel to calculate the middle point */
__global__ void Square_6(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int squareInRow = SIZE/rect;
  	if (idx < squareInRow * squareInRow){
		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, nj, mi, mj;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		nj = j + rect;
		mi = i + half;
		mj = j + half;

		/* set check value */
		check1[idx] = mi;
		check2[idx] = mj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);
		rng[idx] = localState;

	    /* set height map */
		hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 + rand;
		__syncthreads();
  	}
}

/* smart diamond kernel calculate the diamond vertex and each thread only calculate one vertex */
__global__ void Diamond_6(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int squareInRow = SIZE/rect;
  	if (idx < 4 * squareInRow * squareInRow){
		/* initialize vairable */
		int half = rect/2;
		int i, j;
		int pmi, pmj;
		float hm_p;
		int num_p;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;

		/* Calculate the diamond vertex use idx */
		int tid = idx/(squareInRow*squareInRow)%4;
		pmi = i + (1 - tid%2)*half + tid/2*half;
		pmj = j + tid%2*half + tid/2*half;

		/* Set the value */
		hm_p = 0;
		num_p = 0;
		if (pmi - half >= 0){
			hm_p += hm[(pmi-half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmi + half <= SIZE){
			hm_p += hm[(pmi+half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmj - half >= 0){
			hm_p += hm[pmi + (pmj-half)*(SIZE+1)];
			num_p++;
		}
		if (pmj + half <= SIZE){
			hm_p += hm[pmi + (pmj+half)*(SIZE+1)];
			num_p++;
		}

		/* set check value */
		check1[idx] = pmi;
		check2[idx] = pmj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);

		/* get height for  */
		hm[pmi + pmj*(SIZE+1)] = hm_p/num_p + rand;
		rng[idx] = localState;
		__syncthreads();    
  	}
}

/* the host code for version 3: 1 square kernel + 1 smart diamond kernel */
int version_6 (void) {
	printf("Version 6: square kernel + smart diamond kernel (active less threads) \n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = CORNER;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = CORNER;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
 	for (int i=SIZE; i>1; i=i/2){
		Square_6<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_6<<<ceil((float)N/256),256>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	for (int i=0; i<N; i++){
	  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
	}
	// printf("\n");	
	// for (int i=0; i<N; i++){
	//   printf("%d: pmi = %f, pmj = %f\n", i, check1[i], check2[i]);
	// }

	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_6: %0.20f\n", runTime);
	return EXIT_SUCCESS;
}


int version_7 (void) {
	printf("7\n");
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 8.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 8.0:
*			8.0 Smarter Kernel Version: 1 * sqaure kernel + 1 * smart diamond kernel (1 thread => 1 vertex);
*			This version reconstruct the diamond kernel to use different threads for different vertx. Each 
*			thread in diamond kernel only need to calculate one vertex. (A simple revised 2D version of version 3)
*/
__global__ void Square_8(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx_temp = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
  	if (idx_temp < SIZE+1 && idy < SIZE+1){
  		int idx = idy*(SIZE+1) + idx_temp;
		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, nj, mi, mj;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		nj = j + rect;
		mi = i + half;
		mj = j + half;

		/* set check value */
		check1[idx] = mi;
		check2[idx] = mj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);
		rng[idx] = localState;

	    /* set height map */
		hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 +rand;
		__syncthreads();
  	}
}

__global__ void Diamond_8(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx_temp = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
  	if (idx_temp < SIZE+1 && idy < SIZE+1){
  		int idx = idy*(SIZE+1) + idx_temp;
		/* initialize vairable */
		int half = rect/2;
		int i, j;
		int pmi, pmj;
		float hm_p;
		int num_p;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;

		/* Calculate the diamond vertex use idx */
		int tid = idx/(squareInRow*squareInRow)%4;
		pmi = i + (1 - tid%2)*half + tid/2*half;
		pmj = j + tid%2*half + tid/2*half;

		/* Set the value */
		hm_p = 0;
		num_p = 0;
		if (pmi - half >= 0){
			hm_p += hm[(pmi-half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmi + half <= SIZE){
			hm_p += hm[(pmi+half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmj - half >= 0){
			hm_p += hm[pmi + (pmj-half)*(SIZE+1)];
			num_p++;
		}
		if (pmj + half <= SIZE){
			hm_p += hm[pmi + (pmj+half)*(SIZE+1)];
			num_p++;
		}

		/* set check value */
		check1[idx] = pmi;
		check2[idx] = pmj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);

		/* get height for  */
		hm[pmi + pmj*(SIZE+1)] = hm_p/num_p +rand;
		rng[idx] = localState;
		__syncthreads();    
  	}
}

/* the host code for version 8: 2D + 1 square kernel + 1 smart diamond kernel. */
int version_8 (bool print, int block_size) {
	printf("Version 8: square kernel + smart diamond kernel\n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = 1;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = 2;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = 3;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = 4;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
	dim3 DimGrid(ceil(((float)SIZE)/block_size),ceil(((float)SIZE)/block_size), 1); 
	dim3 DimBlock(block_size,block_size,1);
 	for (int i=SIZE; i>1; i=i/2){
		Square_8<<<DimGrid,DimBlock>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_8<<<DimGrid,DimBlock>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	if(print){
		for (int i=0; i<N; i++){
		  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
		}
	}
	// printf("\n");	
//	for (int i=0; i<SIZE+1; i++){
//	   printf("%d: pmi = %f, pmj = %f\n", i, check1[i], check2[i]);
//	}

	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_8: %0.20f\n", runTime);
	return EXIT_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// VERSION 9.0 ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	VERSION 9.0:
*			9.0 Smarter Kernel Version: 1 * sqaure kernel + 1 * smart diamond kernel (1 thread => 1 vertex);
*			This version reconstruct the diamond kernel to use different threads for different vertx. Each 
*			thread in diamond kernel only need to calculate one vertex. (A simple revised 2D version of version 3)
*/
__global__ void Square_9(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx_temp = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
  	if (idx_temp < SIZE+1 && idy < SIZE+1){
  		int idx = idy*(SIZE+1) + idx_temp;
		/* initialize vairable */
		int half = rect/2;
		int i, j, ni, nj, mi, mj;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;
		ni = i + rect;
		nj = j + rect;
		mi = i + half;
		mj = j + half;

		/* set check value */
		check1[idx] = mi;
		check2[idx] = mj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);
		rng[idx] = localState;

	    /* set height map */
		hm[mi + mj*(SIZE+1)] = (hm[i + j*(SIZE+1)] + hm[ni + j*(SIZE+1)] + hm[i + nj*(SIZE+1)] + hm[ni + nj*(SIZE+1)])/4 +rand;
		__syncthreads();
  	}
}

__global__ void Diamond_9(curandState* rng, float* hm, int rect, float* check1, float* check2){
	/* set idx */
	int idx_temp = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
  	if (idx_temp < SIZE+1 && idy < SIZE+1){
  		int idx = idy*(SIZE+1) + idx_temp;
		/* initialize vairable */
		int half = rect/2;
		int i, j;
		int pmi, pmj;
		float hm_p;
		int num_p;
		int squareInRow = SIZE/rect;
	  
		/* calculate vertex */
		i = (idx%squareInRow*rect)%SIZE;
		j = (idx/squareInRow*rect)%SIZE;

		/* Calculate the diamond vertex use idx */
		int tid = idx/(squareInRow*squareInRow)%4;
		pmi = i + (1 - tid%2)*half + tid/2*half;
		pmj = j + tid%2*half + tid/2*half;

		/* Set the value */
		hm_p = 0;
		num_p = 0;
		if (pmi - half >= 0){
			hm_p += hm[(pmi-half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmi + half <= SIZE){
			hm_p += hm[(pmi+half) + pmj*(SIZE+1)];
			num_p++;
		}
		if (pmj - half >= 0){
			hm_p += hm[pmi + (pmj-half)*(SIZE+1)];
			num_p++;
		}
		if (pmj + half <= SIZE){
			hm_p += hm[pmi + (pmj+half)*(SIZE+1)];
			num_p++;
		}

		/* set check value */
		check1[idx] = pmi;
		check2[idx] = pmj;

		/* set random generator */
		float v1 = (0.0f - (float)ROUGHNESS)/2;
		float v2 = ((float)ROUGHNESS)/2;
		curandState localState = rng[idx];
	    float rand = v1 + (v2 - v1) * curand_uniform(&localState);

		/* get height for  */
		hm[pmi + pmj*(SIZE+1)] = hm_p/num_p +rand;
		rng[idx] = localState;
		__syncthreads();    
  	}
}

/* the host code for version 8: 2D + 1 square kernel + 1 smart diamond kernel. */
int version_9 (bool print, int block_size) {
	printf("Version 8: square kernel + smart diamond kernel\n");
	/* initialize variables */
	float check1[N];
	float check2[N];
	float heightMap[N];
	/* initialize device */
	float *dev_heightMap;
	float *dev_check1;
	float *dev_check2;
	/* initialize time*/
	clock_t start, end;
	double runTime;
	/* initial height map */
	for (int i=0; i<N; i++){
	  heightMap[i] = 0;
	}

	/* set height for corner */
	heightMap[0 + 0 * (SIZE+1)] = 1;
	printf("heightMap_corner0: %f\n", heightMap[0 + 0 * (SIZE+1)]);
	heightMap[SIZE + 0 * (SIZE+1)] = 2;
	printf("heightMap_corner1: %f\n", heightMap[SIZE + 0 * (SIZE+1)]);
	heightMap[0 + SIZE * (SIZE+1)] = 3;
	printf("heightMap_corner3: %f\n", heightMap[0 + SIZE * (SIZE+1)]);
	heightMap[SIZE + SIZE * (SIZE+1)] = 4;
	printf("heightMap_corner2: %f\n", heightMap[SIZE + SIZE * (SIZE+1)]);

	curandState* rng;
	/* allocate memory for device */
	cudaMalloc(&rng, N * sizeof(curandState));
	cudaMalloc((void**)&dev_heightMap, N * sizeof(float));
	cudaMalloc((void**)&dev_check1, N * sizeof(float));
	cudaMalloc((void**)&dev_check2, N * sizeof(float));

	/* memory copy from host to device */
	cudaMemcpy(dev_heightMap, heightMap, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check1, check1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_check2, check2, N * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	/* run kernel */
	dim3 DimGrid(ceil(((float)SIZE)/block_size),ceil(((float)SIZE)/block_size), 1); 
	dim3 DimBlock(block_size,block_size,1);
 	for (int i=SIZE; i>1; i=i/2){
		Square_9<<<DimGrid,DimBlock>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
		Diamond_9<<<DimGrid,DimBlock>>>(rng, (float*)dev_heightMap, i, dev_check1, dev_check2);
		cudaDeviceSynchronize();
 	}
 	end = clock();

	/* memory copy from device to host*/
	cudaMemcpy(heightMap, dev_heightMap, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check1, dev_check1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(check2, dev_check2, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* print the output */
	if(print){
		for (int i=0; i<N; i++){
		  printf("%d: x = %d, y = %d; hm = %f\n", i, i%(SIZE+1), i/(SIZE+1), heightMap[i]);
		}
	}
	// printf("\n");	
//	for (int i=0; i<SIZE+1; i++){
//	   printf("%d: pmi = %f, pmj = %f\n", i, check1[i], check2[i]);
//	}

	// printf("%f\n", cpu_time_used);
	cudaFree(dev_heightMap);
	cudaFree(dev_check1);
	cudaFree(dev_check2);

 	runTime = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Run time for Version_8: %0.20f\n", runTime);
	return EXIT_SUCCESS;
}
