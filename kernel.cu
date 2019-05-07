#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//The number of character in the encrypted text
#define N 1024

void checkCUDAError(const char*);


#define A 15
#define B 27
#define M 128
#define A_MMI_M 111







char *input_file_name;//inputfilename string 
char *output_file_name;

void get_image_dimensions(FILE *fp, int *x, int *y) {// to get the dimensions of the image
	int char_len;
	char ch, ch1;
	ch = fgetc(fp);
	ch1 = fgetc(fp);

	while (fgetc(fp) != '\n') {
		continue;
	}
	char cc;
	cc = fgetc(fp);
	while (cc == '#') {
		while (getc(fp) != '\n');
		cc = getc(fp);
	}
	ungetc(cc, fp);
	fscanf(fp, "%d %d", x, y);
	fscanf(fp, "%d", &char_len);

	//below two lines are just there to make sure there is no warnings
	ch1 = ch + 1;
	ch = ch1 + 1;
}


__global__ void image_func(int *red_cuda)
{

	int index_x =blockDim.x*blockIdx.x + threadIdx.x;
	int index_y = blockDim.y*blockIdx.y + threadIdx.y;
	int offset= (index_x * 16) + index_y;

	red_cuda[offset] = 24;



	
	//printf("x is %d y is %d \n", x, y);
	//int index = threadIdx.x;
	
	/*
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int position =  (i * 16) + j;
			sum+=red_cuda[position]
		}
	}

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int position = (i * 16) + j;
			red_cuda[position] = sum;
		}
	}*/
	
}




int main(int argc, char *argv[])
{
	//reading the input file
	input_file_name = "SheffieldPlainText16x16.ppm";
	printf("%s", input_file_name);
	FILE *fp = fopen(input_file_name, "r");
	int x, y;
	get_image_dimensions(fp, &x, &y);
	printf("%d %d", x, y);
	int* red = (int*)malloc(x*y*sizeof(int));
	int* green = (int*)malloc(x *y* sizeof(int));
	int* blue = (int*)malloc(x *y* sizeof(int));

	int* red_cuda;


	int *copy_red = red;
	int *copy_green = green;
	int *copy_blue = blue;


	int* out_red = (int*)malloc(x*y* sizeof(int));
	int* out_green = (int*)malloc(x*y* sizeof(int));
	int* out_blue = (int*)malloc(x*y* sizeof(int));
	int *out_copy_red = red;
	int *out_copy_green = green;
	int *out_copy_blue = blue;


	//free(copy_red); in the end
	for (int i = 0; i<x; i++) {
		for (int j = 0; j<y; j++) {
			int r, g, b;
			fscanf(fp, "%d %d %d", &r, &g, &b);
			char useless;
			useless = getc(fp);
			useless = 1 + useless;
			int position = (i * y) + j;

			red[position] = r;
			green[position] = g;
			blue[position] = b;
		}
	}
	fclose(fp);

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			int position = (i * y) + j;
			printf("%d ", red[position]);
		}
		printf("\n");
	}
	printf("____________________________________________\n");


	int size_of_image = x * y * sizeof(int);
	printf("%d", size_of_image);


	cudaMalloc((void **)&red_cuda,size_of_image);
	checkCUDAError("Memory allocation");

	/* copy host input to device input */
	cudaMemcpy(red_cuda, red, size_of_image, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");


	/* Configure the grid of thread blocks and run the GPU kernel */
	dim3 blocksPerGrid(1, 1, 1);
	dim3 threadsPerBlock(4, 4, 1);
	image_func << <blocksPerGrid, threadsPerBlock >> >(red_cuda);


	/* wait for all threads to complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");

	/* copy the gpu output back to the host */
	cudaMemcpy(red, red_cuda, size_of_image, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");

	
	printf("\n_______________________\n\n\n");


	for(int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
		int position = (i * y) + j;
			printf("%d ", red[position]);
		}
		printf("\n");
	}

	
	/* free device memory */
	cudaFree(red_cuda);
	checkCUDAError("Free memory");

	/* free host buffers */
	free(copy_red);


	getchar();
	
	return 0;


	
	//int *h_input, *h_output;
//	int *d_input, *d_output;
	//unsigned int size;
	//int i;




//	size = N * sizeof(int);

	/* allocate the host memory */
//	h_input = (int *)malloc(size);
	//h_output = (int *)malloc(size);

	/* allocate device memory */
	//cudaMalloc((void **)&d_input, size);
	//cudaMalloc((void **)&d_output, size);
//	checkCUDAError("Memory allocation");

	/* read the encryted text */
	//read_encrypted_file(h_input);

	/* copy host input to device input */
	//cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	//checkCUDAError("Input transfer to device");

	/* Configure the grid of thread blocks and run the GPU kernel */
	//dim3 blocksPerGrid(8, 1, 1);
	//dim3 threadsPerBlock(N / 8, 1, 1);
	//affine_decrypt_multiblock << <blocksPerGrid, threadsPerBlock >> >(d_input, d_output);

	/* wait for all threads to complete */
	//cudaThreadSynchronize();
	//checkCUDAError("Kernel execution");

	/* copy the gpu output back to the host */
//	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	//checkCUDAError("Result transfer to host");

	/* print out the result to screen */
	//for (i = 0; i < N; i++) {
	//	printf("%c", (char)h_output[i]);
	//}
	//printf("\n");

	/* free device memory */
	//cudaFree(d_input);
	//cudaFree(d_output);
	//checkCUDAError("Free memory");

	/* free host buffers */
	//free(h_input);
	//free(h_output);

	//return 0;
	
}


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
