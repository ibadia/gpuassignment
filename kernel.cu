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
__device__ int c = 0;
__device__ int row = 0;
__device__ int column = 0;

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

__global__ void change_c_value(int newc, int r, int co) {
	c = newc;
	row = r;
	column = co;
}
__global__ void image_func(int *red_cuda, int *green_cuda, int *blue_cuda)
{

	int index_x = c * (blockDim.x*blockIdx.x + threadIdx.x);
	int index_y = c * (blockDim.y*blockIdx.y + threadIdx.y);
	int offset = (index_x * 16) + index_y;

	int sum_red = 0;
	int sum_green = 0;
	int sum_blue = 0;
	for (int i = index_x; i < index_x + c; i++) {
		for (int j = index_y; j < index_y + c; j++) {
			int off = (i * column) + j;
			sum_red += red_cuda[off];
			sum_green += green_cuda[off];
			sum_blue += blue_cuda[off];
		}
	}
	for (int i = index_x; i < index_x + c; i++) {
		for (int j = index_y; j < index_y + c; j++) {
			int off = (i * column) + j;
			red_cuda[off] = sum_red / (c*c);
			green_cuda[off] = sum_green / (c*c);
			blue_cuda[off] = sum_blue / (c*c);
		}
	}

}




int main(int argc, char *argv[])
{
	//reading the input file
	input_file_name = "DogPlainText2048x2048.ppm";
	printf("%s", input_file_name);
	FILE *fp = fopen(input_file_name, "r");
	int x, y;
	get_image_dimensions(fp, &x, &y);
	printf("%d %d\n\n", x, y);
	int* red = (int*)malloc(x*y * sizeof(int));
	int* green = (int*)malloc(x *y * sizeof(int));
	int* blue = (int*)malloc(x *y * sizeof(int));
	int *copy_red = red;
	int *copy_green = green;
	int *copy_blue = blue;


	int* red_cuda, *green_cuda, *blue_cuda;


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
	
	printf("____________________________________________\n");


	int size_of_image = x * y * sizeof(int);
	printf("%d", size_of_image);



	cudaMalloc((void **)&red_cuda, size_of_image);
	cudaMalloc((void **)&green_cuda, size_of_image);
	cudaMalloc((void **)&blue_cuda, size_of_image);

	checkCUDAError("Memory allocation");

	/* copy host input to device input */
	cudaMemcpy(red_cuda, red, size_of_image, cudaMemcpyHostToDevice);
	cudaMemcpy(green_cuda, green, size_of_image, cudaMemcpyHostToDevice);
	cudaMemcpy(blue_cuda, blue, size_of_image, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");


	/* Configure the grid of thread blocks and run the GPU kernel */
	printf("hello");
	getchar();
	
	int c = 64;
	dim3 blocksPerGrid(1,1, 1);
	dim3 threadsPerBlock(4, 4, 1);
	
	change_c_value << <blocksPerGrid, threadsPerBlock >> > (c,x,y);


	image_func << <blocksPerGrid, threadsPerBlock >> >(red_cuda, green_cuda, blue_cuda);


	/* wait for all threads to complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");

	/* copy the gpu output back to the host */
	cudaMemcpy(red, red_cuda, size_of_image, cudaMemcpyDeviceToHost);
	cudaMemcpy(green, green_cuda, size_of_image, cudaMemcpyDeviceToHost);
	cudaMemcpy(blue, blue_cuda, size_of_image, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");


	printf("\n_______________________\n\n\n");

	/*
	for(int i = 0; i < x; i++) {
	for (int j = 0; j < y; j++) {
	int position = (i * y) + j;
	printf("%d %d %d    ", red[position], green[position], blue[position]);
	}
	printf("\n");
	}*/

	output_file_name = "out.ppm";
	FILE *outfile = fopen(output_file_name, "w");
	fprintf(outfile, "P3\n");// since it is plain text
	fprintf(outfile, "# COM4521 Assignment test output\n");
	fprintf(outfile, "%d\n%d\n", x, y);
	fprintf(outfile, "255");//by default
	for (int i = 0; i<x; i++) {
		fprintf(outfile, "\n");
		for (int j = 0; j<y; j++) {
			int r, g, b;
			int position = (i*y) + j;
			r = red[position];
			g = green[position];
			b = blue[position];
			if (j == (y - 1)) {
				fprintf(outfile, "%d %d %d", r, g, b);//for last value no need to add tab
			}
			else {
				fprintf(outfile, "%d %d %d\t", r, g, b);
			}
		}
	}
	fclose(outfile);



	/* free device memory */
	cudaFree(red_cuda);
	cudaFree(green_cuda);
	cudaFree(blue_cuda);
	checkCUDAError("Free memory");

	/* free host buffers */
	free(copy_red);
	free(copy_green);
	free(copy_blue);


	getchar();

	return 0;
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