#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
//#include <unistd.h>
#include <stdbool.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acp18iur"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
MODE execution_mode = CPU;
int x, y;
__device__ unsigned long long int red_val_cuda, green_val_cuda, blue_val_cuda; // for saving the averages of the global averages.

//function declaration for file reading and previous cpu openmp codes
int check_file_mode();
void get_image_dimensions(FILE *fp, int *x, int *y);
void read_plain_image(uchar4 *, FILE *);
void read_binary_image(uchar4 *);
void writing_plain_text_file(uchar4 *);
void writing_binary_file(uchar4 *image);
void calculate_mosaic_CPU(uchar4 *, int *, int *, int *);
void calculate_mosaic_OPENMP(uchar4 *, int * ,int *, int *);

char *input_file_name;//inputfilename string 
char *output_file_name;//output filename string
char *output_format = "PPM_BINARY";//output format by default set to binary
bool OUTPUT_FORMAT_BINARY = 1;//1 means binary, zero means plain text default binary or 1
typedef struct pixel {// pixel structure to save each cell Red,green,blue value
	int r;
	int g;
	int b;
}pixel;


void checkCUDAError(const char*);


__global__ void image_func_big_c(uchar4 *image_cuda, int row, int column, const int c) {
	int index_x = (c*blockIdx.x) + threadIdx.x;
	int index_y = c * blockIdx.y;
	__shared__ unsigned int red_s, blue_s, green_s;
	
	red_s = 0;
	blue_s = 0;
	green_s = 0;
	__syncthreads();


	int h_c = c;
	if ((index_y + c) >= row) {
		h_c = row - index_y;
	}
	int r_c = c;
	if ((c*blockIdx.x) + c >= column) {
		r_c = column - (c*blockIdx.x);
	}


	for (int i = index_y; i < (index_y + c) && (i < row); i++) {
		if (index_x >= column)continue;
		int position = (i*column) + index_x;
		atomicAdd(&red_s, image_cuda[position].x);
		atomicAdd(&green_s, image_cuda[position].y);
		atomicAdd(&blue_s, image_cuda[position].z);

		atomicAdd(&red_val_cuda, image_cuda[position].x);
		atomicAdd(&green_val_cuda, image_cuda[position].y);
		atomicAdd(&blue_val_cuda, image_cuda[position].z);



		if ((threadIdx.x + 1024) < (r_c)) {
			position = (i*column) + (index_x + 1024);
			atomicAdd(&red_s, image_cuda[position].x);
			atomicAdd(&green_s, image_cuda[position].y);
			atomicAdd(&blue_s, image_cuda[position].z);

			atomicAdd(&red_val_cuda, image_cuda[position].x);
			atomicAdd(&green_val_cuda, image_cuda[position].y);
			atomicAdd(&blue_val_cuda, image_cuda[position].z);


		}


	}

	for (int i = index_y; i < (index_y + c) && (i < row); i++) {
		if (index_x >= column)continue;
		int position = (i*column) + index_x;
		image_cuda[position].x = (red_s / (h_c*r_c));
		image_cuda[position].y = (green_s / (h_c*r_c));
		image_cuda[position].z = (blue_s / (h_c*r_c));

		if ((threadIdx.x + 1024) < c) {
			position = (i*column) + (index_x + 1024);
			image_cuda[position].x = red_s / (h_c*r_c);
			image_cuda[position].y = green_s / (h_c*r_c);
			image_cuda[position].z = blue_s / (h_c*r_c);
		}


	}


}



__global__ void image_func_optimized_reduction(uchar4 *image_cuda, int row, int column, const int c) {
	int index_x = (blockDim.x*blockIdx.x) + threadIdx.x;
	int index_y = c * blockIdx.y;

	// a float 3 array is a array of structures having 3 floats namely x y z perfect for our problem.
	extern __shared__ float3 unified_ar[];
	
	unified_ar[threadIdx.x].x = 0;
	unified_ar[threadIdx.x].y = 0;
	unified_ar[threadIdx.x].z = 0;
	// synchronizing threads to make sure that all values are initialized to zero for shared variable unified_ar.
	__syncthreads();


	//boundary condition for the partial mosaic
	int h_c = c;
	if ((index_y + c) >= row) h_c = row - index_y;
	
	//boundary condition for the partial mosaic
	int r_c = c;
	if ((blockDim.x*blockIdx.x) + c >= column) r_c = column - (blockDim.x*blockIdx.x);
	
	//adding up the values in the row
	for (int i = index_y; i < (index_y + c) && (i < row); i++) {
		if (index_x >= column)continue;//if it is out of bound of the image then ignore
		int position = (i*column) + index_x;

		//adding up the values for global average calculation
		atomicAdd(&red_val_cuda, image_cuda[position].x);
		atomicAdd(&green_val_cuda, image_cuda[position].y);
		atomicAdd(&blue_val_cuda, image_cuda[position].z);
		
		unified_ar[threadIdx.x].x += image_cuda[position].x;
		unified_ar[threadIdx.x].y += image_cuda[position].y;
		unified_ar[threadIdx.x].z += image_cuda[position].z;
	}

	//synchronizing threads to make sure that all the values are successfully added in the particular column.
	__syncthreads();
	

	//applying reduction on code
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			unified_ar[threadIdx.x].x += unified_ar[threadIdx.x + stride].x;
			unified_ar[threadIdx.x].y += unified_ar[threadIdx.x + stride].y;
			unified_ar[threadIdx.x].z += unified_ar[threadIdx.x + stride].z;		
		}
		__syncthreads();
	}

	//adding up the values back
	for (int i = index_y; i < (index_y + c) && (i < row); i++) {
		if (index_x >= column)continue;//if it is out of bound of the image then ignore
		int position = (i*column) + index_x;
		image_cuda[position].x = unified_ar[0].x / (h_c*r_c);
		image_cuda[position].y = unified_ar[0].y / (h_c*r_c);
		image_cuda[position].z = unified_ar[0].z / (h_c*r_c);
	}
}



__global__ void image_func_optimized(uchar4 *image_cuda, int row, int column, const int cc) {
	int index_x = (blockDim.x*blockIdx.x) + threadIdx.x;
	int index_y = cc * blockIdx.y;

	__shared__ int red_s, blue_s, green_s;
	red_s = 0;
	blue_s = 0;
	green_s = 0;
	__syncthreads();

	int h_c = cc;
	if ((index_y + cc) >= row) {
		h_c = row - index_y;
	}
	int r_c = cc;
	if ((blockDim.x*blockIdx.x) + cc >= column) {
		r_c = column - (blockDim.x*blockIdx.x);
	}

	for (int i = index_y; i < (index_y + cc) && (i < row); i++) {
		if (index_x >= column)continue;

		int position = (i*column) + index_x;
		atomicAdd(&red_s, image_cuda[position].x);
		atomicAdd(&green_s, image_cuda[position].y);
		atomicAdd(&blue_s, image_cuda[position].z);
	}
	__syncthreads();

	for (int i = index_y; i < (index_y + cc) && (i < row); i++) {
		if (index_x >= column)continue;
		int position = (i*column) + index_x;
		image_cuda[position].x = (red_s / (h_c*r_c));
		image_cuda[position].y = (green_s / (h_c*r_c));
		image_cuda[position].z = (blue_s / (h_c*r_c));
	}


}


__global__ void image_func(int *red_cuda, int *green_cuda, int *blue_cuda, int row, int column, const int cc)
{
	const int ccc = cc * cc;
	__shared__ int red_s;
	__shared__ int blue_s;
	__shared__ int green_s;
	red_s = 0;
	blue_s = 0;
	green_s = 0;
	__syncthreads();

	int index_x = blockDim.x*blockIdx.x + threadIdx.x;
	int index_y = blockDim.y*blockIdx.y + threadIdx.y;
	int position = (index_x*column) + index_y;
	//	int p2 = (threadIdx.x*blockDim.y) + threadIdx.y;
	/*
	if (threadIdx.x == 0 && threadIdx.y == 0){// && blockIdx.x==2 && blockIdx.y==1) {
	//printf("Adding it\n");
	int red_s = 0;
	int green_s = 0;
	int blue_s = 0;
	//printf("__%d %d__", index_x, index_y);

	for (int i = index_x; i < index_x + cc; i++) {
	for (int j = index_y; j < index_y + cc; j++) {

	//	printf("%d %d __", i, j);
	red_s += red_cuda[j*column + i];
	green_s += green_cuda[j*column + i];
	blue_s += blue_cuda[j*column + i];
	}
	//	printf("\n");
	}
	//printf("\n\n\n");
	//atomicAdd(&red_s, red_cuda[position]);
	//atomicAdd(&green_s, green_cuda[position]);
	//atomicAdd(&blue_s, blue_cuda[position]);
	//__syncthreads();


	for (int i = index_x; i < index_x + cc; i++) {
	for (int j = index_y; j < index_y + cc; j++) {
	red_cuda[j*column + i] = (red_s / ccc);
	green_cuda[j*column + i] = (green_s / ccc);
	blue_cuda[j*column + i] =  (blue_s / ccc);
	}
	}
	}*/
	position = (index_y*column) + index_x;

	atomicAdd(&red_s, red_cuda[position]);
	atomicAdd(&green_s, green_cuda[position]);
	atomicAdd(&blue_s, blue_cuda[position]);
	__syncthreads();
	red_cuda[position] = (red_s / ccc);
	green_cuda[position] = (green_s / ccc);
	blue_cuda[position] = (blue_s / ccc);
	/*
	int sum_red = 0;
	int sum_green = 0;
	int sum_blue = 0;

	for (int i = index_x; i < index_x + cc; i++) {
	for (int j = index_y; j < index_y + cc; j++) {
	int off = (i * column) + j;
	sum_red += red_cuda[off];
	sum_green += green_cuda[off];
	sum_blue += blue_cuda[off];
	}
	}
	for (int i = index_x; i < index_x + cc; i++) {
	for (int j = index_y; j < index_y + cc; j++) {
	int off = (i * column) + j;
	red_cuda[off] = sum_red / (cc*cc);
	green_cuda[off] = sum_green / (cc*cc);
	blue_cuda[off] = sum_blue / (cc*cc);
	}
	}
	*/

}

// the function for handling the cuda input.
float cuda_mode(uchar4 *input_image, unsigned long long int *avg_r, unsigned long long int *avg_g, unsigned long long int *avg_b) {
	
	uchar4 *image_cuda;
	int size_of_image = x * y * sizeof(uchar4);
	*avg_r = 0;
	*avg_g = 0;
	*avg_b = 0;
	

	
	cudaMalloc((void **)&image_cuda, size_of_image);
	checkCUDAError("Memory allocation");

	// copy host input to device input 
	
	cudaMemcpyToSymbol(red_val_cuda, avg_r, sizeof(unsigned long long int));
	cudaMemcpyToSymbol(green_val_cuda, avg_g, sizeof(unsigned long long int));
	cudaMemcpyToSymbol(blue_val_cuda, avg_b, sizeof(unsigned long long int));


	cudaMemcpy(image_cuda, input_image, size_of_image, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");

	//printf("Value of c is %d\n", c);
	//printf("Starting kernel\n");

	int blockdimx = x / c;
	int blockdimy = y / c;
	if (x%c != 0) {
		blockdimx += 1;
	}
	if (y%c != 0) {
		blockdimy += 1;
	}
	//printf("\nThe block dimensions set are: %d %d\n", blockdimx, blockdimy);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	dim3 blocksPerGrid(blockdimx, blockdimy, 1);
	unsigned int sm_size = sizeof(float3)*c ;
	if (c > 1024) {
		//printf("C is greater than 1024 hence calling another kernel function for big c values...\n");
		dim3 threadsPerBlock(1024, 1, 1);
		image_func_big_c << <blocksPerGrid, threadsPerBlock >> > (image_cuda, y, x, c);
	}
	else {
	//	printf("C is less than 1024 calling function for this C..\n");
		dim3 threadsPerBlock(c, 1, 1);
		//image_func_optimized << <blocksPerGrid, threadsPerBlock >> > (image_cuda, y, x, c);
		image_func_optimized_reduction << <blocksPerGrid, threadsPerBlock, sm_size >> > (image_cuda, y, x, c);
	}
	
	

	//	image_func << <blocksPerGrid, threadsPerBlock >> >(red_cuda, green_cuda, blue_cuda, x, y, c);


	//wait for all threads to complete
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// copy the gpu output back to the host 
	cudaMemcpyFromSymbol(avg_r, red_val_cuda, sizeof(unsigned long long int));
	cudaMemcpyFromSymbol(avg_g, green_val_cuda, sizeof(unsigned long long int));
	cudaMemcpyFromSymbol(avg_b, blue_val_cuda, sizeof(unsigned long long int));

	*avg_r /= (x*y);
	*avg_g /= (x*y);
	*avg_b /= (x*y);


	cudaMemcpy(input_image, image_cuda, size_of_image, cudaMemcpyDeviceToHost);

	checkCUDAError("Result transfer to host");

	


	//free device memory
	cudaFree(image_cuda);
	//cudaFree(red_cuda);
	//cudaFree(green_cuda);
	//cudaFree(blue_cuda);
	checkCUDAError("Free memory");
	return milliseconds;
}
void print_image_pretty(uchar4 *image) {
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			int position = (i * y) + j;
			printf("%d %d %d    ", image[position].x, image[position].y, image[position].y);
		}
		printf("\n");

	}
}


int main(int argc, char *argv[]) {

	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	int file_mode = check_file_mode();//checks the filemode 
	printf("Reading image.. \n");
	FILE *fp = fopen(input_file_name, "r");
	get_image_dimensions(fp, &x, &y);
	printf("The Dimensions of the image is: %d %d\n", x, y);
	uchar4* input_image = (uchar4*)malloc(x*y * sizeof(uchar4));
	unsigned long long int avg_r, avg_g, avg_b;

	uchar4* red = (uchar4*)malloc(x*y * sizeof(uchar4));
	uchar4* green = (uchar4*)malloc(x *y * sizeof(uchar4));
	uchar4* blue = (uchar4*)malloc(x *y * sizeof(uchar4));
	uchar4 *copy_red = red;
	uchar4 *copy_green = green;
	uchar4 *copy_blue = blue;

	if (file_mode == 1) {
		//printf("the image is plain text");
		read_plain_image(input_image, fp);
	}
	else {
		//printf("the image is binary\n");
		read_binary_image(input_image);
	}
	fclose(fp);






	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		printf("###############################__USING CPU__############################################n\n");
		double start_time = omp_get_wtime();

		int gr, gg, gb;
		printf("\n");
		calculate_mosaic_CPU(input_image, &gr, &gg, &gb);
		double time = omp_get_wtime() - start_time;
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(input_image);
		}
		else {
			printf("Writing plain text: \n");
			//	print_image_pretty(red, green, blue);
			writing_plain_text_file(input_image);
		}
		printf("CPU Average image colour red = %d, green = %d, blue = %d \n", gr, gg, gb);
		printf("CPU mode execution time took %f ms\n", time * 1000);
		break;
	}
	case (OPENMP): {
		printf("######################################____Using OpenMP____#######################################################\n");
		//TODO: starting timing here
		double start_time = omp_get_wtime();
		printf("USING OPENMP");
		int gr, gg, gb;
		calculate_mosaic_OPENMP(input_image, &gr, &gg, &gb);
		double time = omp_get_wtime() - start_time;
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(input_image);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(input_image);
		}
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", gr, gg, gb);
		printf("OPENMP mode execution time %f ms\n", time * 1000);
		break;
	}
	case (CUDA): {
		printf("####################################____CUDA MODE____#################################################################\n");
		float time_taken = cuda_mode(input_image, &avg_r, &avg_g, &avg_b);
		printf("CUDA Average image colour red = %zu, green = %zu, blue = %zu \n", avg_r, avg_g, avg_b);
		printf("CUDA mode execution time took  %f ms\n", time_taken);
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(input_image);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(input_image);
		}
		break;
	}
	case (ALL): {
		printf("####################################____ALL MODE____#################################################################\n");

		double start_time = omp_get_wtime();
		int gr, gg, gb;

		printf("####################################____USING CPU____#################################################################\n");

		calculate_mosaic_CPU(input_image, &gr, &gg, &gb);
		double time1 = omp_get_wtime() - start_time;

		FILE *fp = fopen(input_file_name, "r");
		get_image_dimensions(fp, &x, &y);
		if (file_mode == 1) {
			printf("the image is plain text");
			read_plain_image(input_image,fp);
		}
		else {
			printf("the image is binary\n");
			read_binary_image(input_image);
		}
		fclose(fp);
		printf("####################################____USING OPENMP____#################################################################\n");

		start_time = omp_get_wtime();
		calculate_mosaic_OPENMP(input_image, &gr, &gg, &gb);
		double time2 = omp_get_wtime() - start_time;
		
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(input_image);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(input_image);
		}

		//reading again for cuda
		FILE *fp1 = fopen(input_file_name, "r");
		get_image_dimensions(fp1, &x, &y);
		if (file_mode == 1) {
			read_plain_image(input_image, fp);
		}
		else {
			read_binary_image(input_image);
		}
		fclose(fp);

		printf("####################################____USING CUDA____#################################################################\n");

		float time_taken = cuda_mode(input_image, &avg_r, &avg_g, &avg_b);
		printf("Average image colour red = %zu, green = %zu, blue = %zu \n", avg_r, avg_g, avg_b);
		
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(input_image);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(input_image);
		}


		printf("CPU TIME IS %fms\nOPENMP time is: %fms\n", time1 * 1000, time2 * 1000);
		printf("CUDA TIME IS  %f ms\n", time_taken);
		break;
	}
	}



	free(copy_red);
	free(copy_green);
	free(copy_blue);
	return 0;
}

void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}


	c = (unsigned int)atoi(argv[1]);



	char *argv1 = argv[2];
	//To set the mode user is giving
	if (strcmp(argv1, "CPU") == 0) {
		execution_mode = CPU;
	}
	else if (strcmp(argv1, "OPENMP") == 0) {
		execution_mode = OPENMP;
	}
	else if (strcmp(argv1, "ALL") == 0) {
		execution_mode = ALL;
	}
	else {
		printf("using cuda mode\n\n");
		execution_mode = CUDA;

	}

//	char *mode = argv[3];
	input_file_name = argv[4];
	output_file_name = argv[6];
	char *mode1 = argv[7];
	output_format = argv[8];

	if (mode1 != NULL) {
		if (strcmp(mode1, "-f") == 0) {
			if (strcmp(output_format, "PPM_PLAIN_TEXT") == 0) {
				OUTPUT_FORMAT_BINARY = 0;// zero means plaintext 1 means binary
			}
		}
	}



	int debug = 1;// For sanity check printing the user input given
	if (debug == 1) {
		printf("Some description of user input: \n");
		printf("Input filename given: %s \n", input_file_name);
		printf("Output_filename_ given: %s \n", output_file_name);
		printf("The value of C given is %d\n", c);
		//	printf("output_file_mode: %s \n ", output_format);
		//printf(" THE MODE: %d\n", OUTPUT_FORMAT_BINARY);
	}
	return SUCCESS;

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
	int remove_warnings;
	remove_warnings = fscanf(fp, "%d %d", x, y);
	remove_warnings = fscanf(fp, "%d", &char_len);
	remove_warnings++;

	//below two lines are just there to make sure there is no warnings
	ch1 = ch + 1;
	ch = ch1 + 1;
}


//function to check whether the input file is in binary mode or plain text
int check_file_mode() {
	FILE *check_file = fopen(input_file_name, "rb");
	char ch1, ch;//for reading the mode
	ch = fgetc(check_file);//getting the first character
	ch1 = fgetc(check_file);//getting the second character
	fclose(check_file);

	if (ch1 == '3') {
		return 1;	// if it is binary it returns 1 
	}

	//just to make sure there are no warnings
	ch1 = ch + 1;//useless
	ch = ch1 + 2;//useless
	return 0;
}


void read_plain_image(uchar4 *input_image, FILE *fp) {
	for (int i = 0; i<x; i++) {
		for (int j = 0; j<y; j++) {
			unsigned int r, g, b;
			fscanf(fp, "%d %d %d", &r, &g, &b);
			char useless;
			useless = getc(fp);
			useless = 1 + useless;
			int position = (i * y) + j;
			input_image[position].x = r;
			input_image[position].y = g;
			input_image[position].z = b;
		}
	}
}



void read_binary_image(uchar4 *input_image) {
	FILE *fp_binary = fopen(input_file_name, "rb");
	get_image_dimensions(fp_binary, &x, &y);
	unsigned char currentPixel[3];
	unsigned char a;
	size_t useless_for_warnings = fread(&a, 1, 1, fp_binary);
	for (int i = 0; i<x; i++) {
		for (int j = 0; j<y; j++) {
			useless_for_warnings = fread(currentPixel, 3, 1, fp_binary);
			int r, g, b;
			r = currentPixel[0];
			g = currentPixel[1];
			b = currentPixel[2];
			int position = (i * y) + j;
			input_image[position].x = r;
			input_image[position].y = g;
			input_image[position].z = b;

		}
	}

	fclose(fp_binary);
}



void writing_plain_text_file(uchar4 *image) {
	FILE *outfile = fopen(output_file_name, "w");
	fprintf(outfile, "P3\n");// since it is plain text
	fprintf(outfile, "# COM4521 Assignment test output\n");
	fprintf(outfile, "%d\n%d\n", x, y);
	fprintf(outfile, "255");//by default
	for (int i = 0; i<x; i++) {
		fprintf(outfile, "\n");
		for (int j = 0; j<y; j++) {
			int r, g, b;
			int position = (i * y) + j;
			r = image[position].x;
			g = image[position].y;
			b = image[position].z;
			if (j == (y - 1)) {
				fprintf(outfile, "%d %d %d", r, g, b);//for last value no need to add tab
			}
			else {
				fprintf(outfile, "%d %d %d\t", r, g, b);
			}
		}
	}
	fclose(outfile);
}


//for writing the binary file, it gets the filename globally and pointer to array
void writing_binary_file(uchar4 *image) {
	FILE *outfile = fopen(output_file_name, "wb");//using wb parameter for binary
	fprintf(outfile, "P6\n");//means that is is binary
	fprintf(outfile, "# COM4521 Assignment test output\n");
	fprintf(outfile, "%d\n%d\n", x, y);
	fprintf(outfile, "255\n");//by default
	for (int i = 0; i<x; i++) {
		for (int j = 0; j<y; j++) {
			int r, g, b;
			int position = (i * y) + j;
			r = image[position].x;
			g = image[position].y;
			b = image[position].z;
			static unsigned char color[3];
			//to get the bytes representation of rgb
			color[0] = r % 256;  /* red */
			color[1] = g % 256;  /* green */
			color[2] = b % 256;  /* blue */
			(void)fwrite(color, 1, 3, outfile);
		}
	}
	fclose(outfile);//closing the file
}


//to calculate the mosaic and average in CPU mode
//the gr ,gg,and gb is used to get the average of r g and b values
void calculate_mosaic_CPU(uchar4 *input_image, int *gr, int *gg, int *gb) {
	printf("STARTING THE MOSAIC OPERATION USING CPU APPROACH\n\n\n");
	int global_average_r = 0;//initialized to zero
	int global_average_g = 0;
	int global_average_b = 0;
	for (int i = 0; i<x; i += c) {//accesing matrix by squares
		for (int j = 0; j<y; j += c) {
			int average_r, average_b, average_g;
			average_b = 0;
			average_r = 0;
			average_g = 0;
			int cell_count = 0;
			for (int ii = i; (ii< (i + c)) && (ii<x); ii++) {//accessing the squares
				for (int jj = j; jj<(j + c) && jj<y; jj++) {
					int position = (ii * y) + jj;
					int r = input_image[position].x;
					int g = input_image[position].y;
					int b = input_image[position].z;
					average_r += r;
					average_b += b;
					average_g += g;
					global_average_r += r;
					global_average_g += g;
					global_average_b += b;
					cell_count++;
				}
			}
			average_b = average_b / (cell_count);//averaging
			average_r = average_r / (cell_count);//averaging
			average_g = average_g / (cell_count);//averaging

												 //updating
			for (int ii = i; ii< (i + c) && ii<x; ii++) {//updating the value in main matrix
				for (int jj = j; jj<(j + c) && jj<y; jj++) {
					int position = (ii * y) + jj;
					input_image[position].x = average_r;
					input_image[position].y = average_g;
					input_image[position].z = average_b;
				}
			}
		}
	}
	global_average_r /= (x*y);//finding the average for the matrix
	global_average_g /= (x*y);//finding the average for the matrix
	global_average_b /= (x*y);//finding the average for the matrix
	*gg = global_average_g;//setting up the pointer
	*gr = global_average_r;//setting up the pointer
	*gb = global_average_b;//setting up the pointer

}






void calculate_mosaic_OPENMP(uchar4 *input_image, int *gr, int *gg, int *gb) {
	printf("STARTING THE MOSAIC OPERATION USING OPENMP APPROACH\n\n\n");
	int global_average_r = 0;
	int global_average_g = 0;
	int global_average_b = 0;
	int NUM_THREADS = 50;
	NUM_THREADS += 5;
	NUM_THREADS -= 5;
	int cell_count = 0;
	int i = 0;
	int j = 0;
	int ii = 0;
	int jj = 0;
	int average_r = 0;
	int average_g = 0;
	int average_b = 0;

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) private(i,j,ii,jj) reduction(+: average_r, average_b, average_g,cell_count,global_average_r,global_average_g,global_average_b)
	for (i = 0; i<x; i += c) {
		for (j = 0; j<y; j += c) {
			//int average_r, average_b, average_g;
			average_b = 0;
			average_r = 0;
			average_g = 0;
			cell_count = 0;
			for (ii = i; ii< (i + c) && ii<x; ii++) {
				for (jj = j; jj<(j + c) && jj<y; jj++) {
					int position = (ii * y) + jj;
					int r = input_image[position].x;
					int g = input_image[position].y;
					int b = input_image[position].z;
					average_r += r;
					average_b += b;
					average_g += g;
					global_average_r += r;
					global_average_g += g;
					global_average_b += b;
					cell_count++;
				}
			}
			average_b = average_b / (cell_count);
			average_r = average_r / (cell_count);
			average_g = average_g / (cell_count);

			//updating
			for (int ii = i; ii< (i + c) && ii<x; ii++) {
				for (int jj = j; jj<(j + c) && jj<y; jj++) {
					int position = (ii * y) + jj;
					input_image[position].x = average_r;
					input_image[position].y = average_g;
					input_image[position].z = average_b;

				}
			}
		}
	}
	global_average_r /= (x*y);
	global_average_g /= (x*y);
	global_average_b /= (x*y);
	*gg = global_average_g;
	*gr = global_average_r;
	*gb = global_average_b;
}

