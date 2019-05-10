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

#define USER_NAME "aca00pr"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
//__device__ int cc = 0;
//__device__ int row = 0;
//__device__ int column = 0;
unsigned int c = 0;
MODE execution_mode = CPU;
int x, y;


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
void read_plain_image(int *red, int *green, int *blue, FILE *fp) {
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
}
void read_binary_image(int *red, int *green, int *blue) {
	FILE *fp_binary = fopen(input_file_name, "rb");
	get_image_dimensions(fp_binary, &x, &y);
	unsigned char currentPixel[3];
	unsigned char a;
	int useless_for_warnings = fread(&a, 1, 1, fp_binary);
	for (int i = 0; i<x; i++) {
		for (int j = 0; j<y; j++) {
			useless_for_warnings = fread(currentPixel, 3, 1, fp_binary);
			int r, g, b;
			r = currentPixel[0];
			g = currentPixel[1];
			b = currentPixel[2];
			int position = (i * y) + j;
			red[position] = r;
			green[position] = g;
			blue[position] = b;
		}
	}

	fclose(fp_binary);
}

void writing_plain_text_file(int *red, int *green, int *blue) {
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
}

//for writing the binary file, it gets the filename globally and pointer to array
void writing_binary_file(int *red, int *green, int *blue) {
	FILE *outfile = fopen(output_file_name, "wb");//using wb parameter for binary
	fprintf(outfile, "P6\n");//means that is is binary
	fprintf(outfile, "# COM4521 Assignment test output\n");
	fprintf(outfile, "%d\n%d\n", x, y);
	fprintf(outfile, "255\n");//by default
	for (int i = 0; i<x; i++) {
		for (int j = 0; j<y; j++) {
			int r, g, b;
			int position = (i * y) + j;
			r = red[position];
			g = green[position];
			b = blue[position];
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
void calculate_mosaic_CPU(int *red, int *green, int *blue, int *gr, int *gg, int *gb) {
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
			for (int ii = i; ii< (i + c) && ii<x; ii++) {//accessing the squares
				for (int jj = j; jj<(j + c) && jj<y; jj++) {
					int position = (ii * y) + jj;
					int r = red[position];
					int g = green[position];
					int b = blue[position];
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
					red[position] = average_r;
					blue[position] = average_b;
					green[position] = average_g;
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
void calculate_mosaic_OPENMP(int *red, int *green, int *blue, int *gr, int *gg, int *gb) {
	printf("STARTING THE MOSAIC OPERATION USING OPENMP APPROACH\n\n\n");
	int global_average_r = 0;
	int global_average_g = 0;
	int global_average_b = 0;
	int NUM_THREADS = 50;
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
					int r = red[position];
					int g = green[position];
					int b = blue[position];
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

					red[position] = average_r;
					blue[position] = average_b;
					green[position] = average_g;
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
/*
__global__ void change_c_value(int newc, int r, int co) {
cc = newc;
row = r;
column = co;
}*/
__global__ void image_func(int *red_cuda, int *green_cuda, int *blue_cuda, int row, int column, const int cc)
{
	const int ccc = cc * cc;
	

	//int index_x = cc * (blockDim.x*blockIdx.x + threadIdx.x);
	//int index_y = cc * (blockDim.y*blockIdx.y + threadIdx.y);


	int index_x = blockDim.x*blockIdx.x + threadIdx.x;
	int index_y = blockDim.y*blockIdx.y + threadIdx.y;
	int position = (index_x*column) + index_y;
	int p2 = (threadIdx.x*blockDim.y) + threadIdx.y;
	
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
	}


	//red_cuda[position] = red_s / (ccc);
	//green_cuda[position] = green_s / (ccc);
//	blue_cuda[position] = blue_s / (ccc);



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


void cuda_mode(int *red, int *green, int *blue) {
	int* red_cuda, *green_cuda, *blue_cuda;
	int size_of_image = x * y * sizeof(int);
	printf("dimensions of image is: %d %d \n\n", x, y);
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
	printf("Value of c is %d\n", c);
	printf("Starting kernel\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int blockdimx = x / c;
	int blockdimy = y / c;
	printf("\nThe block dimensions set are as follows: %d %d\n", blockdimx, blockdimy);
	printf("The threads per block is : %d %d\n", c, c);


	dim3 blocksPerGrid(blockdimx, blockdimy, 1);
//	dim3 blocksPerGrid(2, 2, 1);
	dim3 threadsPerBlock(c, c, 1);

	//change_c_value << <blocksPerGrid, threadsPerBlock >> > (c, x, y);


	image_func << <blocksPerGrid, threadsPerBlock >> >(red_cuda, green_cuda, blue_cuda, x, y, c);
	

	/* wait for all threads to complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cuda run sucessfully\n\n\n");
	getchar();

	/* copy the gpu output back to the host */
	cudaMemcpy(red, red_cuda, size_of_image, cudaMemcpyDeviceToHost);
	cudaMemcpy(green, green_cuda, size_of_image, cudaMemcpyDeviceToHost);
	cudaMemcpy(blue, blue_cuda, size_of_image, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");



	/* free device memory */
	cudaFree(red_cuda);
	cudaFree(green_cuda);
	cudaFree(blue_cuda);
	checkCUDAError("Free memory");
	printf("Time taken by the gpu is %f\n", milliseconds);
	getchar();

}
void print_image_pretty(int *red, int *green, int *blue) {
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			int position = (i * y) + j;
			printf("%d %d %d    ", red[position], green[position], blue[position]);
		}
		printf("\n");

	}
}


int main(int argc, char *argv[]) {

	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	int file_mode = check_file_mode();//checks the filemode 
	printf("Reading image");
	FILE *fp = fopen(input_file_name, "r");
	get_image_dimensions(fp, &x, &y);

	int* red = (int*)malloc(x*y * sizeof(int));
	int* green = (int*)malloc(x *y * sizeof(int));
	int* blue = (int*)malloc(x *y * sizeof(int));
	int *copy_red = red;
	int *copy_green = green;
	int *copy_blue = blue;

	if (file_mode == 1) {
		printf("the image is plain text");
		read_plain_image(red, green, blue, fp);
		//print_image_pretty(red, green, blue);
	}
	else {
		printf("the image is binary\n");
		read_binary_image(red, green, blue);
		//print_image_pretty(red, green, blue);

	}
	fclose(fp);






	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		printf("USING CPU");
		//TODO: starting timing here
		double start_time = omp_get_wtime();

		//TODO: calculate the average colour value
		int gr, gg, gb;
		printf("\n");
		//	print_image_pretty(red, green, blue);
		printf("this is the image");
		calculate_mosaic_CPU(red, green, blue, &gr, &gg, &gb);
		double time = omp_get_wtime() - start_time;
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(red, green, blue);
		}
		else {
			printf("Writing plain text: \n");
			//	print_image_pretty(red, green, blue);
			writing_plain_text_file(red, green, blue);
		}

		printf("CPU Average image colour red = %d, green = %d, blue = %d \n", gr, gg, gb);

		printf("CPU mode execution time took %f s and %f ms\n", time, time * 1000);
		break;
	}
	case (OPENMP): {
		printf("Using OpenMP\n");
		//TODO: starting timing here
		double start_time = omp_get_wtime();
		printf("USING OPENMP");
		//TODO: calculate the average colour value
		int gr, gg, gb;
		calculate_mosaic_OPENMP(red, green, blue, &gr, &gg, &gb);
		double time = omp_get_wtime() - start_time;
		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(red, green, blue);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(red, green, blue);
		}
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", gr, gg, gb);
		printf("OPENMP mode execution time took %f s and  %f ms\n", time, time * 1000);
		break;
	}
	case (CUDA): {
		printf("CUDA Implementation\n");
		//print_image_pretty(red, green, blue);
		printf("\n\n");

		printf("Calling cuda function\n");
		cuda_mode(red, green, blue);
		printf("sd");
		//print_image_pretty(red, green, blue);

		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(red, green, blue);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(red, green, blue);
		}


		break;
	}
	case (ALL): {
		double start_time = omp_get_wtime();
		int gr, gg, gb;

		calculate_mosaic_CPU(red, green, blue, &gr, &gg, &gb);
		double time1 = omp_get_wtime() - start_time;



		FILE *fp = fopen(input_file_name, "r");
		get_image_dimensions(fp, &x, &y);
		if (file_mode == 1) {
			printf("the image is plain text");
			read_plain_image(red, green, blue, fp);
			//print_image_pretty(red, green, blue);
		}
		else {
			printf("the image is binary\n");
			read_binary_image(red, green, blue);
			//print_image_pretty(red, green, blue);

		}
		fclose(fp);

		start_time = omp_get_wtime();
		calculate_mosaic_OPENMP(red, green, blue, &gr, &gg, &gb);
		double time2 = omp_get_wtime() - start_time;
		printf("ALL Average image colour red = %d, green = %d, blue = %d \n", gr, gg, gb);
		printf("CPU TIME IS %f milliseconds.. OPENMP time is: %f\n\n", time1 * 1000, time2 * 1000);

		if (OUTPUT_FORMAT_BINARY) {
			printf("Writing Binary File: \n");
			writing_binary_file(red, green, blue);
		}
		else {
			printf("Writing plain text: \n");
			writing_plain_text_file(red, green, blue);
		}

		FILE *fp1 = fopen(input_file_name, "r");
		get_image_dimensions(fp1, &x, &y);
		if (file_mode == 1) {
			printf("the image is plain text");
			read_plain_image(red, green, blue, fp1);
			//print_image_pretty(red, green, blue);
		}
		else {
			printf("the image is binary\n");
			read_binary_image(red, green, blue);
			//print_image_pretty(red, green, blue);

		}
		fclose(fp);






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

	//first argument is always the executable name

	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);



	//TODO: read in the mode
	printf("Mode specified is: %s", argv[2]);
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


	/*Since the file will be in the mode of -i input filename,
	and -o outputfilename -f PPL_PLAIN hence for that the
	c language getopt library is used
	*/
	char *mode = argv[3];
	input_file_name = argv[4];
	output_file_name = argv[6];

	char *mode1 = argv[7];
	output_format = argv[8];
	printf("%s", mode1);
	getchar();
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
		//	printf("output_file_mode: %s \n ", output_format);
		printf(" THE MODE: %d\n", OUTPUT_FORMAT_BINARY);
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