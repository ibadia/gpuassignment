#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <stdbool.h>
#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "aca00pr"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
int x,y;
MODE execution_mode = CPU;
char *input_file_name;//inputfilename string 
char *output_file_name;//output filename string
char *output_format="PPM_BINARY";//output format by default set to binary
bool OUTPUT_FORMAT_BINARY=1;//1 means binary, zero means plain text default binary or 1
typedef struct pixel{// pixel structure to save each cell Red,green,blue value
	int r;
	int g;
	int b;
}pixel;

void allocateMatrix(int n, pixel* (**matrix)[n])
{// to dynamically create a multidimensional array of struct when executing
    *matrix = malloc(n * sizeof(**matrix));
}

void get_image_dimensions(FILE *fp, int *x, int *y){// to get the dimensions of the image
	int char_len;
	char ch,ch1;
	ch=fgetc(fp);
	ch1=fgetc(fp);
	
	while (fgetc(fp)!='\n'){
		continue;
	}
	char cc;
	cc=fgetc(fp);
	while (cc == '#') {
	while (getc(fp) != '\n') ;
         cc = getc(fp);
    }
	ungetc(cc,fp);
	fscanf(fp, "%d %d", x,y);
	fscanf(fp, "%d",&char_len);

	//below two lines are just there to make sure there is no warnings
	ch1=ch+1;
	ch=ch1+1;
}


//for writing the PLAIN TEXT FILE it receives the filename and the pointer to array
void writing_plain_text_file(pixel* (*matrix)[x]){
	FILE *outfile=fopen(output_file_name, "w");
	fprintf(outfile,"P3\n");// since it is plain text
	fprintf(outfile,"# COM4521 Assignment test output\n");
	fprintf(outfile, "%d\n%d\n", x,y);
	fprintf(outfile, "255");//by default
	for (int i=0; i<x; i++){
		fprintf(outfile,"\n");
		for(int j=0; j<y; j++){
			int r,g,b;
			r=matrix[i][j]->r;
			g=matrix[i][j]->g;
			b=matrix[i][j]->b;
			if (j==(y-1)){
				fprintf(outfile,"%d %d %d",r,g,b );//for last value no need to add tab
			}else{
				fprintf(outfile,"%d %d %d\t",r,g,b );
			}
		}
	}
	fclose(outfile);
}


//for writing the binary file, it gets the filename globally and pointer to array
void writing_binary_file(pixel* (*matrix)[x]){
	FILE *outfile=fopen(output_file_name, "wb");//using wb parameter for binary
	fprintf(outfile,"P6\n");//means that is is binary
	fprintf(outfile,"# COM4521 Assignment test output\n");
	fprintf(outfile, "%d\n%d\n", x,y);
	fprintf(outfile, "255\n");//by default
	for(int i=0; i<x; i++){
		for(int j=0; j<y; j++){
			int r,g,b;
			r=matrix[i][j]->r;//accessing the value from matrix
			g=matrix[i][j]->g;
			b=matrix[i][j]->b;
			static unsigned char color[3];
			//to get the bytes representation of rgb
      		color[0] = r % 256;  /* red */
      		color[1] = g % 256;  /* green */
      		color[2] = b % 256;  /* blue */
      		(void) fwrite(color, 1, 3, outfile);
		}
	}
	fclose(outfile);//closing the file
}

//to calculate the mosaic and average in CPU mode
//the gr ,gg,and gb is used to get the average of r g and b values
void calculate_mosaic_CPU(pixel* (*matrix)[x], int *gr,int *gg, int *gb){
	printf("STARTING THE MOSAIC OPERATION USING CPU APPROACH\n\n\n");
	int global_average_r=0;//initialized to zero
	int global_average_g=0;
	int global_average_b=0;
	for(int i=0; i<x; i+=c){//accesing matrix by squares
		for(int j=0; j<y; j+=c){
			int average_r, average_b, average_g;
			average_b=0;
			average_r=0;
			average_g=0;
			int cell_count=0;		
			for(int ii=i; ii< (i+c) && ii<x ; ii++){//accessing the squares
				for (int jj=j; jj<(j+c) && jj<y;jj++){
					average_r+=matrix[ii][jj]->r;
					average_b+=matrix[ii][jj]->b;
					average_g+=matrix[ii][jj]->g;
					global_average_r+=matrix[ii][jj]->r;
					global_average_g+=matrix[ii][jj]->g;
					global_average_b+=matrix[ii][jj]->b;
					cell_count++;
				}
			}
			
			average_b=average_b/(cell_count);//averaging
			average_r=average_r/(cell_count);//averaging
			average_g=average_g/(cell_count);//averaging
	
			//updating
			for(int ii=i; ii< (i+c) && ii<x; ii++){//updating the value in main matrix
				for (int jj=j; jj<(j+c) && jj<y;jj++){
					matrix[ii][jj]->r=average_r;
					matrix[ii][jj]->b=average_b;
					matrix[ii][jj]->g=average_g;
				}
			}
		}
	}
	global_average_r/=(x*y);//finding the average for the matrix
	global_average_g/=(x*y);//finding the average for the matrix
	global_average_b/=(x*y);//finding the average for the matrix
	*gg=global_average_g;//setting up the pointer
	*gr=global_average_r;//setting up the pointer
	*gb=global_average_b;//setting up the pointer

}



void calculate_mosaic_OPENMP(pixel* (*matrix)[x], int *gr,int *gg, int *gb){
	printf("STARTING THE MOSAIC OPERATION USING OPENMP APPROACH\n\n\n");
	int global_average_r=0;
	int global_average_g=0;
	int global_average_b=0;
	//int NUM_THREADS=4;
	//int i,j,ii,jj,average_r,average_g,average_b;

	#pragma omp parallel for //default(shared) num_threads(NUM_THREADS) private(i,j,ii,jj,cell_count) reduction(+: average_r, average_b, average_g)
	for(int i=0; i<x; i+=c){
		//#pragma omp parallel for
		for(int j=0; j<y; j+=c){
			int average_r, average_b, average_g;
			average_b=0;
			average_r=0;
			average_g=0;
			int cell_count=0;		
			for(int ii=i; ii< (i+c) && ii<x ; ii++){
				for (int jj=j; jj<(j+c) && jj<y;jj++){
					average_r+=matrix[ii][jj]->r;
					average_b+=matrix[ii][jj]->b;
					average_g+=matrix[ii][jj]->g;
					global_average_r+=matrix[ii][jj]->r;
					global_average_g+=matrix[ii][jj]->g;
					global_average_b+=matrix[ii][jj]->b;
					cell_count++;
				}
			}
			average_b=average_b/(cell_count);
			average_r=average_r/(cell_count);
			average_g=average_g/(cell_count);

			//updating
			for(int ii=i; ii< (i+c) && ii<x; ii++){
				for (int jj=j; jj<(j+c) && jj<y;jj++){
					matrix[ii][jj]->r=average_r;
					matrix[ii][jj]->b=average_b;
					matrix[ii][jj]->g=average_g;
				}
			}
		}
	}
	global_average_r/=(x*y);
	global_average_g/=(x*y);
	global_average_b/=(x*y);
	*gg=global_average_g;
	*gr=global_average_r;
	*gb=global_average_b;	
}

//function to check whether the input file is in binary mode or plain text
int check_file_mode(){
	FILE *check_file=fopen(input_file_name, "rb");
	char ch1,ch;//for reading the mode
	ch=fgetc(check_file);//getting the first character
	ch1=fgetc(check_file);//getting the second character
	fclose(check_file);
	if (ch1=='3'){
		return 1;	// if it is binary it returns 1 
	}

	//just to make sure there are no warnings
	ch1=ch+1;//useless
	ch=ch1+2;//useless
	return 0;
}

int main(int argc, char *argv[]) {
	
	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	//TODO: read input image file (either binary or plain text PPM)
	
	

	int file_mode=check_file_mode();//checks the filemode 	
	FILE *fp=fopen(input_file_name,"r");
	get_image_dimensions(fp, &x, &y);
	pixel* (*matrix)[x];
	allocateMatrix(y, &matrix);
	if (file_mode==1){
		printf("The input image is binary\n");
		for(int i=0; i<x; i++){
			for(int j=0; j<y; j++){
				matrix[i][j]=malloc(sizeof(pixel));
				int r,g,b;
				fscanf(fp,"%d %d %d", &r, &g, &b);
				char useless;
				useless=getc(fp);
				useless=1+useless;
				matrix[i][j]->r=r;
				matrix[i][j]->g=g;
				matrix[i][j]->b=b;
			}
		}
		fclose(fp);
	}else{
		printf("The input image is PLAIN TEXT\n");
		fclose(fp);
		FILE *fp_binary=fopen(input_file_name,"rb");
		//char buff;
		get_image_dimensions(fp_binary, &x, &y);
		unsigned char currentPixel[3];
		unsigned char a;
		fread(&a, 1, 1, fp_binary);
		for(int i=0; i<x; i++){
			for(int j=0; j<y; j++){
				fread(currentPixel, 3, 1, fp_binary);
				int r,g,b;
				r=currentPixel[0];
				g=currentPixel[1];
				b=currentPixel[2];
				matrix[i][j]=malloc(sizeof(pixel));
				matrix[i][j]->r=r;
				matrix[i][j]->g=g;
				matrix[i][j]->b=b;

			}
		}
		fclose(fp_binary);
	}
	
	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode){
		case (CPU) : {
			printf("USING CPU");
			//TODO: starting timing here
			double start_time = omp_get_wtime();
			
			//TODO: calculate the average colour value
			int gr,gg,gb;
			calculate_mosaic_CPU(matrix,&gr,&gg,&gb);
			double time = omp_get_wtime() - start_time;
			if (OUTPUT_FORMAT_BINARY){
				printf("Writing Binary File: \n");
				writing_binary_file(matrix);
			}else{
				printf("Writing plain text: \n");
				writing_plain_text_file(matrix);
			}
			
			


			// Output the average colour value for the image

			printf("CPU Average image colour red = %d, green = %d, blue = %d \n",gr,gg,gb);

			//TODO: end timing here
		
			printf("CPU mode execution time took %f s and %f ms\n", time, time*1000);
			break;
		}
		case (OPENMP) : {
			printf("Using OpenMP\n");
			//TODO: starting timing here
			double start_time = omp_get_wtime();
			printf("USING OPENMP");
			//TODO: calculate the average colour value
			int gr,gg,gb;
			calculate_mosaic_OPENMP(matrix, &gr, &gg,&gb);
			double time = omp_get_wtime() - start_time;
			if (OUTPUT_FORMAT_BINARY){
				printf("Writing Binary File: \n");
				writing_binary_file(matrix);
			}else{
				printf("Writing plain text: \n");
				writing_plain_text_file(matrix);
			}
			
			

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n",gr,gg,gb);

			//TODO: end timing here
			
			printf("OPENMP mode execution time took %f s and  %f ms\n", time, time*1000);
			
			break;
		}
		case (CUDA) : {
			printf("CUDA Implementation not required for assignment part 1\n");
			break;
		}
		case (ALL) : {
			//TODO
			break;
		}
	}

	//save the output image file (from last executed mode)

	return 0;
}

void print_help(){
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
	printf("\t-f ppm_format  PPM image output format either 	default) or \n"
		   "\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]){
	if (argc < 7){
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	
	//first argument is always the executable name
	
    
	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);
	
	
	//TODO: read in the mode
	printf("Mode specified is: %s",argv[2]);
	char *argv1=argv[2];
	//To set the mode user is giving
	if (strcmp(argv1,"CPU")==0){execution_mode = CPU;
	}else if (strcmp(argv1,"OPENMP")==0){execution_mode = OPENMP;
	}else if (strcmp(argv1,"ALL")==0){execution_mode = ALL;
	}else{printf("Mode not recognized:Using default mode");//since cuda is not required
	}

	
	/*Since the file will be in the mode of -i input filename, 
	and -o outputfilename -f PPL_PLAIN hence for that the 
	c language getopt library is used
	*/
	int opt;
	while((opt = getopt(argc, argv, "i:o:f:")) != -1){
		 switch(opt)  
        {  
            case ('i'):{
				input_file_name=strdup(optarg);//setting up the value of inputfile
				break;
			}
			case('o'):{
				output_file_name=strdup(optarg);//setting up the value of outputfile
				break;
			}
			case('f'):{
				output_format=strdup(optarg);//setting up the value of output format
				break;
			}
        }  
	}

	//outputformat by default is binary hence if user specifies plaintext it changes it
	if (strcmp(output_format,"PPM_PLAIN_TEXT")==0){
		OUTPUT_FORMAT_BINARY=0;// zero means plaintext 1 means binary
	}
	int debug=1;// For sanity check printing the user input given
	if (debug==1){
		printf("Some description of user input: \n");
		printf("Input filename given: %s \n", input_file_name);
		printf("Output_filename_ given: %s \n", output_file_name);
		printf("output_file_mode: %s \n ", output_format);
		printf(" THE MODE: %d\n", OUTPUT_FORMAT_BINARY);
	}
	return SUCCESS;
}