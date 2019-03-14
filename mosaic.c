#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "aca00pr"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
MODE execution_mode = CPU;
char *input_file_name;
char *output_file_name;
typedef struct pixel{
	int r;
	int g;
	int b;
}pixel;
void allocateMatrix(int n, pixel* (**matrix)[n])
{
    *matrix = malloc(n * sizeof(**matrix));
}
int main(int argc, char *argv[]) {
	
	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	//TODO: read input image file (either binary or plain text PPM)
	// For plain files
	printf("Reading a file");
	FILE *fp;
	int x,y,char_len;
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
	fscanf(fp, "%d %d", &x,&y);
	fscanf(fp, "%d",&char_len);
	pixel* (*matrix)[x];
	allocateMatrix(y, &matrix);
	printf("For loop starrting");
	for(int i=0; i<x; i++){
		for(int j=0; j<y; j++){
			matrix[i][j]=malloc(sizeof(pixel));
			int r,g,b;
			fscanf(fp,"%d %d %d", &r, &g, &b);
			char useless;
			useless=getc(fp);
			matrix[i][j]->r=r;
			matrix[i][j]->g=g;
			matrix[i][j]->b=b;
			//printf("%c", useless);
			//printf("%d %d %d", r,g,b);
			//printf("\n\n");
			//getchar();    
		}
	}
	printf("\n%d\n",c);
	printf("STARTIG THE MOSAIC OPERATION\n\n\n");
	for(int i=0; i<x; i+=c){
		for(int j=0; j<y; j+=c){
	//		printf ("%d %d\n", i,j);
	//		getchar();
			int average_r, average_b, average_g;
			for(int ii=i; ii< (i+c); ii++){
				for (int jj=j; jj<(j+c);jj++){
					average_r+=matrix[ii][jj]->r;
					average_b+=matrix[ii][jj]->b;
					//printf("%d\t",matrix[ii][jj]->b);
					average_g+=matrix[ii][jj]->g;
				}
			}
			average_b=average_b/(c*c);
			average_r=average_r/(c*c);
			average_g=average_g/(c*c);
			//updating
			for(int ii=i; ii< (i+c); ii++){
				for (int jj=j; jj<(j+c);jj++){
					matrix[ii][jj]->r=average_r;
					matrix[ii][jj]->b=average_b;
					matrix[ii][jj]->g=average_g;
				}
			}
		}
	}



//now writing the shit madar chot
	FILE *outfile=fopen("out.ppm", "w");
	fprintf(outfile,"P3\n");
	fprintf(outfile,"#randomshitcomment\n");
	fprintf(outfile, "%d %d\n", x,y);
	fprintf(outfile, "255\n");
	for (int i=0; i<x; i++){
		for(int j=0; j<y; j++){
			int r,g,b;
			r=matrix[i][j]->r;
			g=matrix[i][j]->g;
			b=matrix[i][j]->b;
			fprintf(outfile,"%d %d %d\t",r,g,b );
		}
		fprintf(outfile,"\n");
	}
	fclose(outfile);








	

	
	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode){
		case (CPU) : {
			//TODO: starting timing here


			//TODO: calculate the average colour value

			// Output the average colour value for the image
			printf("CPU Average image colour red = ???, green = ???, blue = ??? \n");

			//TODO: end timing here
			printf("CPU mode execution time took ??? s and ???ms\n");
			break;
		}
		case (OPENMP) : {
			//TODO: starting timing here

			//TODO: calculate the average colour value

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = ???, green = ???, blue = ??? \n");

			//TODO: end timing here
			printf("OPENMP mode execution time took ??? s and ?? ?ms\n");
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
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
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
	printf("Reached here");
	
	
	//TODO: read in the mode
	printf("Mode specified is: %s",argv[2]);
	char *argv1=argv[2];
	if (strcmp(argv1,"CPU")==0){MODE execution_mode = CPU;
	}else if (strcmp(argv1,"OPENMP")==0){MODE execution_mode = OPENMP;
	}else if (strcmp(argv1,"ALL")==0){MODE execution_mode = ALL;
	}else{printf("Mode not recognized:Using default mode");
	}
	
	//TODO: read in the input image name
	input_file_name=strdup(argv[3]);

	//TODO: read in the output image name
	output_file_name=strdup(argv[4]);
	printf ("Successfully readed all files\n\n");

	printf("Stackoverflow saved me ");




	//TODO: read in any optional part 3 arguments


	return SUCCESS;
}