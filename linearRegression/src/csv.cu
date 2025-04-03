#include <stdio.h>
#include <stdlib.h>
#include <string.h>


float *loadCSV(const char *filename, int *numRows, int *numCols) {
   FILE *file = fopen(filename, "r");

   if(!file) {
   	printf("Unable to open file - %s\n", filename);
	return NULL;
   }

   char line[1024];

   int rows = 0, cols = 0;
   
   // count number of columns
   if(fgets(line, sizeof(line), file)){
	rows++;
	cols = 1;
	for(char *p = line; *p; p++){
		if(*p == ',') cols++;
	}	
   }

   // count number of rows
   while(fgets(line, sizeof(line), file)) {
	if(strlen(line) <= 1) continue;
	rows++;
   }
   rewind(file);

   float *data = (float *)malloc(rows * cols * sizeof(float));

   if(!data){
      fclose(file);
      return NULL;
   }

   int row = 0;

   while(fgets(line, sizeof(line), file)){
	if(strlen(line) <= 1) continue;
	int col = 0;
	char *token = strtok(line, ",");
	while(token != NULL){
		if(col == 2){
			float label;
			if(strcmp(token, "Yes") == 0) label = 1;
			else label = 0;

			data[row * cols + col] = label;
			break;
		}
		
		data[row * cols + col] = atof(token);
		token = strtok(NULL, ",");
		col++;
	}
	row++;
   }

   fclose(file);
   
   *numRows = rows, *numCols = cols;
   return data;
}
