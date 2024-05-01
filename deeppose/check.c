#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

void main(int argc, char * argv[]) {
	FILE *fp;
	float *OUT;
	float *EXP;
	int i;
    int size = 0;
    int oind;
    int eind;

    int opt;
    int aopt;
    int fopt = 1;

    struct stat st;

    while((opt = getopt(argc, argv, "aifh")) != -1) {
      switch(opt) {
        case 'a' :
          aopt = 1;
          break;
        case 'i' :
          fopt = 0;
          break;
        case 'f' :
          fopt = 1;
          break;
        case 'h' :
          usage();
          return;
        default :
          printf("error! \'%c\' \'%c\'\n",opt, optopt);
          usage();
          return;
       }
    } 

    switch (argc-optind) {
      case 1:
        oind = optind;
        eind = oind;
        break;
      case 2:
        oind = optind;
        if(check_digit(argv[optind+1])) {
          size = atoi(argv[optind+1]);
          eind = oind;
        } else {
          eind = optind+1;
        }
        break;
      case 3:
        oind = optind;
        eind = optind+1;
        size = atoi(argv[optind+2]);
        break;
      default:
        printf("error!\n");
        usage();
        return;
    }

    if(aopt || size==0) {
	  if (stat(argv[optind], &st) != 0) {
	  	fprintf(stderr, "%s is not found.\n", argv[1]);
	  	return;
	  }
	  size = st.st_size/4;    
    }
    

	OUT = (void*)malloc(size*sizeof(float));
	EXP = (void*)malloc(size*sizeof(float));

	fp = fopen(argv[oind],"rb");
	fread(OUT, size*sizeof(float), 1, fp);
	fclose(fp);

	fp = fopen(argv[eind],"rb");
	fread(EXP, size*sizeof(float), 1, fp);
	fclose(fp);

	for(i = 0; i < size; i++) {
        if(fopt) printf("%f,%f",OUT[i],EXP[i]);
        else     printf("%x,%x",*(int*)&OUT[i],*(int*)&EXP[i]);
        if(fabs(OUT[i] - EXP[i]) > 0.1) {
			printf(" > ");
        }
        if(OUT[i] == EXP[i]) {
			printf(" == ");
        }
        printf("\n");
	}
    
}


int check_digit(char *c) {
	int i;

	for(i = 0; i < strlen(c); i++) {
      if(isdigit(c[i]) == 0)  {
        return 0;
      }
    }
    
    return 1;
}

int usage() {
  printf("./check [-a,-i] file1 [file2] [size]\n");
  printf("  -a  : compare all size\n");
  printf("  -i  : output char hex(default char float)\n");
  printf("\n");
  printf("file1 : check file\n");
  printf("file2 : exp   file\n");
  printf("size  : compare size\n");
}
