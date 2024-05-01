#include<stdio.h>
#include<stdlib.h>
#include"setting.h"

#ifndef IO_H
#define IO_H
TYPE *loadMatrix( ITR_SIZE *xSize , ITR_SIZE *ySize , char *fileName );
int countLines( FILE *fp );
int countCols( FILE *fp );
#endif

