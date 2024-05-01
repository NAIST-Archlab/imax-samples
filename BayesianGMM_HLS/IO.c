#include"IO.h"
#include"setting.h"
TYPE *loadMatrix( ITR_SIZE *xSize , ITR_SIZE *ySize , char *fileName ){
	FILE *fp = NULL;
	TYPE *arr;
	fp = fopen( fileName , "r" );
	if( fp == NULL ){
		printf("%sがありません",fileName);
		return NULL;
	}
	*ySize = countLines(fp);
	*xSize = countCols(fp);

	//printf("X:%lld Y:%lld\n", *xSize , *ySize );

	arr = (TYPE*)malloc(sizeof(double) * (*ySize) * (*xSize) );

	if( arr == NULL ){
		printf("IO.cpp:malloc err");
		return NULL;
	}
	else{
		//printf("malloc dekita \n");
	}

	for( long long i = 0 ; i < *xSize * *ySize ; i++ ){
		fscanf(fp,"%f",&arr[i]);
	}

	fclose(fp);
	return arr;
}

//入力データのサンプル数を求める
int countLines( FILE *fp ){
	int c;
	ITR_SIZE num = 0;
	while( 1 ){
		c = fgetc(fp);
		if( c == '\0' || c == EOF ) break;
		else if ( c == '\n' ) num++;
	}
	rewind(fp);
	return num;
}

//入力データの次元数を求める
int countCols( FILE *fp ){
	int c , old = ' ';
	ITR_SIZE num = 0;
	while(1){
		c = fgetc(fp);
		if( c == '\n' || c == EOF ) break;
		else if( c == '\t' ) num++;
		old = c;
	}
	if( old != '\t' && c == '\n' ) num++;
	rewind(fp);
	return num;
}
