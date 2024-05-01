#include<stdio.h>
#include<stdlib.h>

#ifndef IO_H
#define IO_H

//入力データのサンプル数を求める
template<typename T>
T countLines( FILE *fp ){
	int c;
	T num = 0;
	while( 1 ){
		c = fgetc(fp);
		if( c == '\0' || c == EOF ) break;
		else if ( c == '\n' ) num++;
	}
	rewind(fp);
	return num;
}

//入力データの次元数を求める
template<typename T>
T countCols( FILE *fp ){
	int c , old;
	T num = 0;
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

template<typename T>
float *loadMatrix( T *xSize , T *ySize , char *fileName ){
	FILE *fp = NULL;
	float *arr;
	fp = fopen( fileName , "r" );
	if( fp == NULL ){
		printf("%sがありません",fileName);
		return NULL;
	}
	*ySize = countLines<T>(fp);
	*xSize = countCols<T>(fp);
	//printf("X:%lld Y:%lld\n", *xSize , *ySize );
	arr = (float *)malloc(sizeof(float ) * (*ySize) * (*xSize) );
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

#endif

