#include <stdio.h>
#include "gemm.h"

#define TYPE float

#define PRINT_MAT( name , vec , row , col , format ) \
	printf(name);  puts(""); \
	for( int i = 0 ; i < row ; i++ ) { for( int j = 0 ; j < col ; j++ ) printf( format , vec[ i * col + j] ); puts(""); } \
	printf("\n");
int main(){
	int x = 10 , y = 3 , z = 15;
	TYPE a[ x * y ] , b[ y * z ] , c[ z * y ] , d[ x * y ] , out[ x * z ];
	for( int i = 0 ; i < x ; i++ ) for( int j = 0 ; j < y ; j++ ) a[ i * y + j ] = i * y + j;
	for( int i = 0 ; i < x ; i++ ) for( int j = 0 ; j < y ; j++ ) d[ j * x + i ] = i * y + j;
	for( int j = 0 ; j < y ; j++ ) for( int k = 0 ; k < z ; k++ ) b[ j * z + k ] = j * z + k;
	for( int j = 0 ; j < y ; j++ ) for( int k = 0 ; k < z ; k++ ) c[ k * y + j ] = j * z + k;

	PRINT_MAT( "a" ,  a , x , y , "%.3f " );
	PRINT_MAT( "b" ,  b , y , z , "%.3f " );
	PRINT_MAT( "c" ,  c , z , y , "%.3f " );
	PRINT_MAT( "d" ,  d , y , x , "%.3f " );


	gemm( 0 , 0 , x , z , y , a , b , out  );
	PRINT_MAT( "test1" , out , x , z , "%.3f " );
	gemm( 0 , 1 , x , z , y , a , c , out  );
	PRINT_MAT( "test2" , out , x , z , "%.3f " );
	gemm( 1 , 0 , x , z , y , d , b , out  );
	PRINT_MAT( "test3" , out , x , z , "%.3f " );
	gemm( 1 , 1 , x , z , y , d , c , out  );
	PRINT_MAT( "test4" , out , x , z , "%.3f " );



	return 0;
}
