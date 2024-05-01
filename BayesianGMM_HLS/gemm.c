#include <stdio.h>
#include "setting.h"

void gemm( int trans_a , int trans_b , int a_rows , int b_cols , int mutual , TYPE *a , TYPE *b , TYPE *c ){

	//init output matrix
	for( int i = 0 ; i < a_rows ; i++ ) for( int j = 0 ; j < b_cols ; j++ ) c[ i * b_cols + j ] = 0;

	//matrix product
	if( !trans_a ){
		if( !trans_b ){
			for( int i = 0 ; i < a_rows ; i++ )
				for( int j = 0 ; j < b_cols ; j++ )
					for( int k = 0 ;  k < mutual ; k++ ){
						c[ i * b_cols + j ] += a[ i * mutual + k ] * b[ k * b_cols + j ];
					}
		}
		else{
			for( int i = 0 ; i < a_rows ; i++ )
				for( int j = 0 ; j < b_cols ; j++ )
					for( int k = 0 ;  k < mutual ; k++ )
						c[ i * b_cols + j ] += a[ i * mutual + k ] * b[ j * mutual + k ];
		}
	}
	else{
		if( !trans_b ){
			for( int i = 0 ; i < a_rows ; i++ )
				for( int j = 0 ; j < b_cols ; j++ )
					for( int k = 0 ;  k < mutual ; k++ ){
						c[ i * b_cols + j ] += a[ k * a_rows + i ] * b[ k * b_cols + j ];
					}
		}
		else{
			for( int i = 0 ; i < a_rows ; i++ )
				for( int j = 0 ; j < b_cols ; j++ )
					for( int k = 0 ;  k < mutual ; k++ )
						c[ i * b_cols + j ] += a[ k * a_rows + i ] * b[ j * mutual + k ];
		}
	}
}


