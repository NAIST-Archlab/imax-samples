#include <stdio.h>
#include "setting.h"

#if defined(CBLAS_GEMM)
#include "cblas.h"
#endif

void gemm( int trans_a , int trans_b , const int a_rows , const int b_cols , const int mutual , const TYPE *a , const TYPE *b , TYPE *c ){

	//init output matrix
	for( int i = 0 ; i < a_rows ; i++ ) for( int j = 0 ; j < b_cols ; j++ ) c[ i * b_cols + j ] = 0;


#if defined(CBLAS_GEMM)

	const int m = trans_a ? mutual : a_rows;
	const int n = trans_b ? b_cols : mutual;
	const int k = trans_a ? a_rows : mutual;

	const int TransA = trans_a ? CblasTrans : CblasNoTrans;
	const int TransB = trans_b ? CblasTrans : CblasNoTrans;

	const int lda = trans_a ? m : k ;
	const int ldb = trans_b ? k : n ;
	const int ldc = n;

	//printf("CblasTrans:%d CblasNoTrans:%d\n" , CblasTrans , CblasNoTrans );
	//printf("m:%d n:%d k:%d TransA:%d TransB:%d\n",m,n,k,TransA,TransB);

	cblas_sgemm(CblasRowMajor, TransA , TransB , m , n , k , 1.0f, a , lda , b , ldb , 0, c , ldc ) /* C=1.0f*A*B + 0*C */ ;

#else

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

#endif

}


