
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<time.h>
/*#include<iostream>*/
/*#include<vector>*/
/*#include<boost/math/special_functions/digamma.hpp>*/

#include"setting.h"
#include"IO.h"
#include"gemm.h"
#include"macro.h"

#define M_PI  3.14159265358979323846
#define false 0
#define true  1

#define LOG_2 0.302029999566

TYPE max( TYPE a , TYPE b ){
	if( a > b ) return a;
	else return b;
}

void compute_log_prob( TYPE *log_prob , TYPE *log_prob_norm ){

	//logsumexp
	for( int n = 0 ; n < N ; n++ ) {
		TYPE max_of_nth = log_prob[ n * K ];
		for( int k = 1 ; k < K ; k++ ) max_of_nth = max( log_prob[ n * K + k ] , max_of_nth );
		TYPE sum = 0.0;
		for( int k = 0 ; k < K ; k++ ) sum += expf( log_prob[ n * K + k ] - max_of_nth );
		log_prob_norm[n] = logf( sum ) + max_of_nth;
	}

	//compute log prob
	for( int n = 0 ; n < N ; n++ ) {
		for( int k = 0 ; k < K ; k++ ) {
			log_prob[ n * K + k ] -= log_prob_norm[n];
		}
	}

}

void print_result( int converged , int iter , TYPE lower_bound , TYPE *weight_concentration_ ,   TYPE *means_ , TYPE *precision_cholesky_ , TYPE weight_concentration_prior_ ){

	int cnt = 0;

	puts("weights");
	for( int k = 0 ; k < K ; k++ ) {
		if( weight_concentration_[k] > weight_concentration_prior_ ){
			printf("%.3f " , weight_concentration_[k] );
			cnt++;
		}
	}
	puts("");

	puts("means");
	for( int k = 0 ; k < K ; k++ ) {
		if( weight_concentration_[k] > weight_concentration_prior_ ){
			for(int d = 0 ; d < D ; d++ ) printf("%.3f\t" , means_[ k * D + d ] );
			puts("");
		}
	}
	puts("");

	puts("covariances");
	for( int k = 0 ; k < K ; k++ ) {
		if( weight_concentration_[k] > weight_concentration_prior_ ){
			for(int d = 0 ; d < D ; d++ ) printf("%.3f\t" , 1 / powf (precision_cholesky_[ k * D + d ] , 2 ) );
			puts("");
		}
	}
	puts("");

	printf("converged : %d\titer : %d\tlower_bound : %.5f\tinit_class : %d\tfinally_class : %d\n" , converged , iter + 1 , lower_bound , K , cnt );
	printf("N : %d\tK : %d\tD : %d\t\n" , N , K , D );
}

TYPE digamma( TYPE input ){
	return logf( 0.5 * input + 0.4849142) -  1 / ( 0.5 * input * 1.0271785 ) ;
}

void hadamard( TYPE *input_a  , TYPE *input_b , TYPE *output , int row_size , int col_size  ){
	/* this fuction return matrix that is expressed input_a * input_b = output of each element */

	for( int row = 0 ; row < row_size ; row++ )
		for( int col = 0 ; col < col_size ; col++ )
			output[ row * col_size + col ] = input_a[ row * col_size + col ] * input_b[ row * col_size + col ];

}

void _e_step( TYPE *X , TYPE *log_resp , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ ){

	DEB;

	/*** estimate log weights ***/
	TYPE log_weights[K];
	TYPE tmp = 0;
	for( unsigned int k = 0 ; k < K ; k++ ){
		tmp += weight_concentration_[k];
	}
	tmp = digamma( tmp );
	for( unsigned int k = 0 ; k < K ; k++ ){
		log_weights[k] = digamma ( weight_concentration_[k] ) - tmp;
	}

	DEB_PRINT_VEC( "log_weights" , log_weights , K , "%.3f " );

	/*** estimate log lambda ***/
	TYPE log_lambda[K];
	for( unsigned int k = 0 ; k < K ; k++ ){
		TYPE sum_digamma_nu_ = 0.0;
		for( unsigned int d = 0 ; d < D ; d++ ){
			tmp = 0.5 * ( degree_of_freedom_[k] - d + 1 );
			sum_digamma_nu_ += digamma( tmp );
		}
		log_lambda[k] = D * LOG_2 + sum_digamma_nu_;
	}
	for( int k = 0 ; k < K ; k++ ) log_lambda[k] = 0.5 * ( log_lambda[k] - D / mean_precision_[k] );

	DEB_PRINT_VEC( "log_lambda" , log_lambda , K , "%.3f " );

	/*** estimate log gauss ***/
	TYPE log_det_chol[K] = {};
	for( int k = 0 ; k < K ; k++ ) for( int d = 0 ; d < D ; d++ ) log_det_chol[k] += logf(precision_cholesky_[ k * D + d ]);
	TYPE precisions_[ K * D ];
	//for( int k = 0 ; k < K ; k++ ) for( int d = 0 ; d < D ; d++ ) precisions[k * D + d] = powf(precision_cholesky_[k * D + d] , 2 );
	hadamard( precision_cholesky_ , precision_cholesky_ , precisions_ , K , D  );

	DEB_PRINT_MAT("precisions" , precisions_ , K , D , "%.3f " );

	TYPE m1[ K ] = {};
	//for( int k = 0 ; k < K ; k++ ) m1[k] = 0;
	for( int k = 0 ; k < K ; k++ ) for( int d = 0 ; d < D ; d++ ) m1[ k ] += powf ( means_[ k * D + d ] , 2 ) * precisions_[ k * D + d ];

	TYPE tmp2[ K * D ];
	hadamard( means_ , precisions_ ,  tmp2 , K , D  );

	DEB_PRINT_MAT("tmp2" , tmp2 , K , D , "%.3f ");

	TYPE X2[ N * D ];
	hadamard( X , X , X2 , N , D );


	TYPE m2[ N * K ];
	gemm( 0 , 1 , N , K , D , X , tmp2 , m2 );

	TYPE m3[ N * K ];
	gemm( 0 , 1 , N , K , D , X2 , precisions_ , m3 );

	for( int n = 0 ; n < N ; n++ ){
		for( int k = 0 ; k < K ; k++ ){
			log_resp[ n * K + k ] = - 0.5 * ( D * logf( 2 * M_PI ) + ( -2 * m2[ n * K + k ] + m3[ n * K + k ] + m1[k] )) + log_det_chol[k] - 0.5 * D * logf(degree_of_freedom_[k] );
		}
	}

	/*
		 DEB_PRINT_VEC( "m1" , m1 , K , "%.3f ");
		 DEB_PRINT_MAT_S( "m2" , m2 , N , K , "%.3f ");
		 DEB_PRINT_MAT_S( "m3" , m3 , N , K , "%.3f ");
		 */

	//DEB_PRINT_MAT("log_resp" , log_resp , N , K , "%.3f" );

	for( unsigned int n = 0 ; n < N ; n++ ){
		for( unsigned int k = 0 ; k < K ; k++ ){
			log_resp[ n * K + k ] += log_lambda[k];
			log_resp[ n * K + k ] += log_weights[k];
		}
	}

	TYPE log_prob_norm[N] = {};

	compute_log_prob( log_resp , log_prob_norm );

	DEB_PRINT_VEC("log_prob_norm" , log_prob_norm , N , "%.3f " );

}


void _m_step( TYPE *X , TYPE *log_resp , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ ,
		TYPE weight_concentration_prior_ , TYPE mean_precision_prior_ , TYPE degree_of_freedom_prior_ , TYPE *mean_prior_ , TYPE *covariance_prior_ ){
	TYPE nk[K] = {} , xk[ K * D ] = {}  , Sk[ K * D ] = {} , tmp[ K * D ] = {};


	DEB_PRINT_MAT_S("resp" , log_resp , N , K , "%.3f " );

	for( int n = 0 ; n < N ; n++  ){
		for( int k = 0 ; k < K ; k++ ){
			nk[ k ] += log_resp[ n * K + k ];
			for( int d = 0 ; d < D ; d++ ){
				xk[ k * D + d ] += log_resp[ n * K + k ] * X[ n * D + d ];
				Sk[ k * D + d ] += log_resp[ n * K + k ] * powf( X[ n * D + d ] , 2 );
				tmp[ k * D + d ] += log_resp[ n * K + k ] * X[ n * D + d ];
			}
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			if( nk[k] > 0 ){
				xk[ k * D + d ] /= nk[k];
				Sk[ k * D + d ] = Sk[k * D + d ] / nk[k] - 2 * xk[ k * D + d ] * tmp[ k * D + d ] / nk[k] + powf( xk[ k * D + d ] , 2);
				if(  Sk[ k * D + d ] < 0 ) Sk[ k * D + d ] = REG_COVOR;
			}
			else{
				xk[ k * D + d ] = mean_prior_[d];
				Sk[ k * D + d ] = REG_COVOR;
			}
		}
	}

	DEB_PRINT_VEC( "nk" , nk , K , "%.3f ");
	DEB_PRINT_MAT( "xk" , xk , K , D , "%.3f " );
	DEB_PRINT_MAT( "Sk" , Sk , K , D , "%.3f " );

	for( int k = 0 ; k < K ; k++ ){
		weight_concentration_[k] = weight_concentration_prior_ + nk[k];
		mean_precision_[k] = mean_precision_prior_ + nk[k];
		degree_of_freedom_[k] = degree_of_freedom_prior_ + nk[k];
	}

	// m
	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			means_[k * D + d]= ( (mean_precision_prior_ * mean_prior_[d]) + (nk[k] * xk[ k * D + d ]) ) / mean_precision_[k];
		}
	}


	// W
	TYPE diff[ K * D ];
	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] = powf( xk[ k * D + d ]  - mean_prior_[d]  , 2 );
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] *= mean_precision_prior_ / mean_precision_[k];
			Sk[ k * D + d ] += diff[ k * D + d ];
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] = covariance_prior_[d] + (double)nk[k] * Sk[ k * D + d ];
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			precision_cholesky_[k * D + d] = diff[ k * D + d ] / degree_of_freedom_[k];
			precision_cholesky_[k * D + d] = (float) ( 1 / sqrt((double)precision_cholesky_[k * D + d]) );
		}
	}

	DEB_PRINT_MAT("precision_cholesky_" , precision_cholesky_ , K , D , "%.3f ");
		 DEB_PRINT_VEC("weight_concentration_",weight_concentration_,K , "%.3f ");
		 DEB_PRINT_VEC("mean_precision_",mean_precision_,K , "%.3f ");
		 DEB_PRINT_VEC("degree_of_freedom_",degree_of_freedom_,K , "%.3f ");
		 DEB_PRINT_MAT("mean" , means_ , K , D , "%.3f ");

	return;
}

TYPE _log_dirichket_norm( TYPE *weight_concentration_ ){
	double tmp = 0 , sum = 0;
	for( int k = 0 ; k < K ; k++ ){
		tmp += lgamma( weight_concentration_[k] );
		sum += weight_concentration_[k];
	}
	return lgamma( sum ) - tmp;
}

double _log_wishart_norm( TYPE *log_det_precision_chol ,  TYPE *degree_of_freedom_ ){
	double sum_log_wishart = 0.0;

	for( int k = 0 ; k < K ; k++ ){
		double sum_gammaln_nu = 0.0;
		for( unsigned int d = 0 ; d < D ; d++ ) sum_gammaln_nu += lgamma( 0.5 * ( degree_of_freedom_[k] - d + 1 ) );

		sum_log_wishart +=	- ( (double)degree_of_freedom_[k] * log_det_precision_chol[k]
				+ (double)degree_of_freedom_[k] * D * 0.5 * LOG_2
				+ sum_gammaln_nu
				);
	}

	return sum_log_wishart;

}



TYPE _compute_lower_bound( TYPE *weight_concentration_ , TYPE *precision_cholesky_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , double sum_resp){

	TYPE log_det_precision_chol[K] = {};

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0; d < D ; d++ ) log_det_precision_chol[k] += logf( precision_cholesky_[ k * D + d ]);
		log_det_precision_chol[k] -= 0.5 * D * logf( degree_of_freedom_[k]);
	}

	TYPE sum_log_wishart = _log_wishart_norm( log_det_precision_chol ,  degree_of_freedom_);

	TYPE log_weight_norm = _log_dirichket_norm( weight_concentration_ );

	TYPE sum_log_mean_precision = 0;
	for( int k = 0 ; k < K ; k++ ) sum_log_mean_precision += logf(mean_precision_[k]);
#if DEBUG_PRINT
	printf("%.5f %.5f %.5f %.5f\n", sum_resp , sum_log_wishart , log_weight_norm , sum_log_mean_precision );
#endif

	return (float)(-sum_resp - sum_log_wishart - log_weight_norm - 0.5 * D * sum_log_mean_precision);
}


int fit(	TYPE *X , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ ,
		TYPE weight_concentration_prior_ , TYPE mean_precision_prior_ , TYPE degree_of_freedom_prior_ , TYPE *mean_prior_ , TYPE *covariance_prior_ ){
	TYPE old_lower_bound = 0;
	//TODO
	//TYPE log_prob_norm;
	TYPE log_resp[ N * K ];

	int converged = false;

	for( int iter = 0 ; iter < MAX_ITER ; iter++ ){
		DEB;
		DEB;


		_e_step( X , log_resp , weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ );


		DEB_PRINT_MAT_S("e_step ato log_resp" , log_resp , N , K , "%.3f " );

		double sum_resp = 0;
		for( int n = 0 ; n < N ; n++ ) for( int k = 0 ; k < K ; k++ ){
			float tmp = expf( log_resp[ n * K + k ] );
			sum_resp += tmp * log_resp[ n * K + k ];
			log_resp[ n * K + k ] = tmp;
		}

		_m_step( X , log_resp , weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ ,
				weight_concentration_prior_ , mean_precision_prior_ , degree_of_freedom_prior_ , mean_prior_ , covariance_prior_ );

		TYPE lower_bound = _compute_lower_bound(  weight_concentration_ , precision_cholesky_ , mean_precision_ , degree_of_freedom_  , sum_resp);

#if DEBUG_PRINT
		printf("iter:%d lower_bound:%.5f\n" , iter + 1 , lower_bound );
#endif

		if( abs( lower_bound - old_lower_bound ) < TOL ){
#if ON_CPU
#if RESURT_PRINT
			converged = true;
			print_result( converged , iter , lower_bound , weight_concentration_ , means_ , precision_cholesky_ , weight_concentration_prior_ );
#endif
#endif
			break;
		}
		old_lower_bound = lower_bound;
	}

	return true;
}
int main(int argc , char *argv[]){

	TYPE *data;
	char* fileName;

	if( argc !=2  ){
		printf("usage: %s <file>\n", *argv);
		return 1;
	}
	fileName = argv[1];


	/*** input data ***/
	unsigned int col = D;
	unsigned int row = N;
	data = loadMatrix( &col , &row , fileName );

	if( data == NULL ){
		printf("failed input data\n");
		return 1;
	}

	TYPE X[ N * D ];
	for( int n = 0 ; n < N ; n++ ){
		for( int d = 0 ; d < D ; d++ ){
			X[n * D + d] = data[ n * D + d];
		}
	}


	/*** init ***/
	TYPE resp_[ N * K ];

	TYPE weight_concentration_[K];
	TYPE means_[ K * D ];
	TYPE mean_precision_[ K ];
	TYPE degree_of_freedom_[ K ];
	TYPE precision_cholesky_[ K * D ];
	TYPE covariances_[ K * D ];

	TYPE weight_concentration_prior_ = (TYPE)1/K; //alpha_0
	TYPE mean_precision_prior_ = 1.0; // beta_0
	TYPE degree_of_freedom_prior_ = D; //Nu_0
	TYPE mean_prior_[D]; // beta_0
	TYPE covariance_prior_[D];


	for( int d = 0 ; d < D ; d++ ){
		mean_prior_[d] = 0;
		covariance_prior_[d] = 0;
	}
	for( int n = 0 ; n < N ; n++ ) for( int d = 0 ; d < D ; d++ ) mean_prior_[d] += X[ n * D + d ];
	for( int d = 0 ; d < D ; d++ ) mean_prior_[d] /= N;
	for( int n = 0 ; n < N ; n++ ) for( int d = 0 ; d < D ; d++ ) covariance_prior_[d] += powf( ( X[ n * D + d ] - mean_prior_[d] ) , 2 );
	for( int d = 0 ; d < D ; d++ ) covariance_prior_[d] /= N;

	for( int k = 0 ; k < K ; k++ ){
		mean_precision_[k] = 1.0;
		weight_concentration_[k] = 1.0 / K;
	}


	for( int n = 0 ; n < N ; n++ ) for( int k = 0 ; k < K ; k++ ) resp_[ n * K + k ] = (TYPE) 1 / K;

	TYPE nk[K] , xk[ K * D ] , sk[ K * D ];

	for( int k = 0 ; k < K ; k++ ) nk[k] = N / K;


#if RANDOM_INIT
	srand( RANDOM_SEED );
	for( int k = 0 ; k < K ; k++ ){
		int idx = rand() % N;
		for( int d = 0 ; d < D ; d++ ){
			xk[ k * D + d ] = X[idx * D + d];
		}
	}
#else
	for( int k = 0 ; k < K ; k++ ){
		int idx =  N / K * k;
		for( int d = 0 ; d < D ; d++ ){
			xk[ k * D + d ] = X[idx * D + d];
		}
	}
#endif

	/*** estimate_gaussian_covariance_diag ***/
	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			for( int n = 0 ; n < N ; n++ ){
				sk[ k * D + d ] += resp_[ n * K + k ] * powf( X[ n * D + d ] - xk[ k * D + d ] , 2 );
			}
			sk[ k * D + d ] /= nk[k];
		}
	}


	DEB_PRINT_VEC("init nk" , nk , K , "%.5f ");
	DEB_PRINT_MAT("init xk" , xk , K , D , "%.5f " );
	DEB_PRINT_MAT("init sk" , sk , K , D , "%.5f " );

	/*** intialize prior parameters ***/
	// alpha , beta , nu
	for( int k = 0 ; k < K ; k++ ){
		weight_concentration_[k] = weight_concentration_prior_ + nk[k];
		mean_precision_[k] = mean_precision_prior_ + nk[k];
		degree_of_freedom_[k] = degree_of_freedom_prior_ + nk[k];
	}

	// m
	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			means_[k * D + d]= ( (mean_precision_prior_ * mean_prior_[d]) + (nk[k] * xk[ k * D + d ]) ) / mean_precision_[k];
		}
	}

	// W
	TYPE diff[ K * D ];
	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] = powf( xk[ k * D + d ]  - mean_prior_[d]  , 2 );
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] *= mean_precision_prior_ / mean_precision_[k];
			sk[ k * D + d ] += diff[ k * D + d ];
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] = covariance_prior_[d] + (double)nk[k] * sk[ k * D + d ];
		}
	}

	for( int k = 0 ; k < K ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			covariances_[k * D + d] = diff[ k * D + d ] / degree_of_freedom_[k];
			precision_cholesky_[k * D + d] = (float) ( 1 / sqrt((double)covariances_[k * D + d]) );
		}
	}

	DEB_PRINT_VEC("init weight_concentration_" , weight_concentration_ , K , "%.3f ");
	DEB_PRINT_VEC("init mean_precision_" , mean_precision_ , K , "%.3f ");
	DEB_PRINT_VEC("init degree_of_freedom_" , degree_of_freedom_ , K , "%.5f ");
	DEB_PRINT_MAT("init means_" , means_ , K , D , "%.5f ");
	DEB_PRINT_MAT("init precision_cholesky_" , precision_cholesky_ , K , D , "%.5f ");

	/*** end of init ***/
	fit( X , weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ , weight_concentration_prior_ , mean_precision_prior_ , degree_of_freedom_prior_ , mean_prior_ , covariance_prior_ );

	free(data);

	return 0;
}
