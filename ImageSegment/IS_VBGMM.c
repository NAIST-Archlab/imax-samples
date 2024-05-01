
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<time.h>

#include"setting.h"
#include"gemm.h"
#include"macro.h"

#define M_PI  3.14159265358979323846
#define false 0
#define true  1

// xdisp.c 参照
void x11_open(), x11_update(), RGB_to_X(), BOX_to_X();
int x11_checkevent();
int WD, HT , SCRWD, SCRHT;

unsigned short ClusterNum = 2;
unsigned int   BITMAP = 512;


#define LOG_2 0.302029999566

TYPE max( TYPE a , TYPE b ){
	if( a > b ) return a;
	else return b;
}


void GetImageFromStdin( TYPE *Data , int TwiceScreenMode , unsigned int ColorAbstractionLevel ){
#if DEBUG
	printf("[start] %s\n" , __func__);
#endif

	unsigned int abs_level = pow( 2 , ColorAbstractionLevel - 1);

	for (int i=0; i<BITMAP; i++){
		for( int j = 0 ; j < 3 ; j++ ){
			Data[i * D + j ] = getchar() / abs_level;
		}
	}

	// 片目分の入力を破棄
	if( TwiceScreenMode )
		for (int i=0; i<BITMAP; i++)
			for( int j = 0 ; j < 3 ; j++ )
				getchar();

#if DEBUG
	printf("[end] %s\n" , __func__);
#endif
}



void compute_log_prob( TYPE *log_prob , TYPE *log_prob_norm ){

	//logsumexp
	for( int n = 0 ; n < BITMAP ; n++ ) {
		TYPE max_of_nth = log_prob[ n * ClusterNum ];
		for( int k = 1 ; k < ClusterNum ; k++ ) max_of_nth = max( log_prob[ n * ClusterNum + k ] , max_of_nth );
		TYPE sum = 0.0;
		for( int k = 0 ; k < ClusterNum ; k++ ) sum += expf( log_prob[ n * ClusterNum + k ] - max_of_nth );
		log_prob_norm[n] = logf( sum ) + max_of_nth;
	}

	//compute log prob
	for( int n = 0 ; n < BITMAP ; n++ ) {
		for( int k = 0 ; k < ClusterNum ; k++ ) {
			log_prob[ n * ClusterNum + k ] -= log_prob_norm[n];
		}
	}

}

double compute_log_likelihood( TYPE *log_prob_norm ){

	double ll = 0;
	for( int i = 0 ; i < BITMAP ; i++ ){
		ll += log_prob_norm[i];
	}

	return ll;
}


void print_result( int converged , int iter , TYPE lower_bound , TYPE *weight_concentration_ ,   TYPE *means_ , TYPE *precision_cholesky_ , TYPE weight_concentration_prior_ ){

	int cnt = 0;

	puts("weights");
	for( int k = 0 ; k < ClusterNum ; k++ ) {
		if( weight_concentration_[k] > weight_concentration_prior_ ){
			printf("%.3f " , weight_concentration_[k] );
			cnt++;
		}
	}
	puts("");

	puts("means");
	for( int k = 0 ; k < ClusterNum ; k++ ) {
		if( weight_concentration_[k] > weight_concentration_prior_ ){
			for(int d = 0 ; d < D ; d++ ) printf("%.3f\t" , means_[ k * D + d ] );
			puts("");
		}
	}
	puts("");

	puts("covariances");
	for( int k = 0 ; k < ClusterNum ; k++ ) {
		if( weight_concentration_[k] > weight_concentration_prior_ ){
			for(int d = 0 ; d < D ; d++ ) printf("%.3f\t" , 1 / powf (precision_cholesky_[ k * D + d ] , 2 ) );
			puts("");
		}
	}
	puts("");

	printf("converged : %d\titer : %d\tlower_bound : %.5f\tinit_class : %d\tfinally_class : %d\n" , converged , iter + 1 , lower_bound , ClusterNum , cnt );
	printf("BITMAP : %d\tK : %d\tD : %d\t\n" , BITMAP , ClusterNum , D );
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

double _e_step( TYPE *X , TYPE *log_resp , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ ){
	/*** estimate log weights ***/

	TYPE *log_weights , *log_lambda , *log_det_chol , *precisions_ , *m1 , *m2 , *m3 , *tmp2 , *X2;
	log_weights = (TYPE *)malloc( sizeof(TYPE) * ClusterNum);
	log_lambda  = (TYPE *)malloc( sizeof(TYPE) * ClusterNum);
	log_det_chol= (TYPE *)malloc( sizeof(TYPE) * ClusterNum);
	precisions_ = (TYPE *)malloc( sizeof(TYPE) * ClusterNum * D );
	tmp2        = (TYPE *)malloc( sizeof(TYPE) * ClusterNum * D );
	X2          = (TYPE *)malloc( sizeof(TYPE) * BITMAP * D );
	m1          = (TYPE *)malloc( sizeof(TYPE) * ClusterNum);
	m2          = (TYPE *)malloc( sizeof(TYPE) * BITMAP * ClusterNum );
	m3          = (TYPE *)malloc( sizeof(TYPE) * BITMAP * ClusterNum );

	for( int k = 0 ; k < ClusterNum ; k++ ){
		m1[k] = 0;
		log_det_chol[k] = 0;
	}

	TYPE tmp = 0;
	for( unsigned int k = 0 ; k < ClusterNum ; k++ ){
		tmp += weight_concentration_[k];
	}

	tmp = digamma( tmp );
	for( unsigned int k = 0 ; k < ClusterNum ; k++ ){
		log_weights[k] = digamma ( weight_concentration_[k] ) - tmp;
	}

	DEB_PRINT_VEC( "log_weights" , log_weights , ClusterNum , "%.3f " );

	/*** estimate log lambda ***/
	for( unsigned int k = 0 ; k < ClusterNum ; k++ ){
		TYPE sum_digamma_nu_ = 0.0;
		for( unsigned int d = 0 ; d < D ; d++ ){
			tmp = 0.5 * ( degree_of_freedom_[k] - d + 1 );
			sum_digamma_nu_ += digamma( tmp );
		}
		log_lambda[k] = D * LOG_2 + sum_digamma_nu_;
	}
	for( int k = 0 ; k < ClusterNum ; k++ ) log_lambda[k] = 0.5 * ( log_lambda[k] - D / mean_precision_[k] );

	DEB_PRINT_VEC( "log_lambda" , log_lambda , ClusterNum , "%.3f " );

	/*** estimate log gauss ***/
	for( int k = 0 ; k < ClusterNum ; k++ ) for( int d = 0 ; d < D ; d++ ) log_det_chol[k] += logf(precision_cholesky_[ k * D + d ]);
	//for( int k = 0 ; k < ClusterNum ; k++ ) for( int d = 0 ; d < D ; d++ ) precisions[k * D + d] = powf(precision_cholesky_[k * D + d] , 2 );
	hadamard( precision_cholesky_ , precision_cholesky_ , precisions_ , ClusterNum , D  );

	//for( int k = 0 ; k < ClusterNum ; k++ ) m1[k] = 0;
	DEB_PRINT_VEC( "before m1" , m1 , ClusterNum , "%.3f ");
	for( int k = 0 ; k < ClusterNum ; k++ ) for( int d = 0 ; d < D ; d++ ) m1[ k ] += powf ( means_[ k * D + d ] , 2 ) * precisions_[ k * D + d ];

	DEB_PRINT_MAT("precisions" , precisions_ , ClusterNum , D , "%.8f " );
	DEB_PRINT_MAT( "means" , means_ , ClusterNum , D , "%.3f ");
	DEB_PRINT_VEC( "m1" , m1 , ClusterNum , "%.3f ");

	hadamard( means_ , precisions_ ,  tmp2 , ClusterNum , D  );

	DEB_PRINT_MAT("tmp2" , tmp2 , ClusterNum , D , "%.3f ");

	hadamard( X , X , X2 , BITMAP , D );

	gemm( 0 , 1 , BITMAP , ClusterNum , D , X , tmp2 , m2 );
	gemm( 0 , 1 , BITMAP , ClusterNum , D , X2 , precisions_ , m3 );

	for( int n = 0 ; n < BITMAP ; n++ ){
		for( int k = 0 ; k < ClusterNum ; k++ ){
			log_resp[ n * ClusterNum + k ] = - 0.5 * ( D * logf( 2 * M_PI ) + ( -2 * m2[ n * ClusterNum + k ] + m3[ n * ClusterNum + k ] + m1[k] )) + log_det_chol[k] - 0.5 * D * logf(degree_of_freedom_[k] );
		}
	}
	
	//DEB_PRINT_MAT_S( "m2" , m2 , BITMAP , ClusterNum , "%.3f ");
	//DEB_PRINT_MAT_S( "m3" , m3 , BITMAP , ClusterNum , "%.3f ");

	free(log_weights); 
	free(log_lambda ); 
	free(log_det_chol);
	free(precisions_); 
	free(tmp2       ); 
	free(X2         ); 
	free(m1         ); 
	free(m2         ); 
	free(m3         ); 

	for( unsigned int n = 0 ; n < BITMAP ; n++ ){
		for( unsigned int k = 0 ; k < ClusterNum ; k++ ){
			log_resp[ n * ClusterNum + k ] += log_lambda[k];
			log_resp[ n * ClusterNum + k ] += log_weights[k];
		}
	}


	TYPE *log_prob_norm = (TYPE *)malloc( sizeof(TYPE) * BITMAP);
	compute_log_prob( log_resp , log_prob_norm );

	double ll = compute_log_likelihood(log_prob_norm);


	free(log_prob_norm);

	//DEB_PRINT_VEC("log_prob_norm" , log_prob_norm , BITMAP , "%.3f " );
	
	return ll;

}


void _m_step( TYPE *X , TYPE *log_resp , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ ,
		TYPE weight_concentration_prior_ , TYPE mean_precision_prior_ , TYPE degree_of_freedom_prior_ , TYPE *mean_prior_ , TYPE *covariance_prior_ ){


	TYPE *nk , *xk , *sk , *tmp , *diff;
	nk  = (TYPE *)malloc(sizeof(TYPE) * ClusterNum );
	xk  = (TYPE *)malloc(sizeof(TYPE) * ClusterNum * D);
	sk  = (TYPE *)malloc(sizeof(TYPE) * ClusterNum * D);
	tmp = (TYPE *)malloc(sizeof(TYPE) * ClusterNum * D);
	diff= (TYPE *)malloc(sizeof(TYPE) * ClusterNum * D);

	for( int k = 0 ; k < ClusterNum ; k++ ){
		nk[k] = 0.0;
		for( int d = 0 ; d < D ; d++ ){
			xk [ k * D + d ] = 0.0;
			sk [ k * D + d ] = 0.0;
			tmp[ k * D + d ] = 0.0;
		}
	}


	DEB_PRINT_MAT_S("resp" , log_resp , BITMAP , ClusterNum , "%.3f " );

	for( int n = 0 ; n < BITMAP ; n++  ){
		for( int k = 0 ; k < ClusterNum ; k++ ){
			nk[ k ] += log_resp[ n * ClusterNum + k ];
			for( int d = 0 ; d < D ; d++ ){
				xk[ k * D + d ] += log_resp[ n * ClusterNum + k ] * X[ n * D + d ];
				sk[ k * D + d ] += log_resp[ n * ClusterNum + k ] * powf( X[ n * D + d ] , 2 );
				tmp[ k * D + d ] += log_resp[ n * ClusterNum + k ] * X[ n * D + d ];
			}
		}
	}

	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			if( nk[k] > 0 ){
				xk[ k * D + d ] /= nk[k];
				sk[ k * D + d ] = sk[k * D + d ] / nk[k] - 2 * xk[ k * D + d ] * tmp[ k * D + d ] / nk[k] + powf( xk[ k * D + d ] , 2);
				if(  sk[ k * D + d ] <= 0 ) sk[ k * D + d ] = REG_COVOR;
			}
			else{
				xk[ k * D + d ] = mean_prior_[d];
				sk[ k * D + d ] = REG_COVOR;
			}
		}
	}

	DEB_PRINT_VEC( "nk" , nk , ClusterNum , "%.3f ");
	DEB_PRINT_MAT( "xk" , xk , ClusterNum , D , "%.3f " );
	DEB_PRINT_MAT( "sk" , sk , ClusterNum , D , "%.3f " );

	for( int k = 0 ; k < ClusterNum ; k++ ){
		weight_concentration_[k] = weight_concentration_prior_ + nk[k];
		mean_precision_[k] = mean_precision_prior_ + nk[k];
		degree_of_freedom_[k] = degree_of_freedom_prior_ + nk[k];
	}

	// m
	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			means_[k * D + d]= ( (mean_precision_prior_ * mean_prior_[d]) + (nk[k] * xk[ k * D + d ]) ) / mean_precision_[k];
		}
	}


	// W
	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] = powf( xk[ k * D + d ]  - mean_prior_[d]  , 2 );
		}
	}

	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] *= mean_precision_prior_ / mean_precision_[k];
			sk[ k * D + d ] += diff[ k * D + d ];
		}
	}

	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			diff[ k * D + d ] = covariance_prior_[d] + (double)nk[k] * sk[ k * D + d ];
		}
	}

	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0 ; d < D ; d++ ){
			precision_cholesky_[k * D + d] = diff[ k * D + d ] / degree_of_freedom_[k];
			precision_cholesky_[k * D + d] = (float) ( 1 / sqrt((double)precision_cholesky_[k * D + d]) );
		}
	}

	DEB_PRINT_MAT("precision_cholesky_" , precision_cholesky_ , ClusterNum , D , "%.3f ");
	DEB_PRINT_VEC("weight_concentration_",weight_concentration_,ClusterNum , "%.3f ");
	DEB_PRINT_VEC("mean_precision_",mean_precision_,ClusterNum , "%.3f ");
	DEB_PRINT_VEC("degree_of_freedom_",degree_of_freedom_,ClusterNum , "%.3f ");
	DEB_PRINT_MAT("mean" , means_ , ClusterNum , D , "%.3f ");

	free(nk ); 
	free(xk ); 
	free(sk ); 
	free(tmp); 
	free(diff);
	return;
}

TYPE _log_dirichket_norm( TYPE *weight_concentration_ ){
	double tmp = 0 , sum = 0;
	for( int k = 0 ; k < ClusterNum ; k++ ){
		tmp += lgamma( weight_concentration_[k] );
		sum += weight_concentration_[k];
	}
	return lgamma( sum ) - tmp;
}

double _log_wishart_norm( TYPE *log_det_precision_chol ,  TYPE *degree_of_freedom_ ){
	double sum_log_wishart = 0.0;

	for( int k = 0 ; k < ClusterNum ; k++ ){
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

	TYPE *log_det_precision_chol = (TYPE *)malloc( sizeof(TYPE) * ClusterNum );

	for( int k = 0 ; k < ClusterNum ; k++ ){
		log_det_precision_chol[k] = 0.0;
	}

	for( int k = 0 ; k < ClusterNum ; k++ ){
		for( int d = 0; d < D ; d++ ) log_det_precision_chol[k] += logf( precision_cholesky_[ k * D + d ]);
		log_det_precision_chol[k] -= 0.5 * D * logf( degree_of_freedom_[k]);
	}

	TYPE sum_log_wishart = _log_wishart_norm( log_det_precision_chol ,  degree_of_freedom_);

	TYPE log_weight_norm = _log_dirichket_norm( weight_concentration_ );

	TYPE sum_log_mean_precision = 0;
	for( int k = 0 ; k < ClusterNum ; k++ ) sum_log_mean_precision += logf(mean_precision_[k]);
#if DEBUG_PRINT
	printf("%.5f %.5f %.5f %.5f\n", sum_resp , sum_log_wishart , log_weight_norm , sum_log_mean_precision );
#endif

	free(log_det_precision_chol);

	return (float)(-sum_resp - sum_log_wishart - log_weight_norm - 0.5 * D * sum_log_mean_precision);
}


int fit(	TYPE *X , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ ,
		TYPE weight_concentration_prior_ , TYPE mean_precision_prior_ , TYPE degree_of_freedom_prior_ , TYPE *mean_prior_ , TYPE *covariance_prior_ ){
	TYPE old_lower_bound = 0;
	//TODO

	TYPE *log_resp = (TYPE *)malloc(sizeof( TYPE ) * BITMAP * ClusterNum);

	int converged = false;

	for( int iter = 0 ; iter < MAX_ITER ; iter++ ){
		_e_step( X , log_resp , weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ );

		DEB_PRINT_MAT_S("e_step ato log_resp" , log_resp , BITMAP , ClusterNum , "%.3f " );

		double sum_resp = 0;
		for( int n = 0 ; n < BITMAP ; n++ ) for( int k = 0 ; k < ClusterNum ; k++ ){
			float tmp = expf( log_resp[ n * ClusterNum + k ] );
			sum_resp += tmp * log_resp[ n * ClusterNum + k ];
			log_resp[ n * ClusterNum + k ] = tmp;
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

	free(log_resp);
	return true;
}

double predict( TYPE *X , TYPE *weight_concentration_ , TYPE *mean_precision_ , TYPE *degree_of_freedom_ , TYPE *means_ , TYPE *precision_cholesky_ , unsigned int *DisplayData ){
	int *res       = (int  *)malloc( sizeof(int ) * BITMAP );
	TYPE *log_resp = (TYPE *)malloc( sizeof(TYPE) * BITMAP * ClusterNum);

	double ll = _e_step( X , log_resp , weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ );

	for( int n = 0 ; n < BITMAP ; n++ ){
		TYPE res_val = log_resp[n * ClusterNum];
		res[n] = 0;
		for( int k = 1 ; k < ClusterNum ; k++ ){
			if(res_val < log_resp[n * ClusterNum + k ]){
				res_val = log_resp[n*ClusterNum+k];
				res[n] = k;
			}
		}
	}

	for( int n = 0 ; n < BITMAP ; n++ ){
		unsigned int r,g,b;
		r = (res[n] % 3) * (255 / 3);
		g = (res[n] % 4) * (255 / 4);
		b = (res[n] % 5) * (255 / 5);
		DisplayData[n] = ( r << 24 ) | ( g << 16 ) | ( b << 8 );
	}

	free(res);
	free(log_resp);

	ll /= BITMAP;

	return ll;
}

int main(int argc , char *argv[]){

	WD = 320;
	HT = 240;
	unsigned short enable_x11 = 0;
	float Threshold = 1.0;
	unsigned short AlwaysResetParameter = 0;
	unsigned int   ColorAbstractionLevel = 1;

	int TwiceScreenMode = 0;
	int UseCoordinateInfo = 0;
	ClusterNum = 2;

	for(argc--,argv++;argc;argc--,argv++){
		if(**argv == '-'){
			switch(*(*argv+1)){
				case 'w':
					sscanf((*argv+2), "%d", &WD);
					break;
				case 'h':
					sscanf((*argv+2), "%d", &HT);
					break;
				case 's':
					TwiceScreenMode = 1;
					break;
				case 'a':
					sscanf((*argv+2), "%d", &ColorAbstractionLevel);
					if( ColorAbstractionLevel <= 0 && ColorAbstractionLevel >= 8 ){
						printf("ColorAbstractionLevel must larger than 0 and smaller than 9\n");
						exit(1);
					}
					break;
				case 'x':
					enable_x11 = 1;
					break;
				case 'c':
					sscanf((*argv+2), "%d", &ClusterNum );
					break;
				case 't':
					sscanf((*argv+2), "%f", &Threshold );
					break;
				case 'r':
					AlwaysResetParameter = 1;
					break;
				case 'u':
					UseCoordinateInfo = 1;
					break;

				default:
					printf("\nInvalid Input %c\n" , *(*argv+1));
					printf("Options\n");
					printf("  -w<num>      : Width of input image\n");
					printf("  -h<num>      : Height of input image\n");
					printf("  -c<num>      : Init cluster number. default : 2\n");
					printf("  -a<num>      : ColorAbstractionLevel. default : 1\n");
					printf("  -t<num>      : Threshold for reset\n");
					printf("  -u           : if you want to use Coordinate infomation. must $make D=5\n");
					printf("  -x           : Enable X11 window\n");
					printf("  -r           : if likelihood fall below. re:initialize parameters by current frame\n");
					printf("  -s           : TwiceScreenMode\n");
					exit(1);
					break;
			}
		}
	}


	if( UseCoordinateInfo && D != 5 ){
		printf("If use coordinate infomation, please $make D=5\n");
		exit(1);
	}

	if( !UseCoordinateInfo && D != 3 ){
		printf("If don't use coordinate infomation, please $make D=3\n");
		exit(1);
	}

	if( D != 3 && D != 5 ){
		printf("Invalid Dimention\n");
		exit(1);
	}


	BITMAP = WD * HT;

	/*
	if( BITMAP != BITMAP ){
		printf("BITMAP != BITMAP\n");
		exit(1);
	}
	*/

	printf("HT:%d\tWD:%d\tBITMAP:%d\tX_display:%d\n", HT , WD , BITMAP , enable_x11 );
	printf("Input ClusterNum:%d Init ClusterNum:%d\n", ClusterNum , ClusterNum );

	SCRWD = 1;
	SCRHT = 1;
	if (enable_x11)
		x11_open(WD, HT, SCRWD, SCRHT); /*sh_video->disp_w, sh_video->disp_h, # rows of output_screen*/

	TYPE *Data;
	Data = (TYPE *)malloc(sizeof(TYPE) * BITMAP * D );
	unsigned short DoLearning = 1;
	unsigned short Initialize = 1;
	double old_LL = 0;


	/*** init ***/

	TYPE *resp_ , *weight_concentration_ , *means_ , *mean_precision_ , *degree_of_freedom_ , *precision_cholesky_ , *covariances_ , *X ;

	resp_ 					= (TYPE *)malloc( sizeof(TYPE) * BITMAP * ClusterNum );
	weight_concentration_	= (TYPE *)malloc( sizeof(TYPE) * ClusterNum );
	means_ 					= (TYPE *)malloc( sizeof(TYPE) * ClusterNum * D );
	mean_precision_			= (TYPE *)malloc( sizeof(TYPE) * ClusterNum );
	degree_of_freedom_		= (TYPE *)malloc( sizeof(TYPE) * ClusterNum );
	precision_cholesky_		= (TYPE *)malloc( sizeof(TYPE) * ClusterNum * D );
	covariances_			= (TYPE *)malloc( sizeof(TYPE) * ClusterNum * D );
	X						= (TYPE *)malloc( sizeof(TYPE) * BITMAP * D );

	TYPE weight_concentration_prior_ = (TYPE)1/ClusterNum; //alpha_0
	TYPE mean_precision_prior_ = 1.0; // beta_0
	TYPE degree_of_freedom_prior_ = D; //Nu_0
	TYPE mean_prior_[D]; // beta_0
	TYPE covariance_prior_[D];

	unsigned int DisplayData[BITMAP];

	while(1){

		if (feof(stdin)) break;

		GetImageFromStdin(Data , TwiceScreenMode , ColorAbstractionLevel );

		if( Data == NULL ){
			printf("failed input Data\n");
			return 1;
		}


		if(UseCoordinateInfo){
			for( int d = 0 ; d < D ; d++ ){
				for( int n = 0 ; n < BITMAP ; n++ ){
					if( d < 3 )
						X[ n * D + d ] = Data[ n * D + d];
					else if( d == 4 )
						X[ n * D + d ] = (n % WD) / ( (float)HT / 10 ); // Y座標入力
					else
						X[ n * D + d ] = (n / WD) / ( (float)WD / 10 ); // X座標入力
				}
			}
		}
		else{
			for( int d = 0 ; d < D ; d++ ){
				for( int n = 0 ; n < BITMAP ; n++ ){
					X[ n * D + d ] = Data[ n * D + d];
				}
			}
		}

		if( DoLearning ){

			if( Initialize ){

				DEB_PRINT_MAT_S("X" , X , BITMAP , D , "%.5f " );

				for( int d = 0 ; d < D ; d++ ){
					mean_prior_[d] = 0;
					covariance_prior_[d] = 0;
				}
				for( int n = 0 ; n < BITMAP ; n++ ) for( int d = 0 ; d < D ; d++ ) mean_prior_[d] += X[ n * D + d ];
				for( int d = 0 ; d < D ; d++ ) mean_prior_[d] /= BITMAP;
				for( int n = 0 ; n < BITMAP ; n++ ) for( int d = 0 ; d < D ; d++ ) covariance_prior_[d] += powf( ( X[ n * D + d ] - mean_prior_[d] ) , 2 );
				for( int d = 0 ; d < D ; d++ ) covariance_prior_[d] /= BITMAP;

				for( int k = 0 ; k < ClusterNum ; k++ ){
					mean_precision_[k] = 1.0;
					weight_concentration_[k] = 1.0 / ClusterNum;
				}

				for( int n = 0 ; n < BITMAP ; n++ ) for( int k = 0 ; k < ClusterNum ; k++ ) resp_[ n * ClusterNum + k ] = (TYPE) 1 / ClusterNum;


				DEB_PRINT_MAT_S("resp_" , resp_, BITMAP , ClusterNum , "%.5f " );

				TYPE *nk , *xk , *sk;
				nk = (TYPE *)malloc(sizeof(TYPE) * ClusterNum );
				xk = (TYPE *)malloc(sizeof(TYPE) * ClusterNum * D);
				sk = (TYPE *)malloc(sizeof(TYPE) * ClusterNum * D);

				for( int k = 0 ; k < ClusterNum ; k++ ) nk[k] = BITMAP / ClusterNum;

#if RANDOM_INIT
				srand( RANDOM_SEED );
				for( int k = 0 ; k < ClusterNum ; k++ ){
					int idx = rand() % BITMAP;
					for( int d = 0 ; d < D ; d++ ){
						xk[ k * D + d ] = X[idx * D + d];
					}
				}
#else
				for( int k = 0 ; k < ClusterNum ; k++ ){
					int idx =  BITMAP / ClusterNum * k;
					for( int d = 0 ; d < D ; d++ ){
						xk[ k * D + d ] = X[idx * D + d];
					}
				}
#endif

				/*** estimate_gaussian_covariance_diag ***/
				for( int k = 0 ; k < ClusterNum ; k++ ){

					for( int d = 0 ; d < D ; d++ ){
						sk[ k * D + d ] = 0;
					}

					for( int d = 0 ; d < D ; d++ ){
						for( int n = 0 ; n < BITMAP ; n++ )
							sk[ k * D + d ] += resp_[ n * ClusterNum + k ] * powf( X[ n * D + d ] - xk[ k * D + d ] , 2 );
						sk[ k * D + d ] /= nk[k];
					}
				}


				DEB_PRINT_VEC("init nk" , nk , ClusterNum , "%.5f ");
				DEB_PRINT_MAT("init xk" , xk , ClusterNum , D , "%.5f " );
				DEB_PRINT_MAT("init sk" , sk , ClusterNum , D , "%.5f " );

				/*** intialize prior parameters ***/
				// alpha , beta , nu
				for( int k = 0 ; k < ClusterNum ; k++ ){
					weight_concentration_[k] = weight_concentration_prior_ + nk[k];
					mean_precision_[k] = mean_precision_prior_ + nk[k];
					degree_of_freedom_[k] = degree_of_freedom_prior_ + nk[k];
				}

				// m
				for( int k = 0 ; k < ClusterNum ; k++ ){
					for( int d = 0 ; d < D ; d++ ){
						means_[k * D + d]= ( (mean_precision_prior_ * mean_prior_[d]) + (nk[k] * xk[ k * D + d ]) ) / mean_precision_[k];
					}
				}

				// W
				TYPE diff[ ClusterNum * D ];
				for( int k = 0 ; k < ClusterNum ; k++ ){
					for( int d = 0 ; d < D ; d++ ){
						diff[ k * D + d ] = powf( xk[ k * D + d ]  - mean_prior_[d]  , 2 );
					}
				}

				for( int k = 0 ; k < ClusterNum ; k++ ){
					for( int d = 0 ; d < D ; d++ ){
						diff[ k * D + d ] *= mean_precision_prior_ / mean_precision_[k];
						sk[ k * D + d ] += diff[ k * D + d ];
					}
				}

				for( int k = 0 ; k < ClusterNum ; k++ ){
					for( int d = 0 ; d < D ; d++ ){
						diff[ k * D + d ] = covariance_prior_[d] + (double)nk[k] * sk[ k * D + d ];
					}
				}

				for( int k = 0 ; k < ClusterNum ; k++ ){
					for( int d = 0 ; d < D ; d++ ){
						covariances_[k * D + d] = diff[ k * D + d ] / degree_of_freedom_[k];
						precision_cholesky_[k * D + d] = (float) ( 1 / sqrt((double)covariances_[k * D + d]) );
					}
				}

				DEB_PRINT_VEC("init weight_concentration_" , weight_concentration_ , ClusterNum , "%.3f ");
				DEB_PRINT_VEC("init mean_precision_" , mean_precision_ , ClusterNum , "%.3f ");
				DEB_PRINT_VEC("init degree_of_freedom_" , degree_of_freedom_ , ClusterNum , "%.5f ");
				DEB_PRINT_MAT("init means_" , means_ , ClusterNum , D , "%.5f ");
				DEB_PRINT_MAT("init precision_cholesky_" , precision_cholesky_ , ClusterNum , D , "%.5f ");

				Initialize = 0;

				free(nk); free(xk); free(sk);
			} // if( Initialize )

			fit( X , weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ , weight_concentration_prior_ , mean_precision_prior_ , degree_of_freedom_prior_ , mean_prior_ , covariance_prior_ );

		} // if( DoLearning )

		double LL =  predict( X ,  weight_concentration_ , mean_precision_ , degree_of_freedom_ , means_ , precision_cholesky_ , DisplayData);

		printf("log likelihood : %.5lf\n" , LL );

		if( DoLearning ){
			DoLearning = 0;
			old_LL = LL;
		}

		RGB_to_X(0, DisplayData);
		while (x11_checkevent());

		if( fabs( old_LL - LL ) > Threshold ){
			DoLearning = 1;
			if( AlwaysResetParameter ) Initialize = 1;
			printf("reset learning\n");
		}

		old_LL = LL;
	}

	free(Data);

	free(resp_);
	free(weight_concentration_);
	free(means_);
	free(mean_precision_);
	free(degree_of_freedom_);
	free(precision_cholesky_);
	free(covariances_);
	free(X);

	return 0;
}
