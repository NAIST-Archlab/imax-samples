
#include<chrono>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<time.h>
#include<iostream>
#include<vector>
#include<boost/math/special_functions/digamma.hpp>

#include<cuda_runtime.h>
#include<cublas_v2.h>

#ifdef PROF
#include<cuda_profiler_api.h>
#endif

#include"setting.hpp"
#include"cu_sum.hpp"
#include"profile.hpp"
#include"print.hpp"
#include"IO.hpp"

#define B0  1                 /* 以下はBernoulli数 */
#define B1  (-1.0 / 2.0)
#define B2  ( 1.0 / 6.0)
#define B4  (-1.0 / 30.0)
#define B6  ( 1.0 / 42.0)
#define B8  (-1.0 / 30.0)
#define B10 ( 5.0 / 66.0)
#define B12 (-691.0 / 2730.0)
#define B14 ( 7.0 / 6.0)
#define B16 (-3617.0 / 510.0)

template<typename type>
__global__ void divide_arr( type *arr , unsigned int size , ITR_SIZE div_val  ){
	const ITR_SIZE idx = blockIdx.x * blockDim.x + threadIdx.x;
	arr[idx] = arr[idx] / div_val;
}

template<typename type >
__global__
void set_value_matrix( type *mat , ITR_SIZE rows  , ITR_SIZE cols , const type val ){
	const ITR_SIZE row = blockIdx.x * blockDim.x + threadIdx.x;
	for( ITR_SIZE col = 0 ; col < cols ; col++ ){
		mat[ col*rows + row] = val;
	}
}

__host__ __device__ float digamma(float xx ){
	float v, w;
	v = 0;
	while(xx < 100){
		v += 1/xx;
		xx++;
	}
	w = 1/(xx * xx);
	v += ((((((((B16 / 16) * w + (B14 /14)) * w + (B12 / 12)) * w + (B10 / 10)) * w + (B8 / 8)) * w + (B6 / 6)) * w + (B4 / 4))* w + (B2 / 2)) * w + 0.5 / xx;
	return log(xx) - v;
}

template<typename type>
__global__ void pow_self_elements( type *mat , const int rows , const int cols ){
	const int row = blockIdx.x * blockDim.x + threadIdx.x;

	if( row < rows )
		for( int col = 0 ; col < cols ; col++ ) mat[ row * cols + col ] = powf(mat[row * cols + col ] , 2);
}

template<typename type >
__global__ void compute_log_gaussian_prob( type *dev_r_ , type *dev_XK_, type *dev_K_ , type *dev_log_det_chol_ , type *dev_degree_of_freedom_ , const ITR_SIZE N , const int K , const int D ){
	const ITR_SIZE n = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ type share[];

	if( n < N ){

		if( threadIdx.x < K ){
			share[threadIdx.x] = dev_log_det_chol_[threadIdx.x] -  0.5 * D * logf(dev_degree_of_freedom_[threadIdx.x]);
		}

		__syncthreads();

		for( int k = 0 ; k < K ; k++ ){
			//dev_r_[k*N+n] = -0.5 * ( D * logf( 2 * M_PI ) + ( dev_r_[k*N+n] - ( 2 * dev_XK_[k*N+n]) + dev_K_[k] ) )  + dev_log_det_chol_[k] -  0.5 * D * logf(dev_degree_of_freedom_[k]);
			dev_r_[k*N+n] = -0.5 * ( D * logf( 2 * M_PI ) + ( dev_r_[k*N+n] - ( 2 * dev_XK_[k*N+n]) + dev_K_[k] ) ) + share[k];
		}
	}
}

__global__ void get_sum_of_2dim( float *devK , float *dev_KD_ , int K , int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;

	if( k < K ){
		devK[k] = 0;
		for( int d = 0 ; d < D ; d++ ){
			devK[k] += dev_KD_[ d * K + k ];
		}
	}

}

template<typename type>
__global__ void compute_weighted_log_prob( type *dev_log_gauss_, type *dev_log_lambda_ , type *dev_log_weights_ , const ITR_SIZE N , const int K ){
	const ITR_SIZE n = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ double sum_lw[];

	if( n < N ){

		if( threadIdx.x < K )
			sum_lw[threadIdx.x] = (double)dev_log_lambda_[threadIdx.x] + dev_log_weights_[threadIdx.x];
		__syncthreads();

		for( int k = 0 ; k < K ; k++ )
			//dev_log_gauss_[ k * N + n ] = dev_log_gauss_[ k * N + n ] + dev_log_lambda_[k] + dev_log_weights_[k];
			dev_log_gauss_[ k * N + n ] = dev_log_gauss_[ k * N + n ] + sum_lw[k];
	}

}

template<unsigned const int threadNum>
__global__ void compute_log_resp( float *norm , float *resp , ITR_SIZE N , int K ){
	const ITR_SIZE n = blockIdx.x * blockDim.x + threadIdx.x;
	if( n < N ){

		__shared__ float _max_of_kth[threadNum];

		_max_of_kth[threadIdx.x] = resp[n];
		for( int k = 1 ; k < K ; k++ ) _max_of_kth[threadIdx.x] = max( _max_of_kth[threadIdx.x] , resp[ k * N + n ] );
		float _sum = 0.0;
#ifdef FMATHA
		for( int k = 0 ; k < K ; k++ ) _sum += expf( __fsub_rd ( resp[k*N+n] , _max_of_kth[threadIdx.x] ));
		const float _logsumexp = logf( _sum ) + _max_of_kth[threadIdx.x];
		for( int k = 0 ; k < K ; k++ ) resp[k*N+n] = __fsub_rd ( resp[k*N+n] , _logsumexp );
#else
		for( int k = 0 ; k < K ; k++ ) _sum += expf( resp[k*N+n] - _max_of_kth[threadIdx.x] );
		const float _logsumexp = logf( _sum ) + _max_of_kth[threadIdx.x];
		for( int k = 0 ; k < K ; k++ ) resp[k*N+n] =  resp[k*N+n] - _logsumexp ;
#endif
	}
}

template<typename type>
__global__ void pow_substitution_elements( type *to_mat , type *from_mat , const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if( k < K ){
		for( int d = 0 ; d < D ; d++ ){
			to_mat[ d * K + k ] = powf( from_mat[d*K+k] , 2 );
		}
	}
}

template<typename type>
__global__ void hadamard_product_self( type *mat_a , type *mat_b , const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if( k < K ){
		for( int d = 0 ; d < D ; d++ ){
			mat_a[d*K+k] = mat_a[d*K+k] * mat_b[d*K+k];
		}
	}
}

template<typename type>
__global__ void hadamard_product_subtitution( type *to_mat , type *from_a , type *from_b, const int  K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if( k < K ){
		for( int d = 0 ; d < D ; d++ ){
			to_mat[d * K + k] = from_a[d*K+k] * from_b[d*K+k];
		}
	}
}

//calc log Lambda for calc log resp
template<typename type>
__global__ void estimate_log_lambda_( float *log_lambda , float *degree_of_freedom_,  float *mean_precision_, const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ type share[];
	if ( k < K  ){
		double sum_digamma_nu_ = 0.0;

		if(threadIdx.x < K )
			share[k] = degree_of_freedom_[threadIdx.x] + 1;

		for( unsigned int d = 0 ; d < D ; d++ ){
			//digamma fuction
			sum_digamma_nu_ += logf(0.5*(share[k]-d) + 0.4849142) -  1 / ( 0.5*(share[k]-d) * 1.0271785 ) ;
		}
		log_lambda[k] = 0.5 * ( D * logf(2.0) + sum_digamma_nu_ - D / mean_precision_[k] ) ;
	}
}

template<const unsigned int threadNum , typename in_type , typename out_type>
__global__ void substitute_sum_with_idx( in_type *g_idata , out_type *g_odata , const ITR_SIZE N , const ITR_SIZE k){
	__shared__ out_type sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	out_type mysum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mysum += g_idata[i + blockDim.x];
	sdata[tid] = mysum;
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s) {
			sdata[tid] = mysum = mysum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		if(blockDim.x >= 64) mysum += sdata[tid + 32];
		for (int offset = 32/2; offset>0; offset>>=1) {
			mysum += __shfl_down_sync(FULL_MASK, mysum , offset);
		}
	}
	if (tid == 0) g_odata[blockIdx.x + k] = mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}

template<typename in_type , typename out_type >
__global__ void cpy( out_type *to , in_type *from, const ITR_SIZE N , const ITR_SIZE k){
	const ITR_SIZE idx = blockIdx.x * blockDim.x + threadIdx.x;
	to[idx] = from[k * N + idx];
}

//double ver.
template<const unsigned int threadNum , typename in_type , typename out_type >
__global__ void cpy_test( in_type *g_idata, out_type *g_odata, const ITR_SIZE N , const unsigned int k ){
	__shared__ out_type sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	out_type mysum = (i < N) ? g_idata[ k * N + i ] : 0;
	if ( i +  blockDim.x < N) mysum += g_idata[ ( k * N + i ) +  blockDim.x  ]  ;
	sdata[tid] = mysum;
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s) {
			sdata[tid] = mysum = mysum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		if(blockDim.x >= 64) mysum += sdata[tid + 32];
		for (int offset = 32/2; offset>0; offset>>=1) {
			mysum += __shfl_down_sync(FULL_MASK, mysum , offset);
		}
	}
	if (tid == 0) g_odata[blockIdx.x] = mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}



template<typename in_type , typename out_type>
__global__ void calc_for_get_covariance_prior( out_type *dev_N_ , in_type *dev_X_, in_type *dev_mean_prior_ , const ITR_SIZE N , const ITR_SIZE d){
	const ITR_SIZE idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx < N )
		dev_N_[idx] = pow( dev_X_[d * N + idx] - dev_mean_prior_[d]  , 2 );
}

template<typename out_type>
__global__ void calc_for_get_covariance_prior( out_type *dev_N_ , float *dev_X_, float *dev_mean_prior_ , const ITR_SIZE N , const ITR_SIZE d ){
	const ITR_SIZE idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx < N )
		dev_N_[idx] = powf( dev_X_[d * N + idx] - dev_mean_prior_[d]  , 2 );

}

//2 type of cpy_and_process_for_calc_sum_resp
template<typename type , const unsigned int threadNum>
__global__ void cpy_and_process_for_calc_sum_resp( type *from , float *to , const ITR_SIZE N  , const int k ){
	const ITR_SIZE n = blockIdx.x * blockDim.x + threadIdx.x;


	if( n < N ){

#ifdef FMATHB
		to[n] = (float) (__expf( from[k * N + n ] ) * (double)from[ k * N + n ]);
#else 
		to[n] = (float) (expf( from[k * N + n ] ) * (double)from[ k * N + n ]);
#endif

#ifdef FMATHG //非推奨
		from[k*N+n] = __expf(from[k*N+n]);
#else
		from[k*N+n] = expf(from[k*N+n]);
#endif
#ifdef DEBUG
		CHECK_ERROR("cpy_and_process_for_calc_sum_resp" , from[k*N+n]);
#endif
	}

}
template<typename type , const unsigned int threadNum>
__global__ void cpy_and_process_for_calc_sum_resp( type *from , double *to , const ITR_SIZE N  , const int k ){
	const ITR_SIZE n = blockIdx.x * blockDim.x + threadIdx.x;


	if( n < N ){
		__shared__ double sdata[threadNum];
		sdata[threadIdx.x] = exp( (double)from[k * N + n ] );
		to[n] = sdata[threadIdx.x] * from[ k * N + n ];
		from[k*N+n] = sdata[threadIdx.x];
#ifdef DEBUG
		CHECK_ERROR("cpy_and_process_for_calc_sum_resp" , from[k*N+n]);
#endif
	}
}

__global__ void divide_and_check_XBar( float *xk , float *nk,  float *mean_prior_ , const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float share[];

	if(threadIdx.x < K ) share[threadIdx.x] = nk[threadIdx.x];
	__syncthreads();


	if( k < K ){
		for( int d = 0 ; d < D ; d++ ){

			if( share[k] > 1.0 )
				xk[d*K+k] = (double)xk[d*K+k] / share[k];
			else
				xk[d*K+k] = mean_prior_[k];

#ifdef DEBUG
			CHECK_ERROR("divide_and_check_XBar" , xk[d*K+k] );
#endif

		}
	}
}

__global__ void estimate_gaussian_covariance_diag( float *avg_X2 , float *avg_means_2 , float *avg_X_means , float *nk , float reg_covar , const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	for( int d = 0 ; d < D ; d++ ){

		if( nk[k] > 0 )
			avg_X2[d*K+k] = ((double)avg_X2[d*K+k]/nk[k]) - 2 * ( (double)avg_X_means[d*K+k]/nk[k]) + avg_means_2[d*K+k];
		else
			avg_X2[d*K+k] = 1e-6;
		// prevent that S takeing negative value
		if( avg_X2[d*K+k] <= 0 ) {

#ifdef DEBUG
			printf("COLLECTION:estimate_gaussian_covariance_diag\n");
			CHECK_ERROR("estimate_gaussian_covariance_diag" , avg_X2[d*K+k]);
#endif
			avg_X2[d*K+k] = 1e-6;
		}
	}
}

__global__ void estimate_precisions( float *nk , float *xk, float *sk, float *mean_prior , float *precisions_cholesky , float *degree_of_freedom , float *covariance_prior , float *mean_precision ,  float *mean_precision_prior , const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if( k < K ){


		if( nk[k] > 1.0 ){

			double diff[DIM] = {};
			for( int d = 0 ; d < D ; d++ ) diff[d] =  pow( xk[d*K+k] - mean_prior[d] , 2) ;

			for( int d = 0 ; d < D ; d++ ){
				diff[d] *= *mean_precision_prior / mean_precision[k];
				sk[d*K+k] += diff[d];
			}

			for( int d = 0 ; d < D ; d++ ){
				diff[d] = covariance_prior[d] + (double)nk[k] * sk[d*K+k];
			}

			//diff <- covariances
			for( int d = 0 ; d < D ; d++ ){
				diff[d] /= degree_of_freedom[k];
				precisions_cholesky[d*K+k] = 1 / sqrt(diff[d]);

#ifdef DEBUG
				CHECK_ERROR( "estimate_precisions" ,  precisions_cholesky[d*K+k]);
#endif
			}
		}
		else{
			for( int d = 0 ; d < D ; d++ )
				precisions_cholesky[d*K+k] = 1e-6;
		}
	}
}

template<const unsigned int threadNum , typename type>
__global__ void add_sum_resp( type *g_idata , double *sum_resp , const ITR_SIZE N){
	__shared__ type sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	type mysum = 0;
	if( i < N ){
		mysum = g_idata[i];
#ifdef DEBUG
		CHECK_ERROR("add_sum_resp" , mysum );
#endif
	}
	if( i + blockDim.x < N ){
#ifdef FMATHC
		mysum = __fadd_rd ( mysum , g_idata[i + blockDim.x ] );
#else
		mysum = mysum , g_idata[i + blockDim.x ];
#endif
#ifdef DEBUG
		CHECK_ERROR("add_sum_resp" , mysum );
#endif
	}
	sdata[tid] = mysum;
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s) {
#ifdef FMATHC
			sdata[tid] = mysum = __fadd_rd( mysum , sdata[tid + s] );
#else
			sdata[tid] = mysum = mysum + sdata[tid + s];
#endif
		}
		__syncthreads();
	}
	if (tid < 32) {
		if(blockDim.x >= 64) mysum += sdata[tid + 32];
		for (int offset = 32/2; offset>0; offset>>=1) {
#ifdef FMATHD
			mysum = __fadd_rd( mysum , __shfl_down_sync(FULL_MASK, mysum , offset) );
#else
			mysum += __shfl_down_sync(FULL_MASK, mysum , offset);
#endif
		}
	}
	//sum_respが大きくなっている場合が多いので倍精度で加算
	if (tid == 0) *sum_resp = *sum_resp + (double)mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}


/*** 2type of get_sum_of_N ***/
//float ver.
template<const unsigned int threadNum>
__global__ void get_sum_of_N( float *g_idata , float *g_odata , const ITR_SIZE N ){
	__shared__ float sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float mysum = (i < N) ? g_idata[i] : 0;
#ifdef FMATHE
	if (i + blockDim.x < N) mysum = __fadd_rd ( mysum , g_idata[i + blockDim.x] );
#else
	if (i + blockDim.x < N) mysum += g_idata[i + blockDim.x]  ;
#endif
	sdata[tid] = mysum;
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s) {
#ifdef FMATHE
			sdata[tid] = mysum = __fadd_rd ( mysum , sdata[tid + s] );
#else
			sdata[tid] = mysum = mysum + sdata[tid + s];
#endif
		}
		__syncthreads();
	}
	if (tid < 32) {
		if(blockDim.x >= 64) mysum += sdata[tid + 32];
		for (int offset = 32/2; offset>0; offset>>=1) {
#ifdef FMATHF
			mysum = __fadd_rd ( mysum , __shfl_down_sync(FULL_MASK, mysum , offset) );
#else
			mysum += __shfl_down_sync(FULL_MASK, mysum , offset);
#endif
		}
	}
	if (tid == 0) g_odata[blockIdx.x] = mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}
//double ver.
template<const unsigned int threadNum>
__global__ void get_sum_of_N( double *g_idata , double *g_odata , const ITR_SIZE N ){
	__shared__ double sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	double mysum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mysum += g_idata[i + blockDim.x]  ;
	sdata[tid] = mysum;
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s) {
			sdata[tid] = mysum = mysum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		if(blockDim.x >= 64) mysum += sdata[tid + 32];
		for (int offset = 32/2; offset>0; offset>>=1) {
			mysum += __shfl_down_sync(FULL_MASK, mysum , offset);
		}
	}
	if (tid == 0) g_odata[blockIdx.x] = mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}

/*** end 2type of get_sum_of_N ***/

template<typename type>
__global__ void compute_log_det_chol( type *_log_det_chol , type *precision_chol, type *degree_of_freedom_ , const int K , const int D ){
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if( k < K ){
		double sum = 0.0;
		for( int d = 0 ; d < D ; d++ ) sum += log( double( precision_chol[d*K+k])  );
		// for( int d = 0 ; d < D ; d++ ) sum += logf( precision_chol[k] ); なぜかバグらない
		_log_det_chol[k] = sum;
	}

}

template<typename type>
__global__ void substitute_zero( type *a ){
	*a = 0.0;
}


template<typename in_type , typename out_type >
__global__ void get_sum_of_K( in_type *resp  , out_type *dev_N_ , ITR_SIZE N , int K ){
	const ITR_SIZE n = blockIdx.x * blockDim.x + threadIdx.x;
	if( n < N ){
		float _max_of_kth = resp[n];
		for( int k = 1 ; k < K ; k++ ) _max_of_kth = max( _max_of_kth , resp[ k * N + n ] );
		double _sum = 0.0;
		for( int k = 0 ; k < K ; k++ ) _sum += exp( double(resp[k*N+n]) - _max_of_kth );
		dev_N_[n] = log(_sum) + _max_of_kth;
	}
}


template<typename type , typename n_type , unsigned int threadNum_N >
class vbgmm{

	public:
		size_t type_width;
		size_t size_means_ , size_covariances_ , size_precisions_ , size_precision_cholesky_ , size_weights_ , size_weight_concentration_ , size_mean_prior_ , size_degree_of_freedom_ , size_covariance_prior_ , size_mean_precision_ , size_gauss_ , size_X_ , size_lambda_ , size_r_ , size_log_det_chol ;

		//declare hst arrays
		type *means_ , *covariances_ , *precision_cholesky_ , *weights_ , *mean_prior_ , *degree_of_freedom_ ,  *mean_precision_ , *weight_concentration_prior_ , *mean_precision_prior_ , *degree_of_freedom_prior_  ;
		std::vector<type> weight_concentration_;
#ifdef INIT_MEAN_ON_HOST
		type *covariance_prior_;
#endif

		//declare dev arrays
		type	*dev_means_ , *dev_log_resp_ , *dev_X_ , *dev_log_weights_ , *dev_r_ , *dev_log_lambda_ , *dev_precisions_ , *dev_K_, *dev_XK_ , *dev_KD_ , *dev_KD_2_ ,  *dev_log_det_chol_  , *dev_degree_of_freedom_  , *dev_XD_ , *dev_mean_precision_  ,
					*dev_mean_prior_ ,  *dev_covariance_prior ,   *dev_mean_precision_prior;
		n_type *dev_N_;
		double *dev_1_;

		cudaStream_t st_X_ , st_lambda_ , st_weights_ , st_log_lambda_ , st_log_weights_ , st_r_  , st_precisions_ , st_log_det_chol_ , st_KD_ , st_means_ , st_degree_of_freedom_ , st_mean_precision_  , st_prior_;

		bool converged = false;
		int n_iter , K , D , fin_iter = -1;
		ITR_SIZE N;

		//CUDA's parameter
		dim3 block_N = dim3( threadNum_N , 1 , 1) , grid_N;
		unsigned int threadNum_K;
		dim3 block_K , grid_K;
		double lower_bound , log_likelihood;

		const type _a = 1.0 , _b = 0.0;

		vbgmm( const type *input_X , const ITR_SIZE init_N , const unsigned int init_K , const unsigned int init_D , const unsigned int iter ){

			START_TIME(CONST);

			//std::cout << __FUNCTION__ << std::endl;
			N = init_N;
			K = init_K;
			D = init_D;

			if( N % block_N.x == 0 )
				grid_N = dim3( N / block_N.x  , 1 , 1 );
			else
				grid_N = dim3( N / block_N.x + 1 , 1 , 1 );

			threadNum_K = min ( K , 1024 );
			block_K = dim3(threadNum_K , 1 , 1 );
			grid_K = dim3( K / block_K.x  , 1 , 1);

			//set size of sum arrays
			type_width = sizeof(type);
			size_means_ = type_width * K * D;
			size_covariances_ = type_width * K * D;
			size_precisions_ = type_width * K * D;
			size_precision_cholesky_ = type_width * K * D;
			size_weights_ = type_width * K;
			size_weight_concentration_ = type_width * K;
			size_mean_prior_ = type_width * D;
			size_degree_of_freedom_ = type_width * K;
			size_covariance_prior_ = type_width * D;
			size_mean_precision_ = type_width * K;
			size_X_ = type_width * N * D;
			size_lambda_ = type_width * K;
			size_weights_ = type_width * K;
			size_r_ = type_width * N * K;
			size_log_det_chol = type_width * K;

			//malloc hst vec
			END_TIME(CONST);
			START_TIME(HOST_MALLOC);

			cudaMallocHost( &means_ , size_means_ );
			cudaMallocHost( &covariances_ , size_covariances_ );
			cudaMallocHost( &precision_cholesky_ , size_precision_cholesky_ );
			cudaMallocHost( &weights_ , size_weights_ );
			cudaMallocHost( &mean_prior_ , size_mean_prior_ );
			cudaMallocHost( &degree_of_freedom_ , size_degree_of_freedom_);
			cudaMallocHost( &mean_precision_ , size_mean_precision_);
			cudaMallocHost( &weight_concentration_prior_ , type_width );
			cudaMallocHost( &mean_precision_prior_ , type_width );
			cudaMallocHost( &degree_of_freedom_prior_ , type_width);
#ifdef INIT_MEAN_ON_HOST
			cudaMallocHost( &covariance_prior_ , size_covariance_prior_);
#endif

			END_TIME(HOST_MALLOC);
			START_TIME(DEVICE_MALLOC);

			//malloc & init weight_concentration_ as 1/K
			weight_concentration_.assign( K , (float)1/K );

			//malloc dev vec
			cudaMalloc( &dev_X_ , size_X_ );
			cudaMalloc( &dev_log_lambda_ , size_lambda_ );
			cudaMalloc( &dev_means_ , size_means_ );
			cudaMalloc( &dev_log_weights_ , size_weights_ );
			cudaMalloc( &dev_r_ , size_r_ );
			cudaMalloc( &dev_precisions_ , size_precisions_ );
			cudaMalloc( &dev_K_ , type_width * K);
			cudaMalloc( &dev_XK_ , type_width * N * K );
			cudaMalloc( &dev_KD_ , type_width * K * D );
			cudaMalloc( &dev_KD_2_ , type_width * K * D );
			cudaMalloc( &dev_XD_ , type_width * N * D );
			cudaMalloc( &dev_log_det_chol_ , size_log_det_chol );
			cudaMalloc( &dev_degree_of_freedom_ , size_degree_of_freedom_ );
			cudaMalloc( &dev_mean_precision_ , size_mean_precision_ );
			cudaMalloc( &dev_mean_prior_ , size_mean_prior_ );  
			cudaMalloc( &dev_covariance_prior , size_covariance_prior_ );
			cudaMalloc( &dev_mean_precision_prior , type_width );
			cudaMalloc( &dev_1_ , sizeof(double) );

			cudaMalloc( &dev_N_ , sizeof(n_type)* N );

			cudaStreamCreate( &st_X_ );
			cudaStreamCreate( &st_lambda_);
			cudaStreamCreate( &st_weights_);
			cudaStreamCreate( &st_log_lambda_);
			cudaStreamCreate( &st_log_weights_);
			cudaStreamCreate( &st_r_ );
			cudaStreamCreate( &st_precisions_);
			cudaStreamCreate( &st_log_det_chol_);
			cudaStreamCreate( &st_KD_ );
			cudaStreamCreate( &st_means_ );
			cudaStreamCreate( &st_degree_of_freedom_ );
			cudaStreamCreate( &st_mean_precision_ );
			cudaStreamCreate( &st_prior_ );

			END_TIME(DEVICE_MALLOC);
			START_TIME2(CONST);

			//tranport X
			cudaMemcpyAsync( dev_X_ , input_X , size_X_ , cudaMemcpyHostToDevice , st_X_ );

			//dev_XD_をX ** 2 で初期化
			pow_substitution_elements <<< grid_N , block_N , 0 , st_X_ >>> ( dev_XD_ , dev_X_ , N , D );

			//init weight_concentration_ as 1/K
			*weight_concentration_prior_ = (type)1/K; //alpha_0

			//init mean_precision_ as 1.0 & tranfer
			for( int k = 0 ; k < K ; k++ ) mean_precision_[k] = 1.0;
			cudaMemcpyAsync( dev_mean_precision_ , mean_precision_ , size_mean_precision_ , cudaMemcpyHostToDevice , st_mean_precision_ );

			//init mean_precision_prior_ as 1.0
			*mean_precision_prior_ = 1.0; // beta_0
			cudaMemcpyAsync( dev_mean_precision_prior , mean_precision_prior_ , type_width , cudaMemcpyHostToDevice , st_prior_ );

			//init degree_of_freedom_prior_ as D
			*degree_of_freedom_prior_ = D; //Nu_0

#ifndef INIT_MEAN_ON_HOST
			cudaStreamSynchronize(st_X_);

			for( int d = 0 ; d < D ; d++ ){
				ITR_SIZE size = N;
				//cpy <type , n_type> <<< N / threadNum_N , threadNum_N >>> ( dev_N_ , dev_X_ , N , d);
				cpy_test <threadNum_N , type , n_type > <<< size / threadNum_N , threadNum_N >>> (  dev_X_ , dev_N_ ,N , d );
				size /= threadNum_N;
				cudaDeviceSynchronize();

				if(size == 1 ){
					cpy <<<  1 , 1 >>> ( &dev_mean_prior_[d] , dev_N_ , 1 , 0  );
					continue;
				}

				while( size > 1 ){
					dim3 block_S( threadNum_N , 1 , 1 );
					dim3 grid_S( size / block_S.x , 1 , 1 );
					if( size == threadNum_N ){
						substitute_sum_with_idx<threadNum_N , n_type , type ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_mean_prior_ , size , d);
					}
					else{
						get_sum_of_N<threadNum_N> <<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , size);
					}
					cudaDeviceSynchronize();
					size = size / threadNum_N ;
				}
			}
			divide_arr<type> <<< D , ( D / threadNum_N) + 1 , 0 , st_means_ >>> ( dev_mean_prior_ , D ,  N );
			cudaMemcpyAsync( mean_prior_ , dev_mean_prior_ , size_mean_prior_ , cudaMemcpyDeviceToHost , st_prior_ );

			for( int d = 0 ; d < D ; d++ ){
				ITR_SIZE size = N;
				calc_for_get_covariance_prior <type , n_type > <<< N / threadNum_N , threadNum_N , 0 , st_means_>>> ( dev_N_ , dev_X_ , dev_mean_prior_ , N , d);
				cudaDeviceSynchronize();
				while( size > 1 ){
					dim3 block_S( threadNum_N , 1 , 1 );
					dim3 grid_S( size / block_S.x , 1 , 1 );
					if( size == threadNum_N ){
						substitute_sum_with_idx<threadNum_N, n_type , type ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_covariance_prior , size , d);
					}
					else{
						get_sum_of_N<threadNum_N> <<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , size);
					}
					cudaDeviceSynchronize();
					size = size / threadNum_N ;
				}
			}
			divide_arr<type> <<< D , ( D / threadNum_N) + 1 >>> ( dev_covariance_prior , D ,  N );

#else
			//init mean_prior_ as X's mean
			for( int d = 0 ; d < D ; d++ ){
				double sum = 0.0;
				for( ITR_SIZE n = 0 ; n < N ; n++ ) sum += input_X[ d*N + n];
				mean_prior_[d] = (float)(sum/N);
			}
			cudaMemcpyAsync( dev_mean_prior_ , mean_prior_ , size_mean_prior_ , cudaMemcpyHostToDevice , st_prior_ );

			//init covariance_prior_ as X's cov
			for( int d = 0 ; d < D ; d++ ){
				double sum = 0.0;
				for( ITR_SIZE n = 0 ; n < N ; n++ ) sum += powf( input_X[d*N+n] - mean_prior_[d] , 2 );
				covariance_prior_[d] = (float)(sum/N);
			}
			cudaMemcpyAsync( dev_covariance_prior , covariance_prior_ , size_covariance_prior_ , cudaMemcpyHostToDevice , st_prior_ );
#endif

			//init degree_of_freedom_ as 1.0
			for( int k = 0 ; k < K ; k++ ) degree_of_freedom_[k] = 1.0; // Nu
			cudaMemcpyAsync( dev_degree_of_freedom_ , degree_of_freedom_ , size_degree_of_freedom_ , cudaMemcpyHostToDevice , st_degree_of_freedom_);

			//set number of iteration
			n_iter = iter;

			END_TIME2(CONST);

		}


		//free arrays that were malloced
		void free_memory_d(){
			START_TIME(FREE);
			_free_dev_memory();
			_destroy_stream();
			END_TIME(FREE);
		}

		void free_memory_h(){
			_free_hst_memory();
		}

		void print_param(){
			int cnt = 0;
			bool f[K];
			for( int k = 0 ; k < K ; k++ ) f[k] = 0;
			puts("weights_");
			for( int k = 0 ; k < K ; k++ ) {
				if( expf( weights_[k] ) > 1e-10 ){
					f[k] = 1;
					cnt++;
					printf("%.5f\t",expf(weights_[k]));
				}
			}
			puts("\n");
			puts("means_");
			for( int k = 0 ; k < K ; k++ ){
				if( f[k] ){
					for( int d = 0; d < D ; d++ ){
						printf("%.3f\t", means_[d*K+k]);
					}
					puts("");
				}
			}
			puts("");
			puts("covariances_");
			for( int k = 0 ; k < K ; k++ ){
				if( f[k] ){
					for( int d = 0; d < D ; d++ ){
						printf("%.3f\t", 1 / powf( precision_cholesky_[d*K+k] , 2  ) );
					}
					puts("");
				}
			}

			puts("");

			printf("iteration:%d converged:%d log_likelihood:%.5f lower_bound:%.5f ",fin_iter,converged,log_likelihood,lower_bound);
			printf("init_class:%d finally_class:%d ",K,cnt);
		}

		void copy_param(){
			cudaMemcpyAsync( means_ , dev_means_ , size_means_ , cudaMemcpyDeviceToHost , st_means_ );
			cudaMemcpyAsync( precision_cholesky_ , dev_precisions_ , size_precisions_ , cudaMemcpyDeviceToHost , st_precisions_ );
			cudaStreamSynchronize( st_means_ );
			cudaStreamSynchronize( st_precisions_ );
		}

		//fit parameter to X


		bool fit( type *X , type tol , type *init_mean ){
			//std::cout << __FUNCTION__ << std::endl;
			double old_lower_bound = 0;

			START_TIME(FIT);

			START_TIME(INIT);
			_initialize(X , init_mean );
			END_TIME(INIT);

			for( int iter = 0 ; iter < n_iter ; iter++ ){
#ifdef DEBUG
				printf("iter:%d start\n", iter + 1);
#endif

				START_TIME(E_STEP);
				_e_step();

				END_TIME(E_STEP);

				//PRINT_CUDA_MAT_SCOPE( "e_step ato log resp" , dev_r_ , N , K , 3 );

				START_TIME(SR);
				double sum_resp;
				_compute_sum_resp(&sum_resp);
				END_TIME(SR);

				START_TIME(M_STEP);
				_m_step();
				END_TIME(M_STEP);

				START_TIME(LB);
				lower_bound = _compute_lower_bound( &sum_resp );
				END_TIME(LB);

#ifdef DEBUG
				printf("iter:%d log_likelihood:%.5f lower_bound:%.5f\n", iter + 1  , log_likelihood , lower_bound );
#endif

#ifndef TAKE_LL_EVERY_ITER
				//printf("iter:%d lower_bound:%.5f\n", iter + 1  , lower_bound );
#else
				printf("iter:%d log_likelihood:%.5f lower_bound:%.5f\n", iter + 1  , log_likelihood , lower_bound );
#endif
				if( abs( lower_bound - old_lower_bound ) < tol  || std::isnan(lower_bound) ){
					fin_iter = iter + 1;
					if(std::isnan(lower_bound)){
						converged = false;
						log_likelihood = std::numeric_limits<double>::quiet_NaN();
						printf("FAILED:lower_bound is nan\n");
					}
					else{
						START_TIME(LL);
						converged = true;
						get_last_log_likelihood( );
						printf("SUCCES:BGMM is converged\n");
						END_TIME(LL);
					}
					break;
				}
				old_lower_bound = lower_bound;
			}

			if( fin_iter == -1 ){
				converged = false;
				log_likelihood = std::numeric_limits<double>::quiet_NaN();
				fin_iter = n_iter;
				get_last_log_likelihood( );
				printf("FAILED:itertion reach n_iter");
			}

			START_TIME(COPY);
			copy_param();
			END_TIME(COPY);

			//std::cout << "end " << __FUNCTION__ << std::endl;
			END_TIME(FIT);
			return converged;
		}

		void output_clustering_result( char *res_file_name){

			_e_step();

			type *r;
			r = (type *)malloc(sizeof(type) * size_r_ );
			cudaMemcpy( r , dev_r_ , size_r_ , cudaMemcpyDeviceToHost );

			for( int n = 0 ; n < N ; n++ ){
				for( int k = 0 ; k < K ; k++ ){
					r[ k * N + n ] = expf( r[ k * N + n ] );
					//printf("%.1f " , r[ k * N + n ] );
					
				}
				//puts("");
			}

			FILE *res_file;
			if( ( res_file = fopen( res_file_name , "w" ) ) == NULL ){
				return;
			}

			int *res;
			res = (int *)malloc(sizeof(int) * N );

			for( ITR_SIZE n = 0 ; n < N ; n++ ){
				type M = 0;
				for( int k = 0 ; k < K ; k++ ){
					if( M < r[ k * N + n ] ){
						M = r[ k * N + n ];
						res[n] = k;
					}
				}
			}

			for( ITR_SIZE n = 0 ; n < N ; n++ ) fprintf( res_file , "%d," , res[n] );
			fprintf( res_file , "\n" );

			free(r);
			free(res);
		}

	private:

		void _compute_sum_resp( double *sum_resp ){
			substitute_zero <double> <<< 1 , 1 >>> (dev_1_);

			for( int k = 0 ; k < K ; k++ ){
				ITR_SIZE size = N;

				cpy_and_process_for_calc_sum_resp<type , threadNum_N > <<< grid_N , block_N >>> ( dev_r_ , dev_N_ , N , k);
				cudaDeviceSynchronize();
				while( size > 1 ){
					dim3 s_block( threadNum_N , 1 , 1 );
					dim3 s_grid( size / s_block.x , 1 , 1 );
					if( size == threadNum_N )
						add_sum_resp< threadNum_N , n_type> <<< s_grid.x , s_block  >>> ( dev_N_, dev_1_, size );
					else
						get_sum_of_N< threadNum_N > <<< s_grid.x , s_block  >>> ( dev_N_ , dev_N_ , size );
					cudaDeviceSynchronize();
					size = size / threadNum_N ;
				}
			}
			cudaMemcpyAsync( sum_resp , dev_1_ , sizeof(double) , cudaMemcpyDeviceToHost  );
		}

		void _initialize( type *input_X , type *init_mean ){

#ifdef DEBUG
			printf("initalize start\n");
#endif


			set_value_matrix <type> <<< grid_N , block_N >>> ( dev_r_ , N , K , 1.0 / K );
			cudaDeviceSynchronize();

			type xk[K*D];

#if USE_INIT_MEAN_FILE
			for( int k = 0 ; k < K ; k++ ) for( int d = 0 ; d < D ; d++ ) xk[d*K+k] = init_mean[d*K+k];
#else
			std::random_device rnd;
			std::mt19937 mt(rnd());
			for( int k = 0 ; k < K ; k++ ){
				//ITR_SIZE idx = N / K * k;
				ITR_SIZE idx = mt() % N;
				for( int d = 0 ; d < D ; d++ ){
					xk[d * K + k ] = input_X[ d * N + idx];
				}
			}
#endif


			cudaMemcpyAsync( dev_means_ , &xk , size_means_ , cudaMemcpyHostToDevice , st_means_ );

			//PRINT_CUDA_MAT("init mean" , dev_means_ , K , D );
			// update alpha , beta , nu
			type nk[K];
			for( int k = 0 ; k < K ; k++ ){
				nk[k] = N/K;
				//estimate weights
				weight_concentration_[k] = *weight_concentration_prior_ + nk[k];
				mean_precision_[k] = *mean_precision_prior_ + nk[k];
				degree_of_freedom_[k] = *degree_of_freedom_prior_ + nk[k];
			}
			cudaMemcpyAsync( dev_K_ , &nk , sizeof(type) * K , cudaMemcpyHostToDevice , st_KD_ );

			//debug
			//PRINT_HOST_VEC("nk" , nk , K ); 

			/*
				 PRINT_HOST_VEC("weight_concentration_" , weight_concentration_, K ); 
				 PRINT_HOST_VEC("mean_precision_" , mean_precision_ , K ); 
				 PRINT_HOST_VEC("degree_of_freedom_" , degree_of_freedom_ , K ); 
			 */

			cudaMemcpyAsync( dev_mean_precision_ , mean_precision_ , size_mean_precision_ , cudaMemcpyHostToDevice , st_mean_precision_ );
			cudaMemcpyAsync( dev_degree_of_freedom_ , degree_of_freedom_ , size_degree_of_freedom_ , cudaMemcpyHostToDevice , st_degree_of_freedom_ );


			//compute S
			cudaStreamSynchronize(st_means_);
			pow_substitution_elements<type> <<< grid_K , block_K , 0 , st_means_ >>> ( dev_KD_2_ , dev_means_ , K , D  );



			cublasHandle_t handle; //bottle neck
			cublasCreate(&handle);

			START_TIME(F2);
			cudaDeviceSynchronize();
			END_TIME(F2);

			cublasSgemm(
					handle, CUBLAS_OP_T, CUBLAS_OP_N,
					K,D,N,
					&_a,
					dev_r_ ,N,
					dev_XD_ , N,
					&_b,
					dev_KD_ , K); //float

			cublasSgemm(
					handle, CUBLAS_OP_T, CUBLAS_OP_N,
					K,D,N,
					&_a,
					dev_r_ ,N,
					dev_X_ , N,
					&_b,
					dev_precisions_ , K); //float

			// estimate means
			for( int k = 0 ; k < K ; k++ ){
				for( int d = 0 ; d < D ; d++ ){
					means_[d*K+k]= ( ( *mean_precision_prior_ * mean_prior_[d]) + (nk[k] * xk[d*K+k] ) ) / mean_precision_[k];
				}
			}

			cudaStreamSynchronize(st_means_);
			cudaDeviceSynchronize();


			hadamard_product_self<type> <<< grid_K , block_K , 0 , st_KD_ >>> ( dev_precisions_ , dev_means_ , K , D );

			//debug
			/*
				 PRINT_CUDA_MAT("avg_X2" , dev_KD_ , K , D );
				 PRINT_CUDA_MAT("avg_means_2" , dev_KD_2_ , K , D );
				 PRINT_CUDA_MAT("avg_X_means" , dev_precisions_ , K , D );
			 */

			//dev_KD <- S = covariance
			estimate_gaussian_covariance_diag <<< grid_K , block_K , 0 , st_KD_ >>> ( dev_KD_ , dev_KD_2_ , dev_precisions_, dev_K_ , 1e-6, K , D );



			PRINT_CUDA_MAT("init nk", dev_K_ , 1 , K );
			PRINT_CUDA_MAT("init xk", dev_means_ , K , D );
			PRINT_CUDA_MAT("init sk", dev_KD_ , K , D );

			/*
				 PRINT_CUDA_MAT("degree_of_freedom_" , dev_degree_of_freedom_ , 1 , K );
				 PRINT_CUDA_MAT("mean_prior_" , dev_mean_prior_ , 1 , D );
				 PRINT_CUDA_MAT("mean_precision_" , dev_mean_precision_ , 1 , K );
				 PRINT_CUDA_MAT("covariance_prior_" , dev_covariance_prior, 1 , D );
			 */

			//estimate precisions
			//dev_precisions_ <- updated precisions_cholesky
			estimate_precisions <<< grid_K , block_K , 0 , st_KD_>>> ( dev_K_ , dev_means_ , dev_KD_ , dev_mean_prior_ , dev_precisions_ , dev_degree_of_freedom_ , dev_covariance_prior , dev_mean_precision_ ,  dev_mean_precision_prior , K , D ) ;

			//must execute after estimate precisions becase estimate pre need dev_means_ val = xk
			cudaMemcpyAsync( dev_means_ , means_ , size_means_ , cudaMemcpyHostToDevice , st_KD_ );

			/*
				 PRINT_CUDA_MAT("init means_" , dev_means_, K , D );
				 PRINT_CUDA_MAT("init precisions_cholesky" , dev_precisions_ , K , D );
			 */

			cudaDeviceSynchronize();

			//for estimate_log_gaussian_prob
			compute_log_det_chol <<< grid_K , block_K , 0 , st_log_det_chol_ >>> ( dev_log_det_chol_ , dev_precisions_ , dev_degree_of_freedom_ ,  K , D );
#ifdef DEBUG
			printf("initalize end\n");
#endif

		}

		void get_last_log_likelihood( ){
			_estimate_weighted_log_prob_();
			cudaDeviceSynchronize();
			get_log_likelihood();
		}

		void get_log_likelihood( ){
			get_sum_of_K <type, n_type > <<< grid_N , block_N >>> ( dev_r_ , dev_N_, N , K );
			cudaDeviceSynchronize();

			//PRINT_CUDA_MAT_SCOPE("sum_of_K" , dev_N_ , N , 1 , 3 , 3  );

			ITR_SIZE size = N;
			dim3 block_S( threadNum_N , 1 , 1 );
			dim3 grid_S( size / block_S.x , 1 , 1 );
			while( size > 1 ){
				get_sum_of_N<threadNum_N> <<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , N );
				cudaDeviceSynchronize();
				size = size / threadNum_N  ;
			}
			n_type tmp;
			cudaMemcpy( &tmp, dev_N_ , sizeof(n_type) , cudaMemcpyDeviceToHost );
			cudaDeviceSynchronize();
			//printf("sum resp:%lf\n" , tmp );
			log_likelihood = tmp / N ;
			return  ;
		}

		void _free_hst_memory(){
			cudaFree(means_);
			cudaFree(covariances_);
			cudaFree(precision_cholesky_);
			cudaFree(weights_);
			cudaFree(mean_prior_);
			cudaFree(degree_of_freedom_);
			cudaFree(mean_precision_);
#ifdef INIT_MEAN_ON_HOST
			cudaFree(covariance_prior_);
#endif
		}

		void _free_dev_memory(){
			cudaFree(dev_X_);
			cudaFree( dev_log_weights_ );
			cudaFree( dev_r_ );
			cudaFree( dev_precisions_ );
			cudaFree( dev_K_ );
			cudaFree( dev_XK_ );
			cudaFree( dev_KD_  );
			cudaFree( dev_KD_2_  );
			cudaFree( dev_XD_ );
			cudaFree( dev_log_det_chol_ );
			cudaFree( dev_degree_of_freedom_ );
			cudaFree( dev_mean_precision_ );
			cudaFree( dev_N_ );
			cudaFree( dev_mean_prior_ );  
			cudaFree( dev_covariance_prior );
			cudaFree( dev_mean_precision_prior );
			cudaFree( dev_1_ );
		}

		void _destroy_stream(){
			cudaStreamDestroy( st_X_ );
			cudaStreamDestroy( st_X_ );
			cudaStreamDestroy( st_lambda_);
			cudaStreamDestroy( st_weights_);
			cudaStreamDestroy( st_log_lambda_);
			cudaStreamDestroy( st_log_weights_);
			cudaStreamDestroy( st_r_ );
			cudaStreamDestroy( st_precisions_);
			cudaStreamDestroy( st_log_det_chol_);
			cudaStreamDestroy( st_KD_ );
			cudaStreamDestroy( st_means_ );
			cudaStreamDestroy( st_degree_of_freedom_ );
			cudaStreamDestroy( st_mean_precision_ );
			cudaStreamDestroy( st_prior_ );
		}

		double _compute_lower_bound( double *sum_resp ){
			START_TIME(LOG_DET_CHOL);
			float _log_det_chol[K];
			compute_log_det_chol <<< grid_K , block_K , 0 , st_log_det_chol_ >>> ( dev_log_det_chol_ , dev_precisions_ , dev_degree_of_freedom_ ,  K , D );
			cudaMemcpy( &_log_det_chol , dev_log_det_chol_ , size_log_det_chol , cudaMemcpyDeviceToHost );
			cudaDeviceSynchronize();
			END_TIME(LOG_DET_CHOL);

			//PRINT_HOST_VEC("log_det_chol" , _log_det_chol , K );
			START_TIME(LOW);

			for( int k = 0 ; k < K ; k++ ){
				_log_det_chol[k] = (double)_log_det_chol[k] - 0.5 * D * logf( degree_of_freedom_[k]);
			}
			type _sum_log_wishart = _compute_sum_log_wishart_norm( _log_det_chol );

			type _log_weight_norm = _log_dirichket_norm();

			type _sum_log_mean_precision = 0;

			for( int k = 0 ; k < K ; k++ ) _sum_log_mean_precision += logf(mean_precision_[k]);

			cudaDeviceSynchronize();

			END_TIME(LOW);

#ifdef DEBUG
			printf("%.5f %.5f %.5f %.5f\n", *sum_resp , _sum_log_wishart , _log_weight_norm , 0.5 * D * _sum_log_mean_precision );
#endif

			return - *sum_resp - _sum_log_wishart - _log_weight_norm - 0.5 * D * _sum_log_mean_precision;
		}

		type _compute_sum_log_wishart_norm( type *log_det_precision_chol ){
			std::vector<double> log_wishart(K,0);
			for( unsigned int k = 0 ; k < K ; k++ ){
				double sum_gammaln_nu = 0.0;
				for( unsigned int d = 0 ; d < D ; d++ ) sum_gammaln_nu += lgamma( 0.5 * ( degree_of_freedom_[k] - d + 1 ) );
				log_wishart[k] =	- ( (double)degree_of_freedom_[k] * log_det_precision_chol[k] 
						+ (double)degree_of_freedom_[k] * D * 0.5 * logf(2.0)
						+ sum_gammaln_nu
						);
			}
			return (float)std::accumulate( log_wishart.begin() , log_wishart.end() , 0.0);
		}

		type _log_dirichket_norm(){
			double tmp = 0 , sum = 0;
			for( int k = 0; k < K ; k++ ){
				tmp += lgamma( weight_concentration_[k] );
				sum += weight_concentration_[k];
			}
			return (float) ( lgamma( sum ) - tmp );
		}

		/*** E step & its Kernels ***/
		void _e_step(){
			_estimate_weighted_log_prob_();


			START_TIME(LR);
			compute_log_resp <threadNum_N><<< grid_N , block_N >>> ( dev_K_ , dev_r_ , N , K ) ;
			cudaDeviceSynchronize();
			END_TIME(LR);

			return;
		}

		void _estimate_weighted_log_prob_(){
			_estimate_log_gaussian_prob_();

			START_TIME(WEIGHT);
#ifndef DEBUG
			type tmp = boost::math::digamma( std::accumulate(  weight_concentration_.begin() , weight_concentration_.end() , 0.0 ) );
#else
			type tmp = std::accumulate(  weight_concentration_.begin() , weight_concentration_.end() , 0.0 ) ;
			if( tmp == 0 || tmp <= -1) tmp = 1;
			tmp = boost::math::digamma( tmp );
#endif
			for(unsigned int k = 0 ; k < K ; k++ ){
#ifdef DEBUG
				if( this->weight_concentration_[k] != 0 || this->weight_concentration_[k] <= -1 )
					this->weights_[k] = boost::math::digamma(this->weight_concentration_[k]) - tmp;
				else
					this->weights_[k] = boost::math::digamma( 1 ) - tmp;
#endif
				this->weights_[k] = boost::math::digamma(this->weight_concentration_[k]) - tmp;
			}

			cudaMemcpyAsync( dev_log_weights_ , this->weights_ , size_weights_ , cudaMemcpyHostToDevice  , st_log_weights_ );


			END_TIME(WEIGHT);

			START_TIME(LAMBDA);

			estimate_log_lambda_<type> <<< grid_K , block_K , sizeof(type) * K , st_log_lambda_ >>> ( dev_log_lambda_ , dev_degree_of_freedom_ ,  dev_mean_precision_, K , D );

#ifdef DEBUG
			PRINT_CUDA_MAT("log_weight" , dev_log_weights_ , 1 , K );
			PRINT_CUDA_MAT("log_lambda" , dev_log_lambda_ , 1 , K );
			PRINT_CUDA_MAT_SCOPE("log_gauss" , dev_r_ , N , K , 3 , 3 );
#endif


			cudaDeviceSynchronize();
			END_TIME(LAMBDA);

			START_TIME(WLP);
			compute_weighted_log_prob  <<< grid_N , block_N , sizeof(double) * K >>> ( dev_r_ , dev_log_lambda_ , dev_log_weights_ , N , K );
			cudaDeviceSynchronize();
			END_TIME(WLP);

#ifdef TAKE_LL_EVERY_ITER
			get_log_likelihood();
#endif

			/*
				 PRINT_CUDA_MAT("log_lambda" , dev_log_lambda_ , 1 , K );
				 PRINT_HOST_VEC ("weights_" , weights_, K );
				 PRINT_CUDA_MAT_SCOPE("weited_log_prob", dev_r_ , N , K , 5 , 5 );
			 */

			return;
		}

		void _estimate_log_gaussian_prob_( ){

			START_TIME(GAUSS);
			cudaStreamSynchronize( st_precisions_ );
			cudaStreamSynchronize( st_KD_) ;

			//get precisions
			pow_self_elements<type><<< grid_K , block_K , 0 , st_precisions_ >>> ( dev_precisions_ , K , D );

			/*
				 PRINT_CUDA_MAT("log_det_chol" , dev_log_det_chol_ , 1 , K );
				 PRINT_CUDA_MAT("means" , dev_means_ , K , D );
				 PRINT_CUDA_MAT("precisions" , dev_precisions_ , K , D );
			 */


			//get devK its kth componnent is sum k={0~K} means_(d,k) ** 2 * precisions_(d,k)
			pow_substitution_elements<type> <<< grid_K , block_K ,  0 , st_KD_ >>> ( dev_KD_ , dev_means_ , K , D );
			hadamard_product_self<type> <<< grid_K , block_K , 0 , st_KD_ >>> ( dev_KD_ , dev_precisions_ , K , D );
			get_sum_of_2dim <<< grid_K , block_K , 0 , st_KD_ >>> ( dev_K_ , dev_KD_ , K , D );
			cudaDeviceSynchronize();

			//dev_r_ <- X ** 2  * precision^T

			//dev_XD_はコンストラクタ内で計算済み
			cublasHandle_t handle;
			cublasCreate(&handle);

			cudaDeviceSynchronize();

			//m3
			cublasSgemm(
					handle, CUBLAS_OP_N , CUBLAS_OP_T ,
					N , K , D ,
					&_a ,
					dev_XD_, N ,
					dev_precisions_ , K ,
					&_b,
					dev_r_ , N
					);
			cudaDeviceSynchronize();

			/*
				 PRINT_CUDA_MAT_SCOPE("dev_XD" , dev_XD_ , N , D  , 3);
				 PRINT_CUDA_MAT("dev_precisions_" , dev_precisions_ , K , D );
			 */


			//dev_KD_ <- means◦ precisions
			hadamard_product_subtitution<type> <<< grid_K , block_K >>> ( dev_KD_ , dev_precisions_ , dev_means_ , K , D );

			//PRINT_CUDA_MAT("mean had pre " , dev_KD_ , K , D );


			//devXK <- devX * means◦ precisions
			cudaDeviceSynchronize();

			//m2?
			cublasSgemm(
					handle, CUBLAS_OP_N , CUBLAS_OP_T ,
					N , K , D,
					&_a ,
					dev_X_ , N,
					dev_KD_ , K,
					&_b,
					dev_XK_ , N
					);
			cudaDeviceSynchronize();

			/*
				 PRINT_CUDA_MAT("log_det_chol" , dev_log_det_chol_ , 1 , K );
				 PRINT_CUDA_MAT("m1_2" , dev_K_ , 1 , K );
				 PRINT_CUDA_MAT_SCOPE("m2" , dev_XK_ , N , K , 3  );
				 PRINT_CUDA_MAT_SCOPE("m3" , dev_r_ , N , K , 3  );
			 */

			compute_log_gaussian_prob<type> <<< grid_N , block_N , type_width * (K) , st_degree_of_freedom_ >>> ( dev_r_ , dev_XK_ , dev_K_ ,   dev_log_det_chol_ , dev_degree_of_freedom_ , N , K , D);

			//PRINT_CUDA_MAT_SCOPE("e_step resp" , dev_r_ , N , K , 3 );


#if PROFILE
			cudaDeviceSynchronize();
			END_TIME(GAUSS);
#endif
			return;
		}

		void _m_step(){
			//PRINT_CUDA_MAT_SCOPE("resp" , dev_r_ , N , K , 3 );

			START_TIME(NK);

			SUM(N,K,dev_r_,dev_K_,cpy_sum,fold_sum);

			//PRINT_CUDA_MAT("nk" , dev_K_ , 1 , K  );

			//compute_Nk
			for( ITR_SIZE k = 0 ; k < K ; k++ ){
				ITR_SIZE size = N;
				cpy_test <threadNum_N , type , n_type > <<< size / threadNum_N , threadNum_N >>> (  dev_r_ , dev_N_ ,N , k );
				size /= threadNum_N;
				cudaDeviceSynchronize();

				// if N = Thread Num N
				if(size == 1 ){
					cpy <<<  1 , 1 >>> ( &dev_K_[k] , dev_N_ , 1 , 0  );
					continue;
				}

				while( size > 1 ){
					dim3 block_S( threadNum_N , 1 , 1 );
					dim3 grid_S( size / block_S.x , 1 , 1 );
					if( size == threadNum_N ){
						substitute_sum_with_idx< threadNum_N , n_type , type ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_K_ , size , k);
					}
					else{
						substitute_sum_with_idx< threadNum_N , n_type , n_type ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , size, 0);
					}
					cudaDeviceSynchronize();
					size = size / threadNum_N ;
				}
			}
#ifdef PROFILE
			cudaDeviceSynchronize();
			END_TIME(NK);
#endif


			//PRINT_CUDA_MAT("nk" , dev_K_ , 1 , K  );

			type nk[K];
			cudaMemcpyAsync( &nk , dev_K_ , type_width * K , cudaMemcpyDeviceToHost , st_X_  );

			START_TIME(MAKE_HANDLE);
			cublasHandle_t handle;
			cublasCreate(&handle);
			END_TIME(MAKE_HANDLE);

			START_TIME(XK);
			// compute Xbar
			// dev_means_ <- XBar = r^T * X
			cudaDeviceSynchronize();
			cublasSgemm(
					handle, CUBLAS_OP_T, CUBLAS_OP_N,
					K,D,N,
					&_a,
					dev_r_ ,N,
					dev_X_ , N,
					&_b,
					dev_means_ , K); //float

#ifdef PROFILE
			cudaDeviceSynchronize();
			END_TIME(XK);
#endif

			START_TIME(PRI);
			cudaStreamSynchronize(st_X_);
			// update alpha , beta , nu

			/*
				 float nk_sum = 0;
				 printf("nk\n");
				 for(int k = 0 ; k < K ; k++ ){
				 nk_sum += nk[k];
				 printf("%.5f " , nk[k]);
				 }
				 printf("nk_sum:%.5f\n",nk_sum);
			 */

			for( int k = 0 ; k < K ; k++ ){
				weight_concentration_[k] = *weight_concentration_prior_ + nk[k];
				mean_precision_[k] = *mean_precision_prior_ + nk[k];
				degree_of_freedom_[k] = *degree_of_freedom_prior_ + nk[k];
			}
			cudaMemcpyAsync( dev_mean_precision_ , mean_precision_ , size_mean_precision_ , cudaMemcpyHostToDevice , st_mean_precision_ );
			cudaMemcpyAsync( dev_degree_of_freedom_ , degree_of_freedom_ , size_degree_of_freedom_ , cudaMemcpyHostToDevice , st_degree_of_freedom_ );
			cudaDeviceSynchronize();

			END_TIME(PRI);
			START_TIME2(XK);

			divide_and_check_XBar <<< grid_K , block_K , sizeof(float) * K , st_means_ >>> ( dev_means_ , dev_K_ , dev_mean_prior_ , K , D );
			type xk[D*K];
			cudaMemcpyAsync( &xk , dev_means_ , size_means_ , cudaMemcpyDeviceToHost , st_means_ );

#ifdef PROFILE
			cudaDeviceSynchronize();
			END_TIME2(XK);
#endif

			START_TIME(SK);
			//compute S
			pow_substitution_elements<type> <<< grid_K , block_K , 0 , st_means_ >>> (dev_KD_2_ , dev_means_ , K , D  );

			cublasSgemm(
					handle, CUBLAS_OP_T, CUBLAS_OP_N,
					K,D,N,
					&_a,
					dev_r_ ,N,
					dev_XD_ , N,
					&_b,
					dev_KD_ , K); //float

			cublasSgemm(
					handle, CUBLAS_OP_T, CUBLAS_OP_N,
					K,D,N,
					&_a,
					dev_r_ ,N,
					dev_X_ , N,
					&_b,
					dev_precisions_ , K); //float

#if PROFILE
			cudaDeviceSynchronize();
			END_TIME(SK);
#endif

			START_TIME(MEAN);
			cudaStreamSynchronize(st_means_);
			// compute means on hst
			for( int k = 0 ; k < K ; k++ ){
				if( nk[k] > 0 ){
					for( int d = 0 ; d < D ; d++ ){
						means_[d*K+k]= ( ( *mean_precision_prior_ * mean_prior_[d]) + (nk[k] * xk[d*K+k] ) ) / mean_precision_[k];
#ifdef DEBUG
						CHECK_ERROR("calc mean on hst" , means_[d*K+k]);
#endif
					}

				}else{
					for( int d = 0 ; d < D ; d++ )
						means_[d*K+k] = mean_prior_[k];
				}

			}

			END_TIME(MEAN);

			START_TIME2(SK);
			cudaDeviceSynchronize();
			hadamard_product_self<type> <<< grid_K , block_K , 0 , st_KD_ >>> ( dev_precisions_ , dev_means_ , K , D );

			//dev_KD <- S = covariance
			estimate_gaussian_covariance_diag <<< grid_K , block_K , 0 , st_KD_ >>> ( dev_KD_ , dev_KD_2_ , dev_precisions_, dev_K_ , 1e-6, K , D );

#if PROFILE
			cudaDeviceSynchronize();
			END_TIME2(SK);
#endif

			PRINT_CUDA_MAT("nk", dev_K_ , 1 , K );
			PRINT_CUDA_MAT("xk", dev_means_ , K , D );
			PRINT_CUDA_MAT("sk", dev_KD_ , K , D );

			/*
				 PRINT_HOST_VEC("weight_concentration_",weight_concentration_,K);
				 PRINT_HOST_VEC("mean_precision_",mean_precision_,K);
				 PRINT_CUDA_MAT("m_step mean_precision_" , dev_mean_precision_ , 1 , K );
				 PRINT_CUDA_MAT("m_step degree_of_freedom_" , dev_degree_of_freedom_ , 1 , K );
			 */


			START_TIME(PRECISION);
			//dev_precisions_ <- precision_cholesky
			estimate_precisions <<< grid_K , block_K , 0 , st_KD_>>> ( dev_K_ , dev_means_ , dev_KD_ , dev_mean_prior_ , dev_precisions_ , dev_degree_of_freedom_ , dev_covariance_prior , dev_mean_precision_ ,  dev_mean_precision_prior , K , D ) ;

			PRINT_CUDA_MAT("precision_cholesky" , dev_precisions_ , K , D );

#if PROFILE
			cudaDeviceSynchronize();
			END_TIME(PRECISION);
#endif

			START_TIME(PTOE);

			//must do after beestimate_precisions
			cudaMemcpyAsync( dev_means_ , means_ , size_means_ , cudaMemcpyHostToDevice , st_KD_ );
			cudaDeviceSynchronize();


			END_TIME(PTOE);
		}
};

int main(int argc , char *argv[]){
	ITR_SIZE N = 0, D = 0;
	TYPE *data , *tmp_init_mean;
	char* fileName;

#if USE_INIT_MEAN_FILE
	if( argc !=4  ) {
		printf("usage: %s <file> [cluster Num] [init mean file] \n", *argv);
		return 0;
	}
#else
	if( argc !=4    ){
		printf("usage: %s <file> [cluster Num] [clustering result filename] \n", *argv);
		return 0;
	}
#endif

	fileName = argv[1];
	ITR_SIZE K = atoi(argv[2]);

	char* clustering_result_file_name = argv[3];

	std::chrono::system_clock::time_point start_c, end_c;
	double time;

	data = loadMatrix<ITR_SIZE>( &D , &N , fileName );

	if( data == NULL ){
		printf("failed input data\n");
		return 1;
	}

	float *X;
	cudaMallocHost( &X , sizeof(float) * N * D );

	for( int d = 0 ; d < D ; d++ ){
		for( int n = 0 ; n < N ; n++ ){
			X[ d * N + n ] = data[ n * D + d];
		}
	}

#if USE_INIT_MEAN_FILE
	char* init_mean_fileName = argv[3];
	tmp_init_mean = loadMatrix<ITR_SIZE>( &D , &K , init_mean_fileName);
	float* init_mean = (float *)malloc(sizeof(float) * K * D);
	for( int d = 0 ; d < D ; d++ ){
		for( int k = 0 ; k < K ; k++ ){
			init_mean[d*K+k] = tmp_init_mean[k*D+d];
		}
	}
#else
	float *init_mean;
#endif

#ifdef PROF
	cudaProfilerStart();
#endif

	//take time
	/*
		 cudaEvent_t start, stop;
		 cudaEventCreate(&start);
		 cudaEventCreate(&stop);
		 cudaEventRecord(start);
	 */

	start_c = std::chrono::system_clock::now();

	class vbgmm< float , double , THREAD_N > vb( X , N , K , D , 10000 );
	vb.fit(X , 1e-7 , init_mean );

	end_c = std::chrono::system_clock::now();
	time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_c - start_c).count() / 1000.0);

	vb.output_clustering_result( clustering_result_file_name );

	start_c = std::chrono::system_clock::now();
	vb.free_memory_d();
	end_c = std::chrono::system_clock::now();

	time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_c - start_c).count() / 1000.0);

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);

	vb.print_param( );
	printf("time %lf[ms]\n", time);

	/*
		 float milliseconds = 0;
		 printf("cudaEvent time:%.5f\n", milliseconds );
		 cudaEventElapsedTime(&milliseconds, start, stop);
	 */

	vb.free_memory_h();

#ifdef PROF
	cudaProfilerStop();
#endif
	free(data);

	PRINT_TIMES();

	return 0;
}
