#ifndef CU_SUM_H 
#define CU_SUM_H 

template<const unsigned int threadNum , typename in_type , typename out_type >
__global__ void cpy_sum( in_type *g_idata, out_type *g_odata, const unsigned int N , const unsigned int k , unsigned int ofs , unsigned int row_size ){
	__shared__ out_type sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	out_type mysum = ( i < N ) ? g_idata[ k * row_size + i + ofs ] : 0;
	if ( i +  blockDim.x < N) mysum += g_idata[ ( k * row_size + i + ofs ) +  blockDim.x  ]  ;
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

template<const unsigned int threadNum , typename in_type , typename out_type >
__global__ void cpy_sum_exp( in_type *g_idata, out_type *g_odata, const unsigned int N , const unsigned int k , unsigned int ofs , unsigned int row_size ){
	__shared__ out_type sdata[threadNum];
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	out_type mysum = 0;
	if( i < N ){
		in_type tmp = exp(g_idata[ k * row_size + i + ofs ] );
		mysum += tmp * g_idata[ k * row_size + i + ofs ];
		g_idata[ k * row_size + i + ofs ] = tmp;

		if( i +  blockDim.x < N ){

			tmp = exp ( g_idata[ ( k * row_size + i + ofs ) +  blockDim.x  ] );
			mysum += tmp * g_idata[ ( k * row_size + i + ofs ) +  blockDim.x  ];
			g_idata[ ( k * row_size + i + ofs ) +  blockDim.x  ] = tmp;
		}
	}

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

template<const unsigned int threadNum , typename in_type , typename out_type>
__global__ void fold_sum( in_type *g_idata , out_type *g_odata , const unsigned int N , const unsigned int k){
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

	if (tid == 0){
		//printf("mysum:%.1lf\n" , mysum);
		g_odata[blockIdx.x + k] = mysum;
	}
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}

__global__ void add_sum( float *to , double *from , unsigned int col ){
	to[col] += from[0];
}

/*
template<typename in_type , typename out_type >
__global__ void add_sum( out_type *to , in_type *from, const unsigned int N , const unsigned int k){
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	to[idx] += from[k * N + idx];
}
*/


__global__ void get_sum( float *dev_K_ , float *dev_r_, const unsigned int row_size, const unsigned int k , const unsigned int offset ){
	float sum = 0;
	for( unsigned int idx = offset ; idx < row_size ; idx++ ){
		//printf("idx:%d  dev_r: %.3f \n",k * row_size + idx , dev_r_[ k * row_size + idx] );
		sum += dev_r_[ k * row_size + idx ];
	}
	dev_K_[k] += sum;
}

__global__ void get_sum_exp( float *dev_K_ , float *dev_r_, const unsigned int row_size , const unsigned int k , const unsigned int offset ){
	float sum = 0;
	for( unsigned int idx = offset ; idx < row_size ; idx++ ){

		float tmp = exp ( dev_r_[ k * row_size + idx ] );
		sum += ( dev_r_[ k * row_size + idx ] * tmp );

		dev_r_[ k * row_size + idx ] = tmp;
	}

	dev_K_[k] += sum;
}


__global__ void initialize_tomat( float *to_mat, const unsigned int col_size  ){

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx < col_size )
		to_mat[idx] = 0.0;

}

//printf("size : %d offset : %d tmp_size : %d threadNum : %d\n" , size , offset , tmp_size , threadNum );
#define first_sum( a , b , from_mat , col ,  cpy_func , row_size ) \
	cpy_func< a , float , double > <<< b / a , a >>> ( from_mat , dev_N_ , b , col , offset , row_size ); \
	offset += b; \
	size -= b ; \
	tmp_size = b/a; \
	threadNum = a ; \

#define SUM(row_size,col_size, from_mat , to_mat , cpy_func , sum_func ) \
	initialize_tomat <<< 1 , col_size >>> ( to_mat , col_size ); \
	for( int col = 0 ; col < col_size ; col++ ){ \
		unsigned int size = row_size; \
		unsigned int offset = 0; \
		bool f = 0; \
		while( size > 0 ){ \
			if(f) continue; \
			unsigned int tmp_size; \
			unsigned int threadNum; \
			if( size >= 16777216 ){ \
				first_sum(256,16777216,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 2097152 ){ \
				first_sum(128,2097152,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 1048576 ){ \
				first_sum(1024,1048576,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 262144 ){ \
				first_sum(512,262144,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 65536 ){ \
				first_sum(256,65536,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 16384){ \
				first_sum(128,16384,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 4096 ){ \
				first_sum(64 , 4096 ,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 1024 ){ \
				first_sum(1024,1024 ,from_mat,col,cpy_func, row_size); \
			} \
			else{ \
				get_sum <<< 1 , 1 >>> ( to_mat , from_mat , row_size , col , offset ); \
				cudaDeviceSynchronize(); \
				break; \
			} \
			cudaDeviceSynchronize(); \
			while( tmp_size > 1 ){ \
				dim3 block_S(  threadNum , 1 , 1 ); \
				dim3 grid_S( tmp_size / block_S.x , 1 , 1 ); \
				switch(threadNum){ \
					case 1024: \
						sum_func< 1024 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 512: \
						sum_func< 512 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 256: \
						sum_func< 256 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 128: \
						sum_func< 128, double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 64: \
						sum_func< 64 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 32: \
						sum_func< 32 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					default: \
						break; \
				} \
				cudaDeviceSynchronize(); \
				tmp_size = tmp_size / threadNum; \
			} \
			add_sum <<<  1 , 1 >>> ( to_mat , dev_N_ , col ); \
		} \
	}

#define SUM_exp(row_size,col_size, from_mat , to_mat , cpy_func , sum_func ) \
	initialize_tomat <<< 1 , col_size >>> ( to_mat , col_size ); \
	for( int col = 0 ; col < col_size ; col++ ){ \
		unsigned int size = row_size; \
		unsigned int offset = 0; \
		bool f = 0; \
		while( size > 0 ){ \
			if(f) continue; \
			unsigned int tmp_size; \
			unsigned int threadNum; \
			if( size >= 16777216 ){ \
				first_sum(256,16777216,from_mat,col,cpy_func,row_size); \
			} \
			else if( size >= 2097152 ){ \
				first_sum(128,2097152,from_mat,col,cpy_func,row_size); \
			} \
			else if( size >= 1048576 ){ \
				first_sum(1024,1048576,from_mat,col,cpy_func,row_size); \
			} \
			else if( size >= 262144 ){ \
				first_sum(512,262144,from_mat,col,cpy_func,row_size); \
			} \
			else if( size >= 65536 ){ \
				first_sum(256,65536,from_mat,col,cpy_func ,row_size); \
			} \
			else if( size >= 16384){ \
				first_sum(128,16384,from_mat,col,cpy_func ,row_size ); \
			} \
			else if( size >= 4096 ){ \
				first_sum(64 , 4096 ,from_mat,col,cpy_func , row_size); \
			} \
			else if( size >= 1024 ){ \
				first_sum(1024,1024 ,from_mat,col,cpy_func , row_size); \
			} \
			else{ \
				get_sum_exp <<< 1 , 1 >>> ( to_mat , from_mat , row_size , col , offset ); \
				break; \
			} \
			cudaDeviceSynchronize(); \
			while( tmp_size > 1 ){ \
				dim3 block_S(  threadNum , 1 , 1 ); \
				dim3 grid_S( tmp_size / block_S.x , 1 , 1 ); \
				switch(threadNum){ \
					case 1024: \
						sum_func< 1024 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 512: \
						sum_func< 512 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 256: \
						sum_func< 256 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 128: \
						sum_func< 128, double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 64: \
						sum_func< 64 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					case 32: \
						sum_func< 32 , double , double ><<< grid_S.x , block_S  >>> ( dev_N_ , dev_N_ , tmp_size, 0); \
						break; \
					default: \
						break; \
				} \
				cudaDeviceSynchronize(); \
				tmp_size = tmp_size / threadNum; \
			} \
			add_sum <<<  1 , 1 >>> ( to_mat , dev_N_ , col ); \
		} \
	}


#endif
