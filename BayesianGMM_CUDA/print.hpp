#ifndef PRINT_HPP
#define PRINT_HPP

#if PRINT

#define PRINT_CUDA_MAT_SCOPE(name,mat,rows,cols,scope) \
	puts(name); \
	cudaDeviceSynchronize(); \
	print_matrix <<< 1 , 1 >>> (mat,rows,cols,scope ); \
	cudaDeviceSynchronize(); \
	puts("");

#define PRINT_CUDA_MAT(name,mat,rows,cols) \
	puts(name); \
	cudaDeviceSynchronize(); \
	print_matrix <<< 1 , 1 >>> (mat,rows,cols); \
	cudaDeviceSynchronize(); \
	puts("");

#define PRINT_HOST_MAT(name,vec,row,col) \
	puts(name); \
	for( int r = 0 ; r < row ; r++ ){ \
		for( int c = 0 ; c < col ; c++ ){ \
			printf("%.3f " , vec[c*row+r] ); \
		} \
		puts("");\
	} \
	puts("");

#define PRINT_HOST_VEC(name,vec,size) \
	puts(name); \
	printf("["); \
	for( ITR_SIZE idx = 0 ; idx < size ; idx++ ){ \
		printf("%.3f ", vec[idx] ); \
		if( (idx+1) % 5 == 0 ) puts(""); \
	} \
	puts("]")

#define PRINT_HOST_VAL(name,val) \
	printf("%s:%.5f\n",name,val)

__global__
void print_matrix( float *mat , ITR_SIZE rows , ITR_SIZE cols ){
	for( ITR_SIZE i = 0 ; i < rows ; i++ ){
		for( ITR_SIZE j = 0 ; j < cols ; j++ ){
			printf("%.5f\t",mat[ j * rows  + i ] );
		}
		printf("\n");
	}
}
__global__
void print_matrix( float *mat , ITR_SIZE rows , ITR_SIZE cols , int scope  ){
	for( ITR_SIZE i = 0 ; i < scope ; i++ ){
		for( int j = 0 ; j < cols ; j++ ){
			printf("%.3f\t",mat[ j * rows  + i ] );
		}
		printf("\n");
	}
	printf("...\n");
	for( ITR_SIZE i = rows - scope ; i < rows ; i++ ){
		for( int j = 0 ; j < cols ; j++ ){
			printf("%.3f\t",mat[ j * rows  + i ] );
		}
		printf("\n");
	}
}
__global__
void print_matrix( double *mat , ITR_SIZE rows , ITR_SIZE cols , int s_n , int e_n  ){
	for( ITR_SIZE i = 0 ; i < s_n ; i++ ){
		for( int j = 0 ; j < cols ; j++ ){
			printf("%.7lf\t",mat[ j * rows  + i ] );
		}
		printf("\n");
	}
	printf("...\n");
	for( ITR_SIZE i = rows - e_n  ; i < rows ; i++ ){
		for( int j = 0 ; j < cols ; j++ ){
			printf("%.7lf\t",mat[ j * rows  + i ] );
		}
		printf("\n");
	}
}

#else

#define PRINT_CUDA_MAT_SCOPE(name,mat,rows,cols,scope) 
#define PRINT_CUDA_MAT(name,mat,rows,cols) 
#define PRINT_HOST_MAT(name,vec,row,col) 
#define PRINT_HOST_VEC(name,vec,size) 
#define PRINT_HOST_VAL(name,val) 

#endif // if PRINT

#endif // ifndef PRINT
