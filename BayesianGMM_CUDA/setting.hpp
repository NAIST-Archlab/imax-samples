#ifndef SETTING_H
#define SETTING_H

#define TYPE float
#define ITR_SIZE unsigned int

#define CUDA_SAFE_CALL(func) \
	do { \
		cudaError_t err = (func); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
			exit(err); \
		} \
	} while(0)

#define FULL_MASK 0xffffffff

#define PROFILE							0
#define PRINT								0
#define USE_INIT_MEAN_FILE	0

#endif
