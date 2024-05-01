#ifndef GEMM_H

#define GEMM_H
#include "setting.h"
void gemm( int trans_a , int trans_b , int a_rows , int b_cols , int mutual , TYPE *a , TYPE *b , TYPE *c );

#endif
