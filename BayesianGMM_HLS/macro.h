#ifndef MACRO_H
#define MACRO_H

#if ON_CPU

#if DEBUG_PRINT
#define DEB_PRINT_VEC( name , vec , size , format ) \
printf(name ); puts(""); \
for( int i = 0 ; i < size ; i++ ) printf( format , vec[i] ); \
printf("\n");

#define DEB_PRINT_MAT( name , vec , row , col , format ) \
printf(name);  puts(""); \
for( int i = 0 ; i < row ; i++ ) { for( int j = 0 ; j < col ; j++ ) printf( format , vec[ i * col + j] ); puts(""); } \
printf("\n");

#define DEB_PRINT_MAT_S( name , vec , row , col , format ) \
printf(name);  puts(""); \
for( int i = 0 ; i < 3 ; i++ ) { for( int j = 0 ; j < col ; j++ ) printf( format , vec[ i * col + j] ); puts(""); } \
printf("\t...\n"); \
for( int i = row - 4 ; i < row ; i++ ) { for( int j = 0 ; j < col ; j++ ) printf( format , vec[ i * col + j] ); puts(""); } \
printf("\n");

#define DEB \
printf("done %d\n" , __LINE__ );

#else // DEBUG_PRINT

#define DEB_PRINT_VEC( name , vec , size , format )
#define DEB_PRINT_MAT( name , vec , row , col , format )
#define DEB_PRINT_MAT_S( name , vec , row , col , format )
#define DEB

#endif // DEBUG_PRINT

#else // ON_CPU

#define DEB_PRINT_VEC( name , vec , size , format )
#define DEB_PRINT_MAT( name , vec , row , col , format )
#define DEB_PRINT_MAT_S( name , vec , row , col , format )
#define DEB

#endif //ON_CPU
#endif // MACRO_H
