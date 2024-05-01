## Primary writer: Hiroki Nishimoto ##
## nishimoto.hiroki.na7@is.naist.jp ##

PROGRAM 	= ./vbgmm
SRCS		= vbgmm.cu
HEADERS		= 

CC		= g++
NVCC		= nvcc -std=c++11
LIBS		= -lcublas
LIBFLAGS	= -L/usr/lib64
FILE		= data.txt
CHECK		= test_result.txt
RESULT		= result.txt
OPT		= -arch=sm_30
FOR_PROF	= -DINIT_MEAN_ON_HOST=1

D		= 16
T		= 1024
DIM		= "-DDIM=${D}"
THREAD_N	= "-DTHREAD_N=${T}"
ADD_OPT		= 

$(PROGRAM):	vbgmm.cu
		@echo dim=${DIM:#*=} threads size=${THREAD_N:#*=}
		${NVCC} ${LIBS} ${ADD_OPT} ${OPT} ${DIM} ${THREAD_N} -o $@ $^

debug:;		@echo "---  DEBUG MODE ---"
		@make ADD_OPT=-DDEBUG=1

fmath:;		@echo "---  use fast math function MODE ---"
		@make ADD_OPT="-DFMATH=1 -DFMATHA=1 -DFMATHB=1 -DFMATHC=1 -DFMATHE=1 -DFMATHF=1"

every:;		@echo "---  get log likelihood every iteration MODE ---"
		@make ADD_OPT="-DFMATH=1 -DFMATHA=1 -DFMATHB=1 -DFMATHC=1 -DFMATHE=1 -DFMATHF=1 -DTAKE_LL_EVERY_ITER=1"

prof:;		@echo "---  use fast math function MODE ---"
		@echo "---  prof mode ---"
		@make ADD_OPT="-DFMATH=1 -DFMATHA=1 -DFMATHB=1 -DFMATHC=1 -DFMATHE=1 -DFMATHF=1 -DPROF=1"

run:;		$(PROGRAM) res.txt 2 log

clean:;		rm -f *.o *.so *.gmon core *~

##
