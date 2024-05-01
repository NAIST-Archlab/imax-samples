## Primary writer: Hiroki Nishimoto ##
## nishimoto.hiroki.na7@is.naist.jp ##

PROGRAM 	= ./vbgmm
HEADERS		= 

CC			= gcc -std=c99
NVCC		= nvcc -std=c++11
LIBS		= -lcublas 
OPT			= -arch=sm_30
OBJS		= xdisp.o
MAIN 		= vbgmm.cu
SRCS		= ($(OBJS:.o=.c))
CFLAGS		= -O2 -Wall -Wextra -Winit-self -Wno-missing-field-initializers -I/usr/local/include -I/usr/include/openblas -DCBLAS_GEMM
LDFLAGS 	= -L/usr/lib64 -L/usr/local/lib -L/usr/lib64/atlas -lm -lrt -lopenblas -lX11

D			= 16
T			= 1024
DIM			= "-DDIM=${D}"
THREAD_N	= "-DTHREAD_N=${T}"
ADD_OPT		= 

all:		$(PROGRAM)

$(PROGRAM):	$(OBJS) $(MAIN)
		@echo dim=${DIM:#*=} threads size=${THREAD_N:#*=}
		$(NVCC) $(MAIN) -o $(PROGRAM) $(OBJS) $(LDFLAGS) $(LIBS) $(DIM) $(THREAD_N)

fmath:
		@echo "---  use fast math function MODE ---"
		@make ADD_OPT="-DFMATH=1 -DFMATHA=1 -DFMATHB=1 -DFMATHC=1 -DFMATHE=1 -DFMATHF=1"

.c.o:	$(HEADERS)
		$(CC) $(CFLAGS) $(LDFLAGS) -c $<

clean:
		rm -f *.o *.so *.gmon core *~ vbgmm

##
