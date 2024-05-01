## Primary writer: Hiroki Nishimoto ##
## nishimoto.hiroki.na7@is.naist.jp ##

PROGRAM		= ./vbgmm
OBJS		= vbgmm.o gemm.o IO.o
SRCS		= ($(OBJS:.o=.c))
HEADERS		= 

CC		= gcc -std=c99
CP		= g++
CFLAGS		= -O2 -Wall -Wextra -Winit-self -Wno-missing-field-initializers
LIBS		= -lm

all:		$(PROGRAM)

$(PROGRAM):	$(OBJS)
		$(CC) $(OBJS) -o $(PROGRAM) $(CFLAGS) $(LIBS)

.c.o:		$(HEADERS)
		$(CC) $(CFLAGS) -c $<

run:;		$(PROGRAM) data_8_8_512.txt

clean:;		rm -f *.o *.so *.gmon core *~

##
