
## EMAX5 Instruction Simulator         ##
##   Copyright (C) 2014 by NAIST UNIV. ##
##         Primary writer: Y.Nakashima ##
##                nakashim@is.naist.jp ##

## -DARMSIML   ... with SIML+mmap
## -DARMZYNQ   ... with ZYNQ+mmap
## -DEMAX      ... with emax5_start()
## -DEMAX5SIML ... with ZYNQ+mmap+EMAX5siml
## -DEMAX5ASIC ... with ZYNQ+mmap+EMAX5real

## hostnative-C        : none
## hostnative+EMAXNC   : -DEMAX                        
## bsim-C              : -DARMSIML
## bsim+EMAXNC         : -DARMSIML -DEMAX
## bsim+EMAXsiml       : -DARMSIML -DEMAX              emax5_start->emax_start->syscall
##(ZYNQnative-C        : -DARMZYNQ                   )
##(ZYNQnative+EMAXNC   : -DARMZYNQ -DEMAX            )
##(ZYNQnative+EMAXsiml : -DARMZYNQ -DEMAX -DEMAX5SIML) emax5_start->inline emax5siml (N.A.)
##(ZYNQnative+EMAXreal : -DARMZYNQ -DEMAX -DEMAX5ASIC) emax5_start->udev_write

PROJTOP	      = ../../
OPTION        = -DDEBUG -DCYCLECNT -DBIT32 -g
PROGRAM       = deeppose
CC            = gcc
#CFLAGS        = -m32 -I/usr/local/include -O3 $(OPTION)
CFLAGS        = -m32 -I/usr/local/include $(OPTION)
AS            = as
ASFLAGS       = 
LD            = ld
LDFLAGS       = -m32
LIBS          = -lX11 -lm -lc
LIBFLAGS      = -L/usr/local/lib
OBJS	      =	deeppose.o deepposelib.o
SRCS	      =	deeppose.c


all:		$(PROGRAM)

run:;		./$(PROGRAM)

clean:;		rm -f $(OBJS) core *.s


$(PROGRAM):     $(OBJS)
		$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS) $(LDFLAGS) $(LIBFLAGS) $(LIBS)

###
