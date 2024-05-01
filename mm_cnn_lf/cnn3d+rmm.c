
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/cnn.c,v 1.4 2018/02/04 10:28:43 nakashim Exp nakashim $";

/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <math.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <signal.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif

int WD=320, HT=240, BITMAP=320*240, SCRWD=5, SCRHT=5, VECWD=240, VECHT=240, VECSTEP=4;

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif
#if defined(EMAX7)
#include "../../src/conv-c2d/emax7.h"
#include "../../src/conv-c2d/emax7lib.c"
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
#endif

Uchar* membase;

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(EMAX5) && defined(ARMZYNQ)
  if (emax5_open() == NULL)
    exit(1);
#elif defined(EMAX6) && defined(ARMZYNQ)
  if (emax6_open() == NULL)
    exit(1);
#elif defined(EMAX7)
#if defined(ARMZYNQ)
  if ((NLANE = emax7_open(2)) == NULL)
    exit(1);
#else
  NLANE = 1;
#endif
#endif

#if defined(EMAX5) && defined(ARMZYNQ)
  membase = emax_info.hpp_mmap;
  {volatile int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif defined(EMAX6) && defined(ARMZYNQ)
  membase = emax_info.ddr_mmap;
  {volatile int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif defined(EMAX7) && defined(ARMZYNQ)
  membase = emax_info[0].ddr_mmap;
  {volatile int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  membase = (void*)malloc(memsize+alignment);
  if ((int)membase & (alignment-1))
    membase = (void*)(((int)membase & ~(alignment-1))+alignment);
#endif

#if defined(EMAX5)
#if !defined(ARMZYNQ)
  emax_info.hpp_phys = membase;
  emax_info.hpp_mmap = emax_info.hpp_phys;
  emax_info.acp_phys = ACP_BASE2_PHYS; /* defined in emax5lib.h >= ALOCLIMIT */
  emax_info.acp_mmap = emax_info.acp_phys;
#endif
  acp_conf = emax_info.acp_mmap; /* 8KB * 256sets */
  acp_lmmi = emax_info.acp_mmap + 0x200000;
  acp_regv = emax_info.acp_mmap + 0x304000;
#endif

#if defined(EMAX6)
#if !defined(ARMZYNQ)
  emax_info.dma_phys = DMA_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.dma_mmap = emax_info.dma_phys;
  emax_info.reg_phys = REG_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.reg_mmap = emax_info.reg_phys;
  emax_info.lmm_phys = LMM_BASE2_PHYS;
  emax_info.lmm_mmap = emax_info.lmm_phys;
  emax_info.ddr_phys = membase;
  emax_info.ddr_mmap = emax_info.ddr_phys;
#endif
#if defined(ARMSIML) || defined(ARMZYNQ)
  emax6.dma_ctrl  = emax_info.dma_mmap;
  emax6.reg_ctrl  = emax_info.reg_mmap;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // б·б·б· RESET
#if defined(ARMZYNQ)
  usleep(1);
#endif
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].adtr = emax_info.ddr_mmap - emax_info.lmm_phys;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].dmrp = 0LL;
  switch (((struct reg_ctrl*)emax6.reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  case  0:EMAX_DEPTH =  8;break;
  default:
    printf("sysinit: illegal stat=%x for setting EMAX_DEPTH\n",((struct reg_ctrl*)emax6.reg_ctrl)->i[0].stat>>8 & 0xf);
    exit(1);
  }
  printf("EMAX6.DEPTH=%d\n", EMAX_DEPTH);
#endif
#endif

#if defined(EMAX7) && (defined(ARMSIML) || defined(ARMZYNQ))
  { int i;
    for (i=0; i<NLANE; i++) {
      emax7[i].dma_ctrl  = emax_info[i].dma_mmap;
      emax7[i].reg_ctrl  = emax_info[i].reg_mmap;
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].cmd = CMD_RESET;  // б·б·б· RESET
#if defined(ARMZYNQ)
      usleep(1);
#endif
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].adtr = emax_info[i].ddr_mmap - emax_info[i].lmm_phys;
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].dmrp = 0LL;
      switch (((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].stat>>8 & 0xf) {
      case  3:EMAX_DEPTH = 64;break;
      case  2:EMAX_DEPTH = 32;break;
      case  1:EMAX_DEPTH = 16;break;
      case  0:EMAX_DEPTH =  8;break;
      default:
	printf("init_xmax: illegal stat=%x for setting EMAX_DEPTH\n",((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].stat>>8 & 0xf);
	exit(1);
      }
      printf("EMAX7[%d].DEPTH=%d\n", i, EMAX_DEPTH);
    }
  }
  printf("EMAX7: NLANE=%d DEPTH=%d\n", NLANE, EMAX_DEPTH);
#endif
}

#define IC    18
//IMAP should be specified in Makefile* to share src for emax6/emax7.
//#define IMAP  6  for 64unit
//#define IMAP  3  for 32unit

#define OC    16
#define M     66
#define RMGRP 64
//NCHIP should be specified in Makefile* to share src for emax6/emax7.
//#define NCHIP 4 for emax6
//#define NCHIP 1 for emax7

#define K     3
#define W     4
Uint *in;  /*[IC*M*M*M];*/
Uint *ker; /*[IC*OC*K*K*K];*/
Uint *out0;/*[OC*M*M*M];*/
Uint *out1;/*[OC*M*M*M];*/
Uint *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp, *op;
int ic, dep, row, col, oc, z, y, x;
int ztop, top, iset;
int w, kidx;
int count0, count1, count2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

main()
{
  sysinit(IC*M*M*M*sizeof(int)
         +IC*OC*K*K*K*sizeof(int)
         +OC*M*M*M*sizeof(int)
         +OC*M*M*M*sizeof(int),32);
  printf("membase: %08.8x\n", (Uint)membase);
  in   = (Uint*)membase;
  ker  = (Uint*)((Uchar*)in   + IC*M*M*M*sizeof(int));
  out0 = (Uint*)((Uchar*)ker  + IC*OC*K*K*K*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OC*M*M*M*sizeof(int));
  printf("in  : %08.8x\n", in);
  printf("ker : %08.8x\n", ker);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);

  for (ic=0; ic<IC; ic++) {
    for (dep=0; dep<M; dep++) {
      for (row=0; row<M; row++) {
	for (col=0; col<M; col++) {
	  *(float*)&in[ic*M*M*M+dep*M*M+row*M+col] = ic<<12|(((M/2-abs(dep-M/2))*(M/2-abs(row-M/2))*(M/2-abs(col-M/2)))&0xfff);
	}
      }
    }
  }
  for (oc=0; oc<OC; oc++) {
    for (ic=0; ic<IC; ic++) {
      for (z=0; z<K; z++) {
	for (y=0; y<K; y++) {
	  for (x=0; x<K; x++) {
	    *(float*)&ker[ic*OC*K*K*K+oc*K*K*K+z*K*K+y*K+x] = (oc-ic)*((2-abs(z-K/2))*(2-abs(y-K/2))*(2-abs(x-K/2)))/OC;
	  }
	}
      }
    }
  }

#if !defined(ARMSIML)
  x11_open(0);
#endif

#if defined(EMAX6)
  reset_nanosec();
  orig();
  get_nanosec(0);
  show_nanosec();
#endif
#if defined(EMAX7)
  reset_nanosec(0);
  orig();
  get_nanosec(0,0);
  show_nanosec(0);
#endif

#if defined(EMAX6)
  reset_nanosec();
  imax(0);
  get_nanosec(0);
  show_nanosec();
#endif
#if defined(EMAX7)
  reset_nanosec(0);
  imax(0);
  get_nanosec(0,0);
  show_nanosec(0);
#endif

#ifdef ARMSIML
  copy_Z(0, out1); _copyX(0, Z);
  copy_Z(1, out1); _copyX(1, Z);
  copy_Z(2, out1); _copyX(2, Z);
  copy_Z(3, out1); _copyX(3, Z);
  copy_Z(5, out1); _copyX(4, Z);
  copy_Z(6, out1); _copyX(5, Z);
  copy_Z(7, out1); _copyX(6, Z);
  copy_Z(8, out1); _copyX(7, Z);
  copy_Z(10,out1); _copyX(8 ,Z);
  copy_Z(11,out1); _copyX(9 ,Z);
  copy_Z(12,out1); _copyX(10,Z);
  copy_Z(13,out1); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, out1); BGR_to_X(0, Z);
  copy_Z(1, out1); BGR_to_X(1, Z);
  copy_Z(2, out1); BGR_to_X(2, Z);
  copy_Z(3, out1); BGR_to_X(3, Z);
  copy_Z(4, out1); BGR_to_X(4, Z);
  copy_Z(5, out1); BGR_to_X(5, Z);
  copy_Z(6, out1); BGR_to_X(6, Z);
  copy_Z(7, out1); BGR_to_X(7, Z);
  copy_Z(8, out1); BGR_to_X(8, Z);
  copy_Z(9 ,out1); BGR_to_X(9, Z);
  copy_Z(10,out1); BGR_to_X(10,Z);
  copy_Z(11,out1); BGR_to_X(11,Z);
  copy_Z(12,out1); BGR_to_X(12,Z);
  copy_Z(13,out1); BGR_to_X(13,Z);
  copy_Z(14,out1); BGR_to_X(14,Z);
  copy_Z(15,out1); BGR_to_X(15,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (oc=0; oc<OC; oc++) {
    for (dep=1; dep<M-1; dep++) {
      for (row=1; row<M-1; row++) {
	for (col=0; col<M-2; col++) {
	  if (out0[oc*M*M*M+dep*M*M+row*M+col] != out1[oc*M*M*M+dep*M*M+row*M+col]) {
	    count2++;
	    printf("o0[%d]=%f o1[%d]=%f\n",
		   oc*M*M*M+dep*M*M+row*M+col, (double)*(float*)&out0[oc*M*M*M+dep*M*M+row*M+col],
		   oc*M*M*M+dep*M*M+row*M+col, (double)*(float*)&out1[oc*M*M*M+dep*M*M+row*M+col]);
	  }
        }
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");

#if defined(EMAX6)
  show_nanosec();
#endif
#if defined(EMAX7)
  show_nanosec(0);
#endif

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif
}

copy_Z(id, from)
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned int *to = Z;

  switch (id) {
  case 0:                   break;
  case 1:  from += M*M;     break;
  case 2:  from += M*M*2;   break;
  case 3:  from += M*M*3;   break;
  case 4:  from += M*M*4;   break;
  case 5:  from += M*M*5;   break;
  case 6:  from += M*M*6;   break;
  case 7:  from += M*M*7;   break;
  case 8:  from += M*M*8;   break;
  case 9:  from += M*M*9;   break;
  case 10: from += M*M*10;  break;
  case 11: from += M*M*11;  break;
  case 12: from += M*M*12;  break;
  case 13: from += M*M*13;  break;
  case 14: from += M*M*14;  break;
  case 15: from += M*M*15;  break;
  }
  for (i=0; i<HT; i++, from+=M) {
    if (i<M) {
      for (j=0; j<WD; j++) {
        if (j<M) *to++ = (*(from+j))<<2;
        else     *to++ = 0;
      }
    }
    else {
      for (j=0; j<WD; j++)
        *to++ = 0;
    }
  }
}

orig() {
  printf("<<<ORIG>>>\n");
  for (ic=0; ic<IC; ic++) { /* set input channel */
    ip0 = &in[ic*M*M*M]; /* top of input */
    for (dep=1; dep<M-1; dep++) { /* image loop */
      for (row=1; row<M-1; row++) { /* image loop */
	for (col=0; col<M-2; col++) {
	  for (oc=0; oc<OC; oc++) { /* set output channel */
	    op = &out0[oc*M*M*M+dep*M*M+row*M+col]; /* top of output */
	    kp = &ker[(oc*IC+ic)*K*K*K];
	    kidx = 0;
	    for (z=-((K-1)/2); z<=(K-1)/2; z++) { /* kernel loop */
	      for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
		for (x=-((K-1)/2); x<=(K-1)/2; x++) {
		  if (ic == 0 && kidx == 0) {
		    *(float*)&*op  = *(float*)&ip0[(dep+z)*M*M+(row+y)*M+col+x+1] * *(float*)&kp[kidx];
		    /*printf("head [%d %d %d][%d %d %d] out0[%d]=%d\n", ic, row, col, oc, y, x, op-&out0[0], *op);*/
		  }
		  else {
		    *(float*)&*op += *(float*)&ip0[(dep+z)*M*M+(row+y)*M+col+x+1] * *(float*)&kp[kidx];
		    /*printf("     [%d %d %d][%d %d %d] out0[%d]=%d\n", ic, row, col, oc, y, x, op-&out0[0], *op);*/
		  }
		  kidx++;
		  count0++;
		}
	      }
	    }
          }
        }
      }
    }
  }
}

#if 0
/*     +-----------------+                       */
/*     | z-----------------+                     */
/*     | | +--x--------------+                   */
/*     | | | +-------242-------+                 */
/*     | | y |       240       | 968B*18=17424B  */
/*     | | | |                 | 968B            */
/*     | | | |     * * *       | 968B            */
/*     | | | |     * * *       | 968B            */
/*     +-| | |     * * *       | 968B            */
/*       +-| |                 | 968B            */
/*         +-|                 | 968B            */
/*           +-----------------+                 */
imax(int LANE) {
  Ull CHIP;
  Ull rofs;
  printf("<<<IMAX>>>\n");
  for (ztop=1; ztop<M-1; ztop++) { /* Z */
    for (top=1; top<M-1; top+=RMGRP) { /* Y */
      for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
	for (oc=0; oc<OC/NCHIP; oc+=W) { /* set output channel */
    /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
      /*2*/ for (rofs=0; rofs<RMGRP; rofs++) { /* image loop (row) */
        /*1*/ for (col=0; col<M-2; col++) { /* image loop (col) */
                for (w=0; w<W; w++) { /* set output channel */
		  op = &out1[(CHIP*OC/NCHIP+oc+w)*M*M*M+ztop*M*M+(top+rofs)*M+col]; /* top of output */
		  for (ic=0; ic<IMAP; ic++) { /* set offset of input channel */
		    ip0 = &in[(iset+ic)*M*M*M]; /* top of input */
		    kp = &ker[((CHIP*OC/NCHIP+oc+w)*IC+iset+ic)*K*K*K];
		    kidx = 0;
		    for (z=-((K-1)/2); z<=(K-1)/2; z++) { /* kernel loop */
		      for (y=-((K-1)/2); y<=(K-1)/2; y++) {
			for (x=-((K-1)/2); x<=(K-1)/2; x++) {
			  if (iset == 0 && ic == 0 && kidx == 0) {
			    *(float*)&*op  = *(float*)&ip0[(ztop+z)*M*M+(top+rofs+y)*M+col+x+1] * *(float*)&kp[kidx];
			    /*printf("head [%d %d %d][%d %d %d] out1[%d]=%d\n", ic, row, col, (oc+w), y, x, op-&out1[0], *op);*/
			  }
			  else {
			    *(float*)&*op += *(float*)&ip0[(ztop+z)*M*M+(top+rofs+y)*M+col+x+1] * *(float*)&kp[kidx];
			    /*printf("     [%d %d %d][%d %d %d] out1[%d]=%d\n", ic, row, col, (oc+w), y, x, op-&out1[0], *op);*/
			  }
			  kidx++;
			  count1++;
			}
		      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#else

imax(int LANE) {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, oofs, k;
  Ull  Force;
  /*  игибибибибибидconvolutionд╬╛ь╣ч         игибибибибибид                             */
  /*  ивигибибибибикидBдм╩г┐Їд╚╣═диды         ивигибибибибикид                           */
  /*  ививигибибибибикидигибибибибибидид      ививигибибибибикидид                       */
  /*  ивививiii       ививk k k k k ививRMGRP ививив          ививRMGRP                  */
  /*  ивививiii       ийив          ивий      ивививo o o o o ивий                       */
  /*  ивививiii in    ивив k(weight)ивив      ививив   out    ивив mmд╬╛ь╣чд╧╣╘д╟╩м│ф    */
  /*  иживив          ийив          ивий      иживив          ивий cnnд╬╛ь╣чд╧outд╟╩м│ф  */
  /*    ижив          ивив          ивив        ижив          ивив                       */
  /*      ижибибибибибиеижибибибибибиеие          ижибибибибибиеие                       */
  printf("<<<IMAX>>>\n");
  for (ztop=1; ztop<M-1; ztop++) { /* Z */
    for (top=1; top<M-1; top+=RMGRP) {
      for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
	Uint *ip[IMAP], *it[3][IMAP], *ip0[IMAP][K*K*K], *ip1[IMAP][K*K*K];
	for (k=0; k<IMAP; k++) {
	  ip[k] = &in[(iset+k)*M*M*M]; /* top of input#0-5 */
	  it[0][k] = ip[k]+(ztop-1)*M*M+(top-1)*M+1-1;
	  it[1][k] = ip[k]+(ztop+0)*M*M+(top-1)*M+1-1;
	  it[2][k] = ip[k]+(ztop+1)*M*M+(top-1)*M+1-1;
	  ip0[k][ 0] = ip[k]+(ztop-1)*M*M+(top-1)*M+1-1; ip0[k][ 1] = ip[k]+(ztop-1)*M*M+(top-1)*M+1+0; ip0[k][ 2] = ip[k]+(ztop-1)*M*M+(top-1)*M+1+1;
	  ip0[k][ 3] = ip[k]+(ztop-1)*M*M+(top+0)*M+1-1; ip0[k][ 4] = ip[k]+(ztop-1)*M*M+(top+0)*M+1+0; ip0[k][ 5] = ip[k]+(ztop-1)*M*M+(top+0)*M+1+1;
	  ip0[k][ 6] = ip[k]+(ztop-1)*M*M+(top+1)*M+1-1; ip0[k][ 7] = ip[k]+(ztop-1)*M*M+(top+1)*M+1+0; ip0[k][ 8] = ip[k]+(ztop-1)*M*M+(top+1)*M+1+1;
	  ip0[k][ 9] = ip[k]+(ztop+0)*M*M+(top-1)*M+1-1; ip0[k][10] = ip[k]+(ztop+0)*M*M+(top-1)*M+1+0; ip0[k][11] = ip[k]+(ztop+0)*M*M+(top-1)*M+1+1;
	  ip0[k][12] = ip[k]+(ztop+0)*M*M+(top+0)*M+1-1; ip0[k][13] = ip[k]+(ztop+0)*M*M+(top+0)*M+1+0; ip0[k][14] = ip[k]+(ztop+0)*M*M+(top+0)*M+1+1;
	  ip0[k][15] = ip[k]+(ztop+0)*M*M+(top+1)*M+1-1; ip0[k][16] = ip[k]+(ztop+0)*M*M+(top+1)*M+1+0; ip0[k][17] = ip[k]+(ztop+0)*M*M+(top+1)*M+1+1;
	  ip0[k][18] = ip[k]+(ztop+1)*M*M+(top-1)*M+1-1; ip0[k][19] = ip[k]+(ztop+1)*M*M+(top-1)*M+1+0; ip0[k][20] = ip[k]+(ztop+1)*M*M+(top-1)*M+1+1;
	  ip0[k][21] = ip[k]+(ztop+1)*M*M+(top+0)*M+1-1; ip0[k][22] = ip[k]+(ztop+1)*M*M+(top+0)*M+1+0; ip0[k][23] = ip[k]+(ztop+1)*M*M+(top+0)*M+1+1;
	  ip0[k][24] = ip[k]+(ztop+1)*M*M+(top+1)*M+1-1; ip0[k][25] = ip[k]+(ztop+1)*M*M+(top+1)*M+1+0; ip0[k][26] = ip[k]+(ztop+1)*M*M+(top+1)*M+1+1;
	  ip1[k][ 0] = ip[k]+(ztop-1)*M*M+(top-1)*M+1+1; ip1[k][ 1] = ip[k]+(ztop-1)*M*M+(top-1)*M+1+2; ip1[k][ 2] = ip[k]+(ztop-1)*M*M+(top-1)*M+1+3;
	  ip1[k][ 3] = ip[k]+(ztop-1)*M*M+(top+0)*M+1+1; ip1[k][ 4] = ip[k]+(ztop-1)*M*M+(top+0)*M+1+2; ip1[k][ 5] = ip[k]+(ztop-1)*M*M+(top+0)*M+1+3;
	  ip1[k][ 6] = ip[k]+(ztop-1)*M*M+(top+1)*M+1+1; ip1[k][ 7] = ip[k]+(ztop-1)*M*M+(top+1)*M+1+2; ip1[k][ 8] = ip[k]+(ztop-1)*M*M+(top+1)*M+1+3;
	  ip1[k][ 9] = ip[k]+(ztop+0)*M*M+(top-1)*M+1+1; ip1[k][10] = ip[k]+(ztop+0)*M*M+(top-1)*M+1+2; ip1[k][11] = ip[k]+(ztop+0)*M*M+(top-1)*M+1+3;
	  ip1[k][12] = ip[k]+(ztop+0)*M*M+(top+0)*M+1+1; ip1[k][13] = ip[k]+(ztop+0)*M*M+(top+0)*M+1+2; ip1[k][14] = ip[k]+(ztop+0)*M*M+(top+0)*M+1+3;
	  ip1[k][15] = ip[k]+(ztop+0)*M*M+(top+1)*M+1+1; ip1[k][16] = ip[k]+(ztop+0)*M*M+(top+1)*M+1+2; ip1[k][17] = ip[k]+(ztop+0)*M*M+(top+1)*M+1+3;
	  ip1[k][18] = ip[k]+(ztop+1)*M*M+(top-1)*M+1+1; ip1[k][19] = ip[k]+(ztop+1)*M*M+(top-1)*M+1+2; ip1[k][20] = ip[k]+(ztop+1)*M*M+(top-1)*M+1+3;
	  ip1[k][21] = ip[k]+(ztop+1)*M*M+(top+0)*M+1+1; ip1[k][22] = ip[k]+(ztop+1)*M*M+(top+0)*M+1+2; ip1[k][23] = ip[k]+(ztop+1)*M*M+(top+0)*M+1+3;
	  ip1[k][24] = ip[k]+(ztop+1)*M*M+(top+1)*M+1+1; ip1[k][25] = ip[k]+(ztop+1)*M*M+(top+1)*M+1+2; ip1[k][26] = ip[k]+(ztop+1)*M*M+(top+1)*M+1+3;
	}

	for (oc=0; oc<OC/NCHIP; oc+=W) { /* set output channel */
	  Uint *kp0[IMAP][NCHIP], *kp1[IMAP][NCHIP], *kp2[IMAP][NCHIP], *kp3[IMAP][NCHIP];
	  Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
	  Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

	  for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	    Uint choc  = CHIP*OC/NCHIP+oc;
	    for (k=0; k<IMAP; k++) {
	      kp0[k][CHIP] = ker+((choc+0)*IC+iset+k)*K*K*K;
	      kp1[k][CHIP] = ker+((choc+1)*IC+iset+k)*K*K*K;
	      kp2[k][CHIP] = ker+((choc+2)*IC+iset+k)*K*K*K;
	      kp3[k][CHIP] = ker+((choc+3)*IC+iset+k)*K*K*K;
	    }
	    op0[CHIP] = out1+(choc+0)*M*M*M+ztop*M*M+top*M+0; op1[CHIP] = out1+(choc+1)*M*M*M+ztop*M*M+top*M+0; op2[CHIP] = out1+(choc+2)*M*M*M+ztop*M*M+top*M+0; op3[CHIP] = out1+(choc+3)*M*M*M+ztop*M*M+top*M+0;
	    ot0[CHIP] = out1+(choc+0)*M*M*M+ztop*M*M+top*M+0; ot1[CHIP] = out1+(choc+1)*M*M*M+ztop*M*M+top*M+0; ot2[CHIP] = out1+(choc+2)*M*M*M+ztop*M*M+top*M+0; ot3[CHIP] = out1+(choc+3)*M*M*M+ztop*M*M+top*M+0;
	  }
	  Force = 1;

#define cnn_core1(r, i, j, ofs, k, rp1) \
            mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)kp0[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K);\
            mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)kp1[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K);\
            mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)kp2[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K);\
            mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)kp3[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K);\
            mop(OP_LDR,    1, &BR[r][2][1],  (Ull)ip1[i][k], oofs, MSK_W0, (Ull)it[j][i], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2));\
            mop(OP_LDR,    1, &BR[r][2][0],  (Ull)ip0[i][k], oofs, MSK_W0, (Ull)it[j][i], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2));\
            exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][0][1], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][0][0], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][1][1], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][1][0], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn_final(r, rp1, Force) \
            mop(OP_LDR,  1, &BR[rp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            mop(OP_LDR,  1, &BR[rp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            mop(OP_LDR,  1, &BR[rp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            mop(OP_LDR,  1, &BR[rp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210, BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210, BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210, BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210, BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_STR,  3, &AR[rp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            mop(OP_STR,  3, &AR[rp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            mop(OP_STR,  3, &AR[rp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP);\
            mop(OP_STR,  3, &AR[rp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], M*RMGRP, 0, Force, (Ull)NULL, M*RMGRP)

//EMAX5A begin cnn mapdist=0
    /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
      /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=0-M*4; LOOP1--; INIT1=0) {            /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
        /*1*/ for (INIT0=1,LOOP0=(M-2)/2,cofs=0-8; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
                exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 8, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);/* stage#0 */
		exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?M*4:0,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#0 */
		exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);  /* stage#1 */

		mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp0[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K); /* stage#2 */
		mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp1[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K); /* stage#2 */
		mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp2[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K); /* stage#2 */
		mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp3[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K*K, 0, 0, (Ull)NULL, IC*OC*K*K*K); /* stage#2 10KB */
		mop(OP_LDR,    1, &BR[2][2][1],  (Ull)ip1[0][0], oofs, MSK_W0, (Ull)it[0][0], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2)); /* stage#2 8KB *//* unaligned load */
		mop(OP_LDR,    1, &BR[2][2][0],  (Ull)ip0[0][0], oofs, MSK_W0, (Ull)it[0][0], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2)); /* stage#2 8KB *//* unaligned load */
		exe(OP_FML, &AR[3][0], BR[2][2][0], EXP_H3210, BR[2][0][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		exe(OP_FML, &AR[3][1], BR[2][2][0], EXP_H3210, BR[2][0][0], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		exe(OP_FML, &AR[3][2], BR[2][2][0], EXP_H3210, BR[2][1][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		exe(OP_FML, &AR[3][3], BR[2][2][0], EXP_H3210, BR[2][1][0], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */

		cnn_core1( 3, 0, 0,  4LL, 1,  4);
		cnn_core1( 4, 0, 0,  8LL, 2,  5);
		cnn_core1( 5, 0, 0, 12LL, 3,  6);
		cnn_core1( 6, 0, 0, 16LL, 4,  7);
		cnn_core1( 7, 0, 0, 20LL, 5,  8);
		cnn_core1( 8, 0, 0, 24LL, 6,  9);
		cnn_core1( 9, 0, 0, 28LL, 7, 10);
		cnn_core1(10, 0, 0, 32LL, 8, 11);

		cnn_core1(11, 0, 1, 36LL, 9, 12);
		cnn_core1(12, 0, 1, 40LL,10, 13);
		cnn_core1(13, 0, 1, 44LL,11, 14);
		cnn_core1(14, 0, 1, 48LL,12, 15);
		cnn_core1(15, 0, 1, 52LL,13, 16);
		cnn_core1(16, 0, 1, 56LL,14, 17);
		cnn_core1(17, 0, 1, 60LL,15, 18);
		cnn_core1(18, 0, 1, 64LL,16, 19);
		cnn_core1(19, 0, 1, 68LL,17, 20);

		cnn_core1(20, 0, 2, 72LL,18, 21);
		cnn_core1(21, 0, 2, 76LL,19, 22);
		cnn_core1(22, 0, 2, 80LL,20, 23);
		cnn_core1(23, 0, 2, 84LL,21, 24);
		cnn_core1(24, 0, 2, 88LL,22, 25);
		cnn_core1(25, 0, 2, 92LL,23, 26);
		cnn_core1(26, 0, 2, 96LL,24, 27);
		cnn_core1(27, 0, 2,100LL,25, 28);
		cnn_core1(28, 0, 2,104LL,26, 29);
#if (IMAP==1)
		cnn_final(29,     30, Force);
#endif
#if (IMAP>1)
		cnn_core1(29, 1, 0,  0LL, 0, 30);
		cnn_core1(30, 1, 0,  4LL, 1, 31);
		cnn_core1(31, 1, 0,  8LL, 2, 32);
		cnn_core1(32, 1, 0, 12LL, 3, 33);
		cnn_core1(33, 1, 0, 16LL, 4, 34);
		cnn_core1(34, 1, 0, 20LL, 5, 35);
		cnn_core1(35, 1, 0, 24LL, 6, 36);
		cnn_core1(36, 1, 0, 28LL, 7, 37);
		cnn_core1(37, 1, 0, 32LL, 8, 38);

		cnn_core1(38, 1, 1, 36LL, 9, 39);
		cnn_core1(39, 1, 1, 40LL,10, 40);
		cnn_core1(40, 1, 1, 44LL,11, 41);
		cnn_core1(41, 1, 1, 48LL,12, 42);
		cnn_core1(42, 1, 1, 52LL,13, 43);
		cnn_core1(43, 1, 1, 56LL,14, 44);
		cnn_core1(44, 1, 1, 60LL,15, 45);
		cnn_core1(45, 1, 1, 64LL,16, 46);
		cnn_core1(46, 1, 1, 68LL,17, 47);

		cnn_core1(47, 1, 2, 72LL,18, 48);
		cnn_core1(48, 1, 2, 76LL,19, 49);
		cnn_core1(49, 1, 2, 80LL,20, 50);
		cnn_core1(50, 1, 2, 84LL,21, 51);
		cnn_core1(51, 1, 2, 88LL,22, 52);
		cnn_core1(52, 1, 2, 92LL,23, 53);
		cnn_core1(53, 1, 2, 96LL,24, 54);
		cnn_core1(54, 1, 2,100LL,25, 55);
		cnn_core1(55, 1, 2,104LL,26, 56);
#endif
#if (IMAP==2)
		cnn_final(56,     57, Force);
#endif
              }
            }
          }
//EMAX5A end
          if (Force) Force = 0; /* reset wdat load to LMM */
        }
      }
    }
  }
//EMAX5A drain_dirty_lmm
}
#endif
