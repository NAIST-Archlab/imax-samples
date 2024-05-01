#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <float.h>
#ifndef ARMSIML
#include <sys/mman.h>
#include <sys/time.h>
#endif
#include "deepposelib.h"

/*#include "../../src/conv-c2d_32bit_64st/emax5.h"*/
/*#include "../../src/conv-c2d_32bit_64st/emax5lib.c"*/


/* --------------------- メモリ領域の確保 ------------------------- */
void sysinit(int isize, int osize, int wsize, int bsize, float **IN, float **OUT, float **BUF, float **WEIGHT, float **BIAS, float **IN_EMAX, float **OUT_EMAX)
{
  Uint memsize;
  char *membase;

#ifdef ARMZYNQ
  if (emax5_open() == NULL) 
    exit(1);
  // ACP + HPP
  //acp_conf = emax_info.acp_mmap; /* 8KB * 256sets */
  //acp_lmmi = emax_info.acp_mmap + 0x200000;
  //acp_regv = emax_info.acp_mmap + 0x304000;
  //membase = emax_info.hpp_mmap;
  // HPP only
  acp_conf = emax_info.hpp_mmap; /* 8KB * 256sets */
  acp_lmmi = emax_info.hpp_mmap + 0x200000;
  acp_regv = emax_info.hpp_mmap + 0x304000;
  membase =  emax_info.hpp_mmap + 0x600000;
  emax_info.acp_mmap = emax_info.hpp_mmap;
  emax_info.acp_phys = emax_info.hpp_phys;   //in driver:phys=(Ull)virt_to_phys(buf);

#else
  memsize = sizeof(float)*(isize+osize*2+wsize+bsize); /* IN, OUT, BUF, WEIGHT, BIAS */
  membase = (void*)malloc(memsize+32); /* 32bytes alignment */
#ifdef BIT32
  if ((int)membase & 31) /* 32bytes alignment */
    membase = (void*)(((int)membase & ~31)+32);
#else
  if ((unsigned long long)membase & 31) /* 32bytes alignment */
    membase = (void*)(((unsigned long long)membase & ~31)+32);
#endif
#ifdef EMAX
  emax_info.hpp_phys = membase;
  emax_info.hpp_mmap = emax_info.hpp_phys;
  emax_info.acp_phys = (void*)malloc(0x400000+32); /* 32bytes alignment */
  if ((int)emax_info.acp_phys & 31) /* 32bytes alignment */
    emax_info.acp_phys = (void*)(((int)emax_info.acp_phys & ~31)+32);
  emax_info.acp_mmap = emax_info.acp_phys;
  acp_conf = emax_info.acp_mmap; /* 8KB * 256sets */
  acp_lmmi = emax_info.acp_mmap + 0x200000;
  acp_regv = emax_info.acp_mmap + 0x304000;
#endif
#endif
  printf("membase: %08.8x_%08.8x\n", (Uint)((Ull)membase>>32),(Uint)membase);
#if defined(ARMSIML) || defined(ARMZYNQ)
  *IN     = (Sfloat*)malloc(sizeof(float)*isize);
  *OUT    = (Sfloat*)malloc(sizeof(float)*osize);
  *BUF    = (Sfloat*)malloc(sizeof(float)*osize);
  *WEIGHT = (Sfloat*)malloc(sizeof(float)*wsize);
  *BIAS   = (Sfloat*)malloc(sizeof(float)*bsize);
  *IN_EMAX  = (Sfloat*)((Uchar*)membase);
  *OUT_EMAX = (Ull)(Sfloat*)((Uchar*)*IN_EMAX  + sizeof(float)*isize*4) & ~4095;
  printf("IN_EMAX  : %08.8x_%08.8x\n", (Uint)((Ull)*IN_EMAX>>32),(Uint)*IN_EMAX);
  printf("OUT_EMAX : %08.8x_%08.8x\n", (Uint)((Ull)*OUT_EMAX>>32),(Uint)*OUT_EMAX);
#else
  *IN     = (Sfloat*)((Uchar*)membase);
  *OUT    = (Sfloat*)((Uchar*)*IN     + sizeof(float)*isize);
  *BUF    = (Sfloat*)((Uchar*)*OUT    + sizeof(float)*osize);
  *WEIGHT = (Sfloat*)((Uchar*)*BUF    + sizeof(float)*osize);
  *BIAS   = (Sfloat*)((Uchar*)*WEIGHT + sizeof(float)*wsize);
#endif
  printf("IN       : %08.8x_%08.8x\n", (Uint)((Ull)*IN>>32),(Uint)*IN);
  printf("OUT      : %08.8x_%08.8x\n", (Uint)((Ull)*OUT>>32),(Uint)*OUT);
  printf("BUF      : %08.8x_%08.8x\n", (Uint)((Ull)*BUF>>32),(Uint)*BUF);
  printf("WEIGHT   : %08.8x_%08.8x\n", (Uint)((Ull)*WEIGHT>>32),(Uint)*WEIGHT);
  printf("BIAS     : %08.8x_%08.8x\n", (Uint)((Ull)*BIAS>>32),(Uint)*BIAS);

#if defined(ARMSIML) || defined(ARMZYNQ)
  int i;
  for(i = 0; i < 27*27*256; i++) {
    *(*OUT_EMAX+i) = 0.0f;
  }
#endif
}

void systerm()
{
#ifdef ARMZYNQ
#endif
}


inline int max(int a, int b) {
	return a < b ? b : a;
}

inline int min(int a, int b) {
	return a > b ? b : a;
}

void SetParamLayer(p_layer_t param_layer[]) {
	FILE *fp;
	char buf[1024];
	int i,j;

	if ( (fp = fopen("param.csv","r")) == NULL ) {
		printf("Can't open param.csv\n");
		exit(1);
	}

	i = 0;
	while( fgets( buf, sizeof(buf), fp) ) {
		if ( strncmp(buf,"//",2) == 0 ) continue;
		if ( sscanf( buf, "%[^,],%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
					  (char*)&param_layer[i].name,
					  &param_layer[i].icn,
					  &param_layer[i].inpH,
					  &param_layer[i].inpW,
					  &param_layer[i].padH,
					  &param_layer[i].padW,
					  &param_layer[i].strideH,
					  &param_layer[i].strideW,
					  &param_layer[i].num,
					  &param_layer[i].group,
					  &param_layer[i].ocn,
					  &param_layer[i].outH,
					  &param_layer[i].outW,
					  &param_layer[i].kerH,
					  &param_layer[i].kerW,
					  &param_layer[i].bias ) != 16) {
			printf("Invalid format\n");
			break;
		}
		i++;
	}
	fclose(fp);
}

void GetMaxSize(int *insize, int *outsize, int *weightsize, int *biassize, p_layer_t param_layer[]) {
	int i,j;
	int tmp[4];
	int max[4];

	max[0] = 0;
	max[1] = 0;
	max[2] = 0;
	max[3] = 0;
	for (i = 0; i < LAYERNUM; i++) {
		tmp[0] = param_layer[i].icn*param_layer[i].inpH*param_layer[i].inpW;
		tmp[1] = param_layer[i].ocn*param_layer[i].outH*param_layer[i].outW;
		if ( strstr(param_layer[i].name, "conv") != NULL){
			tmp[2] = param_layer[i].icn*param_layer[i].kerH*param_layer[i].kerW*param_layer[i].ocn;
		} else if ( strstr(param_layer[i].name, "fc") != NULL){
			tmp[2] = param_layer[i].icn*param_layer[i].inpH*param_layer[i].inpW*param_layer[i].ocn*param_layer[i].outH*param_layer[i].outW;
		} else {
			tmp[2] = 0;
		}
		tmp[3] = param_layer[i].ocn;
		for(j = 0; j < 4; j++) {
			if ( max[j] < tmp[j] ) max[j] = tmp[j];
		}
	}
	*insize     = max[0];
	*outsize    = max[1];
	*weightsize = max[2];
	*biassize   = max[3];

	return;

}

int GetSize(char const *inout, int layernum, p_layer_t param_layer[]) {

	if ( strcmp(inout,"in") == 0 ) {
      return (param_layer[layernum].inpH+param_layer[layernum].padH*2)*(param_layer[layernum].inpW+param_layer[layernum].padW*2)*param_layer[layernum].icn;
	} else if ( strcmp(inout,"out") == 0 ) {
      return param_layer[layernum].outH*param_layer[layernum].outW*param_layer[layernum].ocn;
	} else if ( strcmp(inout,"wconv") == 0 ) {
      return param_layer[layernum].kerH*param_layer[layernum].kerW*param_layer[layernum].ocn*param_layer[layernum].icn/param_layer[layernum].group;
	} else if ( strcmp(inout,"wfc") == 0 ) {
      return param_layer[layernum].icn*param_layer[layernum].inpH*param_layer[layernum].inpW*param_layer[layernum].ocn;
	} else if ( strcmp(inout,"in2") == 0 ) {
      return param_layer[layernum].inpH*param_layer[layernum].inpW*param_layer[layernum].icn;
    } else {
      return 0;
    }

}

int ReadWBFILE(float *w, float *b, int num, char const *layer, int layernum, p_layer_t param_layer[]) {

  FILE *fp;
  char str[100];
  int size;
  float *buf;
  int i;

  sprintf(str, "data/w%s%d",layer,num);
  //printf("start ReadWBFIILE %s\n",str);
  if ((fp = fopen(str,"rb")) != NULL) {
    if ( strcmp(layer,"conv") == 0) {
      size = param_layer[layernum].kerH*param_layer[layernum].kerW*param_layer[layernum].ocn*param_layer[layernum].icn/param_layer[layernum].group;
    } else if ( strcmp(layer,"fc") == 0) {
      size = param_layer[layernum].icn*param_layer[layernum].inpH*param_layer[layernum].inpW*param_layer[layernum].ocn;
    }
//#ifndef ARMZYNQ
    fread(w, sizeof(float)*size, 1, fp);
//#else
//    buf = malloc(sizeof(float)*size);
//    printf("malloc %x %d\n",buf, size);
//    fread(buf, sizeof(float)*size, 1, fp);
//    for(i = 0; i < size; i++) {
//      w[i] = buf[i];
//      if ( (i % 1000) == 0 | i > 307000) printf("i = %d\n",i);
//    }
//    printf("copy end\n");
//    free(buf);
//    printf("free end\n");
//#endif
    fclose(fp);
  } else {
    return(-1);
  }

  sprintf(str, "data/b%s%d",layer,num);
  if ((fp = fopen(str,"rb")) != NULL) {
//#ifndef ARMZYNQ
    fread(b, (sizeof(float)*param_layer[layernum].ocn), 1, fp);
//#else
//    printf("malloc2\n");
//    buf = (float*)malloc(sizeof(float)*param_layer[layernum].ocn);
//    printf("malloc %x %d\n",buf, param_layer[layernum].ocn);
//    fread(buf, sizeof(float)*param_layer[layernum].ocn, 1, fp);
//    for(i = 0; i < param_layer[layernum].ocn; i++) {
//      b[i] = buf[i];
//    }
//    free(buf);
//#endif
    fclose(fp);
  } else {
    return(-1);
  }
  return 0;
}

void AddPad(float **in, float **buf, int layernum, p_layer_t param_layer[]) {

  int hsize;
  int wsize;
  int i = 0;
  int c, h, w;
  float *tmp;

  hsize = (param_layer[layernum].padH*2+param_layer[layernum].inpH);
  wsize = (param_layer[layernum].padW*2+param_layer[layernum].inpW);

  printf("Add pad %p %p\n",*in,*buf);  
  for(c = 0; c < param_layer[layernum].icn; c++) {
    for(h = 0; h < hsize; h++) {
      for(w = 0; w < wsize; w++) {
        if(h < param_layer[layernum].padH || h >= (param_layer[layernum].padH+param_layer[layernum].inpH) ||
           w < param_layer[layernum].padW || w >= (param_layer[layernum].padW+param_layer[layernum].inpW) ) {
           *(*buf+h*wsize+w + c*(hsize*wsize)) = 0.0f;
        } else {
           *(*buf+h*wsize+w + c*(hsize*wsize)) = *(*in+i);
           i++;
        }
      }
    }
  }

  tmp = *in;
  *in = *buf;  
  *buf = tmp;
  return;

}

void AddPad_malloc(float **in, int layernum, p_layer_t param_layer[]) {

  int hsize;
  int wsize;
  int i = 0;
  int c, h, w;
  float *buf;

  hsize = (param_layer[layernum].padH*2+param_layer[layernum].inpH);
  wsize = (param_layer[layernum].padW*2+param_layer[layernum].inpW);
  buf   = (void *)malloc(hsize*wsize*param_layer[layernum].icn*sizeof(float));
  
  for(c = 0; c < param_layer[layernum].icn; c++) {
    for(h = 0; h < hsize; h++) {
      for(w = 0; w < wsize; w++) {
        if(h < param_layer[layernum].padH || h >= (param_layer[layernum].padH+param_layer[layernum].inpH) ||
           w < param_layer[layernum].padW || w >= (param_layer[layernum].padW+param_layer[layernum].inpW) ) {
           buf[h*wsize+w + c*(hsize*wsize)] = 0.0f;
        } else {
           buf[h*wsize+w + c*(hsize*wsize)] = *(*in+i);
           i++;
        }
      }
    }
  }

  *in = buf;  
  //FILE *fp;
  //fp = fopen("apb","wb");
  //fwrite(buf,sizeof(float)*param_layer[layernum].icn*hsize*wsize, 1, fp);
  //fclose(fp);
  //exit(1);
  return;

}

void AddPad_emax(float *in, float *out, int layernum, p_layer_t param_layer[]) {

  int hsize;
  int wsize;
  int i = 0;
  int c, h, w;
  float *buf;

  hsize = (param_layer[layernum].padH*2+param_layer[layernum].inpH);
  wsize = (param_layer[layernum].padW*2+param_layer[layernum].inpW);
  
  for(c = 0; c < param_layer[layernum].icn; c++) {
    for(h = 0; h < hsize; h++) {
      for(w = 0; w < wsize; w++) {
        if(h < param_layer[layernum].padH || h >= (param_layer[layernum].padH+param_layer[layernum].inpH) ||
           w < param_layer[layernum].padW || w >= (param_layer[layernum].padW+param_layer[layernum].inpW) ) {
           out[h*wsize+w + c*(hsize*wsize)] = 0.0f;
        } else {
           out[h*wsize+w + c*(hsize*wsize)] = in[i];
           i++;
        }
      }
    }
  }

  return;

}

void AddPad_Split4KB(float *in, float *out, int layernum, p_layer_t param_layer[]) {

  int hsize;
  int wsize;
  int i = 0;
  int c, h, w, p;
  int parallel = 1;
#ifdef TIMEMEASURE_APAD
  struct timeval s, e;

  gettimeofday(&s, NULL);
#endif
  hsize = (param_layer[layernum].padH*2+param_layer[layernum].inpH);
  wsize = (param_layer[layernum].padW*2+param_layer[layernum].inpW);


  //printf("Add pad\n");  
  switch(layernum) {
    case 8:
    case 10:
    case 12:
      parallel = 4;
    break;
  }
  for(c = 0; c < param_layer[layernum].icn/parallel; c++) {
    for(p =0 ; p < parallel; p++) {
      for(h = 0; h < hsize; h++) {
        for(w = 0; w < wsize; w++) {
          if(h < param_layer[layernum].padH || h >= (param_layer[layernum].padH+param_layer[layernum].inpH) ||
             w < param_layer[layernum].padW || w >= (param_layer[layernum].padW+param_layer[layernum].inpW) ) {
             out[h*wsize + w + p*(hsize*wsize) + c*1024] = 0.0f;
          } else {
             out[h*wsize + w + p*(hsize*wsize) + c*1024] = in[i];
             i++;
          }
        }
      }
    }
  }
  //FILE *fp;
  //fp = fopen("apa_emax","wb");
  //fwrite(out,sizeof(float)*param_layer[layernum].icn*1024, 1, fp);
  //fclose(fp);
  //exit(1);
#ifdef TIMEMEASURE_APAD
  gettimeofday(&e, NULL);
  printf("AddPad_Split4KB %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
#endif
  return;

}

void Pack_For_Cpu(float *in, float *out, int layernum, p_layer_t param_layer[]) {

  int hsize;
  int wsize;
  int i = 0;
  int c, h, w, p;
  int parallel = 1;
#ifdef TIMEMEASURE_PACK
  struct timeval s, e;

  gettimeofday(&s, NULL);
#endif

  hsize = param_layer[layernum].outH;
  wsize = param_layer[layernum].outW;


  //printf("Pack\n");  
  switch(layernum) {
    case 8:
    case 10:
    case 12:
      parallel = 4;
    break;
  }
  for(c = 0; c < param_layer[layernum].ocn/parallel; c++) {
    for(p =0 ; p < parallel; p++) {
      for(h = 0; h < hsize; h++) {
        for(w = 0; w < wsize; w++) {
          out[i] = in[h*wsize + w +  p*(hsize*wsize) + c*1024];
          i++;
        }
      }
    }
  }
  //printf("%d \n",i);
#ifdef TIMEMEASURE_PACK
  gettimeofday(&e, NULL);
  printf("Pack_For_Cpu %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
#endif

  return;

}

void relu(float *in, float *out, float negative_slope, int layernum, p_layer_t param_layer[]) {

  p_layer_t p;
  p = param_layer[layernum];
  int n,i;
  int size;
  size = p.ocn*p.outH*p.outW;

  for (n = 0; n < p.num; n++) {
    for (i = 0; i < size; i++) {
      out[i + n * size] = (in[i + n * size] >= 0.0f) ? in[i + n * size] : negative_slope * in[i + n * size];
    } 
  }

  return;
}
void Emax2Cpu(float *in, float *out, int layernum, p_layer_t param_layer[]) {

  int hsize;
  int wsize;
  int i = 0;
  int c, h, w;
#ifdef TIMEMEASURE_PACK
  struct timeval s, e;

  gettimeofday(&s, NULL);
#endif

  hsize = param_layer[layernum].outH;
  wsize = param_layer[layernum].outW;


  for(c = 0; c < param_layer[layernum].ocn; c++) {
    for(h = 0; h < hsize; h++) {
      for(w = 0; w < wsize; w++) {
        out[i] = in[h*wsize + w +  c*(hsize*wsize)];
        i++;
      }
    }
  }
  //printf("%d \n",i);
#ifdef TIMEMEASURE_PACK
  gettimeofday(&e, NULL);
  printf("Emax2Cput %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
#endif

  return;

}

void pool(float *in, float *out, int layernum, p_layer_t param_layer[]) {

  p_layer_t p;
  p = param_layer[layernum];
  int n, c, ph, pw, h, w;
  int isize, osize;
  float *srcData, *dstData;
  int poolIndex;
  int hstart;
  int wstart;
  int hend;
  int wend;
  float max_val;
  int index, index2;

  isize = GetSize( "in", layernum, param_layer);
  osize = GetSize("out", layernum, param_layer);

  for (n = 0; n < p.num; n++) {
    for (c = 0; c < p.icn; c++) {
      srcData =  &in[n * isize + c * p.inpH * p.inpW];
      dstData = &out[n * osize + c * p.outH * p.outW];

      poolIndex = 0;
      for (ph = 0; ph < p.outH; ++ph) {
        for (pw = 0; pw < p.outW; ++pw) {
          hstart = ph * p.strideH - p.padH;   // pad対応未
          wstart = pw * p.strideW - p.padW; // pad対応未
          hend = min(hstart + p.kerH, p.inpH);
          wend = min(wstart + p.kerW, p.inpW);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          max_val = -FLT_MAX;

          //printf("H[%d,%d]\n",hstart,hend);
          //printf("W[%d,%d]\n",wstart,wend);
          index = hstart * p.inpW;
          for (h = hstart; h < hend; ++h){
            index2 = index + wstart;
            for (w = wstart; w < wend; ++w) {
              if (srcData[index2] > max_val)
                max_val = srcData[index2];
              ++index2;
            }
            index += p.inpW;
          }
          dstData[poolIndex] = max_val;
          ++poolIndex;
        }
      }
    }
  }

  return;

}

void lrn(float *in, float *out, int size, float alpha, float beta, int layernum, p_layer_t param_layer[]) {

  int isize, osize;
  p_layer_t p;
  p = param_layer[layernum];
  int num = p.num;
  int channels = p.icn;
  int ksize = (size - 1) / 2;
  int n, cn, y, x;
  int index;
  float *srcData;
  float *accData;
  float *dstData;
  accData = (void *)malloc(sizeof(float)*p.outH*p.outW);
  memset(accData, 0x00, sizeof(float)*p.outH*p.outW);

  isize = GetSize( "in", layernum, param_layer);
  osize = GetSize("out", layernum, param_layer);
        
  for (n = 0; n < num; n++) {
    for (cn = 0; cn < min(ksize, channels); cn++) {
      srcData =  &in[n * isize + cn * p.inpH * p.inpW];
      index = 0;
      for (y = 0; y < p.inpH; ++y) {
        for (x = 0; x < p.inpW; ++x) {
          accData[index] += srcData[index] * srcData[index];
          index++;
        }
      }
    }
   

    for (cn = 0; cn < channels; cn++) {
      if (cn + ksize < channels) {
        srcData =  &in[n * isize + (cn + ksize) * p.inpH * p.inpW];
        index = 0;
        for (y = 0; y < p.inpH; ++y) {
          for (x = 0; x < p.inpW; ++x) {
            accData[index] += srcData[index] * srcData[index];
            index++;
          }
        }
      }

      if (cn - ksize - 1 >= 0) {
        srcData =  &in[n * isize + (cn - ksize -1) * p.inpH * p.inpW];
        index = 0;
        for (y = 0; y < p.inpH; ++y) {
          for (x = 0; x < p.inpW; ++x) {
            accData[index] -= srcData[index] * srcData[index];
            index++;
          }
        }
      }
      srcData =  &in[n * isize + cn * p.inpH * p.inpW];
      dstData = &out[n * osize + cn * p.outH * p.outW];
      index = 0;
      for (y = 0; y < p.inpH; ++y) {
        for (x = 0; x < p.inpW; ++x) {
          dstData[index] = 1 + alpha/size * accData[index];
          //dstData[index] = std::pow(dstData[index], beta);
          dstData[index] = expf(beta * logf(dstData[index])); //x^y = exp(y * ln(x))
          dstData[index] = srcData[index] / dstData[index];
          //printf("dst src acc %f %f %f \n",dstData[index],srcData[index],accData[index]);
          index++;
        }
      }
    }
  }
  free(accData);
  return;

}

void fc(float *in, float *w, float *b, float *out, int layernum, p_layer_t param_layer[]) {

  p_layer_t p;
  p = param_layer[layernum];
  int n, ocn, icn;
  float *p_src, *p_dst, *p_wgt;
  int isize = GetSize( "in", layernum, param_layer);
  int osize = GetSize("out", layernum, param_layer);
  int wsize = GetSize("wfc", layernum, param_layer);
  float src;

  for (n = 0; n < p.num; n++) {
    for (ocn = 0; ocn < p.ocn; ocn++) {
      p_src = &in[n * isize];
      p_dst = &out[n * osize];
      p_wgt = &w[n * wsize + ocn * p.inpH * p.inpW * p.icn];
      for (icn = 0; icn < p.inpH * p.inpW * p.icn; icn++) {
        if (icn == 0) {
          p_dst[ocn] = p_src[icn] * p_wgt[icn];  
        } else {
          p_dst[ocn] += p_src[icn] * p_wgt[icn];  
        }
      }
      if (p.bias) {
        p_dst[ocn] += b[n * p.ocn + ocn];
      }
    }
  }


  return;
}

void sigmoid(float *in, float *out, int layernum, p_layer_t param_layer[]) {

  p_layer_t p;
  p = param_layer[layernum];
  int n,i;
  int size;
  size = p.ocn * p.outH * p.outW;

  for (n = 0; n < p.num; n++) {
    for (i = 0; i < size; i++) {
      out[i + n * size] = 1.0f / (1.0f + expf(-in[i]));
    } 
  }

  return;
}

#ifdef EMAX
void monitor_float(int i, Ull r0, Ull r1, Ull r2, Ull r3) {

  union { Uint i; float f; } f3, f2, f1, f0;

  f0.i = (Uint)(r0);
  f1.i = (Uint)(r1);
  f2.i = (Uint)(r2);
  f3.i = (Uint)(r3);
  printf("FL1F [%d] %f %f %f %f\n",i++,f0.f, f1.f, f2.f, f3.f);
  printf("FL1I [%d] %x %x %x %x\n",i++,f0.i, f1.i, f2.i, f3.i);
  f0.i = (Uint)(r0>>32);
  f1.i = (Uint)(r1>>32);
  f2.i = (Uint)(r2>>32);
  f3.i = (Uint)(r3>>32);
  //printf("[%d] %f %f %f %f\n",i,f0.f, f1.f, f2.f, f3.f);

}
void monitor_float2(int i, Ull r0, Ull r1, Ull r2, Ull r3) {

  union { Uint i; float f; } f3, f2, f1, f0;

  f0.i = (Uint)(r0);
  f1.i = (Uint)(r1);
  f2.i = (Uint)(r2);
  f3.i = (Uint)(r3);
  printf("FL2F [%d] %f %f %f %f\n",i  ,f0.f, f1.f, f2.f, f3.f);
  printf("FL2I [%d] %x %x %x %x\n",i++,f0.i, f1.i, f2.i, f3.i);

  f0.i = (Uint)(r0>>32);
  f1.i = (Uint)(r1>>32);
  f2.i = (Uint)(r2>>32);
  f3.i = (Uint)(r3>>32);
  //printf("[%d] %f %f %f %f\n",i,f0.f, f1.f, f2.f, f3.f);

}
void monitor_fsum(int i, Ull r1, Ull r2, Ull r3, Ull r4) {

  union { Uint i; float f; } f4, f3, f2, f1, f0;

  f1.i = (Uint)(r1);
  f2.i = (Uint)(r2);
  f3.i = (Uint)(r3);
  f4.i = (Uint)(r4);
  f0.f = f1.f + f2.f + f3.f +f4.f;
  printf("FS [%d] %f %f %f %f %f\n",i,f0.f, f1.f, f2.f, f3.f, f4.f);
  //f1.i = (Uint)(r1>>32);
  //f2.i = (Uint)(r2>>32);
  //f3.i = (Uint)(r3>>32);
  //f0.f += f1.f + f2.f + f3.f;
  //printf("[%d] %f %f %f %f\n",i,f0.f, f1.f, f2.f, f3.f);

}
#endif


#ifdef EMAX
conv1_emax(float *in, float *wgt, float bias, float *out, int iwd, int stridew, int owd, int icn)
{
  //printf("start conv1_emax\n");
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  // Line00
  Uint *in00_00    = (Uint*)&in[iwd*0];
  Uint *in00_00mm = (Uint*)&in[iwd*0      -4];
  Uint *in00_01mm = (Uint*)&in[iwd*0 + 1  -4];
  Uint *in00_02mm = (Uint*)&in[iwd*0 + 2  -4];
  Uint *in00_03mm = (Uint*)&in[iwd*0 + 3  -4];
  Uint *in00_04mm = (Uint*)&in[iwd*0 + 4  -4];
  Uint *in00_05mm = (Uint*)&in[iwd*0 + 5  -4];
  Uint *in00_06mm = (Uint*)&in[iwd*0 + 6  -4];
  Uint *in00_07mm = (Uint*)&in[iwd*0 + 7  -4];
  Uint *in00_08mm = (Uint*)&in[iwd*0 + 8  -4];
  Uint *in00_09mm = (Uint*)&in[iwd*0 + 9  -4];
  Uint *in00_10mm = (Uint*)&in[iwd*0 + 10 -4];
  Uint wgt00_00   = (Uint)*(Uint*)&wgt[0];
  Uint wgt00_01   = (Uint)*(Uint*)&wgt[1];
  Uint wgt00_02   = (Uint)*(Uint*)&wgt[2];
  Uint wgt00_03   = (Uint)*(Uint*)&wgt[3];
  Uint wgt00_04   = (Uint)*(Uint*)&wgt[4];
  Uint wgt00_05   = (Uint)*(Uint*)&wgt[5];
  Uint wgt00_06   = (Uint)*(Uint*)&wgt[6];
  Uint wgt00_07   = (Uint)*(Uint*)&wgt[7];
  Uint wgt00_08   = (Uint)*(Uint*)&wgt[8];
  Uint wgt00_09   = (Uint)*(Uint*)&wgt[9];
  Uint wgt00_10   = (Uint)*(Uint*)&wgt[10];
  // Line01
  Uint *in01_00   = (Uint*)&in[iwd*1];
  Uint *in01_00mm = (Uint*)&in[iwd*1      -4];
  Uint *in01_01mm = (Uint*)&in[iwd*1 + 1  -4];
  Uint *in01_02mm = (Uint*)&in[iwd*1 + 2  -4];
  Uint *in01_03mm = (Uint*)&in[iwd*1 + 3  -4];
  Uint *in01_04mm = (Uint*)&in[iwd*1 + 4  -4];
  Uint *in01_05mm = (Uint*)&in[iwd*1 + 5  -4];
  Uint *in01_06mm = (Uint*)&in[iwd*1 + 6  -4];
  Uint *in01_07mm = (Uint*)&in[iwd*1 + 7  -4];
  Uint *in01_08mm = (Uint*)&in[iwd*1 + 8  -4];
  Uint *in01_09mm = (Uint*)&in[iwd*1 + 9  -4];
  Uint *in01_10mm = (Uint*)&in[iwd*1 + 10 -4];
  Uint wgt01_00   = (Uint)*(Uint*)&wgt[11];
  Uint wgt01_01   = (Uint)*(Uint*)&wgt[12];
  Uint wgt01_02   = (Uint)*(Uint*)&wgt[13];
  Uint wgt01_03   = (Uint)*(Uint*)&wgt[14];
  Uint wgt01_04   = (Uint)*(Uint*)&wgt[15];
  Uint wgt01_05   = (Uint)*(Uint*)&wgt[16];
  Uint wgt01_06   = (Uint)*(Uint*)&wgt[17];
  Uint wgt01_07   = (Uint)*(Uint*)&wgt[18];
  Uint wgt01_08   = (Uint)*(Uint*)&wgt[19];
  Uint wgt01_09   = (Uint)*(Uint*)&wgt[20];
  Uint wgt01_10   = (Uint)*(Uint*)&wgt[21];
  // Line02
  Uint *in02_00   = (Uint*)&in[iwd*2];
  Uint *in02_00mm = (Uint*)&in[iwd*2      -4];
  Uint *in02_01mm = (Uint*)&in[iwd*2 + 1  -4];
  Uint *in02_02mm = (Uint*)&in[iwd*2 + 2  -4];
  Uint *in02_03mm = (Uint*)&in[iwd*2 + 3  -4];
  Uint *in02_04mm = (Uint*)&in[iwd*2 + 4  -4];
  Uint *in02_05mm = (Uint*)&in[iwd*2 + 5  -4];
  Uint *in02_06mm = (Uint*)&in[iwd*2 + 6  -4];
  Uint *in02_07mm = (Uint*)&in[iwd*2 + 7  -4];
  Uint *in02_08mm = (Uint*)&in[iwd*2 + 8  -4];
  Uint *in02_09mm = (Uint*)&in[iwd*2 + 9  -4];
  Uint *in02_10mm = (Uint*)&in[iwd*2 + 10 -4];
  Uint wgt02_00   = (Uint)*(Uint*)&wgt[22];
  Uint wgt02_01   = (Uint)*(Uint*)&wgt[23];
  Uint wgt02_02   = (Uint)*(Uint*)&wgt[24];
  Uint wgt02_03   = (Uint)*(Uint*)&wgt[25];
  Uint wgt02_04   = (Uint)*(Uint*)&wgt[26];
  Uint wgt02_05   = (Uint)*(Uint*)&wgt[27];
  Uint wgt02_06   = (Uint)*(Uint*)&wgt[28];
  Uint wgt02_07   = (Uint)*(Uint*)&wgt[29];
  Uint wgt02_08   = (Uint)*(Uint*)&wgt[30];
  Uint wgt02_09   = (Uint)*(Uint*)&wgt[31];
  Uint wgt02_10   = (Uint)*(Uint*)&wgt[32];
  // Line03
  Uint *in03_00   = (Uint*)&in[iwd*3];
  Uint *in03_00mm = (Uint*)&in[iwd*3      -4];
  Uint *in03_01mm = (Uint*)&in[iwd*3 + 1  -4];
  Uint *in03_02mm = (Uint*)&in[iwd*3 + 2  -4];
  Uint *in03_03mm = (Uint*)&in[iwd*3 + 3  -4];
  Uint *in03_04mm = (Uint*)&in[iwd*3 + 4  -4];
  Uint *in03_05mm = (Uint*)&in[iwd*3 + 5  -4];
  Uint *in03_06mm = (Uint*)&in[iwd*3 + 6  -4];
  Uint *in03_07mm = (Uint*)&in[iwd*3 + 7  -4];
  Uint *in03_08mm = (Uint*)&in[iwd*3 + 8  -4];
  Uint *in03_09mm = (Uint*)&in[iwd*3 + 9  -4];
  Uint *in03_10mm = (Uint*)&in[iwd*3 + 10 -4];
  Uint wgt03_00   = (Uint)*(Uint*)&wgt[33];
  Uint wgt03_01   = (Uint)*(Uint*)&wgt[34];
  Uint wgt03_02   = (Uint)*(Uint*)&wgt[35];
  Uint wgt03_03   = (Uint)*(Uint*)&wgt[36];
  Uint wgt03_04   = (Uint)*(Uint*)&wgt[37];
  Uint wgt03_05   = (Uint)*(Uint*)&wgt[38];
  Uint wgt03_06   = (Uint)*(Uint*)&wgt[39];
  Uint wgt03_07   = (Uint)*(Uint*)&wgt[40];
  Uint wgt03_08   = (Uint)*(Uint*)&wgt[41];
  Uint wgt03_09   = (Uint)*(Uint*)&wgt[42];
  Uint wgt03_10   = (Uint)*(Uint*)&wgt[43];
  // Line04
  Uint *in04_00   = (Uint*)&in[iwd*4];
  Uint *in04_00mm = (Uint*)&in[iwd*4      -4];
  Uint *in04_01mm = (Uint*)&in[iwd*4 + 1  -4];
  Uint *in04_02mm = (Uint*)&in[iwd*4 + 2  -4];
  Uint *in04_03mm = (Uint*)&in[iwd*4 + 3  -4];
  Uint *in04_04mm = (Uint*)&in[iwd*4 + 4  -4];
  Uint *in04_05mm = (Uint*)&in[iwd*4 + 5  -4];
  Uint *in04_06mm = (Uint*)&in[iwd*4 + 6  -4];
  Uint *in04_07mm = (Uint*)&in[iwd*4 + 7  -4];
  Uint *in04_08mm = (Uint*)&in[iwd*4 + 8  -4];
  Uint *in04_09mm = (Uint*)&in[iwd*4 + 9  -4];
  Uint *in04_10mm = (Uint*)&in[iwd*4 + 10 -4];
  Uint wgt04_00   = (Uint)*(Uint*)&wgt[44];
  Uint wgt04_01   = (Uint)*(Uint*)&wgt[45];
  Uint wgt04_02   = (Uint)*(Uint*)&wgt[46];
  Uint wgt04_03   = (Uint)*(Uint*)&wgt[47];
  Uint wgt04_04   = (Uint)*(Uint*)&wgt[48];
  Uint wgt04_05   = (Uint)*(Uint*)&wgt[49];
  Uint wgt04_06   = (Uint)*(Uint*)&wgt[50];
  Uint wgt04_07   = (Uint)*(Uint*)&wgt[51];
  Uint wgt04_08   = (Uint)*(Uint*)&wgt[52];
  Uint wgt04_09   = (Uint)*(Uint*)&wgt[53];
  Uint wgt04_10   = (Uint)*(Uint*)&wgt[54];
  // Line05
  Uint *in05_00   = (Uint*)&in[iwd*5];
  Uint *in05_00mm = (Uint*)&in[iwd*5      -4];
  Uint *in05_01mm = (Uint*)&in[iwd*5 + 1  -4];
  Uint *in05_02mm = (Uint*)&in[iwd*5 + 2  -4];
  Uint *in05_03mm = (Uint*)&in[iwd*5 + 3  -4];
  Uint *in05_04mm = (Uint*)&in[iwd*5 + 4  -4];
  Uint *in05_05mm = (Uint*)&in[iwd*5 + 5  -4];
  Uint *in05_06mm = (Uint*)&in[iwd*5 + 6  -4];
  Uint *in05_07mm = (Uint*)&in[iwd*5 + 7  -4];
  Uint *in05_08mm = (Uint*)&in[iwd*5 + 8  -4];
  Uint *in05_09mm = (Uint*)&in[iwd*5 + 9  -4];
  Uint *in05_10mm = (Uint*)&in[iwd*5 + 10 -4];
  Uint wgt05_00   = (Uint)*(Uint*)&wgt[55];
  Uint wgt05_01   = (Uint)*(Uint*)&wgt[56];
  Uint wgt05_02   = (Uint)*(Uint*)&wgt[57];
  Uint wgt05_03   = (Uint)*(Uint*)&wgt[58];
  Uint wgt05_04   = (Uint)*(Uint*)&wgt[59];
  Uint wgt05_05   = (Uint)*(Uint*)&wgt[60];
  Uint wgt05_06   = (Uint)*(Uint*)&wgt[61];
  Uint wgt05_07   = (Uint)*(Uint*)&wgt[62];
  Uint wgt05_08   = (Uint)*(Uint*)&wgt[63];
  Uint wgt05_09   = (Uint)*(Uint*)&wgt[64];
  Uint wgt05_10   = (Uint)*(Uint*)&wgt[65];
  // Line06
  Uint *in06_00   = (Uint*)&in[iwd*6];
  Uint *in06_00mm = (Uint*)&in[iwd*6      -4];
  Uint *in06_01mm = (Uint*)&in[iwd*6 + 1  -4];
  Uint *in06_02mm = (Uint*)&in[iwd*6 + 2  -4];
  Uint *in06_03mm = (Uint*)&in[iwd*6 + 3  -4];
  Uint *in06_04mm = (Uint*)&in[iwd*6 + 4  -4];
  Uint *in06_05mm = (Uint*)&in[iwd*6 + 5  -4];
  Uint *in06_06mm = (Uint*)&in[iwd*6 + 6  -4];
  Uint *in06_07mm = (Uint*)&in[iwd*6 + 7  -4];
  Uint *in06_08mm = (Uint*)&in[iwd*6 + 8  -4];
  Uint *in06_09mm = (Uint*)&in[iwd*6 + 9  -4];
  Uint *in06_10mm = (Uint*)&in[iwd*6 + 10 -4];
  Uint wgt06_00   = (Uint)*(Uint*)&wgt[66];
  Uint wgt06_01   = (Uint)*(Uint*)&wgt[67];
  Uint wgt06_02   = (Uint)*(Uint*)&wgt[68];
  Uint wgt06_03   = (Uint)*(Uint*)&wgt[69];
  Uint wgt06_04   = (Uint)*(Uint*)&wgt[70];
  Uint wgt06_05   = (Uint)*(Uint*)&wgt[71];
  Uint wgt06_06   = (Uint)*(Uint*)&wgt[72];
  Uint wgt06_07   = (Uint)*(Uint*)&wgt[73];
  Uint wgt06_08   = (Uint)*(Uint*)&wgt[74];
  Uint wgt06_09   = (Uint)*(Uint*)&wgt[75];
  Uint wgt06_10   = (Uint)*(Uint*)&wgt[76];
  // Line07
  Uint *in07_00   = (Uint*)&in[iwd*7];
  Uint *in07_00mm = (Uint*)&in[iwd*7      -4];
  Uint *in07_01mm = (Uint*)&in[iwd*7 + 1  -4];
  Uint *in07_02mm = (Uint*)&in[iwd*7 + 2  -4];
  Uint *in07_03mm = (Uint*)&in[iwd*7 + 3  -4];
  Uint *in07_04mm = (Uint*)&in[iwd*7 + 4  -4];
  Uint *in07_05mm = (Uint*)&in[iwd*7 + 5  -4];
  Uint *in07_06mm = (Uint*)&in[iwd*7 + 6  -4];
  Uint *in07_07mm = (Uint*)&in[iwd*7 + 7  -4];
  Uint *in07_08mm = (Uint*)&in[iwd*7 + 8  -4];
  Uint *in07_09mm = (Uint*)&in[iwd*7 + 9  -4];
  Uint *in07_10mm = (Uint*)&in[iwd*7 + 10 -4];
  Uint wgt07_00   = (Uint)*(Uint*)&wgt[77];
  Uint wgt07_01   = (Uint)*(Uint*)&wgt[78];
  Uint wgt07_02   = (Uint)*(Uint*)&wgt[79];
  Uint wgt07_03   = (Uint)*(Uint*)&wgt[80];
  Uint wgt07_04   = (Uint)*(Uint*)&wgt[81];
  Uint wgt07_05   = (Uint)*(Uint*)&wgt[82];
  Uint wgt07_06   = (Uint)*(Uint*)&wgt[83];
  Uint wgt07_07   = (Uint)*(Uint*)&wgt[84];
  Uint wgt07_08   = (Uint)*(Uint*)&wgt[85];
  Uint wgt07_09   = (Uint)*(Uint*)&wgt[86];
  Uint wgt07_10   = (Uint)*(Uint*)&wgt[87];
  // Line08
  Uint *in08_00   = (Uint*)&in[iwd*8];
  Uint *in08_00mm = (Uint*)&in[iwd*8      -4];
  Uint *in08_01mm = (Uint*)&in[iwd*8 + 1  -4];
  Uint *in08_02mm = (Uint*)&in[iwd*8 + 2  -4];
  Uint *in08_03mm = (Uint*)&in[iwd*8 + 3  -4];
  Uint *in08_04mm = (Uint*)&in[iwd*8 + 4  -4];
  Uint *in08_05mm = (Uint*)&in[iwd*8 + 5  -4];
  Uint *in08_06mm = (Uint*)&in[iwd*8 + 6  -4];
  Uint *in08_07mm = (Uint*)&in[iwd*8 + 7  -4];
  Uint *in08_08mm = (Uint*)&in[iwd*8 + 8  -4];
  Uint *in08_09mm = (Uint*)&in[iwd*8 + 9  -4];
  Uint *in08_10mm = (Uint*)&in[iwd*8 + 10 -4];
  Uint wgt08_00   = (Uint)*(Uint*)&wgt[88];
  Uint wgt08_01   = (Uint)*(Uint*)&wgt[89];
  Uint wgt08_02   = (Uint)*(Uint*)&wgt[90];
  Uint wgt08_03   = (Uint)*(Uint*)&wgt[91];
  Uint wgt08_04   = (Uint)*(Uint*)&wgt[92];
  Uint wgt08_05   = (Uint)*(Uint*)&wgt[93];
  Uint wgt08_06   = (Uint)*(Uint*)&wgt[94];
  Uint wgt08_07   = (Uint)*(Uint*)&wgt[95];
  Uint wgt08_08   = (Uint)*(Uint*)&wgt[96];
  Uint wgt08_09   = (Uint)*(Uint*)&wgt[97];
  Uint wgt08_10   = (Uint)*(Uint*)&wgt[98];
  // Line09
  Uint *in09_00   = (Uint*)&in[iwd*9];
  Uint *in09_00mm = (Uint*)&in[iwd*9      -4];
  Uint *in09_01mm = (Uint*)&in[iwd*9 + 1  -4];
  Uint *in09_02mm = (Uint*)&in[iwd*9 + 2  -4];
  Uint *in09_03mm = (Uint*)&in[iwd*9 + 3  -4];
  Uint *in09_04mm = (Uint*)&in[iwd*9 + 4  -4];
  Uint *in09_05mm = (Uint*)&in[iwd*9 + 5  -4];
  Uint *in09_06mm = (Uint*)&in[iwd*9 + 6  -4];
  Uint *in09_07mm = (Uint*)&in[iwd*9 + 7  -4];
  Uint *in09_08mm = (Uint*)&in[iwd*9 + 8  -4];
  Uint *in09_09mm = (Uint*)&in[iwd*9 + 9  -4];
  Uint *in09_10mm = (Uint*)&in[iwd*9 + 10 -4];
  Uint wgt09_00   = (Uint)*(Uint*)&wgt[99];
  Uint wgt09_01   = (Uint)*(Uint*)&wgt[100];
  Uint wgt09_02   = (Uint)*(Uint*)&wgt[101];
  Uint wgt09_03   = (Uint)*(Uint*)&wgt[102];
  Uint wgt09_04   = (Uint)*(Uint*)&wgt[103];
  Uint wgt09_05   = (Uint)*(Uint*)&wgt[104];
  Uint wgt09_06   = (Uint)*(Uint*)&wgt[105];
  Uint wgt09_07   = (Uint)*(Uint*)&wgt[106];
  Uint wgt09_08   = (Uint)*(Uint*)&wgt[107];
  Uint wgt09_09   = (Uint)*(Uint*)&wgt[108];
  Uint wgt09_10   = (Uint)*(Uint*)&wgt[109];
  // Line10
  Uint *in10_00   = (Uint*)&in[iwd*10];
  Uint *in10_00mm = (Uint*)&in[iwd*10      -4];
  Uint *in10_01mm = (Uint*)&in[iwd*10 + 1  -4];
  Uint *in10_02mm = (Uint*)&in[iwd*10 + 2  -4];
  Uint *in10_03mm = (Uint*)&in[iwd*10 + 3  -4];
  Uint *in10_04mm = (Uint*)&in[iwd*10 + 4  -4];
  Uint *in10_05mm = (Uint*)&in[iwd*10 + 5  -4];
  Uint *in10_06mm = (Uint*)&in[iwd*10 + 6  -4];
  Uint *in10_07mm = (Uint*)&in[iwd*10 + 7  -4];
  Uint *in10_08mm = (Uint*)&in[iwd*10 + 8  -4];
  Uint *in10_09mm = (Uint*)&in[iwd*10 + 9  -4];
  Uint *in10_10mm = (Uint*)&in[iwd*10 + 10 -4];
  Uint wgt10_00   = (Uint)*(Uint*)&wgt[110];
  Uint wgt10_01   = (Uint)*(Uint*)&wgt[111];
  Uint wgt10_02   = (Uint)*(Uint*)&wgt[112];
  Uint wgt10_03   = (Uint)*(Uint*)&wgt[113];
  Uint wgt10_04   = (Uint)*(Uint*)&wgt[114];
  Uint wgt10_05   = (Uint)*(Uint*)&wgt[115];
  Uint wgt10_06   = (Uint)*(Uint*)&wgt[116];
  Uint wgt10_07   = (Uint)*(Uint*)&wgt[117];
  Uint wgt10_08   = (Uint)*(Uint*)&wgt[118];
  Uint wgt10_09   = (Uint)*(Uint*)&wgt[119];
  Uint wgt10_10   = (Uint)*(Uint*)&wgt[120];

  Uint *in11_00    = (Uint*)&in[iwd*11];
  Uint *in12_00    = (Uint*)&in[iwd*12];
  Uint *in13_00    = (Uint*)&in[iwd*13];
  Uint *in14_00    = (Uint*)&in[iwd*14];


  Ull o    = (Ull)out;
  Ull o_mm = (Ull)(out - 1);
  Ull o_pl = (Ull)(out + owd);
  Ull o_ps = (Ull)(out - owd);

  Uint loop = owd;
  Uint str = (Uint)stridew * sizeof(float);
  Uint b   = (Uint)*(Uint*)&bias;


  Uint  x = 0;
  Uint  xol = 0;
  Uint  xos = 0;
  Uint  AR[64][4];                     /* output of EX     in each unit */
  Uint  BR[64][4][4];                  /* output registers in each unit */
  Uint  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Uint  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33;
  int  i = 0;

if ( icn == 0 ) {
  goto conv1_ch0;
} 

//EMAX5A begin conv1 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
#ifdef DEB_NC
    printf("x = %x\n",x);
#endif
    // Line00
    exe(OP_ADD,  &x, x, EXP_H3210, str, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0);       /* stage#0 */
    mop(OP_LDR,  3, &BR[1][0][1], (Uint)in00_00mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r1,          (Uint)in00_01mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r2,          (Uint)in00_02mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r3,          (Uint)in00_03mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    exe(OP_FMA,  &r10, 0, EXP_H3210, BR[1][0][1], EXP_H3210, wgt00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r11, 0, EXP_H3210, r1,          EXP_H3210, wgt00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r12, 0, EXP_H3210, r2,          EXP_H3210, wgt00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r13, 0, EXP_H3210, r3,          EXP_H3210, wgt00_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    mop(OP_LDR,  3, &BR[2][0][1], (Uint)in00_04mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r15,         (Uint)in00_05mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r16,         (Uint)in00_06mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r17,         (Uint)in00_07mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r24,         (Uint)in00_08mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r25,         (Uint)in00_09mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r26,         (Uint)in00_10mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[2][0][1], EXP_H3210, wgt00_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,         EXP_H3210, wgt00_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,         EXP_H3210, wgt00_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,         EXP_H3210, wgt00_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,         EXP_H3210, wgt00_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,         EXP_H3210, wgt00_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,         EXP_H3210, wgt00_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line01
    mop(OP_LDR,  3, &BR[4][0][1], (Uint)in01_00mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r1,          (Uint)in01_01mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r2,          (Uint)in01_02mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r3,          (Uint)in01_03mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[4][0][1], EXP_H3210, wgt01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,          EXP_H3210, wgt01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,          EXP_H3210, wgt01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,          EXP_H3210, wgt01_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    mop(OP_LDR,  3, &BR[5][0][1], (Uint)in01_04mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r15,         (Uint)in01_05mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r16,         (Uint)in01_06mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r17,         (Uint)in01_07mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r24,         (Uint)in01_08mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r25,         (Uint)in01_09mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r26,         (Uint)in01_10mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[5][0][1], EXP_H3210, wgt01_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,         EXP_H3210, wgt01_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,         EXP_H3210, wgt01_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,         EXP_H3210, wgt01_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,         EXP_H3210, wgt01_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,         EXP_H3210, wgt01_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,         EXP_H3210, wgt01_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line02
    mop(OP_LDR,  3, &BR[7][0][1], (Uint)in02_00mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r1,          (Uint)in02_01mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r2,          (Uint)in02_02mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r3,          (Uint)in02_03mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[7][0][1], EXP_H3210, wgt02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,          EXP_H3210, wgt02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,          EXP_H3210, wgt02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,          EXP_H3210, wgt02_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    mop(OP_LDR,  3, &BR[8][0][1], (Uint)in02_04mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r15,         (Uint)in02_05mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r16,         (Uint)in02_06mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r17,         (Uint)in02_07mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r24,         (Uint)in02_08mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r25,         (Uint)in02_09mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r26,         (Uint)in02_10mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[8][0][1], EXP_H3210, wgt02_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,         EXP_H3210, wgt02_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,         EXP_H3210, wgt02_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,         EXP_H3210, wgt02_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,         EXP_H3210, wgt02_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,         EXP_H3210, wgt02_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,         EXP_H3210, wgt02_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line03
    mop(OP_LDR,  3, &BR[10][0][1], (Uint)in03_00mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r1,          (Uint)in03_01mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r2,          (Uint)in03_02mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r3,          (Uint)in03_03mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[10][0][1], EXP_H3210, wgt03_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt03_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt03_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt03_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    mop(OP_LDR,  3, &BR[11][0][1], (Uint)in03_04mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r15,          (Uint)in03_05mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r16,          (Uint)in03_06mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r17,          (Uint)in03_07mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r24,          (Uint)in03_08mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r25,          (Uint)in03_09mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r26,          (Uint)in03_10mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[11][0][1], EXP_H3210, wgt03_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt03_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt03_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt03_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt03_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt03_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt03_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line04
    mop(OP_LDR,  3, &BR[13][0][1], (Uint)in04_00mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r1,           (Uint)in04_01mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r2,           (Uint)in04_02mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r3,           (Uint)in04_03mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[13][0][1], EXP_H3210, wgt04_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt04_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt04_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt04_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    mop(OP_LDR,  3, &BR[14][0][1], (Uint)in04_04mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r15,          (Uint)in04_05mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r16,          (Uint)in04_06mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r17,          (Uint)in04_07mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r24,          (Uint)in04_08mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r25,          (Uint)in04_09mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r26,          (Uint)in04_10mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[14][0][1], EXP_H3210, wgt04_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt04_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt04_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt04_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt04_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt04_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt04_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line05
    mop(OP_LDR,  3, &BR[16][0][1], (Uint)in05_00mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r1,           (Uint)in05_01mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r2,           (Uint)in05_02mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r3,           (Uint)in05_03mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[16][0][1], EXP_H3210, wgt05_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt05_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt05_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt05_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    mop(OP_LDR,  3, &BR[17][0][1], (Uint)in05_04mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r15,          (Uint)in05_05mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r16,          (Uint)in05_06mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r17,          (Uint)in05_07mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r24,          (Uint)in05_08mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r25,          (Uint)in05_09mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r26,          (Uint)in05_10mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[17][0][1], EXP_H3210, wgt05_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt05_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt05_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt05_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt05_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt05_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt05_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line06
    mop(OP_LDR,  3, &BR[19][0][1], (Uint)in06_00mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r1,           (Uint)in06_01mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r2,           (Uint)in06_02mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r3,           (Uint)in06_03mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[19][0][1], EXP_H3210, wgt06_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt06_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt06_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt06_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    mop(OP_LDR,  3, &BR[20][0][1], (Uint)in06_04mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r15,          (Uint)in06_05mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r16,          (Uint)in06_06mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r17,          (Uint)in06_07mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r24,          (Uint)in06_08mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r25,          (Uint)in06_09mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r26,          (Uint)in06_10mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[20][0][1], EXP_H3210, wgt06_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt06_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt06_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt06_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt06_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt06_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt06_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line07
    mop(OP_LDR,  3, &BR[22][0][1], (Uint)in07_00mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r1,           (Uint)in07_01mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r2,           (Uint)in07_02mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r3,           (Uint)in07_03mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[22][0][1], EXP_H3210, wgt07_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt07_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt07_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt07_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    mop(OP_LDR,  3, &BR[23][0][1], (Uint)in07_04mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r15,          (Uint)in07_05mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r16,          (Uint)in07_06mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r17,          (Uint)in07_07mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r24,          (Uint)in07_08mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r25,          (Uint)in07_09mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r26,          (Uint)in07_10mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[23][0][1], EXP_H3210, wgt07_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt07_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt07_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt07_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt07_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt07_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt07_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line08
    mop(OP_LDR,  3, &BR[25][0][1], (Uint)in08_00mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r1,           (Uint)in08_01mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r2,           (Uint)in08_02mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r3,           (Uint)in08_03mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[25][0][1], EXP_H3210, wgt08_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt08_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt08_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt08_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    mop(OP_LDR,  3, &BR[26][0][1], (Uint)in08_04mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r15,          (Uint)in08_05mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r16,          (Uint)in08_06mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r17,          (Uint)in08_07mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r24,          (Uint)in08_08mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r25,          (Uint)in08_09mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r26,          (Uint)in08_10mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[26][0][1], EXP_H3210, wgt08_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt08_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt08_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt08_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt08_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt08_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt08_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line09
    mop(OP_LDR,  3, &BR[28][0][1], (Uint)in09_00mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r1,           (Uint)in09_01mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r2,           (Uint)in09_02mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r3,           (Uint)in09_03mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[28][0][1], EXP_H3210, wgt09_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt09_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt09_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt09_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    mop(OP_LDR,  3, &BR[29][0][1], (Uint)in09_04mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r15,          (Uint)in09_05mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r16,          (Uint)in09_06mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r17,          (Uint)in09_07mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r24,          (Uint)in09_08mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r25,          (Uint)in09_09mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r26,          (Uint)in09_10mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[29][0][1], EXP_H3210, wgt09_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt09_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt09_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt09_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt09_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt09_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt09_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line10
    mop(OP_LDR,  3, &BR[31][0][1], (Uint)in10_00mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_01mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_02mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r3,           (Uint)in10_03mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[31][0][1], EXP_H3210, wgt10_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt10_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt10_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt10_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    mop(OP_LDR,  3, &BR[32][0][1], (Uint)in10_04mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r15,          (Uint)in10_05mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r16,          (Uint)in10_06mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r17,          (Uint)in10_07mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r24,          (Uint)in10_08mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r25,          (Uint)in10_09mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r26,          (Uint)in10_10mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[32][0][1], EXP_H3210, wgt10_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt10_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt10_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt10_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt10_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt10_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt10_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, b,            EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
#ifdef DEB_NC
    monitor_float(88, b, 0, 0, 0);
#endif
    
    exe(OP_FAD,  &r10, r30, EXP_H3210, r31, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    exe(OP_FAD,  &r11, r32, EXP_H3210, r33, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */

    exe(OP_ADD,  &xol, xol, EXP_H3210, 4, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    
    exe(OP_FAD,  &r20, r10, EXP_H3210, r11, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    mop(OP_LDR,  3, &BR[36][0][1], (Uint)o_mm, xol, MSK_W0, (Ull)o, 0, 55, 0, (Ull)NULL); /* stage#36 */
#ifdef DEB_NC
    monitor_float(99, BR[36][0][1], 0, 0, 0);
#endif
    //exe(OP_FAD,  &r10, BR[36][0][1], EXP_H3210, 0, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    //exe(OP_FAD,  &r11, r20,          EXP_H3210, 0, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    //exe(OP_FAD,  &r12, r10,          EXP_H3210, 0, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    //exe(OP_FAD,  &r13, r11,          EXP_H3210, 0, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_ADD,  &xos, xos,          EXP_H3210, 4, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    //exe(OP_FAD,  &AR[39][0], r12,    EXP_H3210, r13, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    exe(OP_FAD,  &AR[39][0], r20,    EXP_H3210, BR[36][0][1], EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    mop(OP_STR,  3, &AR[39][0], (Uint)o_mm, xos, MSK_W0, (Ull)o, 0, 55, 0, (Ull)NULL); /* stage#39 */
#ifdef DEB_NC
    monitor_float(i, AR[39][0], 0, 0, 0);
    i=0;
#endif
  }
//EMAX5A end

  return(0);

conv1_ch0:
{}


//EMAX5A begin conv1_ch0 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
#ifdef DEB_NC
    printf("x = %x\n",x);
#endif
    // Line00
    exe(OP_ADD,  &x, x, EXP_H3210, str, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0);       /* stage#0 */
    mop(OP_LDR,  3, &BR[1][0][1], (Uint)in00_00mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r1,          (Uint)in00_01mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r2,          (Uint)in00_02mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r3,          (Uint)in00_03mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#1 */
    exe(OP_FMA,  &r10, 0, EXP_H3210, BR[1][0][1], EXP_H3210, wgt00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r11, 0, EXP_H3210, r1,          EXP_H3210, wgt00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r12, 0, EXP_H3210, r2,          EXP_H3210, wgt00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r13, 0, EXP_H3210, r3,          EXP_H3210, wgt00_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    mop(OP_LDR,  3, &BR[2][0][1], (Uint)in00_04mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r15,         (Uint)in00_05mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r16,         (Uint)in00_06mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r17,         (Uint)in00_07mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r24,         (Uint)in00_08mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r25,         (Uint)in00_09mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r26,         (Uint)in00_10mm, x, MSK_W0, (Ull)in00_00, 0, 228, 0, (Ull)NULL); /* stage#2 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[2][0][1], EXP_H3210, wgt00_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,         EXP_H3210, wgt00_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,         EXP_H3210, wgt00_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,         EXP_H3210, wgt00_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,         EXP_H3210, wgt00_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,         EXP_H3210, wgt00_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,         EXP_H3210, wgt00_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line01
    mop(OP_LDR,  3, &BR[4][0][1], (Uint)in01_00mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r1,          (Uint)in01_01mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r2,          (Uint)in01_02mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r3,          (Uint)in01_03mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#4 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[4][0][1], EXP_H3210, wgt01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,          EXP_H3210, wgt01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,          EXP_H3210, wgt01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,          EXP_H3210, wgt01_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    mop(OP_LDR,  3, &BR[5][0][1], (Uint)in01_04mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r15,         (Uint)in01_05mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r16,         (Uint)in01_06mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r17,         (Uint)in01_07mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r24,         (Uint)in01_08mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r25,         (Uint)in01_09mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r26,         (Uint)in01_10mm, x, MSK_W0, (Ull)in01_00, 0, 228, 0, (Ull)NULL); /* stage#5 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[5][0][1], EXP_H3210, wgt01_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,         EXP_H3210, wgt01_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,         EXP_H3210, wgt01_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,         EXP_H3210, wgt01_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,         EXP_H3210, wgt01_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,         EXP_H3210, wgt01_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,         EXP_H3210, wgt01_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line02
    mop(OP_LDR,  3, &BR[7][0][1], (Uint)in02_00mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r1,          (Uint)in02_01mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r2,          (Uint)in02_02mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r3,          (Uint)in02_03mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#7 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[7][0][1], EXP_H3210, wgt02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,          EXP_H3210, wgt02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,          EXP_H3210, wgt02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,          EXP_H3210, wgt02_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    mop(OP_LDR,  3, &BR[8][0][1], (Uint)in02_04mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r15,         (Uint)in02_05mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r16,         (Uint)in02_06mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r17,         (Uint)in02_07mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r24,         (Uint)in02_08mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r25,         (Uint)in02_09mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r26,         (Uint)in02_10mm, x, MSK_W0, (Ull)in02_00, 0, 228, 0, (Ull)NULL); /* stage#8 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[8][0][1], EXP_H3210, wgt02_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,         EXP_H3210, wgt02_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,         EXP_H3210, wgt02_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,         EXP_H3210, wgt02_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,         EXP_H3210, wgt02_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,         EXP_H3210, wgt02_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,         EXP_H3210, wgt02_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line03
    mop(OP_LDR,  3, &BR[10][0][1], (Uint)in03_00mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r1,          (Uint)in03_01mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r2,          (Uint)in03_02mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r3,          (Uint)in03_03mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#10 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[10][0][1], EXP_H3210, wgt03_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt03_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt03_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt03_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    mop(OP_LDR,  3, &BR[11][0][1], (Uint)in03_04mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r15,          (Uint)in03_05mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r16,          (Uint)in03_06mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r17,          (Uint)in03_07mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r24,          (Uint)in03_08mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r25,          (Uint)in03_09mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r26,          (Uint)in03_10mm, x, MSK_W0, (Ull)in03_00, 0, 228, 0, (Ull)NULL); /* stage#11 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[11][0][1], EXP_H3210, wgt03_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt03_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt03_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt03_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt03_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt03_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt03_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line04
    mop(OP_LDR,  3, &BR[13][0][1], (Uint)in04_00mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r1,           (Uint)in04_01mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r2,           (Uint)in04_02mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r3,           (Uint)in04_03mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#13 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[13][0][1], EXP_H3210, wgt04_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt04_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt04_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt04_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    mop(OP_LDR,  3, &BR[14][0][1], (Uint)in04_04mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r15,          (Uint)in04_05mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r16,          (Uint)in04_06mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r17,          (Uint)in04_07mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r24,          (Uint)in04_08mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r25,          (Uint)in04_09mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r26,          (Uint)in04_10mm, x, MSK_W0, (Ull)in04_00, 0, 228, 0, (Ull)NULL); /* stage#14 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[14][0][1], EXP_H3210, wgt04_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt04_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt04_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt04_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt04_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt04_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt04_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line05
    mop(OP_LDR,  3, &BR[16][0][1], (Uint)in05_00mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r1,           (Uint)in05_01mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r2,           (Uint)in05_02mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r3,           (Uint)in05_03mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#16 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[16][0][1], EXP_H3210, wgt05_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt05_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt05_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt05_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    mop(OP_LDR,  3, &BR[17][0][1], (Uint)in05_04mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r15,          (Uint)in05_05mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r16,          (Uint)in05_06mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r17,          (Uint)in05_07mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r24,          (Uint)in05_08mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r25,          (Uint)in05_09mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r26,          (Uint)in05_10mm, x, MSK_W0, (Ull)in05_00, 0, 228, 0, (Ull)NULL); /* stage#17 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[17][0][1], EXP_H3210, wgt05_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt05_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt05_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt05_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt05_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt05_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt05_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line06
    mop(OP_LDR,  3, &BR[19][0][1], (Uint)in06_00mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r1,           (Uint)in06_01mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r2,           (Uint)in06_02mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r3,           (Uint)in06_03mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#19 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[19][0][1], EXP_H3210, wgt06_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt06_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt06_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt06_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    mop(OP_LDR,  3, &BR[20][0][1], (Uint)in06_04mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r15,          (Uint)in06_05mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r16,          (Uint)in06_06mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r17,          (Uint)in06_07mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r24,          (Uint)in06_08mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r25,          (Uint)in06_09mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r26,          (Uint)in06_10mm, x, MSK_W0, (Ull)in06_00, 0, 228, 0, (Ull)NULL); /* stage#20 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[20][0][1], EXP_H3210, wgt06_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt06_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt06_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt06_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt06_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt06_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt06_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line07
    mop(OP_LDR,  3, &BR[22][0][1], (Uint)in07_00mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r1,           (Uint)in07_01mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r2,           (Uint)in07_02mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r3,           (Uint)in07_03mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#22 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[22][0][1], EXP_H3210, wgt07_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt07_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt07_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt07_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    mop(OP_LDR,  3, &BR[23][0][1], (Uint)in07_04mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r15,          (Uint)in07_05mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r16,          (Uint)in07_06mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r17,          (Uint)in07_07mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r24,          (Uint)in07_08mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r25,          (Uint)in07_09mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r26,          (Uint)in07_10mm, x, MSK_W0, (Ull)in07_00, 0, 228, 0, (Ull)NULL); /* stage#23 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[23][0][1], EXP_H3210, wgt07_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt07_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt07_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt07_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt07_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt07_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt07_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line08
    mop(OP_LDR,  3, &BR[25][0][1], (Uint)in08_00mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r1,           (Uint)in08_01mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r2,           (Uint)in08_02mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r3,           (Uint)in08_03mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#25 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[25][0][1], EXP_H3210, wgt08_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt08_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt08_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt08_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    mop(OP_LDR,  3, &BR[26][0][1], (Uint)in08_04mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r15,          (Uint)in08_05mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r16,          (Uint)in08_06mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r17,          (Uint)in08_07mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r24,          (Uint)in08_08mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r25,          (Uint)in08_09mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r26,          (Uint)in08_10mm, x, MSK_W0, (Ull)in08_00, 0, 228, 0, (Ull)NULL); /* stage#26 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[26][0][1], EXP_H3210, wgt08_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt08_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt08_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt08_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt08_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt08_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt08_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line09
    mop(OP_LDR,  3, &BR[28][0][1], (Uint)in09_00mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r1,           (Uint)in09_01mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r2,           (Uint)in09_02mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r3,           (Uint)in09_03mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#28 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[28][0][1], EXP_H3210, wgt09_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt09_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt09_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt09_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    mop(OP_LDR,  3, &BR[29][0][1], (Uint)in09_04mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r15,          (Uint)in09_05mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r16,          (Uint)in09_06mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r17,          (Uint)in09_07mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r24,          (Uint)in09_08mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r25,          (Uint)in09_09mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r26,          (Uint)in09_10mm, x, MSK_W0, (Ull)in09_00, 0, 228, 0, (Ull)NULL); /* stage#29 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[29][0][1], EXP_H3210, wgt09_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt09_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt09_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt09_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt09_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt09_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt09_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, 0,          EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
#ifdef DEB_NC
    monitor_fsum(i++, r30, r31, r32, r33);
#endif
    // Line10
    mop(OP_LDR,  3, &BR[31][0][1], (Uint)in10_00mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_01mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_02mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r3,           (Uint)in10_03mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#31 */
    exe(OP_FMA,  &r10, r30, EXP_H3210, BR[31][0][1], EXP_H3210, wgt10_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r11, r31, EXP_H3210, r1,           EXP_H3210, wgt10_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r12, r32, EXP_H3210, r2,           EXP_H3210, wgt10_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r13, r33, EXP_H3210, r3,           EXP_H3210, wgt10_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    mop(OP_LDR,  3, &BR[32][0][1], (Uint)in10_04mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r15,          (Uint)in10_05mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r16,          (Uint)in10_06mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r17,          (Uint)in10_07mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r24,          (Uint)in10_08mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r25,          (Uint)in10_09mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r26,          (Uint)in10_10mm, x, MSK_W0, (Ull)in10_00, 0, 228, 0, (Ull)NULL); /* stage#32 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[32][0][1], EXP_H3210, wgt10_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r15,          EXP_H3210, wgt10_05, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r16,          EXP_H3210, wgt10_06, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r17,          EXP_H3210, wgt10_07, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r30, r20, EXP_H3210, r24,          EXP_H3210, wgt10_08, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r25,          EXP_H3210, wgt10_09, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r26,          EXP_H3210, wgt10_10, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FAD,  &r33, r23, EXP_H3210, b,            EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
#ifdef DEB_NC
    monitor_float(88, b, 0, 0, 0);
#endif
    
    exe(OP_FAD,  &r10, r30, EXP_H3210, r31, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    exe(OP_FAD,  &r11, r32, EXP_H3210, r33, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */

    exe(OP_ADD,  &xos, xos, EXP_H3210, 4, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    
    exe(OP_FAD,  &AR[36][0], r10, EXP_H3210, r11, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    mop(OP_STR,  3, &AR[36][0], (Uint)o_mm, xos, MSK_W0, (Ull)o, 0, 55, 0LL, (Ull)NULL); /* stage#36 */
#ifdef DEB_NC
    monitor_float(i, AR[37][0], 0, 0, 0);
    i=0;
#endif
  }
//EMAX5A end

  return(0);
}
#endif

#ifdef EMAX
conv2_emax(float *in, float *wgt, float bias, float *out, int iwd, int stridew, int owd, int icn, int kw)
{
  //printf("conv2emax %p %p\n",in,out);
#if DEB_NC
  int j, k;
  for ( j = 0; j < 5; j++ ) {
    for ( k = 0; k < 5; k++ ) {
      printf("[%d, %d] %f\n", j, k, in[iwd*j + k]);
    }
  }
#endif
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  int ch;
  ch = 0;
  // CH0 Line00
  Uint *in00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint *in00_03mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 3  -1];
  Uint *in00_04mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 4  -1];
  Uint wgt00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  Uint wgt00_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt00_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  // CH0 Line01
  Uint *in01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint *in01_03mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 3  -1];
  Uint *in01_04mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 4  -1];
  Uint wgt01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  Uint wgt01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt01_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];
  Uint wgt01_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 9];
  // CH0 Line02
  Uint *in02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint *in02_03mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 3  -1];
  Uint *in02_04mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 4  -1];
  Uint wgt02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 10];
  Uint wgt02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 11];
  Uint wgt02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 12];
  Uint wgt02_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 13];
  Uint wgt02_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 14];
  // CH0 Line03
  Uint *in03_00   = (Uint*)&in[iwd*iwd*ch + iwd*3];
  Uint *in03_00mm = (Uint*)&in[iwd*iwd*ch + iwd*3      -1];
  Uint *in03_01mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 1  -1];
  Uint *in03_02mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 2  -1];
  Uint *in03_03mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 3  -1];
  Uint *in03_04mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 4  -1];
  Uint wgt03_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 15];
  Uint wgt03_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 16];
  Uint wgt03_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 17];
  Uint wgt03_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 18];
  Uint wgt03_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 19];
  // CH0 Line04
  Uint *in04_00   = (Uint*)&in[iwd*iwd*ch + iwd*4];
  Uint *in04_00mm = (Uint*)&in[iwd*iwd*ch + iwd*4      -1];
  Uint *in04_01mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 1  -1];
  Uint *in04_02mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 2  -1];
  Uint *in04_03mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 3  -1];
  Uint *in04_04mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 4  -1];
  Uint wgt04_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 20];
  Uint wgt04_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 21];
  Uint wgt04_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 22];
  Uint wgt04_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 23];
  Uint wgt04_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 24];

  ch = 1;
  // CH1 Line00
  Uint *in10_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in10_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in10_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in10_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint *in10_03mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 3  -1];
  Uint *in10_04mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 4  -1];
  Uint wgt10_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt10_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt10_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  Uint wgt10_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt10_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  // CH1 Line01
  Uint *in11_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in11_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in11_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in11_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint *in11_03mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 3  -1];
  Uint *in11_04mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 4  -1];
  Uint wgt11_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  Uint wgt11_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt11_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt11_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];
  Uint wgt11_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 9];
  // CH1 Line02
  Uint *in12_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in12_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in12_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in12_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint *in12_03mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 3  -1];
  Uint *in12_04mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 4  -1];
  Uint wgt12_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 10];
  Uint wgt12_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 11];
  Uint wgt12_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 12];
  Uint wgt12_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 13];
  Uint wgt12_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 14];
  // CH1 Line03
  Uint *in13_00   = (Uint*)&in[iwd*iwd*ch + iwd*3];
  Uint *in13_00mm = (Uint*)&in[iwd*iwd*ch + iwd*3      -1];
  Uint *in13_01mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 1  -1];
  Uint *in13_02mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 2  -1];
  Uint *in13_03mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 3  -1];
  Uint *in13_04mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 4  -1];
  Uint wgt13_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 15];
  Uint wgt13_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 16];
  Uint wgt13_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 17];
  Uint wgt13_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 18];
  Uint wgt13_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 19];
  // CH1 Line04
  Uint *in14_00   = (Uint*)&in[iwd*iwd*ch + iwd*4];
  Uint *in14_00mm = (Uint*)&in[iwd*iwd*ch + iwd*4      -1];
  Uint *in14_01mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 1  -1];
  Uint *in14_02mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 2  -1];
  Uint *in14_03mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 3  -1];
  Uint *in14_04mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 4  -1];
  Uint wgt14_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 20];
  Uint wgt14_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 21];
  Uint wgt14_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 22];
  Uint wgt14_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 23];
  Uint wgt14_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 24];

  ch = 2;
  // CH2 Line00
  Uint *in20_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in20_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in20_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in20_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint *in20_03mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 3  -1];
  Uint *in20_04mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 4  -1];
  Uint wgt20_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt20_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt20_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  Uint wgt20_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt20_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  // CH2 Line01
  Uint *in21_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in21_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in21_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in21_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint *in21_03mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 3  -1];
  Uint *in21_04mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 4  -1];
  Uint wgt21_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  Uint wgt21_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt21_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt21_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];
  Uint wgt21_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 9];
  // CH2 Line02
  Uint *in22_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in22_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in22_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in22_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint *in22_03mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 3  -1];
  Uint *in22_04mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 4  -1];
  Uint wgt22_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 10];
  Uint wgt22_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 11];
  Uint wgt22_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 12];
  Uint wgt22_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 13];
  Uint wgt22_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 14];
  // CH2 Line03
  Uint *in23_00   = (Uint*)&in[iwd*iwd*ch + iwd*3];
  Uint *in23_00mm = (Uint*)&in[iwd*iwd*ch + iwd*3      -1];
  Uint *in23_01mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 1  -1];
  Uint *in23_02mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 2  -1];
  Uint *in23_03mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 3  -1];
  Uint *in23_04mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 4  -1];
  Uint wgt23_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 15];
  Uint wgt23_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 16];
  Uint wgt23_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 17];
  Uint wgt23_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 18];
  Uint wgt23_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 19];
  // CH2 Line04
  Uint *in24_00   = (Uint*)&in[iwd*iwd*ch + iwd*4];
  Uint *in24_00mm = (Uint*)&in[iwd*iwd*ch + iwd*4      -1];
  Uint *in24_01mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 1  -1];
  Uint *in24_02mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 2  -1];
  Uint *in24_03mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 3  -1];
  Uint *in24_04mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 4  -1];
  Uint wgt24_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 20];
  Uint wgt24_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 21];
  Uint wgt24_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 22];
  Uint wgt24_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 23];
  Uint wgt24_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 24];

  ch = 3;
  // CH3 Line00
  Uint *in30_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in30_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in30_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in30_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint *in30_03mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 3  -1];
  Uint *in30_04mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 4  -1];
  Uint wgt30_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt30_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt30_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  Uint wgt30_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt30_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  // CH3 Line01
  Uint *in31_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in31_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in31_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in31_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint *in31_03mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 3  -1];
  Uint *in31_04mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 4  -1];
  Uint wgt31_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  Uint wgt31_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt31_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt31_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];
  Uint wgt31_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 9];
  // CH3 Line02
  Uint *in32_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in32_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in32_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in32_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint *in32_03mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 3  -1];
  Uint *in32_04mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 4  -1];
  Uint wgt32_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 10];
  Uint wgt32_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 11];
  Uint wgt32_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 12];
  Uint wgt32_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 13];
  Uint wgt32_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 14];
  // CH3 Line03
  Uint *in33_00   = (Uint*)&in[iwd*iwd*ch + iwd*3];
  Uint *in33_00mm = (Uint*)&in[iwd*iwd*ch + iwd*3      -1];
  Uint *in33_01mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 1  -1];
  Uint *in33_02mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 2  -1];
  Uint *in33_03mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 3  -1];
  Uint *in33_04mm = (Uint*)&in[iwd*iwd*ch + iwd*3 + 4  -1];
  Uint wgt33_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 15];
  Uint wgt33_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 16];
  Uint wgt33_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 17];
  Uint wgt33_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 18];
  Uint wgt33_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 19];
  // CH3 Line04
  Uint *in34_00   = (Uint*)&in[iwd*iwd*ch + iwd*4];
  Uint *in34_00mm = (Uint*)&in[iwd*iwd*ch + iwd*4      -1];
  Uint *in34_01mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 1  -1];
  Uint *in34_02mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 2  -1];
  Uint *in34_03mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 3  -1];
  Uint *in34_04mm = (Uint*)&in[iwd*iwd*ch + iwd*4 + 4  -1];
  Uint wgt34_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 20];
  Uint wgt34_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 21];
  Uint wgt34_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 22];
  Uint wgt34_03   = (Uint)*(Uint*)&wgt[kw*kw*ch + 23];
  Uint wgt34_04   = (Uint)*(Uint*)&wgt[kw*kw*ch + 24];


  Ull o    = (Ull)out;
  Ull o_mm = (Ull)(out - 1);
  Ull o_pl = (Ull)(out + owd);
  Ull o_ps = (Ull)(out - owd);

  Uint loop = owd;
  Uint str = (Uint)stridew * sizeof(float);
  Uint b   = (Uint)*(Uint*)&bias;

  Uint  x = 0;
  Uint  xo = 0;
  Uint  AR[64][4];                     /* output of EX     in each unit */
  Uint  BR[64][4][4];                  /* output registers in each unit */
  Uint  r0, r1, r2, r3, r4;
  Uint  r00, r01, r02, r03, r04;
  Uint  r10, r11, r12, r13, r14;
  Uint  r20, r21, r22, r23, r24;
  Uint  r30, r31, r32, r33, r34;
  Uint  r40, r41, r42, r43, r44;
  int  i = 0;

if ( icn == 0 ) {
  goto conv2_ch0;
} 

  //printf("start conv2_emax\n");
//EMAX5A begin conv2 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
#ifdef DEB_NC
    printf("x = %x\n",x);
#endif
    exe(OP_ADD,  &x, x, EXP_H3210, str, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0);       /* stage#0 */
    // CH0 Line00
    mop(OP_LDR,  3, &BR[1][0][1], (Uint)in00_00mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r1,          (Uint)in00_01mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r2,          (Uint)in00_02mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r3,          (Uint)in00_03mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r4,          (Uint)in00_04mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    exe(OP_FMA,  &r00, 0LL, EXP_H3210, BR[1][0][1], EXP_H3210, wgt00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r01, 0LL, EXP_H3210, r1,          EXP_H3210, wgt00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r02, 0LL, EXP_H3210, r2,          EXP_H3210, wgt00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r03, 0LL, EXP_H3210, r3,          EXP_H3210, wgt00_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,          EXP_H3210, wgt00_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
#ifdef DEB_NC
    monitor_float(0, BR[1][0][1], wgt00_00, r00, 0);
    monitor_float(1, r1,          wgt00_01, r01, 0);
    monitor_float(2, r2,          wgt00_02, r02, 0);
    monitor_float(3, r3,          wgt00_03, r03, 0);
    monitor_float(4, r4,          wgt00_04, r04, 0);
    monitor_fsum(i++, r01, r02, r03, r04);
#endif
    // CH0 Line01
    mop(OP_LDR,  3, &BR[3][0][1], (Uint)in01_00mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r1,          (Uint)in01_01mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r2,          (Uint)in01_02mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r3,          (Uint)in01_03mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r4,          (Uint)in01_04mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[3][0][1], EXP_H3210, wgt01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,          EXP_H3210, wgt01_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,          EXP_H3210, wgt01_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
#ifdef DEB_NC
    monitor_float(0, BR[3][0][1], wgt01_00, r10, 0);
    monitor_float(1, r1,          wgt01_01, r11, 0);
    monitor_float(2, r2,          wgt01_02, r12, 0);
    monitor_float(3, r3,          wgt01_03, r13, 0);
    monitor_float(4, r4,          wgt01_04, r14, 0);
    monitor_fsum(i++, r11, r12, r13, r14);
#endif
    // CH0 Line02
    mop(OP_LDR,  3, &BR[5][0][1], (Uint)in02_00mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r1,          (Uint)in02_01mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r2,          (Uint)in02_02mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r3,          (Uint)in02_03mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r4,          (Uint)in02_04mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[5][0][1], EXP_H3210, wgt02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,          EXP_H3210, wgt02_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,          EXP_H3210, wgt02_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
#ifdef DEB_NC
    monitor_float(0, BR[5][0][1], wgt02_00, r20, 0);
    monitor_float(1, r1,          wgt02_01, r21, 0);
    monitor_float(2, r2,          wgt02_02, r22, 0);
    monitor_float(3, r3,          wgt02_03, r23, 0);
    monitor_float(4, r4,          wgt02_04, r24, 0);
    monitor_fsum(i++, r21, r22, r23, r24);
#endif
    // CH0 Line03
    mop(OP_LDR,  3, &BR[7][0][1], (Uint)in03_00mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r1,          (Uint)in03_01mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r2,          (Uint)in03_02mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r3,          (Uint)in03_03mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r4,          (Uint)in03_04mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[7][0][1], EXP_H3210, wgt03_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,          EXP_H3210, wgt03_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,          EXP_H3210, wgt03_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,          EXP_H3210, wgt03_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,          EXP_H3210, wgt03_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
#ifdef DEB_NC
    monitor_float(0, BR[7][0][1], wgt03_00, r30, 0);
    monitor_float(1, r1,          wgt03_01, r31, 0);
    monitor_float(2, r2,          wgt03_02, r32, 0);
    monitor_float(3, r3,          wgt03_03, r33, 0);
    monitor_float(4, r4,          wgt03_04, r34, 0);
    monitor_fsum(i++, r31, r32, r33, r34);
#endif
    // CH0 Line04
    mop(OP_LDR,  3, &BR[9][0][1],  (Uint)in04_00mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r1,           (Uint)in04_01mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r2,           (Uint)in04_02mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r3,           (Uint)in04_03mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r4,           (Uint)in04_04mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */

    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[9][0][1], EXP_H3210, wgt04_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,          EXP_H3210, wgt04_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,          EXP_H3210, wgt04_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,          EXP_H3210, wgt04_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,          EXP_H3210, wgt04_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
#ifdef DEB_NC
    monitor_float(0, BR[9][0][1], wgt04_00, r40, 0);
    monitor_float(1, r1,          wgt04_01, r41, 0);
    monitor_float(2, r2,          wgt04_02, r42, 0);
    monitor_float(3, r3,          wgt04_03, r43, 0);
    monitor_float(4, r4,          wgt04_04, r44, 0);
    monitor_fsum(i++, r41, r42, r43, r44);
#endif
    // CH1 Line00
    mop(OP_LDR,  3, &BR[11][0][1], (Uint)in10_00mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_01mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_02mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r3,           (Uint)in10_03mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r4,           (Uint)in10_04mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    exe(OP_FMA,  &r00, r41, EXP_H3210, BR[11][0][1], EXP_H3210, wgt10_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r01, r42, EXP_H3210, r1,           EXP_H3210, wgt10_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r02, r43, EXP_H3210, r2,           EXP_H3210, wgt10_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r03, r44, EXP_H3210, r3,           EXP_H3210, wgt10_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,           EXP_H3210, wgt10_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    // CH1 Line01
    mop(OP_LDR,  3, &BR[13][0][1], (Uint)in11_00mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_01mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_02mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r3,           (Uint)in11_03mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r4,           (Uint)in11_04mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[13][0][1], EXP_H3210, wgt11_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt11_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt11_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,           EXP_H3210, wgt11_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,           EXP_H3210, wgt11_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    // CH1 Line02
    mop(OP_LDR,  3, &BR[15][0][1], (Uint)in12_00mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_01mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_02mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r3,           (Uint)in12_03mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r4,           (Uint)in12_04mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[15][0][1], EXP_H3210, wgt12_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt12_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt12_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,           EXP_H3210, wgt12_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,           EXP_H3210, wgt12_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    // CH1 Line03
    mop(OP_LDR,  3, &BR[17][0][1], (Uint)in13_00mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_01mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_02mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r3,           (Uint)in13_03mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r4,           (Uint)in13_04mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[17][0][1], EXP_H3210, wgt13_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,           EXP_H3210, wgt13_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,           EXP_H3210, wgt13_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,           EXP_H3210, wgt13_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,           EXP_H3210, wgt13_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    // CH1 Line04
    mop(OP_LDR,  3, &BR[19][0][1],  (Uint)in14_00mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r1,            (Uint)in14_01mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r2,            (Uint)in14_02mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r3,            (Uint)in14_03mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r4,            (Uint)in14_04mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[19][0][1], EXP_H3210, wgt14_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,           EXP_H3210, wgt14_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,           EXP_H3210, wgt14_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,           EXP_H3210, wgt14_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,           EXP_H3210, wgt14_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */

    // CH2 Line00
    mop(OP_LDR,  3, &BR[21][0][1], (Uint)in20_00mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r1,           (Uint)in20_01mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r2,           (Uint)in20_02mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r3,           (Uint)in20_03mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r4,           (Uint)in20_04mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    exe(OP_FMA,  &r00, r41, EXP_H3210, BR[21][0][1], EXP_H3210, wgt20_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r01, r42, EXP_H3210, r1,           EXP_H3210, wgt20_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r02, r43, EXP_H3210, r2,           EXP_H3210, wgt20_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r03, r44, EXP_H3210, r3,           EXP_H3210, wgt20_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,           EXP_H3210, wgt20_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    // CH2 Line01
    mop(OP_LDR,  3, &BR[23][0][1], (Uint)in21_00mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r1,           (Uint)in21_01mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r2,           (Uint)in21_02mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r3,           (Uint)in21_03mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r4,           (Uint)in21_04mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[23][0][1], EXP_H3210, wgt21_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt21_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt21_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,           EXP_H3210, wgt21_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,           EXP_H3210, wgt21_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    // CH2 Line02
    mop(OP_LDR,  3, &BR[25][0][1], (Uint)in22_00mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r1,           (Uint)in22_01mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r2,           (Uint)in22_02mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r3,           (Uint)in22_03mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r4,           (Uint)in22_04mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[25][0][1], EXP_H3210, wgt22_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt22_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt22_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,           EXP_H3210, wgt22_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,           EXP_H3210, wgt22_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    // CH2 Line03
    mop(OP_LDR,  3, &BR[27][0][1], (Uint)in23_00mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r1,           (Uint)in23_01mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r2,           (Uint)in23_02mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r3,           (Uint)in23_03mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r4,           (Uint)in23_04mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[27][0][1], EXP_H3210, wgt23_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,           EXP_H3210, wgt23_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,           EXP_H3210, wgt23_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,           EXP_H3210, wgt23_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,           EXP_H3210, wgt23_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    // CH2 Line04
    mop(OP_LDR,  3, &BR[29][0][1],  (Uint)in24_00mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r1,            (Uint)in24_01mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r2,            (Uint)in24_02mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r3,            (Uint)in24_03mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r4,            (Uint)in24_04mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[29][0][1], EXP_H3210, wgt24_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,           EXP_H3210, wgt24_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,           EXP_H3210, wgt24_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,           EXP_H3210, wgt24_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,           EXP_H3210, wgt24_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */

    // CH3 Line00
    mop(OP_LDR,  3, &BR[31][0][1], (Uint)in30_00mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r1,           (Uint)in30_01mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r2,           (Uint)in30_02mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r3,           (Uint)in30_03mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r4,           (Uint)in30_04mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    exe(OP_FMA,  &r00, r41, EXP_H3210, BR[31][0][1], EXP_H3210, wgt30_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r01, r42, EXP_H3210, r1,           EXP_H3210, wgt30_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r02, r43, EXP_H3210, r2,           EXP_H3210, wgt30_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r03, r44, EXP_H3210, r3,           EXP_H3210, wgt30_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,           EXP_H3210, wgt30_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    // CH3 Line01
    mop(OP_LDR,  3, &BR[33][0][1], (Uint)in31_00mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r1,           (Uint)in31_01mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r2,           (Uint)in31_02mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r3,           (Uint)in31_03mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r4,           (Uint)in31_04mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[33][0][1], EXP_H3210, wgt31_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt31_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt31_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,           EXP_H3210, wgt31_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,           EXP_H3210, wgt31_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    // CH3 Line02
    mop(OP_LDR,  3, &BR[35][0][1], (Uint)in32_00mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r1,           (Uint)in32_01mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r2,           (Uint)in32_02mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r3,           (Uint)in32_03mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r4,           (Uint)in32_04mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[35][0][1], EXP_H3210, wgt32_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt32_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt32_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,           EXP_H3210, wgt32_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,           EXP_H3210, wgt32_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    // CH3 Line03
    mop(OP_LDR,  3, &BR[37][0][1], (Uint)in33_00mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r1,           (Uint)in33_01mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r2,           (Uint)in33_02mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r3,           (Uint)in33_03mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r4,           (Uint)in33_04mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[37][0][1], EXP_H3210, wgt33_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,           EXP_H3210, wgt33_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,           EXP_H3210, wgt33_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,           EXP_H3210, wgt33_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,           EXP_H3210, wgt33_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    // CH3 Line04
    mop(OP_LDR,  3, &BR[39][0][1],  (Uint)in34_00mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r1,            (Uint)in34_01mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r2,            (Uint)in34_02mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r3,            (Uint)in34_03mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r4,            (Uint)in34_04mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[39][0][1], EXP_H3210, wgt34_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,           EXP_H3210, wgt34_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,           EXP_H3210, wgt34_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,           EXP_H3210, wgt34_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,           EXP_H3210, wgt34_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */

    exe(OP_FAD,  &r00, r41, EXP_H3210, r42,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    exe(OP_FAD,  &r01, r43, EXP_H3210, b,           EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */

    exe(OP_FAD,  &r02, r00, EXP_H3210, r01,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */

    exe(OP_ADD,  &xo,  xo,  EXP_H3210, 4LL,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */

    mop(OP_LDR,  3, &BR[43][0][1], (Uint)o_mm, xo, MSK_W0, (Ull)o, 0, 27, 0, (Ull)NULL); /* stage#43 */

    exe(OP_FAD,  &r10, r02, EXP_H3210, r44,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */

    exe(OP_FAD,  &AR[44][0], r10,    EXP_H3210, BR[43][0][1], EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    mop(OP_STR,   3, &AR[44][0],    (Uint)o_mm, xo, MSK_W0, (Ull)o, 0, 27, 0, (Ull)NULL); /* stage#44 */
#ifdef DEB_NC
    printf("o = %x \n", o);
    monitor_float2(i, AR[44][0], 0, 0, 0);
    i=0;
#endif
  }
//EMAX5A end

  return(0);

conv2_ch0:
{}
  //printf("start conv2_emax_ch0\n");
//EMAX5A begin conv2_ch0 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
#ifdef DEB_NC
    printf("x = %x\n",x);
#endif
    exe(OP_ADD,  &x, x, EXP_H3210, str, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0);       /* stage#0 */
    // CH0 Line00
    mop(OP_LDR,  3, &BR[1][0][1], (Uint)in00_00mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r1,          (Uint)in00_01mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r2,          (Uint)in00_02mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r3,          (Uint)in00_03mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r4,          (Uint)in00_04mm, x, MSK_W0, (Ull)in00_00, 0, 31, 0, (Ull)NULL); /* stage#1 */
    exe(OP_FMA,  &r00, 0LL, EXP_H3210, BR[1][0][1], EXP_H3210, wgt00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r01, 0LL, EXP_H3210, r1,          EXP_H3210, wgt00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r02, 0LL, EXP_H3210, r2,          EXP_H3210, wgt00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r03, 0LL, EXP_H3210, r3,          EXP_H3210, wgt00_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,          EXP_H3210, wgt00_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
#ifdef DEB_NC
    monitor_float(0, BR[1][0][1], wgt00_00, r00, 0);
    monitor_float(1, r1,          wgt00_01, r01, 0);
    monitor_float(2, r2,          wgt00_02, r02, 0);
    monitor_float(3, r3,          wgt00_03, r03, 0);
    monitor_float(4, r4,          wgt00_04, r04, 0);
    monitor_fsum(i++, r01, r02, r03, r04);
#endif
    // CH0 Line01
    mop(OP_LDR,  3, &BR[3][0][1], (Uint)in01_00mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r1,          (Uint)in01_01mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r2,          (Uint)in01_02mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r3,          (Uint)in01_03mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r4,          (Uint)in01_04mm, x, MSK_W0, (Ull)in01_00, 0, 31, 0, (Ull)NULL); /* stage#3 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[3][0][1], EXP_H3210, wgt01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,          EXP_H3210, wgt01_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,          EXP_H3210, wgt01_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
#ifdef DEB_NC
    monitor_float(0, BR[3][0][1], wgt01_00, r10, 0);
    monitor_float(1, r1,          wgt01_01, r11, 0);
    monitor_float(2, r2,          wgt01_02, r12, 0);
    monitor_float(3, r3,          wgt01_03, r13, 0);
    monitor_float(4, r4,          wgt01_04, r14, 0);
    monitor_fsum(i++, r11, r12, r13, r14);
#endif
    // CH0 Line02
    mop(OP_LDR,  3, &BR[5][0][1], (Uint)in02_00mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r1,          (Uint)in02_01mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r2,          (Uint)in02_02mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r3,          (Uint)in02_03mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r4,          (Uint)in02_04mm, x, MSK_W0, (Ull)in02_00, 0, 31, 0, (Ull)NULL); /* stage#5 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[5][0][1], EXP_H3210, wgt02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,          EXP_H3210, wgt02_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,          EXP_H3210, wgt02_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
#ifdef DEB_NC
    monitor_float(0, BR[5][0][1], wgt02_00, r20, 0);
    monitor_float(1, r1,          wgt02_01, r21, 0);
    monitor_float(2, r2,          wgt02_02, r22, 0);
    monitor_float(3, r3,          wgt02_03, r23, 0);
    monitor_float(4, r4,          wgt02_04, r24, 0);
    monitor_fsum(i++, r21, r22, r23, r24);
#endif
    // CH0 Line03
    mop(OP_LDR,  3, &BR[7][0][1], (Uint)in03_00mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r1,          (Uint)in03_01mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r2,          (Uint)in03_02mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r3,          (Uint)in03_03mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r4,          (Uint)in03_04mm, x, MSK_W0, (Ull)in03_00, 0, 31, 0, (Ull)NULL); /* stage#7 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[7][0][1], EXP_H3210, wgt03_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,          EXP_H3210, wgt03_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,          EXP_H3210, wgt03_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,          EXP_H3210, wgt03_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,          EXP_H3210, wgt03_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
#ifdef DEB_NC
    monitor_float(0, BR[7][0][1], wgt03_00, r30, 0);
    monitor_float(1, r1,          wgt03_01, r31, 0);
    monitor_float(2, r2,          wgt03_02, r32, 0);
    monitor_float(3, r3,          wgt03_03, r33, 0);
    monitor_float(4, r4,          wgt03_04, r34, 0);
    monitor_fsum(i++, r31, r32, r33, r34);
#endif
    // CH0 Line04
    mop(OP_LDR,  3, &BR[9][0][1],  (Uint)in04_00mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r1,           (Uint)in04_01mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r2,           (Uint)in04_02mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r3,           (Uint)in04_03mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r4,           (Uint)in04_04mm, x, MSK_W0, (Ull)in04_00, 0, 31, 0, (Ull)NULL); /* stage#9 */

    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[9][0][1], EXP_H3210, wgt04_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,          EXP_H3210, wgt04_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,          EXP_H3210, wgt04_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,          EXP_H3210, wgt04_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,          EXP_H3210, wgt04_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
#ifdef DEB_NC
    monitor_float(0, BR[9][0][1], wgt04_00, r40, 0);
    monitor_float(1, r1,          wgt04_01, r41, 0);
    monitor_float(2, r2,          wgt04_02, r42, 0);
    monitor_float(3, r3,          wgt04_03, r43, 0);
    monitor_float(4, r4,          wgt04_04, r44, 0);
    monitor_fsum(i++, r41, r42, r43, r44);
#endif
    // CH1 Line00
    mop(OP_LDR,  3, &BR[11][0][1], (Uint)in10_00mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_01mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_02mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r3,           (Uint)in10_03mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r4,           (Uint)in10_04mm, x, MSK_W0, (Ull)in10_00, 0, 31, 0, (Ull)NULL); /* stage#11 */
    exe(OP_FMA,  &r00, r41, EXP_H3210, BR[11][0][1], EXP_H3210, wgt10_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r01, r42, EXP_H3210, r1,           EXP_H3210, wgt10_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r02, r43, EXP_H3210, r2,           EXP_H3210, wgt10_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r03, r44, EXP_H3210, r3,           EXP_H3210, wgt10_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,           EXP_H3210, wgt10_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    // CH1 Line01
    mop(OP_LDR,  3, &BR[13][0][1], (Uint)in11_00mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_01mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_02mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r3,           (Uint)in11_03mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r4,           (Uint)in11_04mm, x, MSK_W0, (Ull)in11_00, 0, 31, 0, (Ull)NULL); /* stage#13 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[13][0][1], EXP_H3210, wgt11_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt11_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt11_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,           EXP_H3210, wgt11_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,           EXP_H3210, wgt11_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    // CH1 Line02
    mop(OP_LDR,  3, &BR[15][0][1], (Uint)in12_00mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_01mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_02mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r3,           (Uint)in12_03mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r4,           (Uint)in12_04mm, x, MSK_W0, (Ull)in12_00, 0, 31, 0, (Ull)NULL); /* stage#15 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[15][0][1], EXP_H3210, wgt12_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt12_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt12_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,           EXP_H3210, wgt12_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,           EXP_H3210, wgt12_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    // CH1 Line03
    mop(OP_LDR,  3, &BR[17][0][1], (Uint)in13_00mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_01mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_02mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r3,           (Uint)in13_03mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r4,           (Uint)in13_04mm, x, MSK_W0, (Ull)in13_00, 0, 31, 0, (Ull)NULL); /* stage#17 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[17][0][1], EXP_H3210, wgt13_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,           EXP_H3210, wgt13_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,           EXP_H3210, wgt13_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,           EXP_H3210, wgt13_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,           EXP_H3210, wgt13_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    // CH1 Line04
    mop(OP_LDR,  3, &BR[19][0][1],  (Uint)in14_00mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r1,            (Uint)in14_01mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r2,            (Uint)in14_02mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r3,            (Uint)in14_03mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r4,            (Uint)in14_04mm, x, MSK_W0, (Ull)in14_00, 0, 31, 0, (Ull)NULL); /* stage#19 */
    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[19][0][1], EXP_H3210, wgt14_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,           EXP_H3210, wgt14_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,           EXP_H3210, wgt14_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,           EXP_H3210, wgt14_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,           EXP_H3210, wgt14_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */

    // CH2 Line00
    mop(OP_LDR,  3, &BR[21][0][1], (Uint)in20_00mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r1,           (Uint)in20_01mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r2,           (Uint)in20_02mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r3,           (Uint)in20_03mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r4,           (Uint)in20_04mm, x, MSK_W0, (Ull)in20_00, 0, 31, 0, (Ull)NULL); /* stage#21 */
    exe(OP_FMA,  &r00, r41, EXP_H3210, BR[21][0][1], EXP_H3210, wgt20_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r01, r42, EXP_H3210, r1,           EXP_H3210, wgt20_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r02, r43, EXP_H3210, r2,           EXP_H3210, wgt20_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r03, r44, EXP_H3210, r3,           EXP_H3210, wgt20_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,           EXP_H3210, wgt20_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    // CH2 Line01
    mop(OP_LDR,  3, &BR[23][0][1], (Uint)in21_00mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r1,           (Uint)in21_01mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r2,           (Uint)in21_02mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r3,           (Uint)in21_03mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r4,           (Uint)in21_04mm, x, MSK_W0, (Ull)in21_00, 0, 31, 0, (Ull)NULL); /* stage#23 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[23][0][1], EXP_H3210, wgt21_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt21_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt21_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,           EXP_H3210, wgt21_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,           EXP_H3210, wgt21_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    // CH2 Line02
    mop(OP_LDR,  3, &BR[25][0][1], (Uint)in22_00mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r1,           (Uint)in22_01mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r2,           (Uint)in22_02mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r3,           (Uint)in22_03mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r4,           (Uint)in22_04mm, x, MSK_W0, (Ull)in22_00, 0, 31, 0, (Ull)NULL); /* stage#25 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[25][0][1], EXP_H3210, wgt22_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt22_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt22_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,           EXP_H3210, wgt22_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,           EXP_H3210, wgt22_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    // CH2 Line03
    mop(OP_LDR,  3, &BR[27][0][1], (Uint)in23_00mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r1,           (Uint)in23_01mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r2,           (Uint)in23_02mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r3,           (Uint)in23_03mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r4,           (Uint)in23_04mm, x, MSK_W0, (Ull)in23_00, 0, 31, 0, (Ull)NULL); /* stage#27 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[27][0][1], EXP_H3210, wgt23_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,           EXP_H3210, wgt23_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,           EXP_H3210, wgt23_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,           EXP_H3210, wgt23_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,           EXP_H3210, wgt23_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    // CH2 Line04
    mop(OP_LDR,  3, &BR[29][0][1],  (Uint)in24_00mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r1,            (Uint)in24_01mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r2,            (Uint)in24_02mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r3,            (Uint)in24_03mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r4,            (Uint)in24_04mm, x, MSK_W0, (Ull)in24_00, 0, 31, 0, (Ull)NULL); /* stage#29 */
    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[29][0][1], EXP_H3210, wgt24_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,           EXP_H3210, wgt24_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,           EXP_H3210, wgt24_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,           EXP_H3210, wgt24_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,           EXP_H3210, wgt24_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */

    // CH3 Line00
    mop(OP_LDR,  3, &BR[31][0][1], (Uint)in30_00mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r1,           (Uint)in30_01mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r2,           (Uint)in30_02mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r3,           (Uint)in30_03mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r4,           (Uint)in30_04mm, x, MSK_W0, (Ull)in30_00, 0, 31, 0, (Ull)NULL); /* stage#31 */
    exe(OP_FMA,  &r00, r41, EXP_H3210, BR[31][0][1], EXP_H3210, wgt30_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r01, r42, EXP_H3210, r1,           EXP_H3210, wgt30_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r02, r43, EXP_H3210, r2,           EXP_H3210, wgt30_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r03, r44, EXP_H3210, r3,           EXP_H3210, wgt30_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r04, r00, EXP_H3210, r4,           EXP_H3210, wgt30_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    // CH3 Line01
    mop(OP_LDR,  3, &BR[33][0][1], (Uint)in31_00mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r1,           (Uint)in31_01mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r2,           (Uint)in31_02mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r3,           (Uint)in31_03mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r4,           (Uint)in31_04mm, x, MSK_W0, (Ull)in31_00, 0, 31, 0, (Ull)NULL); /* stage#33 */
    exe(OP_FMA,  &r10, r04, EXP_H3210, BR[33][0][1], EXP_H3210, wgt31_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt31_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt31_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r13, r03, EXP_H3210, r3,           EXP_H3210, wgt31_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r14, r10, EXP_H3210, r4,           EXP_H3210, wgt31_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    // CH3 Line02
    mop(OP_LDR,  3, &BR[35][0][1], (Uint)in32_00mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r1,           (Uint)in32_01mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r2,           (Uint)in32_02mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r3,           (Uint)in32_03mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r4,           (Uint)in32_04mm, x, MSK_W0, (Ull)in32_00, 0, 31, 0, (Ull)NULL); /* stage#35 */
    exe(OP_FMA,  &r20, r14, EXP_H3210, BR[35][0][1], EXP_H3210, wgt32_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt32_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt32_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r23, r13, EXP_H3210, r3,           EXP_H3210, wgt32_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r24, r20, EXP_H3210, r4,           EXP_H3210, wgt32_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    // CH3 Line03
    mop(OP_LDR,  3, &BR[37][0][1], (Uint)in33_00mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r1,           (Uint)in33_01mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r2,           (Uint)in33_02mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r3,           (Uint)in33_03mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r4,           (Uint)in33_04mm, x, MSK_W0, (Ull)in33_00, 0, 31, 0, (Ull)NULL); /* stage#37 */
    exe(OP_FMA,  &r30, r24, EXP_H3210, BR[37][0][1], EXP_H3210, wgt33_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r31, r21, EXP_H3210, r1,           EXP_H3210, wgt33_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r32, r22, EXP_H3210, r2,           EXP_H3210, wgt33_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r33, r23, EXP_H3210, r3,           EXP_H3210, wgt33_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r34, r30, EXP_H3210, r4,           EXP_H3210, wgt33_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    // CH3 Line04
    mop(OP_LDR,  3, &BR[39][0][1],  (Uint)in34_00mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r1,            (Uint)in34_01mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r2,            (Uint)in34_02mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r3,            (Uint)in34_03mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r4,            (Uint)in34_04mm, x, MSK_W0, (Ull)in34_00, 0, 31, 0, (Ull)NULL); /* stage#39 */
    exe(OP_FMA,  &r40, r34, EXP_H3210, BR[39][0][1], EXP_H3210, wgt34_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r41, r31, EXP_H3210, r1,           EXP_H3210, wgt34_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r42, r32, EXP_H3210, r2,           EXP_H3210, wgt34_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r43, r33, EXP_H3210, r3,           EXP_H3210, wgt34_03, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r44, r40, EXP_H3210, r4,           EXP_H3210, wgt34_04, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */


    exe(OP_FAD,  &r00, r41, EXP_H3210, r42,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    exe(OP_FAD,  &r01, r43, EXP_H3210, b,           EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */

    exe(OP_FAD,  &r02, r00, EXP_H3210, r01,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */

    exe(OP_ADD,  &xo,  xo,  EXP_H3210, 4LL,         EXP_H3210, 0,      EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */

    exe(OP_FAD,  &AR[44][0], r02,    EXP_H3210, r44, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    mop(OP_STR,   3, &AR[44][0],    (Uint)o_mm, xo, MSK_W0, (Ull)o, 0, 27, 0LL, (Ull)NULL); /* stage#44 */
#ifdef DEB_NC
    printf("o = %x \n", o);
    monitor_float2(i, AR[44][0], 0, 0, 0);
    i=0;
#endif
  }
//EMAX5A end

  return(0);
}

conv345_emax(float *in, float *wgt, float bias, float *out, int iwd, int stridew, int owd, int icn, int kw)
{
  //printf("start conv345_emax\n");
#if DEB_NC
  int j, k;
  for ( j = 0; j < 15; j++ ) {
    for ( k = 0; k < 15; k++ ) {
      printf("[%d, %d] %f\n", j, k, in[iwd*j + k]);
    }
  }
#endif
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  int ch;
  ch = 0;
  // CH0 Line00
  Uint *in0_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in0_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in0_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in0_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt0_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt0_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt0_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in0_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in0_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in0_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in0_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt0_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt0_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt0_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in0_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in0_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in0_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in0_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt0_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt0_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt0_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 1;
  // CH1 Line00
  Uint *in1_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in1_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in1_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in1_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt1_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt1_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt1_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in1_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in1_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in1_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in1_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt1_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt1_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt1_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in1_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in1_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in1_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in1_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt1_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt1_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt1_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 2;
  // CH2 Line00
  Uint *in2_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in2_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in2_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in2_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt2_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt2_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt2_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in2_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in2_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in2_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in2_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt2_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt2_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt2_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in2_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in2_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in2_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in2_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt2_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt2_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt2_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 3;
  // CH3 Line00
  Uint *in3_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in3_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in3_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in3_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt3_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt3_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt3_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in3_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in3_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in3_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in3_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt3_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt3_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt3_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in3_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in3_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in3_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in3_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt3_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt3_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt3_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 4;
  // CH4 Line00
  Uint *in4_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in4_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in4_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in4_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt4_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt4_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt4_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in4_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in4_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in4_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in4_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt4_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt4_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt4_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in4_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in4_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in4_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in4_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt4_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt4_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt4_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 5;
  // CH5 Line00
  Uint *in5_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in5_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in5_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in5_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt5_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt5_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt5_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in5_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in5_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in5_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in5_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt5_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt5_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt5_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in5_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in5_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in5_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in5_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt5_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt5_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt5_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 6;
  // CH6 Line00
  Uint *in6_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in6_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in6_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in6_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt6_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt6_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt6_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in6_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in6_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in6_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in6_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt6_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt6_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt6_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in6_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in6_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in6_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in6_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt6_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt6_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt6_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 7;
  // CH7 Line00
  Uint *in7_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in7_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in7_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in7_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt7_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt7_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt7_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in7_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in7_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in7_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in7_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt7_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt7_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt7_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in7_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in7_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in7_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in7_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt7_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt7_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt7_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 8;
  // CH8 Line00
  Uint *in8_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in8_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in8_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in8_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt8_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt8_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt8_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in8_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in8_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in8_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in8_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt8_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt8_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt8_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in8_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in8_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in8_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in8_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt8_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt8_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt8_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 9;
  // CH9 Line00
  Uint *in9_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in9_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in9_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in9_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt9_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt9_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt9_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in9_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in9_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in9_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in9_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt9_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt9_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt9_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in9_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in9_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in9_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in9_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt9_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt9_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt9_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 10;
  // CH10 Line00
  Uint *in10_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in10_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in10_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in10_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt10_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt10_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt10_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in10_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in10_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in10_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in10_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt10_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt10_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt10_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in10_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in10_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in10_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in10_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt10_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt10_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt10_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 11;
  // CH11 Line00
  Uint *in11_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in11_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in11_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in11_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt11_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt11_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt11_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in11_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in11_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in11_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in11_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt11_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt11_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt11_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in11_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in11_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in11_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in11_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt11_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt11_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt11_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 12;
  // CH12 Line00
  Uint *in12_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in12_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in12_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in12_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt12_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt12_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt12_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in12_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in12_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in12_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in12_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt12_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt12_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt12_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in12_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in12_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in12_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in12_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt12_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt12_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt12_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 13;
  // CH13 Line00
  Uint *in13_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in13_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in13_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in13_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt13_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt13_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt13_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in13_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in13_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in13_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in13_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt13_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt13_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt13_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in13_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in13_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in13_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in13_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt13_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt13_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt13_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 14;
  // CH14 Line00
  Uint *in14_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in14_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in14_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in14_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt14_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt14_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt14_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in14_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in14_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in14_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in14_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt14_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt14_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt14_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in14_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in14_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in14_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in14_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt14_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt14_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt14_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  ch = 15;
  // CH15 Line00
  Uint *in15_00_00   = (Uint*)&in[iwd*iwd*ch + iwd*0];
  Uint *in15_00_00mm = (Uint*)&in[iwd*iwd*ch + iwd*0      -1];
  Uint *in15_00_01mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 1  -1];
  Uint *in15_00_02mm = (Uint*)&in[iwd*iwd*ch + iwd*0 + 2  -1];
  Uint wgt15_00_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 0];
  Uint wgt15_00_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 1];
  Uint wgt15_00_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 2];
  // Line01
  Uint *in15_01_00   = (Uint*)&in[iwd*iwd*ch + iwd*1];
  Uint *in15_01_00mm = (Uint*)&in[iwd*iwd*ch + iwd*1      -1];
  Uint *in15_01_01mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 1  -1];
  Uint *in15_01_02mm = (Uint*)&in[iwd*iwd*ch + iwd*1 + 2  -1];
  Uint wgt15_01_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 3];
  Uint wgt15_01_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 4];
  Uint wgt15_01_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 5];
  // Line02
  Uint *in15_02_00   = (Uint*)&in[iwd*iwd*ch + iwd*2];
  Uint *in15_02_00mm = (Uint*)&in[iwd*iwd*ch + iwd*2      -1];
  Uint *in15_02_01mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 1  -1];
  Uint *in15_02_02mm = (Uint*)&in[iwd*iwd*ch + iwd*2 + 2  -1];
  Uint wgt15_02_00   = (Uint)*(Uint*)&wgt[kw*kw*ch + 6];
  Uint wgt15_02_01   = (Uint)*(Uint*)&wgt[kw*kw*ch + 7];
  Uint wgt15_02_02   = (Uint)*(Uint*)&wgt[kw*kw*ch + 8];

  Ull o    = (Ull)out;
  Ull o_mm = (Ull)(out - 1);
  Ull o_pl = (Ull)(out + owd);
  Ull o_ps = (Ull)(out - owd);

  Uint loop = owd;
  Uint str = (Uint)stridew * sizeof(float);
  Uint b   = (Uint)*(Uint*)&bias;

  Uint  x = 0;
  Uint  xo = 0;
  Uint  AR[64][4];                     /* output of EX     in each unit */
  Uint  BR[64][4][4];                  /* output registers in each unit */
  Uint  r0, r1, r2, r3, r4;
  Uint  r00, r01, r02, r03, r04;
  Uint  r10, r11, r12, r13, r14;
  Uint  r20, r21, r22, r23, r24;
  Uint  r30, r31, r32, r33, r34;
  Uint  r40, r41, r42, r43, r44;
  int  i = 0;

if ( icn == 0 ) {
  goto conv345_ch0;
} 

//EMAX5A begin conv345 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
#ifdef DEB_NC
    printf("x = %x\n",x);
#endif
    exe(OP_ADD,  &x, x, EXP_H3210, str, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0);       /* stage#0 */
    // CH0 Line00
    mop(OP_LDR,  3, &BR[1][0][1], (Uint)in0_00_00mm, x, MSK_W0, (Ull)in0_00_00, 0, 15, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r1,          (Uint)in0_00_01mm, x, MSK_W0, (Ull)in0_00_00, 0, 15, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r2,          (Uint)in0_00_02mm, x, MSK_W0, (Ull)in0_00_00, 0, 15, 0, (Ull)NULL); /* stage#1 */
    exe(OP_FMA,  &r00, 0LL, EXP_H3210, BR[1][0][1], EXP_H3210, wgt0_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r01, 0LL, EXP_H3210, r1,          EXP_H3210, wgt0_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r02, 0LL, EXP_H3210, r2,          EXP_H3210, wgt0_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
#ifdef DEB_NC
    monitor_float(0, BR[1][0][1], wgt0_00_00, r00, 0);
    monitor_float(1, r1,          wgt0_00_01, r01, 0);
    monitor_float(2, r2,          wgt0_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH0 Line01
    mop(OP_LDR,  3, &BR[2][0][1], (Uint)in0_01_00mm, x, MSK_W0, (Ull)in0_01_00, 0, 15, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r1,          (Uint)in0_01_01mm, x, MSK_W0, (Ull)in0_01_00, 0, 15, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r2,          (Uint)in0_01_02mm, x, MSK_W0, (Ull)in0_01_00, 0, 15, 0, (Ull)NULL); /* stage#2 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[2][0][1], EXP_H3210, wgt0_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt0_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt0_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
#ifdef DEB_NC
    monitor_float(0, BR[2][0][1], wgt0_01_00, r10, 0);
    monitor_float(1, r1,          wgt0_01_01, r11, 0);
    monitor_float(2, r2,          wgt0_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH0 Line02
    mop(OP_LDR,  3, &BR[3][0][1], (Uint)in0_02_00mm, x, MSK_W0, (Ull)in0_02_00, 0, 15, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r1,          (Uint)in0_02_01mm, x, MSK_W0, (Ull)in0_02_00, 0, 15, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r2,          (Uint)in0_02_02mm, x, MSK_W0, (Ull)in0_02_00, 0, 15, 0, (Ull)NULL); /* stage#3 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[3][0][1], EXP_H3210, wgt0_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt0_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt0_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
#ifdef DEB_NC
    monitor_float(0, BR[3][0][1], wgt0_02_00, r20, 0);
    monitor_float(1, r1,          wgt0_02_01, r21, 0);
    monitor_float(2, r2,          wgt0_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, 0);
#endif
    // CH1 Line00
    mop(OP_LDR,  3, &BR[4][0][1], (Uint)in1_00_00mm, x, MSK_W0, (Ull)in1_00_00, 0, 15, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r1,          (Uint)in1_00_01mm, x, MSK_W0, (Ull)in1_00_00, 0, 15, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r2,          (Uint)in1_00_02mm, x, MSK_W0, (Ull)in1_00_00, 0, 15, 0, (Ull)NULL); /* stage#4 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[4][0][1], EXP_H3210, wgt1_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,          EXP_H3210, wgt1_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,          EXP_H3210, wgt1_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
#ifdef DEB_NC
    monitor_float(0, BR[4][0][1], wgt1_00_00, r00, 0);
    monitor_float(1, r1,          wgt1_00_01, r01, 0);
    monitor_float(2, r2,          wgt1_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH1 Line01
    mop(OP_LDR,  3, &BR[5][0][1], (Uint)in1_01_00mm, x, MSK_W0, (Ull)in1_01_00, 0, 15, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r1,          (Uint)in1_01_01mm, x, MSK_W0, (Ull)in1_01_00, 0, 15, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r2,          (Uint)in1_01_02mm, x, MSK_W0, (Ull)in1_01_00, 0, 15, 0, (Ull)NULL); /* stage#5 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[5][0][1], EXP_H3210, wgt1_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt1_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt1_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
#ifdef DEB_NC
    monitor_float(0, BR[5][0][1], wgt1_01_00, r10, 0);
    monitor_float(1, r1,          wgt1_01_01, r11, 0);
    monitor_float(2, r2,          wgt1_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH1 Line02
    mop(OP_LDR,  3, &BR[6][0][1], (Uint)in1_02_00mm, x, MSK_W0, (Ull)in1_02_00, 0, 15, 0, (Ull)NULL); /* stage#6 */
    mop(OP_LDR,  3, &r1,          (Uint)in1_02_01mm, x, MSK_W0, (Ull)in1_02_00, 0, 15, 0, (Ull)NULL); /* stage#6 */
    mop(OP_LDR,  3, &r2,          (Uint)in1_02_02mm, x, MSK_W0, (Ull)in1_02_00, 0, 15, 0, (Ull)NULL); /* stage#6 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[6][0][1], EXP_H3210, wgt1_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt1_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt1_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
#ifdef DEB_NC
    monitor_float(0, BR[6][0][1], wgt1_02_00, r20, 0);
    monitor_float(1, r1,          wgt1_02_01, r21, 0);
    monitor_float(2, r2,          wgt1_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, 0);
#endif
    // CH2 Line00
    mop(OP_LDR,  3, &BR[7][0][1], (Uint)in2_00_00mm, x, MSK_W0, (Ull)in2_00_00, 0, 15, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r1,          (Uint)in2_00_01mm, x, MSK_W0, (Ull)in2_00_00, 0, 15, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r2,          (Uint)in2_00_02mm, x, MSK_W0, (Ull)in2_00_00, 0, 15, 0, (Ull)NULL); /* stage#7 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[7][0][1], EXP_H3210, wgt2_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,          EXP_H3210, wgt2_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,          EXP_H3210, wgt2_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
#ifdef DEB_NC
    monitor_float(0, BR[7][0][1], wgt2_00_00, r00, 0);
    monitor_float(1, r1,          wgt2_00_01, r01, 0);
    monitor_float(2, r2,          wgt2_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH2 Line01
    mop(OP_LDR,  3, &BR[8][0][1], (Uint)in2_01_00mm, x, MSK_W0, (Ull)in2_01_00, 0, 15, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r1,          (Uint)in2_01_01mm, x, MSK_W0, (Ull)in2_01_00, 0, 15, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r2,          (Uint)in2_01_02mm, x, MSK_W0, (Ull)in2_01_00, 0, 15, 0, (Ull)NULL); /* stage#8 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[8][0][1], EXP_H3210, wgt2_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt2_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt2_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
#ifdef DEB_NC
    monitor_float(0, BR[8][0][1], wgt2_01_00, r10, 0);
    monitor_float(1, r1,          wgt2_01_01, r11, 0);
    monitor_float(2, r2,          wgt2_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH2 Line02
    mop(OP_LDR,  3, &BR[9][0][1], (Uint)in2_02_00mm, x, MSK_W0, (Ull)in2_02_00, 0, 15, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r1,          (Uint)in2_02_01mm, x, MSK_W0, (Ull)in2_02_00, 0, 15, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r2,          (Uint)in2_02_02mm, x, MSK_W0, (Ull)in2_02_00, 0, 15, 0, (Ull)NULL); /* stage#9 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[9][0][1], EXP_H3210, wgt2_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt2_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt2_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
#ifdef DEB_NC
    monitor_float(0, BR[9][0][1], wgt2_02_00, r20, 0);
    monitor_float(1, r1,          wgt2_02_01, r21, 0);
    monitor_float(2, r2,          wgt2_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, 0);
#endif
    // CH3 Line00
    mop(OP_LDR,  3, &BR[10][0][1], (Uint)in3_00_00mm, x, MSK_W0, (Ull)in3_00_00, 0, 15, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r1,           (Uint)in3_00_01mm, x, MSK_W0, (Ull)in3_00_00, 0, 15, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r2,           (Uint)in3_00_02mm, x, MSK_W0, (Ull)in3_00_00, 0, 15, 0, (Ull)NULL); /* stage#10 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[10][0][1], EXP_H3210, wgt3_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt3_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt3_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
#ifdef DEB_NC
    monitor_float(0, BR[10][0][1], wgt3_00_00, r00, 0);
    monitor_float(1, r1,           wgt3_00_01, r01, 0);
    monitor_float(2, r2,           wgt3_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH3 Line01
    mop(OP_LDR,  3, &BR[11][0][1], (Uint)in3_01_00mm, x, MSK_W0, (Ull)in3_01_00, 0, 15, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r1,           (Uint)in3_01_01mm, x, MSK_W0, (Ull)in3_01_00, 0, 15, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r2,           (Uint)in3_01_02mm, x, MSK_W0, (Ull)in3_01_00, 0, 15, 0, (Ull)NULL); /* stage#11 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[11][0][1], EXP_H3210, wgt3_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt3_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt3_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
#ifdef DEB_NC
    monitor_float(0, BR[11][0][1], wgt3_01_00, r10, 0);
    monitor_float(1, r1,           wgt3_01_01, r11, 0);
    monitor_float(2, r2,           wgt3_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH3 Line02
    mop(OP_LDR,  3, &BR[12][0][1], (Uint)in3_02_00mm, x, MSK_W0, (Ull)in3_02_00, 0, 15, 0, (Ull)NULL); /* stage#12 */
    mop(OP_LDR,  3, &r1,           (Uint)in3_02_01mm, x, MSK_W0, (Ull)in3_02_00, 0, 15, 0, (Ull)NULL); /* stage#12 */
    mop(OP_LDR,  3, &r2,           (Uint)in3_02_02mm, x, MSK_W0, (Ull)in3_02_00, 0, 15, 0, (Ull)NULL); /* stage#12 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[12][0][1], EXP_H3210, wgt3_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt3_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt3_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
#ifdef DEB_NC
    monitor_float(0, BR[12][0][1], wgt3_02_00, r20, 0);
    monitor_float(1, r1,           wgt3_02_01, r21, 0);
    monitor_float(2, r2,           wgt3_02_02, r22, 0);
    monitor_float(3, r3,           b,          r23, 0);
    monitor_fsum(i++, r20, r21, r22, r23);
#endif

    // CH4 Line00
    mop(OP_LDR,  3, &BR[13][0][1], (Uint)in4_00_00mm, x, MSK_W0, (Ull)in4_00_00, 0, 15, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r1,           (Uint)in4_00_01mm, x, MSK_W0, (Ull)in4_00_00, 0, 15, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r2,           (Uint)in4_00_02mm, x, MSK_W0, (Ull)in4_00_00, 0, 15, 0, (Ull)NULL); /* stage#13 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[13][0][1], EXP_H3210, wgt4_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt4_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt4_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    // CH4 Line01
    mop(OP_LDR,  3, &BR[14][0][1], (Uint)in4_01_00mm, x, MSK_W0, (Ull)in4_01_00, 0, 15, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r1,           (Uint)in4_01_01mm, x, MSK_W0, (Ull)in4_01_00, 0, 15, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r2,           (Uint)in4_01_02mm, x, MSK_W0, (Ull)in4_01_00, 0, 15, 0, (Ull)NULL); /* stage#14 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[14][0][1], EXP_H3210, wgt4_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt4_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt4_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    // CH4 Line02
    mop(OP_LDR,  3, &BR[15][0][1], (Uint)in4_02_00mm, x, MSK_W0, (Ull)in4_02_00, 0, 15, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r1,           (Uint)in4_02_01mm, x, MSK_W0, (Ull)in4_02_00, 0, 15, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r2,           (Uint)in4_02_02mm, x, MSK_W0, (Ull)in4_02_00, 0, 15, 0, (Ull)NULL); /* stage#15 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[15][0][1], EXP_H3210, wgt4_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt4_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt4_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */

    // CH5 Line00
    mop(OP_LDR,  3, &BR[16][0][1], (Uint)in5_00_00mm, x, MSK_W0, (Ull)in5_00_00, 0, 15, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r1,           (Uint)in5_00_01mm, x, MSK_W0, (Ull)in5_00_00, 0, 15, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r2,           (Uint)in5_00_02mm, x, MSK_W0, (Ull)in5_00_00, 0, 15, 0, (Ull)NULL); /* stage#16 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[16][0][1], EXP_H3210, wgt5_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt5_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt5_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    // CH5 Line01
    mop(OP_LDR,  3, &BR[17][0][1], (Uint)in5_01_00mm, x, MSK_W0, (Ull)in5_01_00, 0, 15, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r1,           (Uint)in5_01_01mm, x, MSK_W0, (Ull)in5_01_00, 0, 15, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r2,           (Uint)in5_01_02mm, x, MSK_W0, (Ull)in5_01_00, 0, 15, 0, (Ull)NULL); /* stage#17 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[17][0][1], EXP_H3210, wgt5_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt5_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt5_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    // CH5 Line02
    mop(OP_LDR,  3, &BR[18][0][1], (Uint)in5_02_00mm, x, MSK_W0, (Ull)in5_02_00, 0, 15, 0, (Ull)NULL); /* stage#18 */
    mop(OP_LDR,  3, &r1,           (Uint)in5_02_01mm, x, MSK_W0, (Ull)in5_02_00, 0, 15, 0, (Ull)NULL); /* stage#18 */
    mop(OP_LDR,  3, &r2,           (Uint)in5_02_02mm, x, MSK_W0, (Ull)in5_02_00, 0, 15, 0, (Ull)NULL); /* stage#18 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[18][0][1], EXP_H3210, wgt5_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt5_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt5_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */

    // CH6 Line00
    mop(OP_LDR,  3, &BR[19][0][1], (Uint)in6_00_00mm, x, MSK_W0, (Ull)in6_00_00, 0, 15, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r1,           (Uint)in6_00_01mm, x, MSK_W0, (Ull)in6_00_00, 0, 15, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r2,           (Uint)in6_00_02mm, x, MSK_W0, (Ull)in6_00_00, 0, 15, 0, (Ull)NULL); /* stage#19 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[19][0][1], EXP_H3210, wgt6_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt6_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt6_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    // CH6 Line01
    mop(OP_LDR,  3, &BR[20][0][1], (Uint)in6_01_00mm, x, MSK_W0, (Ull)in6_01_00, 0, 15, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r1,           (Uint)in6_01_01mm, x, MSK_W0, (Ull)in6_01_00, 0, 15, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r2,           (Uint)in6_01_02mm, x, MSK_W0, (Ull)in6_01_00, 0, 15, 0, (Ull)NULL); /* stage#20 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[20][0][1], EXP_H3210, wgt6_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt6_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt6_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    // CH6 Line02
    mop(OP_LDR,  3, &BR[21][0][1], (Uint)in6_02_00mm, x, MSK_W0, (Ull)in6_02_00, 0, 15, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r1,           (Uint)in6_02_01mm, x, MSK_W0, (Ull)in6_02_00, 0, 15, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r2,           (Uint)in6_02_02mm, x, MSK_W0, (Ull)in6_02_00, 0, 15, 0, (Ull)NULL); /* stage#21 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[21][0][1], EXP_H3210, wgt6_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt6_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt6_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */

    // CH7 Line00
    mop(OP_LDR,  3, &BR[22][0][1], (Uint)in7_00_00mm, x, MSK_W0, (Ull)in7_00_00, 0, 15, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r1,           (Uint)in7_00_01mm, x, MSK_W0, (Ull)in7_00_00, 0, 15, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r2,           (Uint)in7_00_02mm, x, MSK_W0, (Ull)in7_00_00, 0, 15, 0, (Ull)NULL); /* stage#22 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[22][0][1], EXP_H3210, wgt7_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt7_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt7_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    // CH7 Line01
    mop(OP_LDR,  3, &BR[23][0][1], (Uint)in7_01_00mm, x, MSK_W0, (Ull)in7_01_00, 0, 15, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r1,           (Uint)in7_01_01mm, x, MSK_W0, (Ull)in7_01_00, 0, 15, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r2,           (Uint)in7_01_02mm, x, MSK_W0, (Ull)in7_01_00, 0, 15, 0, (Ull)NULL); /* stage#23 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[23][0][1], EXP_H3210, wgt7_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt7_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt7_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    // CH7 Line02
    mop(OP_LDR,  3, &BR[24][0][1], (Uint)in7_02_00mm, x, MSK_W0, (Ull)in7_02_00, 0, 15, 0, (Ull)NULL); /* stage#24 */
    mop(OP_LDR,  3, &r1,           (Uint)in7_02_01mm, x, MSK_W0, (Ull)in7_02_00, 0, 15, 0, (Ull)NULL); /* stage#24 */
    mop(OP_LDR,  3, &r2,           (Uint)in7_02_02mm, x, MSK_W0, (Ull)in7_02_00, 0, 15, 0, (Ull)NULL); /* stage#24 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[24][0][1], EXP_H3210, wgt7_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt7_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt7_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */

    // CH8 Line00
    mop(OP_LDR,  3, &BR[25][0][1], (Uint)in8_00_00mm, x, MSK_W0, (Ull)in8_00_00, 0, 15, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r1,           (Uint)in8_00_01mm, x, MSK_W0, (Ull)in8_00_00, 0, 15, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r2,           (Uint)in8_00_02mm, x, MSK_W0, (Ull)in8_00_00, 0, 15, 0, (Ull)NULL); /* stage#25 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[25][0][1], EXP_H3210, wgt8_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt8_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt8_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    // CH8 Line01
    mop(OP_LDR,  3, &BR[26][0][1], (Uint)in8_01_00mm, x, MSK_W0, (Ull)in8_01_00, 0, 15, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r1,           (Uint)in8_01_01mm, x, MSK_W0, (Ull)in8_01_00, 0, 15, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r2,           (Uint)in8_01_02mm, x, MSK_W0, (Ull)in8_01_00, 0, 15, 0, (Ull)NULL); /* stage#26 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[26][0][1], EXP_H3210, wgt8_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt8_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt8_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    // CH8 Line02
    mop(OP_LDR,  3, &BR[27][0][1], (Uint)in8_02_00mm, x, MSK_W0, (Ull)in8_02_00, 0, 15, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r1,           (Uint)in8_02_01mm, x, MSK_W0, (Ull)in8_02_00, 0, 15, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r2,           (Uint)in8_02_02mm, x, MSK_W0, (Ull)in8_02_00, 0, 15, 0, (Ull)NULL); /* stage#27 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[27][0][1], EXP_H3210, wgt8_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt8_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt8_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */

    // CH9 Line00
    mop(OP_LDR,  3, &BR[28][0][1], (Uint)in9_00_00mm, x, MSK_W0, (Ull)in9_00_00, 0, 15, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r1,           (Uint)in9_00_01mm, x, MSK_W0, (Ull)in9_00_00, 0, 15, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r2,           (Uint)in9_00_02mm, x, MSK_W0, (Ull)in9_00_00, 0, 15, 0, (Ull)NULL); /* stage#28 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[28][0][1], EXP_H3210, wgt9_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt9_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt9_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    // CH9 Line01
    mop(OP_LDR,  3, &BR[29][0][1], (Uint)in9_01_00mm, x, MSK_W0, (Ull)in9_01_00, 0, 15, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r1,           (Uint)in9_01_01mm, x, MSK_W0, (Ull)in9_01_00, 0, 15, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r2,           (Uint)in9_01_02mm, x, MSK_W0, (Ull)in9_01_00, 0, 15, 0, (Ull)NULL); /* stage#29 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[29][0][1], EXP_H3210, wgt9_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt9_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt9_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    // CH9 Line02
    mop(OP_LDR,  3, &BR[30][0][1], (Uint)in9_02_00mm, x, MSK_W0, (Ull)in9_02_00, 0, 15, 0, (Ull)NULL); /* stage#30 */
    mop(OP_LDR,  3, &r1,           (Uint)in9_02_01mm, x, MSK_W0, (Ull)in9_02_00, 0, 15, 0, (Ull)NULL); /* stage#30 */
    mop(OP_LDR,  3, &r2,           (Uint)in9_02_02mm, x, MSK_W0, (Ull)in9_02_00, 0, 15, 0, (Ull)NULL); /* stage#30 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[30][0][1], EXP_H3210, wgt9_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt9_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt9_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */

    // CH10 Line00
    mop(OP_LDR,  3, &BR[31][0][1], (Uint)in10_00_00mm, x, MSK_W0, (Ull)in10_00_00, 0, 15, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_00_01mm, x, MSK_W0, (Ull)in10_00_00, 0, 15, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_00_02mm, x, MSK_W0, (Ull)in10_00_00, 0, 15, 0, (Ull)NULL); /* stage#31 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[31][0][1], EXP_H3210, wgt10_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt10_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt10_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    // CH10 Line01
    mop(OP_LDR,  3, &BR[32][0][1], (Uint)in10_01_00mm, x, MSK_W0, (Ull)in10_01_00, 0, 15, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_01_01mm, x, MSK_W0, (Ull)in10_01_00, 0, 15, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_01_02mm, x, MSK_W0, (Ull)in10_01_00, 0, 15, 0, (Ull)NULL); /* stage#32 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[32][0][1], EXP_H3210, wgt10_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt10_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt10_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    // CH10 Line02
    mop(OP_LDR,  3, &BR[33][0][1], (Uint)in10_02_00mm, x, MSK_W0, (Ull)in10_02_00, 0, 15, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_02_01mm, x, MSK_W0, (Ull)in10_02_00, 0, 15, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_02_02mm, x, MSK_W0, (Ull)in10_02_00, 0, 15, 0, (Ull)NULL); /* stage#33 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[33][0][1], EXP_H3210, wgt10_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt10_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt10_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */

    // CH11 Line00
    mop(OP_LDR,  3, &BR[34][0][1], (Uint)in11_00_00mm, x, MSK_W0, (Ull)in11_00_00, 0, 15, 0, (Ull)NULL); /* stage#34 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_00_01mm, x, MSK_W0, (Ull)in11_00_00, 0, 15, 0, (Ull)NULL); /* stage#34 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_00_02mm, x, MSK_W0, (Ull)in11_00_00, 0, 15, 0, (Ull)NULL); /* stage#34 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[34][0][1], EXP_H3210, wgt11_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt11_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt11_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    // CH11 Line01
    mop(OP_LDR,  3, &BR[35][0][1], (Uint)in11_01_00mm, x, MSK_W0, (Ull)in11_01_00, 0, 15, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_01_01mm, x, MSK_W0, (Ull)in11_01_00, 0, 15, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_01_02mm, x, MSK_W0, (Ull)in11_01_00, 0, 15, 0, (Ull)NULL); /* stage#35 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[35][0][1], EXP_H3210, wgt11_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt11_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt11_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    // CH11 Line02
    mop(OP_LDR,  3, &BR[36][0][1], (Uint)in11_02_00mm, x, MSK_W0, (Ull)in11_02_00, 0, 15, 0, (Ull)NULL); /* stage#36 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_02_01mm, x, MSK_W0, (Ull)in11_02_00, 0, 15, 0, (Ull)NULL); /* stage#36 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_02_02mm, x, MSK_W0, (Ull)in11_02_00, 0, 15, 0, (Ull)NULL); /* stage#36 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[36][0][1], EXP_H3210, wgt11_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt11_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt11_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */

    // CH12 Line00
    mop(OP_LDR,  3, &BR[37][0][1], (Uint)in12_00_00mm, x, MSK_W0, (Ull)in12_00_00, 0, 15, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_00_01mm, x, MSK_W0, (Ull)in12_00_00, 0, 15, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_00_02mm, x, MSK_W0, (Ull)in12_00_00, 0, 15, 0, (Ull)NULL); /* stage#37 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[37][0][1], EXP_H3210, wgt12_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt12_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt12_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    // CH12 Line01
    mop(OP_LDR,  3, &BR[38][0][1], (Uint)in12_01_00mm, x, MSK_W0, (Ull)in12_01_00, 0, 15, 0, (Ull)NULL); /* stage#38 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_01_01mm, x, MSK_W0, (Ull)in12_01_00, 0, 15, 0, (Ull)NULL); /* stage#38 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_01_02mm, x, MSK_W0, (Ull)in12_01_00, 0, 15, 0, (Ull)NULL); /* stage#38 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[38][0][1], EXP_H3210, wgt12_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt12_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt12_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    // CH12 Line02
    mop(OP_LDR,  3, &BR[39][0][1], (Uint)in12_02_00mm, x, MSK_W0, (Ull)in12_02_00, 0, 15, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_02_01mm, x, MSK_W0, (Ull)in12_02_00, 0, 15, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_02_02mm, x, MSK_W0, (Ull)in12_02_00, 0, 15, 0, (Ull)NULL); /* stage#39 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[39][0][1], EXP_H3210, wgt12_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt12_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt12_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */

    // CH13 Line00
    mop(OP_LDR,  3, &BR[40][0][1], (Uint)in13_00_00mm, x, MSK_W0, (Ull)in13_00_00, 0, 15, 0, (Ull)NULL); /* stage#40 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_00_01mm, x, MSK_W0, (Ull)in13_00_00, 0, 15, 0, (Ull)NULL); /* stage#40 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_00_02mm, x, MSK_W0, (Ull)in13_00_00, 0, 15, 0, (Ull)NULL); /* stage#40 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[40][0][1], EXP_H3210, wgt13_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt13_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt13_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    // CH13 Line01
    mop(OP_LDR,  3, &BR[41][0][1], (Uint)in13_01_00mm, x, MSK_W0, (Ull)in13_01_00, 0, 15, 0, (Ull)NULL); /* stage#41 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_01_01mm, x, MSK_W0, (Ull)in13_01_00, 0, 15, 0, (Ull)NULL); /* stage#41 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_01_02mm, x, MSK_W0, (Ull)in13_01_00, 0, 15, 0, (Ull)NULL); /* stage#41 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[41][0][1], EXP_H3210, wgt13_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt13_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt13_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */
    // CH13 Line02
    mop(OP_LDR,  3, &BR[42][0][1], (Uint)in13_02_00mm, x, MSK_W0, (Ull)in13_02_00, 0, 15, 0, (Ull)NULL); /* stage#42 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_02_01mm, x, MSK_W0, (Ull)in13_02_00, 0, 15, 0, (Ull)NULL); /* stage#42 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_02_02mm, x, MSK_W0, (Ull)in13_02_00, 0, 15, 0, (Ull)NULL); /* stage#42 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[42][0][1], EXP_H3210, wgt13_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt13_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt13_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */

    // CH14 Line00
    mop(OP_LDR,  3, &BR[43][0][1], (Uint)in14_00_00mm, x, MSK_W0, (Ull)in14_00_00, 0, 15, 0, (Ull)NULL); /* stage#43 */
    mop(OP_LDR,  3, &r1,           (Uint)in14_00_01mm, x, MSK_W0, (Ull)in14_00_00, 0, 15, 0, (Ull)NULL); /* stage#43 */
    mop(OP_LDR,  3, &r2,           (Uint)in14_00_02mm, x, MSK_W0, (Ull)in14_00_00, 0, 15, 0, (Ull)NULL); /* stage#43 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[43][0][1], EXP_H3210, wgt14_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt14_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt14_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    // CH14 Line01
    mop(OP_LDR,  3, &BR[44][0][1], (Uint)in14_01_00mm, x, MSK_W0, (Ull)in14_01_00, 0, 15, 0, (Ull)NULL); /* stage#44 */
    mop(OP_LDR,  3, &r1,           (Uint)in14_01_01mm, x, MSK_W0, (Ull)in14_01_00, 0, 15, 0, (Ull)NULL); /* stage#44 */
    mop(OP_LDR,  3, &r2,           (Uint)in14_01_02mm, x, MSK_W0, (Ull)in14_01_00, 0, 15, 0, (Ull)NULL); /* stage#44 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[44][0][1], EXP_H3210, wgt14_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#45 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt14_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#45 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt14_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#45 */
    // CH14 Line02
    mop(OP_LDR,  3, &BR[45][0][1], (Uint)in14_02_00mm, x, MSK_W0, (Ull)in14_02_00, 0, 15, 0, (Ull)NULL); /* stage#45 */
    mop(OP_LDR,  3, &r1,           (Uint)in14_02_01mm, x, MSK_W0, (Ull)in14_02_00, 0, 15, 0, (Ull)NULL); /* stage#45 */
    mop(OP_LDR,  3, &r2,           (Uint)in14_02_02mm, x, MSK_W0, (Ull)in14_02_00, 0, 15, 0, (Ull)NULL); /* stage#45 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[45][0][1], EXP_H3210, wgt14_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#46 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt14_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#46 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt14_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#46 */

    // CH15 Line00
    mop(OP_LDR,  3, &BR[46][0][1], (Uint)in15_00_00mm, x, MSK_W0, (Ull)in15_00_00, 0, 15, 0, (Ull)NULL); /* stage#46 */
    mop(OP_LDR,  3, &r1,           (Uint)in15_00_01mm, x, MSK_W0, (Ull)in15_00_00, 0, 15, 0, (Ull)NULL); /* stage#46 */
    mop(OP_LDR,  3, &r2,           (Uint)in15_00_02mm, x, MSK_W0, (Ull)in15_00_00, 0, 15, 0, (Ull)NULL); /* stage#46 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[46][0][1], EXP_H3210, wgt15_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#47 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt15_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#47 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt15_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#47 */
    // CH15 Line01
    mop(OP_LDR,  3, &BR[47][0][1], (Uint)in15_01_00mm, x, MSK_W0, (Ull)in15_01_00, 0, 15, 0, (Ull)NULL); /* stage#47 */
    mop(OP_LDR,  3, &r1,           (Uint)in15_01_01mm, x, MSK_W0, (Ull)in15_01_00, 0, 15, 0, (Ull)NULL); /* stage#47 */
    mop(OP_LDR,  3, &r2,           (Uint)in15_01_02mm, x, MSK_W0, (Ull)in15_01_00, 0, 15, 0, (Ull)NULL); /* stage#47 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[47][0][1], EXP_H3210, wgt15_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#48 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt15_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#48 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt15_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#48 */
    // CH15 Line02
    mop(OP_LDR,  3, &BR[48][0][1], (Uint)in15_02_00mm, x, MSK_W0, (Ull)in15_02_00, 0, 15, 0, (Ull)NULL); /* stage#48 */
    mop(OP_LDR,  3, &r1,           (Uint)in15_02_01mm, x, MSK_W0, (Ull)in15_02_00, 0, 15, 0, (Ull)NULL); /* stage#48 */
    mop(OP_LDR,  3, &r2,           (Uint)in15_02_02mm, x, MSK_W0, (Ull)in15_02_00, 0, 15, 0, (Ull)NULL); /* stage#48 */
    mop(OP_LDR,  3, &r3,           (Uint)o_mm,         x, MSK_W0, (Ull)o,          0, 13, 0, (Ull)NULL); /* stage#48 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[48][0][1], EXP_H3210, wgt15_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt15_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt15_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */
    exe(OP_FAD,  &r23, r3,  EXP_H3210, b,            EXP_H3210, 0,           EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */

    exe(OP_FAD,  &r00, r20, EXP_H3210, r21, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#50 */
    exe(OP_FAD,  &r01, r22, EXP_H3210, r23, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#50 */

    exe(OP_FAD,  &AR[51][0], r00,    EXP_H3210, r01, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#51 */
    mop(OP_STR,   3, &AR[51][0],    (Uint)o_mm, x, MSK_W0, (Ull)o, 0, 13, 0, (Ull)NULL); /* stage#51 */
#ifdef DEB_NC
    printf("o = %x \n", o);
    monitor_float(i, AR[51][0], 0, 0, 0);
    i=0;
#endif
  }
//EMAX5A end

  return(0);

conv345_ch0:
{}
//EMAX5A begin conv345_ch0 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
#ifdef DEB_NC
    printf("x = %x\n",x);
#endif
    exe(OP_ADD,  &x, x, EXP_H3210, str, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0);       /* stage#0 */
    // CH0 Line00
    mop(OP_LDR,  3, &BR[1][0][1], (Uint)in0_00_00mm, x, MSK_W0, (Ull)in0_00_00, 0, 15, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r1,          (Uint)in0_00_01mm, x, MSK_W0, (Ull)in0_00_00, 0, 15, 0, (Ull)NULL); /* stage#1 */
    mop(OP_LDR,  3, &r2,          (Uint)in0_00_02mm, x, MSK_W0, (Ull)in0_00_00, 0, 15, 0, (Ull)NULL); /* stage#1 */
    exe(OP_FMA,  &r00, 0LL, EXP_H3210, BR[1][0][1], EXP_H3210, wgt0_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r01, 0LL, EXP_H3210, r1,          EXP_H3210, wgt0_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
    exe(OP_FMA,  &r02, 0LL, EXP_H3210, r2,          EXP_H3210, wgt0_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#2 */
#ifdef DEB_NC
    monitor_float(0, BR[1][0][1], wgt0_00_00, r00, 0);
    monitor_float(1, r1,          wgt0_00_01, r01, 0);
    monitor_float(2, r2,          wgt0_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH0 Line01
    mop(OP_LDR,  3, &BR[2][0][1], (Uint)in0_01_00mm, x, MSK_W0, (Ull)in0_01_00, 0, 15, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r1,          (Uint)in0_01_01mm, x, MSK_W0, (Ull)in0_01_00, 0, 15, 0, (Ull)NULL); /* stage#2 */
    mop(OP_LDR,  3, &r2,          (Uint)in0_01_02mm, x, MSK_W0, (Ull)in0_01_00, 0, 15, 0, (Ull)NULL); /* stage#2 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[2][0][1], EXP_H3210, wgt0_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt0_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt0_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#3 */
#ifdef DEB_NC
    monitor_float(0, BR[2][0][1], wgt0_01_00, r10, 0);
    monitor_float(1, r1,          wgt0_01_01, r11, 0);
    monitor_float(2, r2,          wgt0_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH0 Line02
    mop(OP_LDR,  3, &BR[3][0][1], (Uint)in0_02_00mm, x, MSK_W0, (Ull)in0_02_00, 0, 15, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r1,          (Uint)in0_02_01mm, x, MSK_W0, (Ull)in0_02_00, 0, 15, 0, (Ull)NULL); /* stage#3 */
    mop(OP_LDR,  3, &r2,          (Uint)in0_02_02mm, x, MSK_W0, (Ull)in0_02_00, 0, 15, 0, (Ull)NULL); /* stage#3 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[3][0][1], EXP_H3210, wgt0_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt0_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt0_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#4 */
#ifdef DEB_NC
    monitor_float(0, BR[3][0][1], wgt0_02_00, r20, 0);
    monitor_float(1, r1,          wgt0_02_01, r21, 0);
    monitor_float(2, r2,          wgt0_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, 0);
#endif
    // CH1 Line00
    mop(OP_LDR,  3, &BR[4][0][1], (Uint)in1_00_00mm, x, MSK_W0, (Ull)in1_00_00, 0, 15, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r1,          (Uint)in1_00_01mm, x, MSK_W0, (Ull)in1_00_00, 0, 15, 0, (Ull)NULL); /* stage#4 */
    mop(OP_LDR,  3, &r2,          (Uint)in1_00_02mm, x, MSK_W0, (Ull)in1_00_00, 0, 15, 0, (Ull)NULL); /* stage#4 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[4][0][1], EXP_H3210, wgt1_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,          EXP_H3210, wgt1_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,          EXP_H3210, wgt1_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#5 */
#ifdef DEB_NC
    monitor_float(0, BR[4][0][1], wgt1_00_00, r00, 0);
    monitor_float(1, r1,          wgt1_00_01, r01, 0);
    monitor_float(2, r2,          wgt1_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH1 Line01
    mop(OP_LDR,  3, &BR[5][0][1], (Uint)in1_01_00mm, x, MSK_W0, (Ull)in1_01_00, 0, 15, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r1,          (Uint)in1_01_01mm, x, MSK_W0, (Ull)in1_01_00, 0, 15, 0, (Ull)NULL); /* stage#5 */
    mop(OP_LDR,  3, &r2,          (Uint)in1_01_02mm, x, MSK_W0, (Ull)in1_01_00, 0, 15, 0, (Ull)NULL); /* stage#5 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[5][0][1], EXP_H3210, wgt1_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt1_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt1_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#6 */
#ifdef DEB_NC
    monitor_float(0, BR[5][0][1], wgt1_01_00, r10, 0);
    monitor_float(1, r1,          wgt1_01_01, r11, 0);
    monitor_float(2, r2,          wgt1_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH1 Line02
    mop(OP_LDR,  3, &BR[6][0][1], (Uint)in1_02_00mm, x, MSK_W0, (Ull)in1_02_00, 0, 15, 0, (Ull)NULL); /* stage#6 */
    mop(OP_LDR,  3, &r1,          (Uint)in1_02_01mm, x, MSK_W0, (Ull)in1_02_00, 0, 15, 0, (Ull)NULL); /* stage#6 */
    mop(OP_LDR,  3, &r2,          (Uint)in1_02_02mm, x, MSK_W0, (Ull)in1_02_00, 0, 15, 0, (Ull)NULL); /* stage#6 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[6][0][1], EXP_H3210, wgt1_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt1_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt1_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#7 */
#ifdef DEB_NC
    monitor_float(0, BR[6][0][1], wgt1_02_00, r20, 0);
    monitor_float(1, r1,          wgt1_02_01, r21, 0);
    monitor_float(2, r2,          wgt1_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, 0);
#endif
    // CH2 Line00
    mop(OP_LDR,  3, &BR[7][0][1], (Uint)in2_00_00mm, x, MSK_W0, (Ull)in2_00_00, 0, 15, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r1,          (Uint)in2_00_01mm, x, MSK_W0, (Ull)in2_00_00, 0, 15, 0, (Ull)NULL); /* stage#7 */
    mop(OP_LDR,  3, &r2,          (Uint)in2_00_02mm, x, MSK_W0, (Ull)in2_00_00, 0, 15, 0, (Ull)NULL); /* stage#7 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[7][0][1], EXP_H3210, wgt2_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,          EXP_H3210, wgt2_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,          EXP_H3210, wgt2_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#8 */
#ifdef DEB_NC
    monitor_float(0, BR[7][0][1], wgt2_00_00, r00, 0);
    monitor_float(1, r1,          wgt2_00_01, r01, 0);
    monitor_float(2, r2,          wgt2_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH2 Line01
    mop(OP_LDR,  3, &BR[8][0][1], (Uint)in2_01_00mm, x, MSK_W0, (Ull)in2_01_00, 0, 15, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r1,          (Uint)in2_01_01mm, x, MSK_W0, (Ull)in2_01_00, 0, 15, 0, (Ull)NULL); /* stage#8 */
    mop(OP_LDR,  3, &r2,          (Uint)in2_01_02mm, x, MSK_W0, (Ull)in2_01_00, 0, 15, 0, (Ull)NULL); /* stage#8 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[8][0][1], EXP_H3210, wgt2_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,          EXP_H3210, wgt2_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,          EXP_H3210, wgt2_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#9 */
#ifdef DEB_NC
    monitor_float(0, BR[8][0][1], wgt2_01_00, r10, 0);
    monitor_float(1, r1,          wgt2_01_01, r11, 0);
    monitor_float(2, r2,          wgt2_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH2 Line02
    mop(OP_LDR,  3, &BR[9][0][1], (Uint)in2_02_00mm, x, MSK_W0, (Ull)in2_02_00, 0, 15, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r1,          (Uint)in2_02_01mm, x, MSK_W0, (Ull)in2_02_00, 0, 15, 0, (Ull)NULL); /* stage#9 */
    mop(OP_LDR,  3, &r2,          (Uint)in2_02_02mm, x, MSK_W0, (Ull)in2_02_00, 0, 15, 0, (Ull)NULL); /* stage#9 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[9][0][1], EXP_H3210, wgt2_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,          EXP_H3210, wgt2_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,          EXP_H3210, wgt2_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#10 */
#ifdef DEB_NC
    monitor_float(0, BR[9][0][1], wgt2_02_00, r20, 0);
    monitor_float(1, r1,          wgt2_02_01, r21, 0);
    monitor_float(2, r2,          wgt2_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, 0);
#endif
    // CH3 Line00
    mop(OP_LDR,  3, &BR[10][0][1], (Uint)in3_00_00mm, x, MSK_W0, (Ull)in3_00_00, 0, 15, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r1,           (Uint)in3_00_01mm, x, MSK_W0, (Ull)in3_00_00, 0, 15, 0, (Ull)NULL); /* stage#10 */
    mop(OP_LDR,  3, &r2,           (Uint)in3_00_02mm, x, MSK_W0, (Ull)in3_00_00, 0, 15, 0, (Ull)NULL); /* stage#10 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[10][0][1], EXP_H3210, wgt3_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt3_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt3_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#11 */
#ifdef DEB_NC
    monitor_float(0, BR[10][0][1], wgt3_00_00, r00, 0);
    monitor_float(1, r1,           wgt3_00_01, r01, 0);
    monitor_float(2, r2,           wgt3_00_02, r02, 0);
    monitor_fsum(i++, r00, r01, r02, 0);
#endif
    // CH3 Line01
    mop(OP_LDR,  3, &BR[11][0][1], (Uint)in3_01_00mm, x, MSK_W0, (Ull)in3_01_00, 0, 15, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r1,           (Uint)in3_01_01mm, x, MSK_W0, (Ull)in3_01_00, 0, 15, 0, (Ull)NULL); /* stage#11 */
    mop(OP_LDR,  3, &r2,           (Uint)in3_01_02mm, x, MSK_W0, (Ull)in3_01_00, 0, 15, 0, (Ull)NULL); /* stage#11 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[11][0][1], EXP_H3210, wgt3_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt3_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt3_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#12 */
#ifdef DEB_NC
    monitor_float(0, BR[11][0][1], wgt3_01_00, r10, 0);
    monitor_float(1, r1,           wgt3_01_01, r11, 0);
    monitor_float(2, r2,           wgt3_01_02, r12, 0);
    monitor_fsum(i++, r10, r11, r12, 0);
#endif
    // CH3 Line02
    mop(OP_LDR,  3, &BR[12][0][1], (Uint)in3_02_00mm, x, MSK_W0, (Ull)in3_02_00, 0, 15, 0, (Ull)NULL); /* stage#12 */
    mop(OP_LDR,  3, &r1,           (Uint)in3_02_01mm, x, MSK_W0, (Ull)in3_02_00, 0, 15, 0, (Ull)NULL); /* stage#12 */
    mop(OP_LDR,  3, &r2,           (Uint)in3_02_02mm, x, MSK_W0, (Ull)in3_02_00, 0, 15, 0, (Ull)NULL); /* stage#12 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[12][0][1], EXP_H3210, wgt3_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt3_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt3_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#13 */
#ifdef DEB_NC
    monitor_float(0, BR[12][0][1], wgt3_02_00, r20, 0);
    monitor_float(1, r1,           wgt3_02_01, r21, 0);
    monitor_float(2, r2,           wgt3_02_02, r22, 0);
    monitor_fsum(i++, r20, r21, r22, r23);
#endif
    // CH4 Line00
    mop(OP_LDR,  3, &BR[13][0][1], (Uint)in4_00_00mm, x, MSK_W0, (Ull)in4_00_00, 0, 15, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r1,           (Uint)in4_00_01mm, x, MSK_W0, (Ull)in4_00_00, 0, 15, 0, (Ull)NULL); /* stage#13 */
    mop(OP_LDR,  3, &r2,           (Uint)in4_00_02mm, x, MSK_W0, (Ull)in4_00_00, 0, 15, 0, (Ull)NULL); /* stage#13 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[13][0][1], EXP_H3210, wgt4_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt4_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt4_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#14 */
    // CH4 Line01
    mop(OP_LDR,  3, &BR[14][0][1], (Uint)in4_01_00mm, x, MSK_W0, (Ull)in4_01_00, 0, 15, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r1,           (Uint)in4_01_01mm, x, MSK_W0, (Ull)in4_01_00, 0, 15, 0, (Ull)NULL); /* stage#14 */
    mop(OP_LDR,  3, &r2,           (Uint)in4_01_02mm, x, MSK_W0, (Ull)in4_01_00, 0, 15, 0, (Ull)NULL); /* stage#14 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[14][0][1], EXP_H3210, wgt4_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt4_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt4_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#15 */
    // CH4 Line02
    mop(OP_LDR,  3, &BR[15][0][1], (Uint)in4_02_00mm, x, MSK_W0, (Ull)in4_02_00, 0, 15, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r1,           (Uint)in4_02_01mm, x, MSK_W0, (Ull)in4_02_00, 0, 15, 0, (Ull)NULL); /* stage#15 */
    mop(OP_LDR,  3, &r2,           (Uint)in4_02_02mm, x, MSK_W0, (Ull)in4_02_00, 0, 15, 0, (Ull)NULL); /* stage#15 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[15][0][1], EXP_H3210, wgt4_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt4_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt4_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#16 */

    // CH5 Line00
    mop(OP_LDR,  3, &BR[16][0][1], (Uint)in5_00_00mm, x, MSK_W0, (Ull)in5_00_00, 0, 15, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r1,           (Uint)in5_00_01mm, x, MSK_W0, (Ull)in5_00_00, 0, 15, 0, (Ull)NULL); /* stage#16 */
    mop(OP_LDR,  3, &r2,           (Uint)in5_00_02mm, x, MSK_W0, (Ull)in5_00_00, 0, 15, 0, (Ull)NULL); /* stage#16 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[16][0][1], EXP_H3210, wgt5_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt5_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt5_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#17 */
    // CH5 Line01
    mop(OP_LDR,  3, &BR[17][0][1], (Uint)in5_01_00mm, x, MSK_W0, (Ull)in5_01_00, 0, 15, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r1,           (Uint)in5_01_01mm, x, MSK_W0, (Ull)in5_01_00, 0, 15, 0, (Ull)NULL); /* stage#17 */
    mop(OP_LDR,  3, &r2,           (Uint)in5_01_02mm, x, MSK_W0, (Ull)in5_01_00, 0, 15, 0, (Ull)NULL); /* stage#17 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[17][0][1], EXP_H3210, wgt5_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt5_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt5_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#18 */
    // CH5 Line02
    mop(OP_LDR,  3, &BR[18][0][1], (Uint)in5_02_00mm, x, MSK_W0, (Ull)in5_02_00, 0, 15, 0, (Ull)NULL); /* stage#18 */
    mop(OP_LDR,  3, &r1,           (Uint)in5_02_01mm, x, MSK_W0, (Ull)in5_02_00, 0, 15, 0, (Ull)NULL); /* stage#18 */
    mop(OP_LDR,  3, &r2,           (Uint)in5_02_02mm, x, MSK_W0, (Ull)in5_02_00, 0, 15, 0, (Ull)NULL); /* stage#18 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[18][0][1], EXP_H3210, wgt5_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt5_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt5_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#19 */

    // CH6 Line00
    mop(OP_LDR,  3, &BR[19][0][1], (Uint)in6_00_00mm, x, MSK_W0, (Ull)in6_00_00, 0, 15, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r1,           (Uint)in6_00_01mm, x, MSK_W0, (Ull)in6_00_00, 0, 15, 0, (Ull)NULL); /* stage#19 */
    mop(OP_LDR,  3, &r2,           (Uint)in6_00_02mm, x, MSK_W0, (Ull)in6_00_00, 0, 15, 0, (Ull)NULL); /* stage#19 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[19][0][1], EXP_H3210, wgt6_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt6_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt6_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#20 */
    // CH6 Line01
    mop(OP_LDR,  3, &BR[20][0][1], (Uint)in6_01_00mm, x, MSK_W0, (Ull)in6_01_00, 0, 15, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r1,           (Uint)in6_01_01mm, x, MSK_W0, (Ull)in6_01_00, 0, 15, 0, (Ull)NULL); /* stage#20 */
    mop(OP_LDR,  3, &r2,           (Uint)in6_01_02mm, x, MSK_W0, (Ull)in6_01_00, 0, 15, 0, (Ull)NULL); /* stage#20 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[20][0][1], EXP_H3210, wgt6_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt6_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt6_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#21 */
    // CH6 Line02
    mop(OP_LDR,  3, &BR[21][0][1], (Uint)in6_02_00mm, x, MSK_W0, (Ull)in6_02_00, 0, 15, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r1,           (Uint)in6_02_01mm, x, MSK_W0, (Ull)in6_02_00, 0, 15, 0, (Ull)NULL); /* stage#21 */
    mop(OP_LDR,  3, &r2,           (Uint)in6_02_02mm, x, MSK_W0, (Ull)in6_02_00, 0, 15, 0, (Ull)NULL); /* stage#21 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[21][0][1], EXP_H3210, wgt6_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt6_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt6_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#22 */

    // CH7 Line00
    mop(OP_LDR,  3, &BR[22][0][1], (Uint)in7_00_00mm, x, MSK_W0, (Ull)in7_00_00, 0, 15, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r1,           (Uint)in7_00_01mm, x, MSK_W0, (Ull)in7_00_00, 0, 15, 0, (Ull)NULL); /* stage#22 */
    mop(OP_LDR,  3, &r2,           (Uint)in7_00_02mm, x, MSK_W0, (Ull)in7_00_00, 0, 15, 0, (Ull)NULL); /* stage#22 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[22][0][1], EXP_H3210, wgt7_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt7_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt7_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#23 */
    // CH7 Line01
    mop(OP_LDR,  3, &BR[23][0][1], (Uint)in7_01_00mm, x, MSK_W0, (Ull)in7_01_00, 0, 15, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r1,           (Uint)in7_01_01mm, x, MSK_W0, (Ull)in7_01_00, 0, 15, 0, (Ull)NULL); /* stage#23 */
    mop(OP_LDR,  3, &r2,           (Uint)in7_01_02mm, x, MSK_W0, (Ull)in7_01_00, 0, 15, 0, (Ull)NULL); /* stage#23 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[23][0][1], EXP_H3210, wgt7_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt7_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt7_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#24 */
    // CH7 Line02
    mop(OP_LDR,  3, &BR[24][0][1], (Uint)in7_02_00mm, x, MSK_W0, (Ull)in7_02_00, 0, 15, 0, (Ull)NULL); /* stage#24 */
    mop(OP_LDR,  3, &r1,           (Uint)in7_02_01mm, x, MSK_W0, (Ull)in7_02_00, 0, 15, 0, (Ull)NULL); /* stage#24 */
    mop(OP_LDR,  3, &r2,           (Uint)in7_02_02mm, x, MSK_W0, (Ull)in7_02_00, 0, 15, 0, (Ull)NULL); /* stage#24 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[24][0][1], EXP_H3210, wgt7_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt7_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt7_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#25 */

    // CH8 Line00
    mop(OP_LDR,  3, &BR[25][0][1], (Uint)in8_00_00mm, x, MSK_W0, (Ull)in8_00_00, 0, 15, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r1,           (Uint)in8_00_01mm, x, MSK_W0, (Ull)in8_00_00, 0, 15, 0, (Ull)NULL); /* stage#25 */
    mop(OP_LDR,  3, &r2,           (Uint)in8_00_02mm, x, MSK_W0, (Ull)in8_00_00, 0, 15, 0, (Ull)NULL); /* stage#25 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[25][0][1], EXP_H3210, wgt8_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt8_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt8_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#26 */
    // CH8 Line01
    mop(OP_LDR,  3, &BR[26][0][1], (Uint)in8_01_00mm, x, MSK_W0, (Ull)in8_01_00, 0, 15, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r1,           (Uint)in8_01_01mm, x, MSK_W0, (Ull)in8_01_00, 0, 15, 0, (Ull)NULL); /* stage#26 */
    mop(OP_LDR,  3, &r2,           (Uint)in8_01_02mm, x, MSK_W0, (Ull)in8_01_00, 0, 15, 0, (Ull)NULL); /* stage#26 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[26][0][1], EXP_H3210, wgt8_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt8_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt8_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#27 */
    // CH8 Line02
    mop(OP_LDR,  3, &BR[27][0][1], (Uint)in8_02_00mm, x, MSK_W0, (Ull)in8_02_00, 0, 15, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r1,           (Uint)in8_02_01mm, x, MSK_W0, (Ull)in8_02_00, 0, 15, 0, (Ull)NULL); /* stage#27 */
    mop(OP_LDR,  3, &r2,           (Uint)in8_02_02mm, x, MSK_W0, (Ull)in8_02_00, 0, 15, 0, (Ull)NULL); /* stage#27 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[27][0][1], EXP_H3210, wgt8_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt8_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt8_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#28 */

    // CH9 Line00
    mop(OP_LDR,  3, &BR[28][0][1], (Uint)in9_00_00mm, x, MSK_W0, (Ull)in9_00_00, 0, 15, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r1,           (Uint)in9_00_01mm, x, MSK_W0, (Ull)in9_00_00, 0, 15, 0, (Ull)NULL); /* stage#28 */
    mop(OP_LDR,  3, &r2,           (Uint)in9_00_02mm, x, MSK_W0, (Ull)in9_00_00, 0, 15, 0, (Ull)NULL); /* stage#28 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[28][0][1], EXP_H3210, wgt9_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt9_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt9_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#29 */
    // CH9 Line01
    mop(OP_LDR,  3, &BR[29][0][1], (Uint)in9_01_00mm, x, MSK_W0, (Ull)in9_01_00, 0, 15, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r1,           (Uint)in9_01_01mm, x, MSK_W0, (Ull)in9_01_00, 0, 15, 0, (Ull)NULL); /* stage#29 */
    mop(OP_LDR,  3, &r2,           (Uint)in9_01_02mm, x, MSK_W0, (Ull)in9_01_00, 0, 15, 0, (Ull)NULL); /* stage#29 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[29][0][1], EXP_H3210, wgt9_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt9_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt9_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#30 */
    // CH9 Line02
    mop(OP_LDR,  3, &BR[30][0][1], (Uint)in9_02_00mm, x, MSK_W0, (Ull)in9_02_00, 0, 15, 0, (Ull)NULL); /* stage#30 */
    mop(OP_LDR,  3, &r1,           (Uint)in9_02_01mm, x, MSK_W0, (Ull)in9_02_00, 0, 15, 0, (Ull)NULL); /* stage#30 */
    mop(OP_LDR,  3, &r2,           (Uint)in9_02_02mm, x, MSK_W0, (Ull)in9_02_00, 0, 15, 0, (Ull)NULL); /* stage#30 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[30][0][1], EXP_H3210, wgt9_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt9_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt9_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#31 */

    // CH10 Line00
    mop(OP_LDR,  3, &BR[31][0][1], (Uint)in10_00_00mm, x, MSK_W0, (Ull)in10_00_00, 0, 15, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_00_01mm, x, MSK_W0, (Ull)in10_00_00, 0, 15, 0, (Ull)NULL); /* stage#31 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_00_02mm, x, MSK_W0, (Ull)in10_00_00, 0, 15, 0, (Ull)NULL); /* stage#31 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[31][0][1], EXP_H3210, wgt10_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt10_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt10_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#32 */
    // CH10 Line01
    mop(OP_LDR,  3, &BR[32][0][1], (Uint)in10_01_00mm, x, MSK_W0, (Ull)in10_01_00, 0, 15, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_01_01mm, x, MSK_W0, (Ull)in10_01_00, 0, 15, 0, (Ull)NULL); /* stage#32 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_01_02mm, x, MSK_W0, (Ull)in10_01_00, 0, 15, 0, (Ull)NULL); /* stage#32 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[32][0][1], EXP_H3210, wgt10_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt10_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt10_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#33 */
    // CH10 Line02
    mop(OP_LDR,  3, &BR[33][0][1], (Uint)in10_02_00mm, x, MSK_W0, (Ull)in10_02_00, 0, 15, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r1,           (Uint)in10_02_01mm, x, MSK_W0, (Ull)in10_02_00, 0, 15, 0, (Ull)NULL); /* stage#33 */
    mop(OP_LDR,  3, &r2,           (Uint)in10_02_02mm, x, MSK_W0, (Ull)in10_02_00, 0, 15, 0, (Ull)NULL); /* stage#33 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[33][0][1], EXP_H3210, wgt10_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt10_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt10_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#34 */

    // CH11 Line00
    mop(OP_LDR,  3, &BR[34][0][1], (Uint)in11_00_00mm, x, MSK_W0, (Ull)in11_00_00, 0, 15, 0, (Ull)NULL); /* stage#34 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_00_01mm, x, MSK_W0, (Ull)in11_00_00, 0, 15, 0, (Ull)NULL); /* stage#34 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_00_02mm, x, MSK_W0, (Ull)in11_00_00, 0, 15, 0, (Ull)NULL); /* stage#34 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[34][0][1], EXP_H3210, wgt11_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt11_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt11_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#35 */
    // CH11 Line01
    mop(OP_LDR,  3, &BR[35][0][1], (Uint)in11_01_00mm, x, MSK_W0, (Ull)in11_01_00, 0, 15, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_01_01mm, x, MSK_W0, (Ull)in11_01_00, 0, 15, 0, (Ull)NULL); /* stage#35 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_01_02mm, x, MSK_W0, (Ull)in11_01_00, 0, 15, 0, (Ull)NULL); /* stage#35 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[35][0][1], EXP_H3210, wgt11_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt11_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt11_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#36 */
    // CH11 Line02
    mop(OP_LDR,  3, &BR[36][0][1], (Uint)in11_02_00mm, x, MSK_W0, (Ull)in11_02_00, 0, 15, 0, (Ull)NULL); /* stage#36 */
    mop(OP_LDR,  3, &r1,           (Uint)in11_02_01mm, x, MSK_W0, (Ull)in11_02_00, 0, 15, 0, (Ull)NULL); /* stage#36 */
    mop(OP_LDR,  3, &r2,           (Uint)in11_02_02mm, x, MSK_W0, (Ull)in11_02_00, 0, 15, 0, (Ull)NULL); /* stage#36 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[36][0][1], EXP_H3210, wgt11_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt11_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt11_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#37 */

    // CH12 Line00
    mop(OP_LDR,  3, &BR[37][0][1], (Uint)in12_00_00mm, x, MSK_W0, (Ull)in12_00_00, 0, 15, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_00_01mm, x, MSK_W0, (Ull)in12_00_00, 0, 15, 0, (Ull)NULL); /* stage#37 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_00_02mm, x, MSK_W0, (Ull)in12_00_00, 0, 15, 0, (Ull)NULL); /* stage#37 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[37][0][1], EXP_H3210, wgt12_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt12_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt12_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#38 */
    // CH12 Line01
    mop(OP_LDR,  3, &BR[38][0][1], (Uint)in12_01_00mm, x, MSK_W0, (Ull)in12_01_00, 0, 15, 0, (Ull)NULL); /* stage#38 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_01_01mm, x, MSK_W0, (Ull)in12_01_00, 0, 15, 0, (Ull)NULL); /* stage#38 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_01_02mm, x, MSK_W0, (Ull)in12_01_00, 0, 15, 0, (Ull)NULL); /* stage#38 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[38][0][1], EXP_H3210, wgt12_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt12_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt12_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#39 */
    // CH12 Line02
    mop(OP_LDR,  3, &BR[39][0][1], (Uint)in12_02_00mm, x, MSK_W0, (Ull)in12_02_00, 0, 15, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r1,           (Uint)in12_02_01mm, x, MSK_W0, (Ull)in12_02_00, 0, 15, 0, (Ull)NULL); /* stage#39 */
    mop(OP_LDR,  3, &r2,           (Uint)in12_02_02mm, x, MSK_W0, (Ull)in12_02_00, 0, 15, 0, (Ull)NULL); /* stage#39 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[39][0][1], EXP_H3210, wgt12_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt12_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt12_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#40 */

    // CH13 Line00
    mop(OP_LDR,  3, &BR[40][0][1], (Uint)in13_00_00mm, x, MSK_W0, (Ull)in13_00_00, 0, 15, 0, (Ull)NULL); /* stage#40 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_00_01mm, x, MSK_W0, (Ull)in13_00_00, 0, 15, 0, (Ull)NULL); /* stage#40 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_00_02mm, x, MSK_W0, (Ull)in13_00_00, 0, 15, 0, (Ull)NULL); /* stage#40 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[40][0][1], EXP_H3210, wgt13_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt13_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt13_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#41 */
    // CH13 Line01
    mop(OP_LDR,  3, &BR[41][0][1], (Uint)in13_01_00mm, x, MSK_W0, (Ull)in13_01_00, 0, 15, 0, (Ull)NULL); /* stage#41 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_01_01mm, x, MSK_W0, (Ull)in13_01_00, 0, 15, 0, (Ull)NULL); /* stage#41 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_01_02mm, x, MSK_W0, (Ull)in13_01_00, 0, 15, 0, (Ull)NULL); /* stage#41 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[41][0][1], EXP_H3210, wgt13_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt13_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt13_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#42 */
    // CH13 Line02
    mop(OP_LDR,  3, &BR[42][0][1], (Uint)in13_02_00mm, x, MSK_W0, (Ull)in13_02_00, 0, 15, 0, (Ull)NULL); /* stage#42 */
    mop(OP_LDR,  3, &r1,           (Uint)in13_02_01mm, x, MSK_W0, (Ull)in13_02_00, 0, 15, 0, (Ull)NULL); /* stage#42 */
    mop(OP_LDR,  3, &r2,           (Uint)in13_02_02mm, x, MSK_W0, (Ull)in13_02_00, 0, 15, 0, (Ull)NULL); /* stage#42 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[42][0][1], EXP_H3210, wgt13_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt13_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt13_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#43 */

    // CH14 Line00
    mop(OP_LDR,  3, &BR[43][0][1], (Uint)in14_00_00mm, x, MSK_W0, (Ull)in14_00_00, 0, 15, 0, (Ull)NULL); /* stage#43 */
    mop(OP_LDR,  3, &r1,           (Uint)in14_00_01mm, x, MSK_W0, (Ull)in14_00_00, 0, 15, 0, (Ull)NULL); /* stage#43 */
    mop(OP_LDR,  3, &r2,           (Uint)in14_00_02mm, x, MSK_W0, (Ull)in14_00_00, 0, 15, 0, (Ull)NULL); /* stage#43 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[43][0][1], EXP_H3210, wgt14_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt14_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt14_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#44 */
    // CH14 Line01
    mop(OP_LDR,  3, &BR[44][0][1], (Uint)in14_01_00mm, x, MSK_W0, (Ull)in14_01_00, 0, 15, 0, (Ull)NULL); /* stage#44 */
    mop(OP_LDR,  3, &r1,           (Uint)in14_01_01mm, x, MSK_W0, (Ull)in14_01_00, 0, 15, 0, (Ull)NULL); /* stage#44 */
    mop(OP_LDR,  3, &r2,           (Uint)in14_01_02mm, x, MSK_W0, (Ull)in14_01_00, 0, 15, 0, (Ull)NULL); /* stage#44 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[44][0][1], EXP_H3210, wgt14_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#45 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt14_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#45 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt14_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#45 */
    // CH14 Line02
    mop(OP_LDR,  3, &BR[45][0][1], (Uint)in14_02_00mm, x, MSK_W0, (Ull)in14_02_00, 0, 15, 0, (Ull)NULL); /* stage#45 */
    mop(OP_LDR,  3, &r1,           (Uint)in14_02_01mm, x, MSK_W0, (Ull)in14_02_00, 0, 15, 0, (Ull)NULL); /* stage#45 */
    mop(OP_LDR,  3, &r2,           (Uint)in14_02_02mm, x, MSK_W0, (Ull)in14_02_00, 0, 15, 0, (Ull)NULL); /* stage#45 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[45][0][1], EXP_H3210, wgt14_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#46 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt14_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#46 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt14_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#46 */

    // CH15 Line00
    mop(OP_LDR,  3, &BR[46][0][1], (Uint)in15_00_00mm, x, MSK_W0, (Ull)in15_00_00, 0, 15, 0, (Ull)NULL); /* stage#46 */
    mop(OP_LDR,  3, &r1,           (Uint)in15_00_01mm, x, MSK_W0, (Ull)in15_00_00, 0, 15, 0, (Ull)NULL); /* stage#46 */
    mop(OP_LDR,  3, &r2,           (Uint)in15_00_02mm, x, MSK_W0, (Ull)in15_00_00, 0, 15, 0, (Ull)NULL); /* stage#46 */
    exe(OP_FMA,  &r00, r20, EXP_H3210, BR[46][0][1], EXP_H3210, wgt15_00_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#47 */
    exe(OP_FMA,  &r01, r21, EXP_H3210, r1,           EXP_H3210, wgt15_00_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#47 */
    exe(OP_FMA,  &r02, r22, EXP_H3210, r2,           EXP_H3210, wgt15_00_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#47 */
    // CH15 Line01
    mop(OP_LDR,  3, &BR[47][0][1], (Uint)in15_01_00mm, x, MSK_W0, (Ull)in15_01_00, 0, 15, 0, (Ull)NULL); /* stage#47 */
    mop(OP_LDR,  3, &r1,           (Uint)in15_01_01mm, x, MSK_W0, (Ull)in15_01_00, 0, 15, 0, (Ull)NULL); /* stage#47 */
    mop(OP_LDR,  3, &r2,           (Uint)in15_01_02mm, x, MSK_W0, (Ull)in15_01_00, 0, 15, 0, (Ull)NULL); /* stage#47 */
    exe(OP_FMA,  &r10, r00, EXP_H3210, BR[47][0][1], EXP_H3210, wgt15_01_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#48 */
    exe(OP_FMA,  &r11, r01, EXP_H3210, r1,           EXP_H3210, wgt15_01_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#48 */
    exe(OP_FMA,  &r12, r02, EXP_H3210, r2,           EXP_H3210, wgt15_01_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#48 */
    // CH15 Line02
    mop(OP_LDR,  3, &BR[48][0][1], (Uint)in15_02_00mm, x, MSK_W0, (Ull)in15_02_00, 0, 15, 0, (Ull)NULL); /* stage#48 */
    mop(OP_LDR,  3, &r1,           (Uint)in15_02_01mm, x, MSK_W0, (Ull)in15_02_00, 0, 15, 0, (Ull)NULL); /* stage#48 */
    mop(OP_LDR,  3, &r2,           (Uint)in15_02_02mm, x, MSK_W0, (Ull)in15_02_00, 0, 15, 0, (Ull)NULL); /* stage#48 */
    exe(OP_FMA,  &r20, r10, EXP_H3210, BR[48][0][1], EXP_H3210, wgt15_02_00, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */
    exe(OP_FMA,  &r21, r11, EXP_H3210, r1,           EXP_H3210, wgt15_02_01, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */
    exe(OP_FMA,  &r22, r12, EXP_H3210, r2,           EXP_H3210, wgt15_02_02, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#49 */

    exe(OP_FAD,  &r00, r20, EXP_H3210, r21, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#50 */
    exe(OP_FAD,  &r01, r22, EXP_H3210, b,   EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#50 */

    exe(OP_FAD,  &AR[51][0], r00,    EXP_H3210, r01, EXP_H3210, 0, EXP_H3210, OP_NOP, 0, OP_NOP, 0); /* stage#51 */
    mop(OP_STR,   3, &AR[51][0],    (Uint)o_mm, x, MSK_W0, (Ull)o, 0, 13, 0, (Ull)NULL); /* stage#51 */
#ifdef DEB_NC
    printf("o = %x \n", o);
    monitor_float(i, AR[51][0], 0, 0, 0);
    i=0;
#endif
  }
//EMAX5A end

  return(0);


}
#endif


void conv(float *in, float *w, float *b, float *out, float *IN_EMAX, float *OUT_EMAX, int layernum, p_layer_t param_layer[]) {

  p_layer_t p;
  p = param_layer[layernum];
  int i;
  int n, g, icn, ocn, x, y, ky, kx;
  int index, index_k;
  float *p_src, *p_dst, *p_wgt;
  float bias;
  float dst;
  float weight[3*3*4];
  union { Uint i; float f; } u_dst, u_src, u_wgt;
  int outGroupCn = p.ocn / p.group;
  int inpGroupCn = p.icn / p.group;
  int isize = GetSize( "in", layernum, param_layer);
  int osize = GetSize("out", layernum, param_layer);
  int icnoffs = 0;
  int cnt = 0;

//  if (p.padH != 0 || p.padW != 0) {
#if defined(ARMSIML) || defined(ARMZYNQ)
//#if defined(ARMZYNQ)
    //AddPad_Split4KB(in, IN_EMAX, layernum, param_layer);
    AddPad_emax(in, IN_EMAX, layernum, param_layer);
#else
    AddPad_malloc(&in, layernum, param_layer);
#endif
//  }

  //for (i = 0; i < 100; i++) {
      //printf("[%d] %f\n",i,IN_EMAX[i]);
  //}
#ifndef TIMEMEASURE
  printf("conv start %d\n", layernum);
#endif

//#define SHORT0
//#define SHORT1
//#define SHORT2
//#define SHORT3

#ifdef ARMSIML
    _getpa();
#endif
  for (n = 0; n < p.num; n++)
  {
#ifndef SHORT0
    for (g = 0; g < p.group; g++)
#else
    for (g = 0; g < 1; g++)
#endif
    {
#ifndef SHORT1
      for (icn = 0; icn < inpGroupCn; ++icn)
#else
      for (icn = 0; icn < 1; ++icn) // MARK
#endif
      {
#if defined(ARMSIML) || defined(ARMZYNQ)
        switch(layernum) {
        case 0 : 
          //p_src = &in[n * isize + (p.inpH + p.padH * 2) * (p.inpW + p.padW * 2) * (icn + inpGroupCn * g)];
          p_src = &IN_EMAX[n * isize + (p.inpH + p.padH * 2) * (p.inpW + p.padW * 2) * (icn + inpGroupCn * g)];
          break;
        case 4 :
          //p_src = &IN_EMAX[n * inpGroupCn*p.group * 1024 + 1024 * (icn + inpGroupCn * g)];
          p_src = &IN_EMAX[n * isize + (p.inpH + p.padH * 2) * (p.inpW + p.padW * 2) * (icn + inpGroupCn * g)];
          break;
        case 8:
        case 10:
        case 12:
          //icnoffs = 3;
          //p_src = &IN_EMAX[n * inpGroupCn*p.group/(icnoffs+1) * 1024 + 1024 * (icn + inpGroupCn * g)/(icnoffs+1)];
          p_src = &IN_EMAX[n * isize + (p.inpH + p.padH * 2) * (p.inpW + p.padW * 2) * (icn + inpGroupCn * g)];
          break;
        }
#else
        p_src = &in[n * isize + (p.inpH + p.padH * 2) * (p.inpW + p.padW * 2) * (icn + inpGroupCn * g)];
#endif
#ifndef SHORT2
        for (y = 0; y < p.outH; ++y)
#else
        for (y = 0; y < 2; ++y)
#endif
        {
#ifndef SHORT3
          for (ocn = 0; ocn < outGroupCn; ++ocn)
#else
          for (ocn = 0; ocn < 4; ++ocn)
#endif
          {
#if defined(ARMSIML) || defined(ARMZYNQ)
            switch(layernum) {
            case 0 : 
              //p_dst = &out[n * osize + p.outH * p.outW * (ocn + outGroupCn * g)];
              p_dst = &OUT_EMAX[n * osize + p.outH * p.outW * (ocn + outGroupCn * g)];
              break;
            case 4 :
              //p_dst = &OUT_EMAX[n * outGroupCn*p.group*1024 + 1024 * (ocn + outGroupCn * g)];
              p_dst = &OUT_EMAX[n * osize + p.outH * p.outW * (ocn + outGroupCn * g)];
              //printf("OUT_EMAX %03d %p\n",ocn,p_dst);
              break;
            case 8:
            case 10:
            case 12:
              //p_dst = &OUT_EMAX[n * (outGroupCn*p.group>>2)*1024 + 1024 * ((ocn + outGroupCn * g)>>2) + p.outH * p.outW * ((ocn + outGroupCn * g)%4)];
              p_dst = &OUT_EMAX[n * osize + p.outH * p.outW * (ocn + outGroupCn * g)];
              //if (ocn < 16) printf("OUT_EMAX %d %p\n",ocn,p_dst);
              break;
            }
#else
            p_dst = &out[n * osize + p.outH * p.outW * (ocn + outGroupCn * g)];
#endif
            p_wgt = &w[(ocn + g * outGroupCn) * p.kerH * p.kerW * inpGroupCn];
#ifndef EMAX
goto not_use_emax;
#endif

            bias =  (p.bias && icn == 0) ? b[n * p.ocn + g * outGroupCn + ocn] : 0.0f;
            switch(layernum) {
            case 0 : 
            //case 4 :
            //case 8:
            //case 10:
            //case 12:
              conv1_emax(&p_src[(y * p.strideH)*(p.inpW + p.padW * 2)], &p_wgt[p.kerW * p.kerH * icn], bias, &p_dst[y * p.outW], (p.inpW + p.padW * 2), p.strideW, p.outW, icn);
              icnoffs = 0;
              break;
            case 4 :
              //memcpy(weight, &p_wgt[p.kerW * p.kerH * icn], p.kerW*p.kerH*sizeof(float)); // 1channel
              //conv2_emax(&p_src[(y * p.strideH)*(p.inpW + p.padW * 2)], weight, bias, &p_dst[y * p.outW], (p.inpW + p.padW * 2), p.strideW, p.outW, icn, ocn, outGroupCn);
              conv2_emax(&p_src[(y * p.strideH)*(p.inpW + p.padW * 2)], &p_wgt[p.kerW * p.kerH * icn], bias, &p_dst[y * p.outW], (p.inpW + p.padW * 2), p.strideW, p.outW, icn, p.kerW);
              icnoffs = 3;
              cnt++;
              break;
            case 8:
            case 10:
            case 12:
              //memcpy(weight, &p_wgt[p.kerW * p.kerH * icn], p.kerW*p.kerH*sizeof(float)*4);	// 4channel
              //conv345_emax(&p_src[(y * p.strideH)*(p.inpW + p.padW * 2)], weight, bias, &p_dst[y * p.outW], (p.inpW + p.padW * 2), p.strideW, p.outW, icn, ocn, outGroupCn, p.kerW);
              conv345_emax(&p_src[(y * p.strideH)*(p.inpW + p.padW * 2)], &p_wgt[p.kerW * p.kerH * icn], bias, &p_dst[y * p.outW], (p.inpW + p.padW * 2), p.strideW, p.outW, icn, p.kerW);
              icnoffs = 15;
              cnt++;
              break;
            }
goto skip_cpu_calc;
not_use_emax:
{}
            for (x = 0; x < p.outW; ++x)
            {
              //if ( layernum == 4 & ocn == 0 && y == 0 && x==0) printf("wgt = %d\n",(ocn + g*outGroupCn)*p.kerH*p.kerW*inpGroupCn);
              index_k = 0;
              index = y*p.outW+x;
              for (ky = 0; ky < p.kerH; ++ky)
              {
                for (kx = 0; kx < p.kerW; ++kx)
                {
                  //if (ocn == 0 && y == 0 && x == 0) printf("[%d,%d,%d] %f,%f,%f\n",icn,ky,kx, p_dst[index],p_src[(y*p.strideH - p.padH + ky)*p.inpW+ x*p.strideW - p.padW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                  //if (&p_dst[index] == &out[0]) printf("[I] [%d,%d,%d] %f,%f,%f\n",icn,ky,kx, p_dst[index],p_src[(y*p.strideH - p.padH + ky)*p.inpW+ x*p.strideW - p.padW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                  if (kx == 0 && ky == 0 && icn == 0){
                      //p_dst[index] = p_src[(y*p.strideH - p.padH + ky)*p.inpW+ x*p.strideW - p.padW + kx]
                      //               * p_wgt[index_k+p.kerW*p.kerH*icn];
                      //p_dst[index] = p_src[(y * p.strideH + ky) * (p.inpW + p.padW * 2)+ x * p.strideW + kx] * p_wgt[index_k + p.kerW * p.kerH * icn];
                      p_dst[index] = p_src[(y * p.strideH + ky) * (p.inpW + p.padW * 2)+ x * p.strideW + kx] * p_wgt[index_k + p.kerW * p.kerH * icn] + b[n * p.ocn + g * outGroupCn + ocn];
                  }
                  else{
                      p_dst[index] += p_src[(y * p.strideH + ky) * (p.inpW + p.padW * 2)+ x * p.strideW + kx] * p_wgt[index_k + p.kerW * p.kerH * icn];
                  }
                  //u_dst.f = p_dst[index];
                  //u_src.f = p_src[(y*p.strideH - p.padH + ky)*p.inpW+x*p.strideW - p.padW + kx];
                  //u_wgt.f = p_wgt[index_k+p.kerW*p.kerH*icn];
                  //if (ocn == 0 && y == 0 && x == 0) printf("[%d,%d,%d,%d] %f,%f,%f\n",g,icn,ky,kx, p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                  //if (&p_dst[index] == &out[0]) printf("[I] [%d,%d,%d] %f,%f,%f\n",icn,ky,kx, p_dst[index],p_src[(y*p.strideH - p.padH + ky)*p.inpW+ x*p.strideW - p.padW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                  //if (ocn == 0 && icn == 0 && y == 0 && layernum == 0) printf("[%d,%d,%d,%d,%d] %f,%f,%f\n",g,ocn ,x, ky,kx, p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                  //if (ky == 10 && kx == 10 && layernum == 0) printf("[%d,%d,%d,%d,%d,%d] %f,%f,%f\n",icn,ocn ,y, x, ky,kx, p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                  //if (g == 0 && icn <= 1  && ocn == 0 && y == 0 & x == 0 & ky <= 4 && kx <= 4 && layernum == 4) printf("[%d,%d,%d,%d,%d,%d,%d] %f,%f,%f %p\n",g,icn, ocn, y,x, ky,kx, p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn],&p_dst[index]);
                  //if (g == 0 && icn == 0  && y == 0 & x == 0 & ky == 2 && kx == 2 && layernum == 8) printf("[%d,%d,%d,%d,%d,%d,%d] %f,%f,%f %p\n",g,icn, ocn, y,x, ky,kx, p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn],&p_dst[index]);
                  //printf("[%d,%d,%d,%d,%d,%d,%d] %f,%f,%f\n",g, icn, ocn, y, x, ky, kx,  p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);

                  ++index_k;
                }
              }
              //if (p.bias && icn == (inpGroupCn - 1))
              if (p.bias && icn == 0)
              {
                //p_dst[index] += b[n * p.ocn + ocn];
                //if (&p_dst[index] == &out[0]) printf("[B] [%d,%d,%d] %f,%f,%f\n",icn,ky,kx, p_dst[index],p_src[(y*p.strideH - p.padH + ky)*p.inpW+ x*p.strideW - p.padW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
                //if (layernum == 0) printf("[B][%d,%d,%d,%d,%d,%d] %f,%f,%f\n",icn,ocn ,y, x, ky,kx, p_dst[index],p_src[(y*p.strideH + ky)*(p.inpW+p.padW*2)+ x*p.strideW + kx], p_wgt[index_k+p.kerW*p.kerH*icn]);
              }
            }
            //if (ocn == 0 & y == 0 ) printf("[%d] %f\n",icn,p_dst[0]);
skip_cpu_calc:
{}
          }
        }
        icn += icnoffs;
      }
    }
  }
//EMAX5A drain_dirty_lmm
//emax5_drain_dirty_lmm();
#ifdef ARMSIML
    _getpa();
#endif

#ifndef EMAX
  if (p.padH != 0 || p.padW != 0) {
    free(in);
  }
#else
 FILE *fp;
 switch(layernum) {
 case 0 :
 case 4 :
 case 8:
 case 10:
 case 12:
 //fp = fopen("out_pb","wb");
 //fwrite(OUT_EMAX,sizeof(float)*outGroupCn*p.group*1024, 1, fp);
 //fclose(fp);
 //Pack_For_Cpu(OUT_EMAX, out, layernum, param_layer);
 Emax2Cpu(OUT_EMAX, out, layernum, param_layer);
 //fp = fopen("out_pa","wb");
 //fwrite(out,sizeof(float)*outGroupCn*p.group*1024, 1, fp);
 //fclose(fp);
 //exit(1);
   break;
 }
#endif
  //printf("cnt = %d\n",cnt);
  return;
}

//int debu_layer_out(Sfloat *OUT, int layernum, char *name) {
//  FILE *fp;
//  char str[100];
// 
//#ifdef DLOON
//
//#ifndef EMAX
//  sprintf(str,"%s",name);
//#else
//  sprintf(str,"%s_emax",name);
//#endif
//  if ((fp = fopen(str,"wb")) != NULL) {
//    fwrite(OUT, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
//    fclose(fp);
//  } else {
//    return(-1);
//  }
//
//#endif
//
//}

void deeppose_init(float **in, float **out, float **weight, float **bias, float **buf, float **in_emax, float **out_emax, p_layer_t param_layer[]) {

  int insize, outsize, weightsize, biassize;

  SetParamLayer(param_layer);
  GetMaxSize(&insize, &outsize, &weightsize, &biassize, param_layer);
  sysinit(insize, outsize, weightsize, biassize, in, out, buf, weight, bias, in_emax, out_emax);

}

#ifndef TIMEMEASURE
void deeppose_start(float *in, float *out, float *weight, float *bias, float *in_emax, float *out_emax, p_layer_t param_layer[]) {
  FILE *fp;
  int layernum  = 0;
  int convfcnum = 1;

// --------
// Layer0-3
// --------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  relu(out, in, 0.0, layernum++, param_layer);
  pool(in, out, layernum++, param_layer);
  lrn(out, in, 5, 0.0001f, 0.75f, layernum++, param_layer);
// --------
// Layer4-7
// --------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  relu(out, in, 0.0, layernum++, param_layer);
  pool(in, out, layernum++, param_layer);
  lrn(out, in, 5, 0.0001f, 0.75f, layernum++, param_layer);
  
// --------
// Layer8-9
// --------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  relu(out, in, 0.0, layernum++, param_layer);

// ----------
// Layer10-11
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  relu(out, in, 0.0, layernum++, param_layer);

// ----------
// Layer12-14
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  relu(out, in, 0.0, layernum++, param_layer);
  pool(in, out, layernum++, param_layer);

// ----------
// Layer15-16
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  fc(out, weight, bias, in, layernum++, param_layer);
  relu(in, out, 0.0, layernum++, param_layer);

// ----------
// Layer17-18
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  fc(out, weight, bias, in, layernum++, param_layer);
  relu(in, out, 0.0, layernum++, param_layer);

// ----------
// Layer19-20
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  fc(out, weight, bias, in, layernum, param_layer);
  sigmoid(in, out, layernum, param_layer);
  
#ifdef EMAX
 fp = fopen("_output_emax","wb");
#else
 fp = fopen("_output","wb");
#endif
 printf("--output--\n");
 fwrite(out, sizeof(float)*param_layer[layernum].outH*param_layer[layernum].outW*param_layer[layernum].ocn, 1, fp);
 fclose(fp);
}
#else
void deeppose_start(float *in, float *out, float *weight, float *bias, float *in_emax, float *out_emax, p_layer_t param_layer[]) {
  FILE *fp;
  int layernum  = 0;
  int convfcnum = 1;
  struct timeval s, e;

// --------
// Layer0-3
// --------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("conv1 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  relu(out, in, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu1 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  pool(in, out, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("pool1 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  lrn(out, in, 5, 0.0001f, 0.75f, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("lrn1 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
// --------
// Layer4-7
// --------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("conv2 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  relu(out, in, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu2 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  pool(in, out, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("pool2 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  lrn(out, in, 5, 0.0001f, 0.75f, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("lrn2 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
  
// --------
// Layer8-9
// --------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("conv3 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
  
  gettimeofday(&s, NULL);
  relu(out, in, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu3 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

// ----------
// Layer10-11
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("conv4 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
  
  gettimeofday(&s, NULL);
  relu(out, in, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu4 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

// ----------
// Layer12-14
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  conv(in, weight, bias, out, in_emax, out_emax, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("conv5 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
  
  gettimeofday(&s, NULL);
  relu(out, in, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu5 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  pool(in, out, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("pool5 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

// ----------
// Layer15-16
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  fc(out, weight, bias, in, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("fc6 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  relu(in, out, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu6 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

// ----------
// Layer17-18
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  fc(out, weight, bias, in, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("fc7 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  relu(in, out, 0.0, layernum++, param_layer);
  gettimeofday(&e, NULL);
  printf("relu7 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

// ----------
// Layer19-20
// ----------
  if ( ReadWBFILE(weight, bias, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  gettimeofday(&s, NULL);
  fc(out, weight, bias, in, layernum, param_layer);
  gettimeofday(&e, NULL);
  printf("fc8 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);

  gettimeofday(&s, NULL);
  sigmoid(in, out, layernum, param_layer);
  gettimeofday(&e, NULL);
  printf("sigmoid8 %d ms\n",(e.tv_sec - s.tv_sec) * 1000 + (e.tv_usec - s.tv_usec) / 1000);
  
#ifdef EMAX
 fp = fopen("_output_emax","wb");
#else
 fp = fopen("_output","wb");
#endif
 printf("--output--\n");
 fwrite(out, sizeof(float)*param_layer[layernum].outH*param_layer[layernum].outW*param_layer[layernum].ocn, 1, fp);
 fclose(fp);
}
#endif
