#ifndef __DEEPPOLIB_H__
#define __DEEPPOLIB_H__

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
typedef double             Sdbl;
typedef float              Sfloat;
#endif

#define LAYERNUM 21
typedef struct {
  char name[32];
  int icn;
  int inpH;
  int inpW;
  int padH;
  int padW;
  int strideH;
  int strideW;
  int num;
  int group;
  int ocn;
  int outH;
  int outW;
  int kerH;
  int kerW;
  int bias;
} p_layer_t;


#ifdef __cplusplus
extern "C" {
#endif
void sysinit(int isize, int osize, int wsize, int bsize, float **IN, float **OUT, float **BUF, float **WEIGHT, float **BIAS, float **IN_EMAX, float **OUT_EMAX);
void systerm();
void SetParamLayer(p_layer_t param_layer[]);
void GetMaxSize(int *insize, int *outsize, int *weightsize, int *biassize, p_layer_t param_layer[]);
int GetSize(char const *inout, int layernum, p_layer_t param_layer[]);
int ReadWBFILE(float *w, float *b, int num, char const *layer, int layernum, p_layer_t param_layer[]);
void AddPad(float **in, float **buf, int layernum, p_layer_t param_layer[]);
void relu(float *in, float *out, float negative_slope, int layernum, p_layer_t param_layer[]);
void pool(float *in, float *out, int layernum, p_layer_t param_layer[]);
void lrn(float *in, float *out, int size, float alpha, float beta, int layernum, p_layer_t param_layer[]);
void fc(float *in, float *w, float *b, float *out, int layernum, p_layer_t param_layer[]);
void sigmoid(float *in, float *out, int layernum, p_layer_t param_layer[]);
void conv(float *in, float *w, float *b, float *out, float *in_emax, float *out_emax, int layernum, p_layer_t param_layer[]);
void deeppose_init(float **in, float **out, float **weight, float **bias, float **buf, float **in_emax, float **out_emax, p_layer_t param_layer[]);
void deeppose_start(float *in, float *out, float *weight, float *bias, float *in_emax, float *out_emax, p_layer_t param_layer[]);

#ifdef __cplusplus
}
#endif
#endif
