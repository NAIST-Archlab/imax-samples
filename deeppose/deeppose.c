static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* Gather data from light-field-camera and display image */
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#undef PRECISE_SCALE

#ifndef ARMSIML
#define _POSIX_THREADS
#endif

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <float.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <sys/mman.h>
#endif
#if !defined(ARMSIML) && !defined(ARMZYNQ)
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif

#if defined(ARMZYNQ) || defined(EMAX)
#include "../../src/conv-c2d_32bit_64st/emax5.h"
#endif

#include "deepposelib.h"
//#include "deepposelib.c"

void *deeppose();
#define abs(a) (((a)<0)?-(a):(a))

#define MAXTHNUM 2048
#ifdef PTHREAD
#define THNUM 8
#ifndef ARMSIML
pthread_t th[MAXTHNUM];
#endif
#else
#define THNUM 1
#endif

struct param_kernel {
  int th;
  int v; /* valid */
  int from;
  int to;
} param_kernel[MAXTHNUM];

p_layer_t param_layer[LAYERNUM];
/****************/
/*** IN/OUT   ***/
/****************/
Sfloat *IN;
Sfloat *OUT;
Sfloat *BUF;
Sfloat *WEIGHT;
Sfloat *BIAS;
Sfloat *IN_EMAX;
Sfloat *OUT_EMAX;

#define CIRCLESIZE   7
#define WD           227
#define HT           227
#define BITMAP       (WD*HT)
#define CHANEL       3
Uint    IMG[BITMAP];
Uchar   X[BITMAP*3*12];

#if !defined(ARMSIML) && !defined(ARMZYNQ)
/***********/
/* for X11 */
/***********/
Display              *disp;          /* display we're sending to */
int                  scrn;           /* screen we're sending to */

typedef struct {
  unsigned int  width;  /* width of image in pixels */
  unsigned int  height; /* height of image in pixels */
  unsigned char *data;  /* data rounded to full byte for each row */
} Image;

typedef struct {
  Display  *disp;       /* destination display */
  int       scrn;       /* destination screen */
  int       depth;      /* depth of drawable we want/have */
  int       dpixlen;    /* bitsPerPixelAtDepth */
  Drawable  drawable;   /* drawable to send image to */
  Colormap  cmap;       /* colormap used for image */
  GC        gc;         /* cached gc for sending image */
  XImage   *ximage;     /* ximage structure */
} XImageInfo;

union {
  XEvent              event;
  XAnyEvent           any;
  XButtonEvent        button;
  XExposeEvent        expose;
  XMotionEvent        motion;
  XResizeRequestEvent resize;
  XClientMessageEvent message;
} event;

unsigned int          redvalue[256], greenvalue[256], bluevalue[256];
XImageInfo            ximageinfo;
Image                 imageinfo;  /* image that will be sent to the display */
unsigned long         doMemToVal(unsigned char *p, unsigned int len);
unsigned long         doValToMem(unsigned long val, unsigned char *p, unsigned int len);
unsigned int          bitsPerPixelAtDepth();

#define TRUE_RED(PIXVAL)      (((PIXVAL) & 0xff0000) >> 16)
#define TRUE_GREEN(PIXVAL)    (((PIXVAL) &   0xff00) >>  8)
#define TRUE_BLUE(PIXVAL)     (((PIXVAL) &     0xff)      )
#define memToVal(PTR,LEN)     ((LEN) == 1 ? (unsigned long)(*(PTR)) : doMemToVal(PTR,LEN))
#define valToMem(VAL,PTR,LEN) ((LEN) == 1 ? (unsigned long)(*(PTR) = (unsigned char)(VAL)) : doValToMem(VAL,PTR,LEN))

x11_open()
{
  if (!(disp = XOpenDisplay(NULL))) {
    printf("%s: Cannot open display\n", XDisplayName(NULL));
    exit(1);
  }
  scrn = DefaultScreen(disp);
  imageinfo.width = WD*4;
  imageinfo.height= HT*3;
  imageinfo.data  = X;
  imageInWindow(&ximageinfo, disp, scrn, &imageinfo);
}

x11_update()
{
  unsigned int  x, y;
  unsigned int  pixval, newpixval;
  unsigned char *destptr, *srcptr;

  destptr = ximageinfo.ximage->data;
  srcptr  = imageinfo.data;
  for (y= 0; y < imageinfo.height; y++) {
    for (x= 0; x < imageinfo.width; x++) {
      pixval= memToVal(srcptr, 3);
      newpixval= redvalue[TRUE_RED(pixval)] | greenvalue[TRUE_GREEN(pixval)] | bluevalue[TRUE_BLUE(pixval)];
      valToMem(newpixval, destptr, ximageinfo.dpixlen);
      srcptr += 3;
      destptr += ximageinfo.dpixlen;
    }
  }
  XPutImage(ximageinfo.disp, ximageinfo.drawable, ximageinfo.gc,
        ximageinfo.ximage, 0, 0, 0, 0, imageinfo.width, imageinfo.height);
}

x11_checkevent()
{
  XNextEvent(disp, &event.event);
  switch (event.any.type) {
  case KeyPress:
    return (0);
  default:
    x11_update();
    return (1);
  }
}

x11_close()
{
  XCloseDisplay(disp);
}

imageInWindow(ximageinfo, disp, scrn, image)
     XImageInfo   *ximageinfo;
     Display      *disp;
     int           scrn;
     Image        *image;
{ 
  Window                ViewportWin;
  Visual               *visual;
  unsigned int          depth;
  unsigned int          dpixlen;
  XSetWindowAttributes  swa_view;
  XSizeHints            sh;
  unsigned int pixval;
  unsigned int redcolors, greencolors, bluecolors;
  unsigned int redstep, greenstep, bluestep;
  unsigned int redbottom, greenbottom, bluebottom;
  unsigned int redtop, greentop, bluetop;
  XColor        xcolor;
  unsigned int  a, b, newmap, x, y;
  XGCValues gcv;

  bestVisual(disp, scrn, &visual, &depth);
  dpixlen = (bitsPerPixelAtDepth(disp, depth) + 7) / 8;

  ximageinfo->disp    = disp;
  ximageinfo->scrn    = scrn;
  ximageinfo->depth   = depth;
  ximageinfo->dpixlen = dpixlen;
  ximageinfo->drawable= None;
  ximageinfo->gc      = NULL;
  ximageinfo->ximage  = XCreateImage(disp, visual, depth, ZPixmap, 0,
                     NULL, image->width, image->height,
                     8, 0);
  ximageinfo->ximage->data= (unsigned char*)malloc(image->width * image->height * dpixlen);
  ximageinfo->ximage->byte_order= MSBFirst; /* trust me, i know what
                         * i'm talking about */

  if (visual == DefaultVisual(disp, scrn))
    ximageinfo->cmap= DefaultColormap(disp, scrn);
  else
    ximageinfo->cmap= XCreateColormap(disp, RootWindow(disp, scrn), visual, AllocNone);
    
  redcolors= greencolors= bluecolors= 1;
  for (pixval= 1; pixval; pixval <<= 1) {
    if (pixval & visual->red_mask)
      redcolors <<= 1;
    if (pixval & visual->green_mask)
      greencolors <<= 1;
    if (pixval & visual->blue_mask)
      bluecolors <<= 1;
  }
    
  redstep= 256 / redcolors;
  greenstep= 256 / greencolors;
  bluestep= 256 / bluecolors;
  redbottom= greenbottom= bluebottom= 0;
  for (a= 0; a < visual->map_entries; a++) {
    if (redbottom < 256)
      redtop= redbottom + redstep;
    if (greenbottom < 256)
      greentop= greenbottom + greenstep;
    if (bluebottom < 256)
      bluetop= bluebottom + bluestep;
    
    xcolor.flags= DoRed | DoGreen | DoBlue;
    xcolor.red  = (redtop - 1) << 8;
    xcolor.green= (greentop - 1) << 8;
    xcolor.blue = (bluetop - 1) << 8;
    XAllocColor(disp, ximageinfo->cmap, &xcolor);
    
    while ((redbottom < 256) && (redbottom < redtop))
      redvalue[redbottom++]= xcolor.pixel & visual->red_mask;
    while ((greenbottom < 256) && (greenbottom < greentop))
      greenvalue[greenbottom++]= xcolor.pixel & visual->green_mask;
    while ((bluebottom < 256) && (bluebottom < bluetop))
      bluevalue[bluebottom++]= xcolor.pixel & visual->blue_mask;
  }

  swa_view.background_pixel= WhitePixel(disp,scrn);
  swa_view.backing_store= WhenMapped;
  swa_view.cursor= XCreateFontCursor(disp, XC_watch);
  swa_view.event_mask= ButtonPressMask | Button1MotionMask | KeyPressMask |
    StructureNotifyMask | EnterWindowMask | LeaveWindowMask | ExposureMask;
  swa_view.save_under= False;
  swa_view.bit_gravity= NorthWestGravity;
  swa_view.save_under= False;
  swa_view.colormap= ximageinfo->cmap;
  swa_view.border_pixel= 0;
  ViewportWin= XCreateWindow(disp, RootWindow(disp, scrn), 0, 0,
                 image->width, image->height, 0,
                 DefaultDepth(disp, scrn), InputOutput,
                 DefaultVisual(disp, scrn),
                 CWBackingStore | CWBackPixel |
                 CWEventMask | CWSaveUnder,
                 &swa_view);
  ximageinfo->drawable= ViewportWin;

  gcv.function= GXcopy;
  ximageinfo->gc= XCreateGC(ximageinfo->disp, ximageinfo->drawable, GCFunction, &gcv);

  sh.width= image->width;
  sh.height= image->height;
  sh.min_width= image->width;
  sh.min_height= image->height;
  sh.max_width= image->width;
  sh.max_height= image->height;
  sh.width_inc= 1;
  sh.height_inc= 1;
  sh.flags= PMinSize | PMaxSize | PResizeInc | PSize;
  XSetNormalHints(disp, ViewportWin, &sh);

  XStoreName(disp, ViewportWin, "ppmdepth");
  XMapWindow(disp, ViewportWin);
  XSync(disp,False);
}

Visual *bestVisualOfClassAndDepth(disp, scrn, class, depth)
     Display      *disp;
     int           scrn;
     int           class;
     unsigned int  depth;
{
  Visual *best= NULL;
  XVisualInfo template, *info;
  int nvisuals;

  template.screen= scrn;
  template.class= class;
  template.depth= depth;
  if (! (info= XGetVisualInfo(disp, VisualScreenMask | VisualClassMask |
                  VisualDepthMask, &template, &nvisuals)))
    return(NULL); /* no visuals of this depth */

  best= info->visual;
  XFree((char *)info);
  return(best);
}

bestVisual(disp, scrn, rvisual, rdepth)
     Display       *disp;
     int            scrn;
     Visual       **rvisual;
     unsigned int  *rdepth;
{ 
  unsigned int  depth, a;
  Screen       *screen;
  Visual       *visual;

  /* figure out the best depth the server supports.  note that some servers
   * (such as the HP 11.3 server) actually say they support some depths but
   * have no visuals that support that depth.  seems silly to me....
   */

  depth= 0;
  screen= ScreenOfDisplay(disp, scrn);
  for (a= 0; a < screen->ndepths; a++) {
    if (screen->depths[a].nvisuals &&
    ((!depth ||
      ((depth < 24) && (screen->depths[a].depth > depth)) ||
      ((screen->depths[a].depth >= 24) &&
       (screen->depths[a].depth < depth)))))
      depth= screen->depths[a].depth;
  }

  visual= bestVisualOfClassAndDepth(disp, scrn, TrueColor, depth);

  *rvisual= visual;
  *rdepth= depth;
}

unsigned int bitsPerPixelAtDepth(disp, depth)
     Display      *disp;
     unsigned int  depth;
{
  XPixmapFormatValues *xf;
  unsigned int nxf, a;

  xf = XListPixmapFormats(disp, (int *)&nxf);
  for (a = 0; a < nxf; a++)
    if (xf[a].depth == depth)
      return(xf[a].bits_per_pixel);

  fprintf(stderr, "bitsPerPixelAtDepth: Can't find pixmap depth info!\n");
  exit(1);
}     

unsigned long doMemToVal(p, len)
     unsigned char *p;
     unsigned int  len;
{
  unsigned int  a;
  unsigned long i;

  i= 0;
  for (a= 0; a < len; a++)
    i= (i << 8) + *(p++);
  return(i);
}

unsigned long doValToMem(val, p, len)
     unsigned long  val;
     unsigned char  *p;
     unsigned int   len;
{
  int a;

  for (a= len - 1; a >= 0; a--) {
    *(p + a)= val & 0xff;
    val >>= 8;
  }
  return(val);
}
#endif

/****************/
/*** ZYNQ     ***/
/****************/


/****************/
/*** MAIN     ***/
/****************/
main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp;
  int i, j, k, fc;
  char dummy[16];
  int insize, outsize, weightsize, biassize;
#ifndef ARMSIML
  fd_set rfds;
  struct timeval tv;
  char cmd[1024];
#endif

  
  SetParamLayer(param_layer);
  GetMaxSize(&insize, &outsize, &weightsize, &biassize, param_layer);
  printf("%d %d %d %d\n",insize, outsize, weightsize, biassize);
    
  sysinit(insize, outsize, weightsize, biassize, &IN, &OUT, &BUF, &WEIGHT, &BIAS, &IN_EMAX, &OUT_EMAX);
  printf("end sysinit\n");

#if !defined(ARMSIML) && !defined(ARMZYNQ)
  x11_open();
#endif

  while (1) {
    int x, y, dx, dy;
    int cvalR, cvalG, cvalB;
                                          /*                    8  9 10 11 12 13 14  */
#ifdef ARMSIML
    _getpa();
#endif
    printf("start deeppose\n");
#ifdef ARMSIML
    _getpa();
#endif
    deeppose(); /* search triangle in {frontier,next} */
    fp = fopen("data/_input","rb");
    fread(IN, sizeof(float)*GetSize("in",0,param_layer), 1, fp);
    deeppose_start(IN, OUT, WEIGHT, BIAS, IN_EMAX, OUT_EMAX, param_layer);

#ifdef ARMSIML
    _getpa();
#endif
    
    puts("--Press Enter--");
    while(getchar() != '\n');
    break;
  }

  systerm();
  exit(0);
}

int debu_layer_out(Sfloat *OUT, int layernum, char *name) {
  FILE *fp;
  char str[100];
 
#ifdef DLOON

#ifndef EMAX
  sprintf(str,"%s",name);
#else
  sprintf(str,"%s_emax",name);
#endif
  if ((fp = fopen(str,"wb")) != NULL) {
    fwrite(OUT, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
    fclose(fp);
  } else {
    return(-1);
  }

#endif

}

void *deeppose()
{
  FILE *fp;
  int layernum = 0;
  int convfcnum  = 1;
  int i, j;
  float *buf;
  
  printf("start file read\n");
//#define START_MIDDLE
#ifndef START_MIDDLE
  fp = fopen("data/_input","rb");
#if 1
  //printf("IN %llx\n",IN);
  fread(IN, sizeof(float)*GetSize("in",layernum,param_layer), 1, fp);
#else
  buf = (void*)malloc(sizeof(float)*GetSize("in",layernum,param_layer));
  fread(buf, sizeof(float)*GetSize("in",layernum,param_layer), 1, fp);
  for(i = 0; i < GetSize("in",layernum,param_layer); i++) {
	IN[i] = buf[i];
  }
  free(buf);
#endif
#else
  fp = fopen("deeppose_/norm1","rb");
  fread(IN, sizeof(float)*GetSize("in2",layernum,param_layer), 1, fp);
  convfcnum = 2;
  layernum  = 4;
  //fp = fopen("deeppose_/norm2","rb");
  //fp = fopen("lrn1_emax","rb");
  //convfcnum = 3;
  //layernum  = 8;
  //fp = fopen("deeppose_/relu3","rb");
  //convfcnum = 4;
  //layernum  = 10;
  //fread(IN, sizeof(float)*GetSize("in",layernum,param_layer), 1, fp);
#endif
  fclose(fp);
  printf("done file read\n");

#if !defined(ARMSIML) && !defined(ARMZYNQ) // 当面スキップ
  GenerateBitmap(IN, IMG);
#endif

#ifdef ARMSIML
  _copyX(0, IMG);
  _updateX();
#endif
#if !defined(ARMSIML) && !defined(ARMZYNQ)
  copy_X(0, IMG);
  x11_update();
#endif

  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
#ifdef ARMSIML
    _getpa();
#endif

// --------
// Layer0-3
// --------
  conv(IN, WEIGHT, BIAS, OUT, IN_EMAX, OUT_EMAX, layernum++, param_layer);

#ifdef START_MIDDLE
  layernum--;
  goto end;
#endif
  debu_layer_out(OUT, layernum, "conv1");

  relu(OUT, IN, 0.0, layernum++, param_layer);
  pool(IN, OUT, layernum++, param_layer);
  lrn(OUT, IN, 5, 0.0001f, 0.75f, layernum++, param_layer);
#ifndef EMAX
  fp = fopen("lrn1","wb");
#else
  fp = fopen("lrn1_emax","wb");
#endif
  fwrite(IN, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
  fclose(fp);
// --------
// Layer4-7
// --------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(IN, WEIGHT, BIAS, OUT, IN_EMAX, OUT_EMAX, layernum++, param_layer);
#ifndef EMAX
  fp = fopen("conv2","wb");
#else
  fp = fopen("conv2_emax","wb");
#endif
  fwrite(OUT, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
  fclose(fp);

  relu(OUT, IN, 0.0, layernum++, param_layer);
  pool(IN, OUT, layernum++, param_layer);
#ifndef EMAX
  fp = fopen("pool2","wb");
#else
  fp = fopen("pool2_emax","wb");
#endif
  fwrite(OUT, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
  fclose(fp);
  lrn(OUT, IN, 5, 0.0001f, 0.75f, layernum++, param_layer);
#ifndef EMAX
  fp = fopen("lrn2","wb");
#else
  fp = fopen("lrn2_emax","wb");
#endif
  fwrite(IN, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
  fclose(fp);
  
// --------
// Layer8-9
// --------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(IN, WEIGHT, BIAS, OUT, IN_EMAX, OUT_EMAX, layernum++, param_layer);
#ifndef EMAX
  fp = fopen("conv3","wb");
#else
  fp = fopen("conv3_emax","wb");
#endif
  fwrite(OUT, sizeof(float)*param_layer[layernum-1].outH*param_layer[layernum-1].outW*param_layer[layernum-1].ocn, 1, fp);
  fclose(fp);
  relu(OUT, IN, 0.0, layernum++, param_layer);

// ----------
// Layer10-11
// ----------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(IN, WEIGHT, BIAS, OUT, IN_EMAX, OUT_EMAX, layernum++, param_layer);
  relu(OUT, IN, 0.0, layernum++, param_layer);

// ----------
// Layer12-14
// ----------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "conv", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  conv(IN, WEIGHT, BIAS, OUT, IN_EMAX, OUT_EMAX, layernum++, param_layer);
  relu(OUT, IN, 0.0, layernum++, param_layer);
  pool(IN, OUT, layernum++, param_layer);

// ----------
// Layer15-16
// ----------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  fc(OUT, WEIGHT, BIAS, IN, layernum++, param_layer);
  relu(IN, OUT, 0.0, layernum++, param_layer);

// ----------
// Layer17-18
// ----------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  fc(OUT, WEIGHT, BIAS, IN, layernum++, param_layer);
  relu(IN, OUT, 0.0, layernum++, param_layer);

// ----------
// Layer19-20
// ----------
  if ( ReadWBFILE(WEIGHT, BIAS, convfcnum++, "fc", layernum, param_layer) != 0 ) {
    printf("[E] \n");
  }
  fc(OUT, WEIGHT, BIAS, IN, layernum, param_layer);
  sigmoid(IN, OUT, layernum, param_layer);
  
  for(i = 0; i < param_layer[layernum].ocn; i +=2) {
    DrawCircle((int)(OUT[i]*WD),(int)(OUT[i+1]*HT));
  }
#ifdef ARMSIML
  _copyX(0, IMG);
  _updateX();
#endif
#if !defined(ARMSIML) && !defined(ARMZYNQ)
  copy_X(0, IMG);
  x11_update();
#endif

end:
{}
#ifdef EMAX
 fp = fopen("_output_emax","wb");
#else
 fp = fopen("_output","wb");
#endif
 printf("--output--\n");
 //fwrite(IN, sizeof(float)*(param_layer[layernum].inpH+param_layer[layernum].padH)*(param_layer[layernum].inpW+param_layer[layernum].padW)*param_layer[layernum].icn, 1, fp);
 fwrite(OUT, sizeof(float)*param_layer[layernum].outH*param_layer[layernum].outW*param_layer[layernum].ocn, 1, fp);
 fclose(fp);
}

copy_X(id, from) 
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned char *to = X;

  switch (id) {
  case 0:                           break;
  case 1:  to += WD*3;              break;
  case 2:  to += WD*3*2;            break;
  case 3:  to += WD*3*3;            break;
  case 4:  to += WD*3*4;            break;
  case 5:  to += BITMAP*3*5;        break;
  case 6:  to += BITMAP*3*5+WD*3;   break;
  case 7:  to += BITMAP*3*5+WD*3*2; break;
   }
  for (i=0; i<HT; i++, to+=WD*3*3) {
    for (j=0; j<WD; j++, from++) {
      *to++ = *from>>24;
      *to++ = *from>>16;
      *to++ = *from>> 8;
    }
  }
}

GenerateBitmap(from, to)
    float *from;
    unsigned int  *to;
{
  int i, j;
  char c;
  
  for(i = 0; i < BITMAP; i++) {
    *to  = ((unsigned char)from[BITMAP * 0 + i]) << 8;
    *to += ((unsigned char)from[BITMAP * 1 + i]) << 16;
    *to += ((unsigned char)from[BITMAP * 2 + i]) << 24;
    //printf("%d %x %x %x %x\n",i, *to, (unsigned char)from[BITMAP * 0 + i], (unsigned char)from[BITMAP * 1 + i], (unsigned char)from[BITMAP * 2 + i]);
    //printf("%d %x %f %f %f\n",i, *to, from[BITMAP * 0 + i], from[BITMAP * 1 + i], from[BITMAP * 2 + i]);
    
    *to++;
  }
}

SetColor(p, r, g, b)
    volatile unsigned int *p;
    char r;
    char g; 
    char b;
{
  *p   = r << 24;
  *p  += g << 16;
  *p  += b << 8;
}

DrawCircle( x, y)
    int x;
    int y;
{
  volatile unsigned int *to = IMG;
  int centor;
  int size = CIRCLESIZE;
  char point[CIRCLESIZE] = {3, 5, 7, 7, 7, 5, 3};
  int i, j;

  for(i = 0; i < size; i++) {
    for(j = 0; j < point[i]; j++) {
      SetColor(&to[(y -size/2 + i)* WD + (x - point[i]/2 + j)], 255, 0 , 0);
    }
  }
  
}
