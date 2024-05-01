
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* xdisp.c 2019/10/18 */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>
#include <sys/times.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>

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
unsigned int          bitsPerPixelAtDepth();
void                  imageInWindow();
void                  bestVisual();

#define TRUE_RED(PIXVAL)      (((PIXVAL) & 0xff0000) >> 16)
#define TRUE_GREEN(PIXVAL)    (((PIXVAL) &   0xff00) >>  8)
#define TRUE_BLUE(PIXVAL)     (((PIXVAL) &     0xff)      )

void x11_open(int width, int height, int screen_wd, int screen_ht)
{
  if (!(disp = XOpenDisplay(NULL))) {
    printf("%s: Cannot open display\n", XDisplayName(NULL));
    exit(1);
  }
  scrn = DefaultScreen(disp);
  imageinfo.width = width*screen_wd;
  imageinfo.height= height*screen_ht;
  /*imageinfo.data  = malloc(sizeof(Uint)*imageinfo.width*imageinfo.height);*/
  imageInWindow(&ximageinfo, disp, scrn, &imageinfo);
}

void x11_update()
{
  XPutImage(ximageinfo.disp, ximageinfo.drawable, ximageinfo.gc,
            ximageinfo.ximage, 0, 0, 0, 0, imageinfo.width, imageinfo.height);
}

int x11_checkevent()
{
  static int stop = 0;

  x11_update();
  while (XPending(disp)) {
    XNextEvent(disp, &event.event);
    switch (event.any.type) {
    case KeyPress:
      stop = 1-stop;
      if   (stop) printf("-stopped- (type any key to continue)\n");
      else        printf("-running-\n");
      break;
    default:
      break;
    }
  }
  return (stop);
}

void x11_close()
{
  XCloseDisplay(disp);
}

void imageInWindow(ximageinfo, disp, scrn, image)
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
  unsigned int  a;
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
  ximageinfo->ximage->data= (char*)malloc(image->width * image->height * dpixlen);
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

  redtop   = 0;
  greentop = 0;
  bluetop  = 0;
  redstep  = 256 / redcolors;
  greenstep= 256 / greencolors;
  bluestep = 256 / bluecolors;
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

  XStoreName(disp, ViewportWin, "ImageSegment");
  XMapWindow(disp, ViewportWin);
  XSync(disp,False);
}

void bestVisual(disp, scrn, rvisual, rdepth)
     Display       *disp;
     int            scrn;
     Visual       **rvisual;
     unsigned int  *rdepth;
{
  unsigned int  depth, a;
  Screen       *screen;
  XVisualInfo template, *info;
  int nvisuals;

  /* figure out the best depth the server supports.  note that some servers
   * (such as the HP 11.3 server) actually say they support some depths but
   * have no visuals that support that depth.  seems silly to me....
   */
  depth = 0;
  screen= ScreenOfDisplay(disp, scrn);
  for (a= 0; a < screen->ndepths; a++) {
    if (screen->depths[a].nvisuals &&
        ((!depth ||
          ((depth < 24) && (screen->depths[a].depth > depth)) ||
          ((screen->depths[a].depth >= 24) &&
           (screen->depths[a].depth < depth)))))
      depth= screen->depths[a].depth;
  }
  template.screen= scrn;
  template.class= TrueColor;
  template.depth= depth;
  if (! (info= XGetVisualInfo(disp, VisualScreenMask | VisualClassMask | VisualDepthMask, &template, &nvisuals)))
    *rvisual= NULL; /* no visuals of this depth */
  else {
    *rvisual= info->visual;
    XFree((char *)info);
  }
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

extern int WD, HT, BITMAP, SCRWD, SCRHT;

void RGB_to_X(int id, Uint *from)
{
  int i, j;
  Uint *to;

  to = (Uint*)(ximageinfo.ximage->data)+BITMAP*SCRWD*(id/SCRWD)+WD*(id%SCRWD);
  for (i=0; i<HT; i++,to+=WD*(SCRWD-1)) {
    for (j=0; j<WD; j++)
      *to++ = *from++;
  }
}

void BOX_to_X(int id, int nx, int ny, int boxsize)
{
//  . . . . . . . . . . .
//  . . . * . * . * . . . ¢ÍWDxHT -> 28x28 x 11x7
//  . . . . . . . . . . .   * 56x56 X 112x112
//  . . . * . X . * . . .
//  . . . . . . . . . . .   nx = 11
//  . . . * . * . * . . .   ny = 7
//  . . . . . . . . . . .   boxsize = 28
  int x, y;
  Uint *top;

  if (boxsize * nx > WD) {
    printf("in xdisp.c BOX_to_X: boxsize * nx > WD\n");
    return;
  }
  if (boxsize * ny > HT) {
    printf("in xdisp.c BOX_to_X: boxsize * ny > HT\n");
    return;
  }

  top = (Uint*)(ximageinfo.ximage->data)+BITMAP*SCRWD*(id/SCRWD)+WD*(id%SCRWD)
      + (HT-boxsize*ny)/2*WD*SCRWD+(WD-boxsize*nx)/2;
  for (y=0; y<=ny; y++) {
    for (x=0; x<=nx*boxsize; x++)
      top[y*boxsize*WD*SCRWD+x] = 0xffffff00;
  }
  for (y=0; y<=ny*boxsize; y++) {
    for (x=0; x<=nx; x++)
      top[y*WD*SCRWD+x*boxsize] = 0xffffff00;
  }
}
