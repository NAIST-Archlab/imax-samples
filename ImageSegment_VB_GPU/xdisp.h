#ifndef XDISP_H
#define XDISP_H

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
typedef long double        Dll;
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

extern "C"{
	void x11_open(int width, int height, int screen_wd, int screen_ht);
	void x11_update();
	void RGB_to_X(int id, Uint *from);
	void BOX_to_X(int id, int nx, int ny, int boxsize);
	int x11_checkevent();
	int WD, HT , SCRWD, SCRHT;
	unsigned int   BITMAP = 512;
}

#endif //XDISP_H
