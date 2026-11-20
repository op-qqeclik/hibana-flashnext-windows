/* Windows host definitions the CUDA sources take for granted elsewhere, force-included into every kernel translation
   unit on the Windows port (flashinfer and hibana's own kernels):
     - ushort/uint/ulong come from <sys/types.h> on Linux; the Windows headers only define them with <windows.h>.
     - M_PI and friends live in <math.h> only with _USE_MATH_DEFINES, and MSVC has no __restrict__ spelling.
*/
#ifndef HIBANA_WIN_COMPAT_H
#define HIBANA_WIN_COMPAT_H
#ifndef _MSC_VER
#error "win_compat.h is for the MSVC host compiler"
#endif
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#endif
