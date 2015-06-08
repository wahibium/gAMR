#ifndef __CONSTANTS__
#define __CONSTANTS__



//#define REAL float
#define REAL double
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#define GET_INITIAL_VAL(x, y, z, nx, ny, nz,            \
                        kx, ky, kz,                     \
                        dx, dy, dz, kappa, time,        \
                        ax, ay, az)                     \
  do {                                                  \
    REAL i = dx*((REAL)((x) + 0.5));                    \
    REAL j = dy*((REAL)((y) + 0.5));                    \
    REAL k = dz*((REAL)((z) + 0.5));                    \
    return (REAL)0.125 *(1.0 - (ax)*cos((kx)*(i)))      \
        * (1.0 - (ay)*cos((ky)*(j)))                    \
        * (1.0 - (az)*cos((k)z*(k)));                   \
  } while (0)


#if __CUDA_ARCH__ >= 350
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif
  
#define GET(x) (x)

#define diffusion_backward()                                            \
  do {                                                                  \
    sb[ps] = s2;                                                        \
    __syncthreads();                                                    \
    f2[p-xy] = cc * s2                                                  \
        + cw * sb[ps+sb_w] + ce * sb[ps+sb_e]                           \
        + cs * sb[ps+sb_s] + cn * sb[ps+sb_n] + cb*s1 + ct*s3;          \
  } while (0)  


#define ull unsigned long long
#define MAX (64*1024*1024)

#define WARPSIZE 32
#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)

#define BLOCK_X 128
#define BLOCK_Y 2

#ifndef NX
#define NX 256 //512
#endif

#ifndef NY
#define NY 256 //512
#endif

#ifndef NZ
#define NZ 256 //512
#endif

#define bdimx (block_x)
#define bdimy (block_y)
#define SHIFT3(x, y, z) x = y; y = z
#define SHIFT4(x, y, z, k) x = y; y = z; z = k


#define index_l(i, j, k) \
  ( (k*(NY+2) + j)*(NX+2) + i )


#define index(i, j, k) \
  ( (k*NY + j)*NX + i )


#define index_c(i, j, k) \
   ( (k*(NX/2) + j)*(NX/2) + i )

#endif
