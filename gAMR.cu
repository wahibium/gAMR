#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
#include "stopwatch.h"

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

#define OFFSET3D(i, j, k, nx, ny) \
  ((i) + (j) * (nx) + (k) * (nx) * (ny))

#if __CUDA_ARCH__ >= 350
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif
  
#define GET(x) (x)

#define bdimx (BLOCK_X)
#define bdimy (BLOCK_Y)
#define SHIFT3(x, y, z) x = y; y = z
#define SHIFT4(x, y, z, k) x = y; y = z; z = k


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

#define block_x 128
#define block_y 2
#ifndef NX
//#define NX 512
#define NX 256
#endif
#define REAL double
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

using std::vector;
using std::string;
using std::map;
using std::make_pair;
using std::cout;

REAL *f1_d, *f2_d;
REAL *f1, *f2;
int block_x_, block_y_, block_z_;
cudaEvent_t ev1_, ev2_;
  int nx;
  int ny;
  int nz;
  REAL kappa;
  REAL dx, dy, dz;
  REAL kx, ky, kz;
  REAL dt;
  REAL ce, cw, cn, cs, ct, cb, cc;


/************************************************************************************/

static void CudaTest(char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

/************************************************************************************/

static int VerifySystemParameters()
{
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));
  int val = 1;
  assert(1 == *((char *)&val));

  int current_device = 0, sm_per_multiproc = 0; 
  int max_compute_perf = 0, max_perf_device = 0; 
  int device_count = 0, best_SM_arch = 0; 
  int arch_cores_sm[3] = { 1, 8, 32 }; 
  cudaDeviceProp deviceProp; 

  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }
   
  // Find the best major SM Architecture GPU device 
  for (current_device = 0; current_device < device_count; current_device++) { 
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) { 
      best_SM_arch = max(best_SM_arch, deviceProp.major); 
    }
  }
   
  // Find the best CUDA capable GPU device 
  for (current_device = 0; current_device < device_count; current_device++) { 
    cudaGetDeviceProperties(&deviceProp, current_device); 
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } 
    else if (deviceProp.major <= 2) { 
      sm_per_multiproc = arch_cores_sm[deviceProp.major]; 
    } 
    else { // Device has SM major > 2 
      sm_per_multiproc = arch_cores_sm[2]; 
    }
      
    int compute_perf = deviceProp.multiProcessorCount * 
                       sm_per_multiproc * deviceProp.clockRate; 
      
    if (compute_perf > max_compute_perf) { 
      // If we find GPU of SM major > 2, search only these 
      if (best_SM_arch > 2) { 
        // If device==best_SM_arch, choose this, or else pass 
        if (deviceProp.major == best_SM_arch) { 
          max_compute_perf = compute_perf; 
          max_perf_device = current_device; 
        } 
      } 
      else { 
        max_compute_perf = compute_perf; 
        max_perf_device = current_device; 
      } 
    } 
  } 
   
  cudaGetDeviceProperties(&deviceProp, max_perf_device); 
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable  device\n");
    exit(-1);
  }
  if (deviceProp.major < 2) {
    fprintf(stderr, "Need at least compute capability 2.0\n");
    exit(-1);
  }
  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
    exit(-1);
  }
  if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE-1) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }

  return max_perf_device;
}

/************************************************************************************/

__global__ void diffusion_kernel_shared(REAL *f1, REAL *f2,
                                        int nx, int ny, int nz,
                                        REAL ce, REAL cw, REAL cn, REAL cs,
                                        REAL ct, REAL cb, REAL cc) {

  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int xy = nx * ny;  
  __shared__ REAL sb[block_x * block_y];
  int c = i + j * nx;
  const int c1 = tid_x + tid_y * blockDim.x;
  REAL t1, t2, t3;
  t2 = t3 = f1[c];
  int w = (i == 0)        ? c1 : c1 - 1;
  int e = (i == nx-1)     ? c1 : c1 + 1;
  int n = (j == 0)        ? c1 : c1 - block_x;
  int s = (j == ny-1)     ? c1 : c1 + block_x;
  int bw = tid_x == 0 && i != 0;
  int be = tid_x == block_x-1 && i != nx - 1;
  int bn = tid_y == 0 && j != 0;
  int bs = tid_y == block_y-1 && j != ny - 1;
  //#pragma unroll 4
  for (int k = 0; k < nz; ++k) {
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = (k < nz-1) ? f1[c+xy] : t3;
    __syncthreads();
    REAL t = cc * t2 + cb * t1 + ct * t3;
    REAL v;
    v = bw ? f1[c-1] : sb[w];
    t += cw * v;
    v = be ? f1[c+1] : sb[e];
    t += ce * v;
    v = bs ? f1[c+nx] : sb[s];
    t += cs * v;
    v = bn ? f1[c-nx] : sb[n];
    t += cn * v;
    f2[c] = t;
    c += xy;
    __syncthreads();
  }
  return;
}
//##################################################################//
// Temporal blocking
// z blocking
// sperate warp for diagonal points
__global__ void diffusion_kernel_shared6(REAL *f1, REAL *f2,
                                         int nx, int ny, int nz,
                                         REAL ce, REAL cw, REAL cn, REAL cs,
                                         REAL ct, REAL cb, REAL cc) {
  extern __shared__ REAL sb[];
  const int sbx = bdimx+4;
  const int tidx = threadIdx.x % bdimx;
  const int tidy = threadIdx.x / bdimx - 1;
  int i = bdimx * blockIdx.x + tidx;
  int j = bdimy * blockIdx.y + tidy;
  j = (j < 0) ? 0 : j;      // max(j, 0)
  j = (j == ny) ? ny - 1 : j; // min(j, ny-1)

  int xy = nx * ny;
  const int block_z = nz / gridDim.z;
  int k = (blockIdx.z == 0) ? 0:
      block_z * blockIdx.z - 1;
  const int k_end = (blockIdx.z == gridDim.z-1) ? nz:
      block_z * (blockIdx.z + 1) + 1;
  int p = i + j * nx + k *xy;
  int ps = tidx+2 + (tidy+1) * sbx;
  
  if (tidy == -1) {
    int s = (j == 0)        ? 0 : -nx;
    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s2, s3;            
    s3 = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * GET(f1[p+s])
        + cn * sb[ps+sbx] + cb*t1 + ct*t3;
    p += xy;
    __syncthreads();
    ++k;

    if (k != 1) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * GET(f1[p+s])
          + cn * sb[ps+sbx] + cb*t1 + ct*t3;
      __syncthreads();       
      p += xy;
      ++k;
    }

    for (; k < k_end; ++k) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * GET(f1[p+s])
          + cn * sb[ps+sbx] + cb*t1 + ct*t3;
      __syncthreads();
      sb[ps] = s2;
      __syncthreads();
      __syncthreads();       
      p += xy;      
    }

    if (k == nz) {
      s2 = s3;
      sb[ps] = s2;
      __syncthreads();
    }
  } else if (tidy == bdimy) {
    int n = (j == ny-1)     ? 0 : nx;

    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s2, s3;      
    s2 = s3 = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * sb[ps-sbx] + cn * GET(f1[p+n]) + cb*t1 + ct*t3;
    p += xy;
    __syncthreads();
    ++k;

    if (k != 1) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx] + cn * GET(f1[p+n]) + cb*t1 + ct*t3;
      p += xy;
      __syncthreads();
      ++k;
    }
    
    for (; k < k_end; ++k) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx] + cn * GET(f1[p+n]) + cb*t1 + ct*t3;
      __syncthreads();
      sb[ps] = s2;
      __syncthreads();
      __syncthreads();      
      p += xy;      
    }

    if (k == nz) {
      s2 = s3;
      sb[ps] = s2;
      __syncthreads();      
    }
  } else if (tidy >= 0 && tidy < bdimy) {
    int sb_s = (j == 0)    ? 0: -sbx;
    int sb_n = (j == ny-1) ? 0:  sbx; 
    int sb_w = (i == 0)    ? 0: -1;
    int sb_e = (i == nx-1) ? 0:  1;

    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s1, s2, s3;
    s2 = s3 = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
        + cb * t1 + ct * t3;
    p += xy;
    __syncthreads();
    ++k;

    if (k != 1) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      SHIFT3(s1, s2, s3);      
      __syncthreads();
    
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
          + cb * t1 + ct * t3;
      p += xy;      
      __syncthreads();
      ++k;
    }
    
    for (; k < k_end; ++k) {          
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      SHIFT3(s1, s2, s3);      
      __syncthreads();
    
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
          + cb * t1 + ct * t3;
      __syncthreads();
      diffusion_backward();
      __syncthreads();
      p += xy;            
    }
    
    if (k == nz) {
      SHIFT3(s1, s2, s3);
      diffusion_backward();
    }
  } else if (tidx < 32 && tidy == bdimy + 1) {
    // horizontal halo
    int xoffset = (tidx & 1) + ((tidx & 2) >> 1) * (bdimx + 2);
    int yoffset = (tidx >> 2) + 1;
    yoffset = (yoffset >= (bdimy+1)) ? bdimy : yoffset;
    i = bdimx * blockIdx.x - 2 + xoffset;
    i = (i < 0) ? 0 : i;
    i = (i >= nx) ? nx - 1 : i;
    j = bdimy * blockIdx.y -1 + yoffset;
    j = (j < 0) ? 0 : j;      // max(j, 0)
    j = (j >= ny) ? ny - 1 : j; // min(j, ny-1)

    int s = -sbx;
    int n = sbx;
    int w = (xoffset == 0) ? 0 : -1;
    int e = (xoffset == sbx-1) ? 0 : 1;
    
    p = i + j * nx + k * xy;
    ps = xoffset + yoffset * sbx;

    float t2 = LDG(f1[p]);
    float t1 = (k == 0) ? t2 : LDG(f1[p-xy]);
    float t3 = (k < nz-1) ? LDG(f1[p+xy]) : t2;
    float t4 = (k < nz-2) ? LDG(f1[p+xy*2]) : t3;
    sb[ps] = t2;
    __syncthreads();
    float s2, s3;
    s2 = s3 = cc * t2
        + cw * sb[ps+w] + ce * sb[ps+e]
        + cs * sb[ps+s] + cn * sb[ps+n]
        + cb*t1 + ct*t3;
    __syncthreads();
    p += xy;
    ++k;

    if (k != 1) {
      SHIFT4(t1, t2, t3, t4);
      t4 = LDG(f1[p+xy*2]);
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      s3 = cc * t2
          + cw * sb[ps+w] + ce * sb[ps+e]
          + cs * sb[ps+s] + cn * sb[ps+n]
          + cb*t1 + ct*t3;
      __syncthreads();
      p += xy;
      ++k;
    }
   #pragma unroll  
    for (; k < k_end-2; ++k) {
      SHIFT4(t1, t2, t3, t4);
      t4 = LDG(f1[p+xy*2]);
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      s3 = cc * t2
          + cw * sb[ps+w] + ce * sb[ps+e]
          + cs * sb[ps+s] + cn * sb[ps+n]
          + cb*t1 + ct*t3;
      __syncthreads();
      sb[ps] = s2;
      __syncthreads();
      __syncthreads();      
      p += xy;      
    }

    SHIFT4(t1, t2, t3, t4);
    t4 = (k < nz-2) ? LDG(f1[p+xy*2]) : t4;
    sb[ps] = t2;
    s2 = s3;
    __syncthreads();
    s3 = cc * t2
        + cw * sb[ps+w] + ce * sb[ps+e]
        + cs * sb[ps+s] + cn * sb[ps+n]
        + cb*t1 + ct*t3;
    __syncthreads();
    sb[ps] = s2;
    __syncthreads();
    __syncthreads();      
    p += xy;      
    ++k;

    SHIFT4(t1, t2, t3, t4);
    sb[ps] = t2;
    s2 = s3;
    __syncthreads();
    s3 = cc * t2
        + cw * sb[ps+w] + ce * sb[ps+e]
        + cs * sb[ps+s] + cn * sb[ps+n]
        + cb*t1 + ct*t3;
    __syncthreads();
    sb[ps] = s2;
    __syncthreads();
    __syncthreads();      
    p += xy;
    ++k;

    if (k == nz) {
      s2 = s3;
      sb[ps] = s2;
      __syncthreads();
    }
  } else {
    const int tidx2 = tidx & 31;
    // 2nd warp
    int xoffset = 1 + (tidx & 1) * (bdimx + 1);
    int yoffset = ((tidx & 2) >> 1) * (bdimy + 1);
    i = bdimx * blockIdx.x - 2 + xoffset;
    i = (i < 0) ? 0 : i;
    i = (i >= nx) ? nx - 1 : i;
    j = bdimy * blockIdx.y -1 + yoffset;
    j = (j < 0) ? 0 : j;      // max(j, 0)
    j = (j >= ny) ? ny - 1 : j; // min(j, ny-1)

    p = i + j * nx + k * xy;
    ps = xoffset + yoffset * sbx;

    float t2, t3, t4;
    //bool active = tidx2 < 4;
    const bool active = 1;

    if (active) {
      t2 = LDG(f1[p]);
      t3 = LDG(f1[p+xy]);
      t4 = LDG(f1[p+xy*2]);      
      sb[ps] = t2;
    }
    __syncthreads();
    __syncthreads();    
    p += xy;
    ++k;

    if (k != 1) {
      SHIFT3(t2, t3, t4);
      if (active) {
        t4 = LDG(f1[p+xy*2]);
        sb[ps] = t2;
      }
      __syncthreads();
      __syncthreads();
      p += xy;
      ++k;
    }
   #pragma unroll  
    for (; k < k_end-2; ++k) {
      SHIFT3(t2, t3, t4);
      if (active) {
        t4 = LDG(f1[p+xy*2]);      
        sb[ps] = t2;
      }
      __syncthreads();
      __syncthreads();
      __syncthreads();
      __syncthreads();      
      p += xy;      
    }

    SHIFT3(t2, t3, t4);
    if (active) {
      sb[ps] = t2;
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();      
    p += xy;
    ++k;

    t2 = t3;
    if (active) {
      sb[ps] = t2;
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();      
    p += xy;      
    
    if (k == nz) {
      __syncthreads();
    }
  }
  return;
}
//##################################################################//
void Initialize(REAL *buff, const int nx, const int ny, const int nz,
                const REAL kx, const REAL ky, const REAL kz,
                const REAL dx, const REAL dy, const REAL dz,
                const REAL kappa, const REAL time){

  REAL ax = exp(-kappa*time*(kx*kx));
  REAL ay = exp(-kappa*time*(ky*ky));
  REAL az = exp(-kappa*time*(kz*kz));
  int jz;  
  for (jz = 0; jz < nz; jz++) {
    int jy;
    for (jy = 0; jy < ny; jy++) {
      int jx;
      for (jx = 0; jx < nx; jx++) {
        int j = jz*nx*ny + jy*nx + jx;
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL z = dz*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
          *(1.0 - ay*cos(ky*y))
          *(1.0 - az*cos(kz*z));
        buff[j] = f0;
      }
    }
  }
}


void InitializeBenchmark() {
  size_t s = sizeof(REAL) * nx * ny * nz;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&f1, s));
  Initialize(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, 0.0);
  CUDA_SAFE_CALL(cudaMalloc((void**)&f1_d, s));
  CUDA_SAFE_CALL(cudaMalloc((void**)&f2_d, s));
  //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared,
                                        cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared6,
                                        cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1_));
  CUDA_SAFE_CALL(cudaEventCreate(&ev2_));
}

void FinalizeBenchmark() {
  assert(f1);
  CUDA_SAFE_CALL(cudaFreeHost(f1));
  assert(f1_d);
  CUDA_SAFE_CALL(cudaFree(f1_d));
  assert(f2_d);
  CUDA_SAFE_CALL(cudaFree(f2_d));
}
std::string GetName(){
    return std::string("cuda_shared");
  }
std::string GetDumpPath() {
    return std::string("diffusion3d_result.")
        + GetName() + std::string(".out");
  }
void Dump() {
  FILE *out = fopen(GetDumpPath().c_str(), "w");
  assert(out);
  long nitems = nx * ny * nz;
  for (long i = 0; i < nitems; ++i) {
    fprintf(out, "%f\n", f1[i]);
  }
  fclose(out);
}


REAL *GetCorrectAnswer(int count){
    REAL *f = (REAL*)malloc(sizeof(REAL) * nx * ny * nz);
    assert(f);
    Initialize(f, nx, ny, nz,
               kx, ky, kz, dx, dy, dz,
               kappa, count * dt);
    return f;
  }
  
float GetThroughput(int count, float time) {
    return (nx * ny * nz) * sizeof(REAL) * 2.0 * ((float)count)
        / time * 1.0e-09;    
  }
  float GetGFLOPS(int count, float time) {
    float f = (nx*ny*nz)*13.0*(float)(count)/time * 1.0e-09;
    return f;
  } 

  REAL GetAccuracy(int count) {
  REAL *ref = GetCorrectAnswer(count);
  REAL err = 0.0;
  long len = nx*ny*nz;
  for (long i = 0; i < len; i++) {
    REAL diff = ref[i] - f1[i];
    err +=  diff * diff;
  }
  return (REAL)sqrt(err/len);
  }

void DisplayResult(int count, float time) {
    printf("Elapsed time : %.3f (s)\n", time);
    printf("FLOPS        : %.3f (GFLOPS)\n",
    GetGFLOPS(count, time));
    printf("Throughput   : %.3f (GB/s)\n",
    GetThroughput(count ,time));
    printf("Accuracy     : %e\n", GetAccuracy(count));
  float time_wo_pci;
  cudaEventElapsedTime(&time_wo_pci, ev1_, ev2_);
  time_wo_pci *= 1.0e-03;
  printf("Kernel-only performance:\n");
  printf("Elapsed time : %.3f (s)\n", time_wo_pci);
  printf("FLOPS        : %.3f (GFLOPS)\n",
         GetGFLOPS(count, time_wo_pci));
  printf("Throughput   : %.3f (GB/s)\n",
         GetThroughput(count ,time_wo_pci));
}

void RunKernel(int count) {
  size_t s = sizeof(REAL) * nx * ny * nz;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d, f1, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x, block_y, 1);
  dim3 grid_dim(nx / block_x, ny / block_y, 1);
  // For calling best-performance kernel
  //dim3 block_dim(bdimx * (bdimy+2) + (32*2));
  //dim3 grid_dim(nx_ / bdimx, ny_ / bdimy, grid_z_);
  CUDA_SAFE_CALL(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    diffusion_kernel_shared<<<grid_dim, block_dim>>>
        (f1_d, f2_d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
    REAL *t = f1_d;
    f1_d = f2_d;
    f2_d = t;
  }
  CUDA_SAFE_CALL(cudaEventRecord(ev2_));
  CUDA_SAFE_CALL(cudaMemcpy(f1, f1_d, s, cudaMemcpyDeviceToHost));
  return;
}

  void RunBenchmark(int count, bool dump) {
    std::cout << "Initializing benchmark input...\n";
    InitializeBenchmark();
    std::cout << "Running diffusion3d/" << GetName() << "\n";
    std::cout << "Iteration count: " << count << "\n";
    std::cout << "Grid size: " << nx << "x" << ny << "x" << nz << "\n";
    Stopwatch st;
    StopwatchStart(&st);
    RunKernel(count);
    float elapsed_time = StopwatchStop(&st);
    std::cout << "Benchmarking finished.\n";
    DisplayResult(count, elapsed_time);
    if (dump) Dump();
    FinalizeBenchmark();
  }

void Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}

void PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--count N   " << "Number of iterations\n"
     << "\t--size N    "  << "Size of each dimension\n"
     << "\t--dump N    "  << "Dump the final data to file\n"
     << "\t--help      "  << "Display this help message\n";
}


void ProcessProgramOptions(int argc, char *argv[],
                           int count, int size,
                           bool dump) {
  int c;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"count", 1, 0, 0},
      {"size", 1, 0, 0},
      {"dump", 0, 0, 0},
      {"help", 0, 0, 0},      
      {0, 0, 0, 0}
    };

    c = getopt_long(argc, argv, "",
                    long_options, &option_index);
    if (c == -1) break;
    if (c != 0) {
      //std::cerr << "Invalid usage\n";
      //PrintUsage(std::cerr, argv[0]);
      //Die();
      continue;
    }

    switch(option_index) {
      case 0:
        count = atoi(optarg);
        break;
      case 1:
        size = atoi(optarg);
        break;
      case 2:
        dump = true;
        break;
      case 3:
        PrintUsage(std::cerr, argv[0]);
        exit(EXIT_SUCCESS);
        break;
      default:
        break;
    }
  }
}

/*************************************************************************************/
/************************************************************************************/
/*
int main(int argc, char *argv[])
{
  int blocks, warpsperblock, dimensionality;
  int device;

  device = VerifySystemParameters();
  cudaSetDevice(device);

  cudaFuncSetCacheConfig(CompressionKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(DecompressionKernel, cudaFuncCachePreferL1);

  if((3 == argc) || (4 == argc)) { // compress 
    char dummy;
    blocks = atoi(argv[1]);
    assert((0 < blocks) && (blocks < 256));
    warpsperblock = atoi(argv[2]);
    assert((0 < warpsperblock) && (warpsperblock < 256));
    if(3 == argc) {
      dimensionality = 1;
    } else {
      dimensionality = atoi(argv[3]);
    }
    assert((0 < dimensionality) && (dimensionality <= WARPSIZE));

    Compress(blocks, warpsperblock, dimensionality);
    assert(0 == fread(&dummy, 1, 1, stdin));
  }
  else if(1 == argc) { // decompress 
    int num, doubles;
    num = fread(&blocks, 1, 1, stdin);
    assert(1 == num);
    blocks &= 255;
    num = fread(&warpsperblock, 1, 1, stdin);
    assert(1 == num);
    warpsperblock &= 255;
    num = fread(&dimensionality, 1, 1, stdin);
    assert(1 == num);
    dimensionality &= 255;
    num = fread(&doubles, 4, 1, stdin);
    assert(1 == num);

    Decompress(blocks, warpsperblock, dimensionality, doubles);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s blocks warps/block (dimensionality) < file.in > file.gfc\n", argv[0]);
    fprintf(stderr, "decompress: %s < file.gfc > file.out\n", argv[0]);
  }

  return 0;
}*/

int main(int argc, char *argv[]) {

  // Stencil Setting
  nx = ny = nz = NX; // default size
  int  count = 1000; // default iteration count
  bool dump = false;
  
  //ProcessProgramOptions(argc, argv, count, nx, dump);

    kappa = 0.1;
    REAL l = 1.0;
    dx = l / nx;
    dy = l / ny;
    dz = l / nz;
    kx = ky = kz = 2.0 * M_PI;
    dt = 0.1 * dx * dx / kappa;
    ce = cw = kappa*dt/(dx*dx);
    cn = cs = kappa*dt/(dy*dy);
    ct = cb = kappa*dt/(dz*dz);
    cc = 1.0 - (ce + cw + cn + cs + ct + cb);

   // Run Stencil operation
  
  RunBenchmark(count, dump);
  //FILE *out = fopen("AfterDiffusion", "w");
  //FILE *out = fopen("AfterDiffusionBinary", "wb");
  //assert(out);
  //long nitems = nx * ny * nz;
  //for (long i = 0; i < nitems; ++i) {
  //  fprintf(out, "%f\n", f1[i]);
  //}
  //fwrite(f1,sizeof(REAL),nitems,out);
  //fclose(out);
  //assert(f1);
  //CUDA_SAFE_CALL(cudaFreeHost(f1));
   
  // Encoding Setting
  /*
  int blocks, warpsperblock, dimensionality;
  int device;



  device = VerifySystemParameters();
  cudaSetDevice(device);

  cudaFuncSetCacheConfig(CompressionKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(DecompressionKernel, cudaFuncCachePreferL1);
  if((3 == argc) || (4 == argc)) { // compress 
    char dummy;
    blocks = atoi(argv[1]);
    assert((0 < blocks) && (blocks < 256));
    warpsperblock = atoi(argv[2]);
    assert((0 < warpsperblock) && (warpsperblock < 256));
    if(3 == argc) {
      dimensionality = 1;
    } else {
      dimensionality = atoi(argv[3]);
    }
    assert((0 < dimensionality) && (dimensionality <= WARPSIZE));  
    Compress(blocks, warpsperblock, dimensionality);
    assert(0 == fread(&dummy, 1, 1, stdin));
  }
  else if(1 == argc) { // decompress 
    int num, doubles;
    num = fread(&blocks, 1, 1, stdin);
    assert(1 == num);
    blocks &= 255;
    num = fread(&warpsperblock, 1, 1, stdin);
    assert(1 == num);
    warpsperblock &= 255;
    num = fread(&dimensionality, 1, 1, stdin);
    assert(1 == num);
    dimensionality &= 255;
    num = fread(&doubles, 4, 1, stdin);
    assert(1 == num);

    Decompress(blocks, warpsperblock, dimensionality, doubles);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s blocks warps/block (dimensionality) < file.in > file.gfc\n", argv[0]);
    fprintf(stderr, "decompress: %s < file.gfc > file.out\n", argv[0]);
  }
  */

  return 0;
}


