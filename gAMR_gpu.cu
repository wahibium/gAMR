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
#include "constants.h"
#include "diffusion3d_gpu.h"


using std::vector;
using std::string;
using std::map;
using std::make_pair;
using std::cout;

namespace benchmark {

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
  __shared__ REAL sb[BLOCK_X * BLOCK_Y];
  int c = i + j * nx;
  const int c1 = tid_x + tid_y * blockDim.x;
  REAL t1, t2, t3;
  t2 = t3 = f1[c];
  int w = (i == 0)        ? c1 : c1 - 1;
  int e = (i == nx-1)     ? c1 : c1 + 1;
  int n = (j == 0)        ? c1 : c1 - BLOCK_X;
  int s = (j == ny-1)     ? c1 : c1 + BLOCK_X;
  int bw = tid_x == 0 && i != 0;
  int be = tid_x == BLOCK_X-1 && i != nx - 1;
  int bn = tid_y == 0 && j != 0;
  int bs = tid_y == BLOCK_Y-1 && j != ny - 1;
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
/************************************************************************************/

__global__ void refine_kernel_shared(REAL *u, REAL *refined_u,
                                     int nx, int ny, int nz){


const int tid_x = threadIdx.x;
const int tid_y = threadIdx.y;
const int i = blockDim.x * blockIdx.x + tid_x;
const int j = blockDim.y * blockIdx.y + tid_y;
const int xy = nx * ny;  
__shared__ REAL sb[BLOCK_X * BLOCK_Y];
REAL sx_l, sx_r, sy_u, sy_d, sz_b, sz_f;
int c = i + j * nx;
const int c1 = tid_x + tid_y * blockDim.x;
REAL t1, t2, t3;
t2 = t3 = u[c];
int w = (i == 0)        ? c1 : c1 - 1;
int e = (i == nx-1)     ? c1 : c1 + 1;
int n = (j == 0)        ? c1 : c1 - BLOCK_X;
int s = (j == ny-1)     ? c1 : c1 + BLOCK_X;
int bw = tid_x == 0 && i != 0;
int be = tid_x == BLOCK_X-1 && i != nx - 1;
int bn = tid_y == 0 && j != 0;
int bs = tid_y == BLOCK_Y-1 && j != ny - 1;


for(int k = 0; k < nz; k++){
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = (k < nz-1) ? u[c+xy] : t3;
    __syncthreads();
    sx_l = bw ? u[c-1] : sb[w];
    sx_r = be ? u[c+1] : sb[e];

    sy_d = bn ? u[c-nx] : sb[n];
    sy_u = bs ? u[c+nx] : sb[s];

    sz_b = t1;
    sz_f = t3;

    refined_u[index(2*i,   2*j  , 2*k)]   = (t2 + sx_l + sy_d + sz_b) / 4;
    refined_u[index(2*i+1, 2*j  , 2*k)]   = (t2 + sx_r + sy_d + sz_b) / 4;
    refined_u[index(2*i,   2*j+1, 2*k)]   = (t2 + sx_l + sy_u + sz_b) / 4;
    refined_u[index(2*i+1, 2*j+1, 2*k)]   = (t2 + sx_r + sy_u + sz_b) / 4;
    refined_u[index(2*i,   2*j  , 2*k+1)] = (t2 + sx_l + sy_d + sz_f) / 4;
    refined_u[index(2*i+1, 2*j  , 2*k+1)] = (t2 + sx_r + sy_d + sz_f) / 4;
    refined_u[index(2*i,   2*j+1, 2*k+1)] = (t2 + sx_l + sy_u + sz_f) / 4;
    refined_u[index(2*i+1, 2*j+1, 2*k+1)] = (t2 + sx_r + sy_u + sz_f) / 4;
    c += xy;
    __syncthreads();
      }
  return;
}
/************************************************************************************/

__global__ void consolidate_kernel_shared(REAL *u, REAL *consolidated_u,
                                          int nx, int ny, int nz){

// nx, ny, and nz are that of array consolidated_u
const int tid_x = threadIdx.x;
const int tid_y = threadIdx.y;
const int i = blockDim.x * blockIdx.x + tid_x;
const int j = blockDim.y * blockIdx.y + tid_y;
for(int k = 0; k < nz; k++)  {
  consolidated_u[index(i, j, k)] = ( u[index(2*i, 2*j, 2*k)]         + u[index(2*i + 1, 2*j, 2*k)] +
                                     u[index(2*i, 2*j + 1, 2*k)]     + u[index(2*i + 1, 2*j + 1, 2*k)] +
                                     u[index(2*i, 2*j, 2*k + 1)]     + u[index(2*i + 1, 2*j, 2*k + 1)] +
                                     u[index(2*i, 2*j + 1, 2*k + 1)] + u[index(2*i + 1, 2*j + 1, 2*k + 1)]) / 8.0;
  }
return;
}



//##################################################################//  
/************************************************************************************/

REAL Diffusion3D_GPU::GetAccuracy(int count) {
    REAL *ref = GetCorrectAnswer(count);
    REAL err = 0.0;
    long len = nx*ny*nz;
    for (long i = 0; i < len; i++) {
      REAL diff = ref[i] - f1[i];
      err +=  diff * diff;
    }
    return (REAL)sqrt(err/len);
  }
/************************************************************************************/

  void Diffusion3D_GPU::Dump() const {
    FILE *out = fopen(GetDumpPath().c_str(), "w");
    assert(out);
    long nitems = nx * ny * nz;
    for (long i = 0; i < nitems; ++i) {
      fprintf(out, "%f\n", f1[i]);
    }
    fclose(out);
  }
/************************************************************************************/

  REAL *Diffusion3D_GPU::GetCorrectAnswer(int count){
    REAL *f1 = (REAL*)malloc(sizeof(REAL) * nx * ny * nz);
    assert(f1);
    Initialize(f1, nx, ny, nz,
               kx, ky, kz, dx, dy, dz,
               kappa, count * dt);
    return f1;
  }
/************************************************************************************/

  float Diffusion3D_GPU::GetThroughput(int count, float time) {
    return (nx * ny * nz) * sizeof(REAL) * 2.0 * ((float)count)
        / time * 1.0e-09;    
  }
/************************************************************************************/

  float Diffusion3D_GPU::GetGFLOPS(int count, float time) {
    float f = (nx*ny*nz)*13.0*(float)(count)/time * 1.0e-09;
    return f;
  }
/************************************************************************************/

  void Diffusion3D_GPU::DisplayResult(int count, float time) {
    printf("Elapsed time : %.3f (s)\n", time);
    printf("FLOPS        : %.3f (GFLOPS)\n",
           GetGFLOPS(count, time));
    printf("Throughput   : %.3f (GB/s)\n",
           GetThroughput(count ,time));
    printf("Accuracy     : %e\n", GetAccuracy(count));
  }
/************************************************************************************/

  void Diffusion3D_GPU::InitializeBenchmark() {
  size_t s = sizeof(REAL) * nx * ny * nz;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&f1, s));
  Initialize(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, 0.0);
  CUDA_SAFE_CALL(cudaMalloc((void**)&f1_d, s));
  CUDA_SAFE_CALL(cudaMalloc((void**)&f2_d, s));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared,
                                        //cudaFuncCachePreferL1));
  //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared6,
                                        cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1));
  CUDA_SAFE_CALL(cudaEventCreate(&ev2));
  }
/************************************************************************************/

  void Diffusion3D_GPU::FinalizeBenchmark() {
  assert(f1);
  CUDA_SAFE_CALL(cudaFreeHost(f1));
  assert(f1_d);
  CUDA_SAFE_CALL(cudaFree(f1_d));
  assert(f2_d);
  CUDA_SAFE_CALL(cudaFree(f2_d));
  }
/************************************************************************************/

  void Diffusion3D_GPU::RunKernel(int count) {
    size_t s = sizeof(REAL) * nx * ny * nz;  
    CUDA_SAFE_CALL(cudaMemcpy(f1_d, f1, s, cudaMemcpyHostToDevice));

    dim3 block_dim(block_x, block_y, 1);
    dim3 grid_dim(nx / block_x, ny / block_y, 1);
    // For calling best-performance kernel
    //dim3 block_dim(bdimx * (bdimy+2) + (32*2));
    //dim3 grid_dim(nx_ / bdimx, ny_ / bdimy, grid_z_);
    CUDA_SAFE_CALL(cudaEventRecord(ev1));
    for (int i = 0; i < count; ++i) {
      diffusion_kernel_shared<<<grid_dim, block_dim>>>
          (f1_d, f2_d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
      REAL *t = f1_d;
      f1_d = f2_d;
      f2_d = t;
    }
    CUDA_SAFE_CALL(cudaEventRecord(ev2));
    CUDA_SAFE_CALL(cudaMemcpy(f1, f1_d, s, cudaMemcpyDeviceToHost));
    return;
  }
/************************************************************************************/

  void Diffusion3D_GPU::RunBenchmark(int count, bool dump) {
    std::cout << "Initializing benchmark input...\n";
    InitializeBenchmark();
    std::cout << "Running Diffusion3D GPU/" << GetName() << "\n";
    std::cout << "Iteration count: " << count << "\n";
    std::cout << "Grid size: " << nx << "x" << ny
              << "x" << nz << "\n";
    Stopwatch st;
    StopwatchStart(&st);
    RunKernel(count);
    float elapsed_time = StopwatchStop(&st);
    std::cout << "Benchmarking finished.\n";
    DisplayResult(count, elapsed_time);
    if (dump) Dump();
    FinalizeBenchmark();
  }
/************************************************************************************/

void Diffusion3D_GPU::CudaTest(char *msg)
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

int Diffusion3D_GPU::VerifySystemParameters()
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

void Diffusion3D_GPU::Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}
/************************************************************************************/

void Diffusion3D_GPU::PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--count N   " << "Number of iterations\n"
     << "\t--size N    "  << "Size of each dimension\n"
     << "\t--dump N    "  << "Dump the final data to file\n"
     << "\t--help      "  << "Display this help message\n";
}
/************************************************************************************/

void Diffusion3D_GPU::ProcessProgramOptions(int argc, char *argv[],
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
void Diffusion3D_GPU::RunRefine(REAL *u, REAL *refined_u, int xstart, int xend, int ystart, int yend, int zstart, int zend, int count){
  
    size_t s = sizeof(REAL) * nx * ny * nz;  
    CUDA_SAFE_CALL(cudaMemcpy(f1_d, f1, s, cudaMemcpyHostToDevice));

    dim3 block_dim(block_x, block_y, 1);
    dim3 grid_dim(nx / block_x, ny / block_y, 1);
    // For calling best-performance kernel
    //dim3 block_dim(bdimx * (bdimy+2) + (32*2));
    //dim3 grid_dim(nx_ / bdimx, ny_ / bdimy, grid_z_);
    CUDA_SAFE_CALL(cudaEventRecord(ev1));
    for (int i = 0; i < count; ++i) {
      diffusion_kernel_shared<<<grid_dim, block_dim>>>
          (f1_d, f2_d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
      REAL *t = f1_d;
      f1_d = f2_d;
      f2_d = t;
    }
    CUDA_SAFE_CALL(cudaEventRecord(ev2));
    CUDA_SAFE_CALL(cudaMemcpy(f1, f1_d, s, cudaMemcpyDeviceToHost));
    return;

}
/************************************************************************************/
void Diffusion3D_GPU::RunConsolidate(REAL *u, REAL *consolidated_u, int xstart, int xend, int ystart, int yend, int zstart, int zend, int count){
    size_t s = sizeof(REAL) * nx * ny * nz;  
    CUDA_SAFE_CALL(cudaMemcpy(f1_d, f1, s, cudaMemcpyHostToDevice));

    dim3 block_dim(block_x, block_y, 1);
    dim3 grid_dim(nx / block_x, ny / block_y, 1);
    // For calling best-performance kernel
    //dim3 block_dim(bdimx * (bdimy+2) + (32*2));
    //dim3 grid_dim(nx_ / bdimx, ny_ / bdimy, grid_z_);
    CUDA_SAFE_CALL(cudaEventRecord(ev1));
    for (int i = 0; i < count; ++i) {
      diffusion_kernel_shared<<<grid_dim, block_dim>>>
          (f1_d, f2_d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
      REAL *t = f1_d;
      f1_d = f2_d;
      f2_d = t;
    }
    CUDA_SAFE_CALL(cudaEventRecord(ev2));
    CUDA_SAFE_CALL(cudaMemcpy(f1, f1_d, s, cudaMemcpyDeviceToHost));
    return;
}

}
