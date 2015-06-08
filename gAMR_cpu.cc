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
#include "stopwatch.h"
#include "constants.h"
#include "diffusion3d_cpu.h"


using std::vector;
using std::string;
using std::map;
using std::make_pair;
using std::cout;

namespace benchmark {

//##################################################################//
REAL* Diffusion3D_CPU::GetCorrectAnswer(int count) {
    REAL *f = (REAL*)malloc(sizeof(REAL) * nx * ny * nz);
    assert(f);
    Initialize(f, nx, ny, nz,
               kx, ky, kz, dx, dy, dz,
               kappa, count * dt);
    return f;
  }
/*************************************************************************************/

  REAL Diffusion3D_CPU::GetAccuracy(int count) {
    REAL *ref = GetCorrectAnswer(count);
    REAL err = 0.0;
    long len = nx*ny*nz;
    for (long i = 0; i < len; i++) {
      REAL diff = ref[i] - f1[i];
      err +=  diff * diff;
    }
    return (REAL)sqrt(err/len);
  }
/*************************************************************************************/

  void Diffusion3D_CPU::Dump() const {
    FILE *out = fopen(GetDumpPath().c_str(), "w");
    assert(out);
    long nitems = nx * ny * nz;
    for (long i = 0; i < nitems; ++i) {
      fprintf(out, "%f\n", f1[i]);
    }
    fclose(out);
  }
/*************************************************************************************/

  float Diffusion3D_CPU::GetThroughput(int count, float time) {
    return (nx * ny * nz) * sizeof(REAL) * 2.0 * ((float)count)
        / time * 1.0e-09;    
  }
/*************************************************************************************/

  float Diffusion3D_CPU::GetGFLOPS(int count, float time) {
    float f = (nx*ny*nz)*13.0*(float)(count)/time * 1.0e-09;
    return f;
  }
/*************************************************************************************/

  void Diffusion3D_CPU::DisplayResult(int count, float time) {
    printf("Elapsed time : %.3f (s)\n", time);
    printf("FLOPS        : %.3f (GFLOPS)\n",
           GetGFLOPS(count, time));
    printf("Throughput   : %.3f (GB/s)\n",
           GetThroughput(count ,time));
    printf("Accuracy     : %e\n", GetAccuracy(count));
  }
/*************************************************************************************/

  void Diffusion3D_CPU::InitializeBenchmark() {
  size_t s = sizeof(REAL) * nx * ny * nz;
  f1 = (REAL*)malloc(s);
  assert(f1);
  f2 = (REAL*)malloc(s);
  assert(f2);
  Initialize(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, 0.0);
  }
/*************************************************************************************/

  void Diffusion3D_CPU::FinalizeBenchmark() {
  assert(f1);
  free(f1);
  assert(f2);
  free(f2);
  }
/*************************************************************************************/

  void Diffusion3D_CPU::RunKernel(int count) {
    int i;
    for (i = 0; i < count; ++i) {
      int z;
  #pragma omp parallel for        
      for (z = 0; z < nz; z++) {
        int y;
        for (y = 0; y <= ny; y++) {
          int x;
          for (x = 0; x <= nx; x++) {
            int c, w, e, n, s, b, t;
            c =  x + y * nx + z * nx * ny;
            w = (x == 0)    ? c : c - 1;
            e = (x == nx-1) ? c : c + 1;
            n = (y == 0)    ? c : c - nx;
            s = (y == ny-1) ? c : c + nx;
            b = (z == 0)    ? c : c - nx * ny;
            t = (z == nz-1) ? c : c + nx * ny;
            f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e]
                + cs * f1[s] + cn * f1[n] + cb * f1[b] + ct * f1[t];
          }
        }
      }
      REAL *t = f1;
      f1 = f2;
      f2 = t;
    }
    return;
  }
/*************************************************************************************/

  void Diffusion3D_CPU::RunBenchmark(int count, bool dump) {
    std::cout << "Initializing benchmark input...\n";
    InitializeBenchmark();
    std::cout << "Running Diffusion3D CPU/" << GetName() << "\n";
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
/*************************************************************************************/

void Diffusion3D_CPU::Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}
/*************************************************************************************/

void Diffusion3D_CPU::PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--count N   " << "Number of iterations\n"
     << "\t--size N    "  << "Size of each dimension\n"
     << "\t--dump N    "  << "Dump the final data to file\n"
     << "\t--help      "  << "Display this help message\n";
}
/*************************************************************************************/

void Diffusion3D_CPU::ProcessProgramOptions(int argc, char *argv[],
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
void Diffusion3D_CPU::Refine(REAL *u, REAL *refined_u, int xstart, int xend, int ystart, int yend, int zstart, int zend){
  REAL sx_l, sx_r, sy_u, sy_d, sz_b, sz_f;
  for(int i=xstart, m=0; i<=xend; i++, m++){
    for(int j=ystart, n=0; j<=yend; j++, n++){
      for(int k=zstart, o=0; k<=zend; k++, o++){
        sx_l = (-u[index(i-1,j,k)]+u[index(i,j,k)]) / 4;
        sx_r = (-u[index(i,j,k)]+u[index(i+1,j,k)]) / 4;

        sy_d = (-u[index(i,j-1,k)]+u[index(i,j,k)]) / 4;
        sy_u = (-u[index(i,j,k)]+u[index(i,j+1,k)]) / 4;

        sz_b = (-u[index(i,j,k-1)]+u[index(i,j,k)]) / 4;
        sz_f = (-u[index(i,j,k)]+u[index(i,j,k+1)]) / 4;

        refined_u[index(2*m,   2*n  , 2*o)]   = u[index(i,j,k)] - sx_l - sy_d - sz_b;
        refined_u[index(2*m+1, 2*n  , 2*o)]   = u[index(i,j,k)] + sx_r - sy_d - sz_b;
        refined_u[index(2*m,   2*n+1, 2*o)]   = u[index(i,j,k)] - sx_l + sy_u - sz_b;
        refined_u[index(2*m+1, 2*n+1, 2*o)]   = u[index(i,j,k)] + sx_r + sy_u - sz_b;
        refined_u[index(2*m,   2*n  , 2*o+1)] = u[index(i,j,k)] - sx_l - sy_d + sz_f;
        refined_u[index(2*m+1, 2*n  , 2*o+1)] = u[index(i,j,k)] + sx_r - sy_d + sz_f;
        refined_u[index(2*m,   2*n+1, 2*o+1)] = u[index(i,j,k)] - sx_l + sy_u + sz_f;
        refined_u[index(2*m+1, 2*n+1, 2*o+1)] = u[index(i,j,k)] + sx_r + sy_u + sz_f;
      }
    }
  }
  return;
}
/************************************************************************************/
void Diffusion3D_CPU::Consolidate(REAL *u, REAL *consolidated_u, int xstart, int xend, int ystart, int yend, int zstart, int zend){
  for(int z=zstart; z<=zend; z+=2)
    for(int y=ystart; y<=yend; y+=2)
      for(int x=xstart; x<= xend; x+=2)
            consolidated_u[index_c(x/2, y/2, z/2)] = ( u[index(x, y, z)]     + u[index(x+1, y, z)] +
                                                       u[index(x, y+1, z)]   + u[index(x+1, y+1, z)] +
                                                       u[index(x, y, z+1)]   + u[index(x+1, y, z+1)] +
                                                       u[index(x, y+1, z+1)] + u[index(x+1, y+1, z+1)]) / 8.0;
  return;
}

}
