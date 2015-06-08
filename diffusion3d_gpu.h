#ifndef BENCHMARKS_DIFFUSION3D_GPU_H_
#define BENCHMARKS_DIFFUSION3D_GPU_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "benchmark.h"
#include "constants.h"
#include "stopwatch.h"

namespace benchmark {


class Diffusion3D_GPU: public Benchmark {
protected:
  int nx;
  int ny;
  int nz;
  REAL kappa;
  REAL *f1;
  REAL *f2;
  int block_x, block_y, block_z;
  REAL dx, dy, dz;
  REAL kx, ky, kz;
  REAL dt;
  REAL ce, cw, cn, cs, ct, cb, cc;
  REAL *f1_d, *f2_d;
  cudaEvent_t ev1, ev2;

 public:
  Diffusion3D_GPU():
      nx(NX), ny(NY), nz(NZ), kappa(0.1), f1(NULL), f2(NULL), 
      block_x(BLOCK_X), block_y(BLOCK_Y), block_z(1) {
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
    //assert(nx_ % block_x_ == 0);
    //assert(ny_ % block_y_ == 0);
    //assert(nz_ % block_z_ == 0);
  }
  ~Diffusion3D_GPU() {}
  std::string GetName() const {
    return std::string("Diffusion3D GPU");
  }
  REAL GetAccuracy(int count);
  void Dump() const;
  REAL *GetCorrectAnswer(int count);
  float GetThroughput(int count, float time); 
  float GetGFLOPS(int count, float time);
  void DisplayResult(int count, float time);
  void InitializeBenchmark();
  void FinalizeBenchmark();
  void RunKernel(int count);
  void RunBenchmark(int count, bool dump);

  void CudaTest(char *msg);
  int  VerifySystemParameters();
  void Die(); 
  void PrintUsage(std::ostream &os, char *prog_name);
  void ProcessProgramOptions(int argc, char *argv[],
                           int count, int size,
                           bool dump);

  void RunConsolidate(REAL *u, REAL *consolidated_u, int xstart, int xend, int ystart, int yend, int zstart, int zend, int count);
  void RunRefine(REAL *u, REAL *refined_u, int xstart, int xend, int ystart, int yend, int zstart, int zend, int count);
};

  
}

#endif