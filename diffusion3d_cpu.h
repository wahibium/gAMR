#ifndef BENCHMARKS_DIFFUSION3D_CPU_H_
#define BENCHMARKS_DIFFUSION3D_CPU_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <string>
#include "benchmark.h"
#include "constants.h"
#include "stopwatch.h"

namespace benchmark {

class Diffusion3D_CPU : public Benchmark {
 protected:
  int nx;
  int ny;
  int nz;
  REAL kappa;
  REAL *f1, *f2;
  REAL dx, dy, dz;
  REAL kx, ky, kz;
  REAL dt;
  REAL ce, cw, cn, cs, ct, cb, cc;
 public:
  Diffusion3D_CPU():
      nx(NX), ny(NY), nz(NZ), kappa(0.1), f1(NULL), f2(NULL) {
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
  }
  ~Diffusion3D_CPU() {}
  std::string GetName() const {
    return std::string("Diffusion3D CPU");
  }
  REAL GetAccuracy(int count);   
  void Dump() const; 
  REAL* GetCorrectAnswer(int count);
  float GetThroughput(int count, float time);
  float GetGFLOPS(int count, float time);
  void DisplayResult(int count, float time);
  void InitializeBenchmark();
  void RunKernel(int count);
  void RunBenchmark(int count, bool dump);
  void FinalizeBenchmark();

  void Die(); 
  void PrintUsage(std::ostream &os, char *prog_name);
  void ProcessProgramOptions(int argc, char *argv[],
                           int count, int size,
                           bool dump);

  void Consolidate(REAL *u, REAL *consolidated_u, int xstart, int xend, int ystart, int yend, int zstart, int zend);
  void Refine(REAL *u, REAL *refined_u, int xstart, int xend, int ystart, int yend, int zstart, int zend);

};

}
#endif