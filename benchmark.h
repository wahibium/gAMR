#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <string>
#include "constants.h"
#include "stopwatch.h"

namespace benchmark {

void Initialize(REAL *buff, const int nx, const int ny, const int nz,
                const REAL kx, const REAL ky, const REAL kz,
                const REAL dx, const REAL dy, const REAL dz,
                const REAL kappa, const REAL time);

class Benchmark {
  
  protected:
  // Operation methods
  virtual void InitializeBenchmark() = 0;  
  virtual void RunKernel(int count) = 0;
  virtual void Dump() const = 0;
  virtual REAL GetAccuracy(int count) = 0;
  virtual void FinalizeBenchmark() = 0;    
  // Performance methods
  virtual float GetThroughput(int count, float time) = 0;
  virtual float GetGFLOPS(int count, float time) = 0;
  virtual void DisplayResult(int count, float time) = 0;
  
  virtual REAL* GetCorrectAnswer(int count) = 0;
  virtual std::string GetDumpPath() const {
    return std::string("benchmark_result.")
        + GetName() + std::string(".out");
  }
  
   public:
  virtual std::string GetName() const = 0;
  virtual void RunBenchmark(int count, bool dump) = 0;

};

}

#endif