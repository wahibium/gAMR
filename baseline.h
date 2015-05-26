#ifndef BENCHMARKS_DIFFUSION3D_BASELINE_H_
#define BENCHMARKS_DIFFUSION3D_BASELINE_H_

#include "diffusion3d.h"

namespace diffusion3d {

class Baseline: public Diffusion3D {
 protected:
  REAL *f1_, *f2_;
 public:
  Baseline(int nx, int ny, int nz):
      Diffusion3D(nx, ny, nz), f1_(NULL), f2_(NULL) {}
  virtual std::string GetName() const {
    return std::string("baseline");
  }
  virtual void InitializeBenchmark();
  virtual void FinalizeBenchmark();
  virtual void RunKernel(int count);
  virtual REAL GetAccuracy(int count);  
  virtual void Dump() const;
};

REAL Baseline::GetAccuracy(int count) {
  REAL *ref = GetCorrectAnswer(count);
  REAL err = 0.0;
  long len = nx_*ny_*nz_;
  for (long i = 0; i < len; i++) {
    REAL diff = ref[i] - f1_[i];
    err +=  diff * diff;
  }
  return (REAL)sqrt(err/len);
}

void Baseline::Dump() const {
  FILE *out = fopen(GetDumpPath().c_str(), "w");
  assert(out);
  long nitems = nx_ * ny_ * nz_;
  for (long i = 0; i < nitems; ++i) {
    fprintf(out, "%f\n", f1_[i]);
  }
  fclose(out);
}

}

#endif /* DIFFUSION3D_DIFFUSION3D_H_ */