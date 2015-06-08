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
#include "diffusion3d_gpu.h"


using std::vector;
using std::string;
using std::map;
using std::make_pair;
using std::cout;

using namespace benchmark;


/************************************************************************************/

//##################################################################//

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
  //nx = ny = nz = NX; // default size
  //int  count = 1000; // default iteration count
  //bool dump = false;
  
  //ProcessProgramOptions(argc, argv, count, nx, dump);

    

   // Run GPU operation:
   // Stencil         = 1
   // Interpolate     = 2
   // Consolidate     = 3
   // Error Fucntions = 4
   
  /////RunBenchmark(count, dump, 1);
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
   

  return 0;
}



