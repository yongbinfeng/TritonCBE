/*
 code to convert p2r to custom backend to be run as a service
*/

#include <cuda_profiler_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include "cuda_runtime.h"

#ifndef EXCLUDE_H2D_TRANSFER
#define MEASURE_H2D_TRANSFER
#endif
#ifndef EXCLUDE_D2H_TRANSFER
#define MEASURE_D2H_TRANSFER
#endif

#ifndef bsize
#define bsize 32
#endif
#ifndef ntrks
#define ntrks 8192
#endif

#define nb (ntrks / bsize)

#ifndef nevts
#define nevts 100
#endif
#define smear 0.1

#ifndef NITER
#define NITER 5
#endif
#ifndef NWARMUP
#define NWARMUP 2
#endif

#ifndef nlayer
#define nlayer 20
#endif

#ifndef num_streams
#define num_streams 1
#endif

#ifndef threadsperblockx
#define threadsperblockx 32
#endif
#ifndef blockspergrid
#define blockspergrid (nevts * nb)
#endif

#ifndef nthreads
#define nthreads 1
#endif

#define HOSTDEV __host__ __device__

#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "triton/backend/backend_common.h"

namespace triton {
namespace backend {
namespace p2r {

#define HOSTDEV __host__ __device__

HOSTDEV constexpr size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i * D + j;
}

HOSTDEV size_t SymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

HOSTDEV size_t SymOffsets66(size_t i) {
  const size_t offs[36] = {0,  1,  3,  6,  10, 15, 1,  2,  4,  7,  11, 16,
                           3,  4,  5,  8,  12, 17, 6,  7,  8,  9,  13, 18,
                           10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}

struct ATRK {
  float par[6];
  float cov[21];
  int q;
  //  int hitidx[22];
};

struct AHIT {
  float pos[3];
  float cov[6];
};

struct MP1I {
  int data[1 * bsize];
};

struct MP22I {
  int data[22 * bsize];
};

struct MP1F {
  float data[1 * bsize];
};

struct MP2F {
  float data[2 * bsize];
};

struct MP3F {
  float data[3 * bsize];
};

struct MP6F {
  float data[6 * bsize];
};

struct MP3x3 {
  float data[9 * bsize];
};
struct MP3x6 {
  float data[18 * bsize];
};

struct MP2x2SF {
  float data[3 * bsize];
};

struct MP3x3SF {
  float data[6 * bsize];
};

struct MP6x6SF {
  float data[21 * bsize];
};

struct MP6x6F {
  float data[36 * bsize];
};

struct MPTRK {
  MP6F par;
  MP6x6SF cov;
  MP1I q;
  //  MP22I   hitidx;
};

struct MPHIT {
  MP3F pos;
  MP3x3SF cov;
};

float randn(float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;
  if (call == 1) {
    call = !call;
    return (mu + sigma * (float)X2);
  }
  do {
    U1 = -1 + ((float)rand() / RAND_MAX) * 2;
    U2 = -1 + ((float)rand() / RAND_MAX) * 2;
    W = pow(U1, 2) + pow(U2, 2);
  } while (W >= 1 || W == 0);
  mult = sqrt((-2 * log(W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
  call = !call;
  return (mu + sigma * (float)X1);
}

HOSTDEV MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb * ev]);
}

HOSTDEV const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb * ev]);
}

HOSTDEV float q(const MP1I* bq, size_t it) { return (*bq).data[it]; }
//
HOSTDEV float par(const MP6F* bpars, size_t it, size_t ipar) {
  return (*bpars).data[it + ipar * bsize];
}
HOSTDEV float x(const MP6F* bpars, size_t it) { return par(bpars, it, 0); }
HOSTDEV float y(const MP6F* bpars, size_t it) { return par(bpars, it, 1); }
HOSTDEV float z(const MP6F* bpars, size_t it) { return par(bpars, it, 2); }
HOSTDEV float ipt(const MP6F* bpars, size_t it) { return par(bpars, it, 3); }
HOSTDEV float phi(const MP6F* bpars, size_t it) { return par(bpars, it, 4); }
HOSTDEV float theta(const MP6F* bpars, size_t it) { return par(bpars, it, 5); }
//
HOSTDEV float par(const MPTRK* btracks, size_t it, size_t ipar) {
  return par(&(*btracks).par, it, ipar);
}
HOSTDEV float x(const MPTRK* btracks, size_t it) { return par(btracks, it, 0); }
HOSTDEV float y(const MPTRK* btracks, size_t it) { return par(btracks, it, 1); }
HOSTDEV float z(const MPTRK* btracks, size_t it) { return par(btracks, it, 2); }
HOSTDEV float ipt(const MPTRK* btracks, size_t it) {
  return par(btracks, it, 3);
}
HOSTDEV float phi(const MPTRK* btracks, size_t it) {
  return par(btracks, it, 4);
}
HOSTDEV float theta(const MPTRK* btracks, size_t it) {
  return par(btracks, it, 5);
}
//
HOSTDEV float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar) {
  size_t ib = tk / bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
HOSTDEV float x(const MPTRK* tracks, size_t ev, size_t tk) {
  return par(tracks, ev, tk, 0);
}
HOSTDEV float y(const MPTRK* tracks, size_t ev, size_t tk) {
  return par(tracks, ev, tk, 1);
}
HOSTDEV float z(const MPTRK* tracks, size_t ev, size_t tk) {
  return par(tracks, ev, tk, 2);
}
HOSTDEV float ipt(const MPTRK* tracks, size_t ev, size_t tk) {
  return par(tracks, ev, tk, 3);
}
HOSTDEV float phi(const MPTRK* tracks, size_t ev, size_t tk) {
  return par(tracks, ev, tk, 4);
}
HOSTDEV float theta(const MPTRK* tracks, size_t ev, size_t tk) {
  return par(tracks, ev, tk, 5);
}
//
HOSTDEV void setpar(MP6F* bpars, size_t it, size_t ipar, float val) {
  (*bpars).data[it + ipar * bsize] = val;
}
HOSTDEV void setx(MP6F* bpars, size_t it, float val) {
  setpar(bpars, it, 0, val);
}
HOSTDEV void sety(MP6F* bpars, size_t it, float val) {
  setpar(bpars, it, 1, val);
}
HOSTDEV void setz(MP6F* bpars, size_t it, float val) {
  setpar(bpars, it, 2, val);
}
HOSTDEV void setipt(MP6F* bpars, size_t it, float val) {
  setpar(bpars, it, 3, val);
}
HOSTDEV void setphi(MP6F* bpars, size_t it, float val) {
  setpar(bpars, it, 4, val);
}
HOSTDEV void settheta(MP6F* bpars, size_t it, float val) {
  setpar(bpars, it, 5, val);
}
//
HOSTDEV void setpar(MPTRK* btracks, size_t it, size_t ipar, float val) {
  setpar(&(*btracks).par, it, ipar, val);
}
HOSTDEV void setx(MPTRK* btracks, size_t it, float val) {
  setpar(btracks, it, 0, val);
}
HOSTDEV void sety(MPTRK* btracks, size_t it, float val) {
  setpar(btracks, it, 1, val);
}
HOSTDEV void setz(MPTRK* btracks, size_t it, float val) {
  setpar(btracks, it, 2, val);
}
HOSTDEV void setipt(MPTRK* btracks, size_t it, float val) {
  setpar(btracks, it, 3, val);
}
HOSTDEV void setphi(MPTRK* btracks, size_t it, float val) {
  setpar(btracks, it, 4, val);
}
HOSTDEV void settheta(MPTRK* btracks, size_t it, float val) {
  setpar(btracks, it, 5, val);
}

HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb * ev]);
}
HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib, size_t lay) {
  return &(hits[lay + (ib * nlayer) + (ev * nlayer * nb)]);
}
//
HOSTDEV float pos(const MP3F* hpos, size_t it, size_t ipar) {
  return (*hpos).data[it + ipar * bsize];
}
HOSTDEV float x(const MP3F* hpos, size_t it) { return pos(hpos, it, 0); }
HOSTDEV float y(const MP3F* hpos, size_t it) { return pos(hpos, it, 1); }
HOSTDEV float z(const MP3F* hpos, size_t it) { return pos(hpos, it, 2); }
//
HOSTDEV float pos(const MPHIT* hits, size_t it, size_t ipar) {
  return pos(&(*hits).pos, it, ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t it) { return pos(hits, it, 0); }
HOSTDEV float y(const MPHIT* hits, size_t it) { return pos(hits, it, 1); }
HOSTDEV float z(const MPHIT* hits, size_t it) { return pos(hits, it, 2); }
//
HOSTDEV float pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar) {
  size_t ib = tk / bsize;
  const MPHIT* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits, it, ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t ev, size_t tk) {
  return pos(hits, ev, tk, 0);
}
HOSTDEV float y(const MPHIT* hits, size_t ev, size_t tk) {
  return pos(hits, ev, tk, 1);
}
HOSTDEV float z(const MPHIT* hits, size_t ev, size_t tk) {
  return pos(hits, ev, tk, 2);
}

#define N bsize
__forceinline__ __device__ void MultHelixProp(const MP6x6F* A, const MP6x6SF* B,
                                              MP6x6F* C) {
  const float* a = (*A).data;  // ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data;  // ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;        // ASSUME_ALIGNED(c, 64);
  // parallel_for(0,N,[&](int n){
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[1 * N + n] +
                   a[3 * N + n] * b[6 * N + n] + a[4 * N + n] * b[10 * N + n];
    c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[2 * N + n] +
                   a[3 * N + n] * b[7 * N + n] + a[4 * N + n] * b[11 * N + n];
    c[2 * N + n] = a[0 * N + n] * b[3 * N + n] + a[1 * N + n] * b[4 * N + n] +
                   a[3 * N + n] * b[8 * N + n] + a[4 * N + n] * b[12 * N + n];
    c[3 * N + n] = a[0 * N + n] * b[6 * N + n] + a[1 * N + n] * b[7 * N + n] +
                   a[3 * N + n] * b[9 * N + n] + a[4 * N + n] * b[13 * N + n];
    c[4 * N + n] = a[0 * N + n] * b[10 * N + n] + a[1 * N + n] * b[11 * N + n] +
                   a[3 * N + n] * b[13 * N + n] + a[4 * N + n] * b[14 * N + n];
    c[5 * N + n] = a[0 * N + n] * b[15 * N + n] + a[1 * N + n] * b[16 * N + n] +
                   a[3 * N + n] * b[18 * N + n] + a[4 * N + n] * b[19 * N + n];
    c[6 * N + n] = a[6 * N + n] * b[0 * N + n] + a[7 * N + n] * b[1 * N + n] +
                   a[9 * N + n] * b[6 * N + n] + a[10 * N + n] * b[10 * N + n];
    c[7 * N + n] = a[6 * N + n] * b[1 * N + n] + a[7 * N + n] * b[2 * N + n] +
                   a[9 * N + n] * b[7 * N + n] + a[10 * N + n] * b[11 * N + n];
    c[8 * N + n] = a[6 * N + n] * b[3 * N + n] + a[7 * N + n] * b[4 * N + n] +
                   a[9 * N + n] * b[8 * N + n] + a[10 * N + n] * b[12 * N + n];
    c[9 * N + n] = a[6 * N + n] * b[6 * N + n] + a[7 * N + n] * b[7 * N + n] +
                   a[9 * N + n] * b[9 * N + n] + a[10 * N + n] * b[13 * N + n];
    c[10 * N + n] =
        a[6 * N + n] * b[10 * N + n] + a[7 * N + n] * b[11 * N + n] +
        a[9 * N + n] * b[13 * N + n] + a[10 * N + n] * b[14 * N + n];
    c[11 * N + n] =
        a[6 * N + n] * b[15 * N + n] + a[7 * N + n] * b[16 * N + n] +
        a[9 * N + n] * b[18 * N + n] + a[10 * N + n] * b[19 * N + n];
    c[12 * N + n] =
        a[12 * N + n] * b[0 * N + n] + a[13 * N + n] * b[1 * N + n] +
        b[3 * N + n] + a[15 * N + n] * b[6 * N + n] +
        a[16 * N + n] * b[10 * N + n] + a[17 * N + n] * b[15 * N + n];
    c[13 * N + n] =
        a[12 * N + n] * b[1 * N + n] + a[13 * N + n] * b[2 * N + n] +
        b[4 * N + n] + a[15 * N + n] * b[7 * N + n] +
        a[16 * N + n] * b[11 * N + n] + a[17 * N + n] * b[16 * N + n];
    c[14 * N + n] =
        a[12 * N + n] * b[3 * N + n] + a[13 * N + n] * b[4 * N + n] +
        b[5 * N + n] + a[15 * N + n] * b[8 * N + n] +
        a[16 * N + n] * b[12 * N + n] + a[17 * N + n] * b[17 * N + n];
    c[15 * N + n] =
        a[12 * N + n] * b[6 * N + n] + a[13 * N + n] * b[7 * N + n] +
        b[8 * N + n] + a[15 * N + n] * b[9 * N + n] +
        a[16 * N + n] * b[13 * N + n] + a[17 * N + n] * b[18 * N + n];
    c[16 * N + n] =
        a[12 * N + n] * b[10 * N + n] + a[13 * N + n] * b[11 * N + n] +
        b[12 * N + n] + a[15 * N + n] * b[13 * N + n] +
        a[16 * N + n] * b[14 * N + n] + a[17 * N + n] * b[19 * N + n];
    c[17 * N + n] =
        a[12 * N + n] * b[15 * N + n] + a[13 * N + n] * b[16 * N + n] +
        b[17 * N + n] + a[15 * N + n] * b[18 * N + n] +
        a[16 * N + n] * b[19 * N + n] + a[17 * N + n] * b[20 * N + n];
    c[18 * N + n] =
        a[18 * N + n] * b[0 * N + n] + a[19 * N + n] * b[1 * N + n] +
        a[21 * N + n] * b[6 * N + n] + a[22 * N + n] * b[10 * N + n];
    c[19 * N + n] =
        a[18 * N + n] * b[1 * N + n] + a[19 * N + n] * b[2 * N + n] +
        a[21 * N + n] * b[7 * N + n] + a[22 * N + n] * b[11 * N + n];
    c[20 * N + n] =
        a[18 * N + n] * b[3 * N + n] + a[19 * N + n] * b[4 * N + n] +
        a[21 * N + n] * b[8 * N + n] + a[22 * N + n] * b[12 * N + n];
    c[21 * N + n] =
        a[18 * N + n] * b[6 * N + n] + a[19 * N + n] * b[7 * N + n] +
        a[21 * N + n] * b[9 * N + n] + a[22 * N + n] * b[13 * N + n];
    c[22 * N + n] =
        a[18 * N + n] * b[10 * N + n] + a[19 * N + n] * b[11 * N + n] +
        a[21 * N + n] * b[13 * N + n] + a[22 * N + n] * b[14 * N + n];
    c[23 * N + n] =
        a[18 * N + n] * b[15 * N + n] + a[19 * N + n] * b[16 * N + n] +
        a[21 * N + n] * b[18 * N + n] + a[22 * N + n] * b[19 * N + n];
    c[24 * N + n] =
        a[24 * N + n] * b[0 * N + n] + a[25 * N + n] * b[1 * N + n] +
        a[27 * N + n] * b[6 * N + n] + a[28 * N + n] * b[10 * N + n];
    c[25 * N + n] =
        a[24 * N + n] * b[1 * N + n] + a[25 * N + n] * b[2 * N + n] +
        a[27 * N + n] * b[7 * N + n] + a[28 * N + n] * b[11 * N + n];
    c[26 * N + n] =
        a[24 * N + n] * b[3 * N + n] + a[25 * N + n] * b[4 * N + n] +
        a[27 * N + n] * b[8 * N + n] + a[28 * N + n] * b[12 * N + n];
    c[27 * N + n] =
        a[24 * N + n] * b[6 * N + n] + a[25 * N + n] * b[7 * N + n] +
        a[27 * N + n] * b[9 * N + n] + a[28 * N + n] * b[13 * N + n];
    c[28 * N + n] =
        a[24 * N + n] * b[10 * N + n] + a[25 * N + n] * b[11 * N + n] +
        a[27 * N + n] * b[13 * N + n] + a[28 * N + n] * b[14 * N + n];
    c[29 * N + n] =
        a[24 * N + n] * b[15 * N + n] + a[25 * N + n] * b[16 * N + n] +
        a[27 * N + n] * b[18 * N + n] + a[28 * N + n] * b[19 * N + n];
    c[30 * N + n] = b[15 * N + n];
    c[31 * N + n] = b[16 * N + n];
    c[32 * N + n] = b[17 * N + n];
    c[33 * N + n] = b[18 * N + n];
    c[34 * N + n] = b[19 * N + n];
    c[35 * N + n] = b[20 * N + n];
  }  //);
}

__forceinline__ __device__ void MultHelixPropTransp(const MP6x6F* A,
                                                    const MP6x6F* B,
                                                    MP6x6SF* C) {
  const float* a = (*A).data;  // ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data;  // ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;        // ASSUME_ALIGNED(c, 64);
  // parallel_for(0,N,[&](int n){
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    c[0 * N + n] = b[0 * N + n] * a[0 * N + n] + b[1 * N + n] * a[1 * N + n] +
                   b[3 * N + n] * a[3 * N + n] + b[4 * N + n] * a[4 * N + n];
    c[1 * N + n] = b[6 * N + n] * a[0 * N + n] + b[7 * N + n] * a[1 * N + n] +
                   b[9 * N + n] * a[3 * N + n] + b[10 * N + n] * a[4 * N + n];
    c[2 * N + n] = b[6 * N + n] * a[6 * N + n] + b[7 * N + n] * a[7 * N + n] +
                   b[9 * N + n] * a[9 * N + n] + b[10 * N + n] * a[10 * N + n];
    c[3 * N + n] = b[12 * N + n] * a[0 * N + n] + b[13 * N + n] * a[1 * N + n] +
                   b[15 * N + n] * a[3 * N + n] + b[16 * N + n] * a[4 * N + n];
    c[4 * N + n] = b[12 * N + n] * a[6 * N + n] + b[13 * N + n] * a[7 * N + n] +
                   b[15 * N + n] * a[9 * N + n] + b[16 * N + n] * a[10 * N + n];
    c[5 * N + n] =
        b[12 * N + n] * a[12 * N + n] + b[13 * N + n] * a[13 * N + n] +
        b[14 * N + n] + b[15 * N + n] * a[15 * N + n] +
        b[16 * N + n] * a[16 * N + n] + b[17 * N + n] * a[17 * N + n];
    c[6 * N + n] = b[18 * N + n] * a[0 * N + n] + b[19 * N + n] * a[1 * N + n] +
                   b[21 * N + n] * a[3 * N + n] + b[22 * N + n] * a[4 * N + n];
    c[7 * N + n] = b[18 * N + n] * a[6 * N + n] + b[19 * N + n] * a[7 * N + n] +
                   b[21 * N + n] * a[9 * N + n] + b[22 * N + n] * a[10 * N + n];
    c[8 * N + n] =
        b[18 * N + n] * a[12 * N + n] + b[19 * N + n] * a[13 * N + n] +
        b[20 * N + n] + b[21 * N + n] * a[15 * N + n] +
        b[22 * N + n] * a[16 * N + n] + b[23 * N + n] * a[17 * N + n];
    c[9 * N + n] =
        b[18 * N + n] * a[18 * N + n] + b[19 * N + n] * a[19 * N + n] +
        b[21 * N + n] * a[21 * N + n] + b[22 * N + n] * a[22 * N + n];
    c[10 * N + n] = b[24 * N + n] * a[0 * N + n] +
                    b[25 * N + n] * a[1 * N + n] +
                    b[27 * N + n] * a[3 * N + n] + b[28 * N + n] * a[4 * N + n];
    c[11 * N + n] =
        b[24 * N + n] * a[6 * N + n] + b[25 * N + n] * a[7 * N + n] +
        b[27 * N + n] * a[9 * N + n] + b[28 * N + n] * a[10 * N + n];
    c[12 * N + n] =
        b[24 * N + n] * a[12 * N + n] + b[25 * N + n] * a[13 * N + n] +
        b[26 * N + n] + b[27 * N + n] * a[15 * N + n] +
        b[28 * N + n] * a[16 * N + n] + b[29 * N + n] * a[17 * N + n];
    c[13 * N + n] =
        b[24 * N + n] * a[18 * N + n] + b[25 * N + n] * a[19 * N + n] +
        b[27 * N + n] * a[21 * N + n] + b[28 * N + n] * a[22 * N + n];
    c[14 * N + n] =
        b[24 * N + n] * a[24 * N + n] + b[25 * N + n] * a[25 * N + n] +
        b[27 * N + n] * a[27 * N + n] + b[28 * N + n] * a[28 * N + n];
    c[15 * N + n] = b[30 * N + n] * a[0 * N + n] +
                    b[31 * N + n] * a[1 * N + n] +
                    b[33 * N + n] * a[3 * N + n] + b[34 * N + n] * a[4 * N + n];
    c[16 * N + n] =
        b[30 * N + n] * a[6 * N + n] + b[31 * N + n] * a[7 * N + n] +
        b[33 * N + n] * a[9 * N + n] + b[34 * N + n] * a[10 * N + n];
    c[17 * N + n] =
        b[30 * N + n] * a[12 * N + n] + b[31 * N + n] * a[13 * N + n] +
        b[32 * N + n] + b[33 * N + n] * a[15 * N + n] +
        b[34 * N + n] * a[16 * N + n] + b[35 * N + n] * a[17 * N + n];
    c[18 * N + n] =
        b[30 * N + n] * a[18 * N + n] + b[31 * N + n] * a[19 * N + n] +
        b[33 * N + n] * a[21 * N + n] + b[34 * N + n] * a[22 * N + n];
    c[19 * N + n] =
        b[30 * N + n] * a[24 * N + n] + b[31 * N + n] * a[25 * N + n] +
        b[33 * N + n] * a[27 * N + n] + b[34 * N + n] * a[28 * N + n];
    c[20 * N + n] = b[35 * N + n];
  }  //);
}

__forceinline__ __device__ void KalmanGainInv(const MP6x6SF* A,
                                              const MP3x3SF* B, MP3x3* C) {
  const float* a = (*A).data;  // ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data;  // ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;        // ASSUME_ALIGNED(c, 64);
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    double det =
        ((a[0 * N + n] + b[0 * N + n]) *
         (((a[6 * N + n] + b[3 * N + n]) * (a[11 * N + n] + b[5 * N + n])) -
          ((a[7 * N + n] + b[4 * N + n]) * (a[7 * N + n] + b[4 * N + n])))) -
        ((a[1 * N + n] + b[1 * N + n]) *
         (((a[1 * N + n] + b[1 * N + n]) * (a[11 * N + n] + b[5 * N + n])) -
          ((a[7 * N + n] + b[4 * N + n]) * (a[2 * N + n] + b[2 * N + n])))) +
        ((a[2 * N + n] + b[2 * N + n]) *
         (((a[1 * N + n] + b[1 * N + n]) * (a[7 * N + n] + b[4 * N + n])) -
          ((a[2 * N + n] + b[2 * N + n]) * (a[6 * N + n] + b[3 * N + n]))));
    double invdet = 1.0 / det;

    c[0 * N + n] =
        invdet *
        (((a[6 * N + n] + b[3 * N + n]) * (a[11 * N + n] + b[5 * N + n])) -
         ((a[7 * N + n] + b[4 * N + n]) * (a[7 * N + n] + b[4 * N + n])));
    c[1 * N + n] =
        -1 * invdet *
        (((a[1 * N + n] + b[1 * N + n]) * (a[11 * N + n] + b[5 * N + n])) -
         ((a[2 * N + n] + b[2 * N + n]) * (a[7 * N + n] + b[4 * N + n])));
    c[2 * N + n] =
        invdet *
        (((a[1 * N + n] + b[1 * N + n]) * (a[7 * N + n] + b[4 * N + n])) -
         ((a[2 * N + n] + b[2 * N + n]) * (a[7 * N + n] + b[4 * N + n])));
    c[3 * N + n] =
        -1 * invdet *
        (((a[1 * N + n] + b[1 * N + n]) * (a[11 * N + n] + b[5 * N + n])) -
         ((a[7 * N + n] + b[4 * N + n]) * (a[2 * N + n] + b[2 * N + n])));
    c[4 * N + n] =
        invdet *
        (((a[0 * N + n] + b[0 * N + n]) * (a[11 * N + n] + b[5 * N + n])) -
         ((a[2 * N + n] + b[2 * N + n]) * (a[2 * N + n] + b[2 * N + n])));
    c[5 * N + n] =
        -1 * invdet *
        (((a[0 * N + n] + b[0 * N + n]) * (a[7 * N + n] + b[4 * N + n])) -
         ((a[2 * N + n] + b[2 * N + n]) * (a[1 * N + n] + b[1 * N + n])));
    c[6 * N + n] =
        invdet *
        (((a[1 * N + n] + b[1 * N + n]) * (a[7 * N + n] + b[4 * N + n])) -
         ((a[2 * N + n] + b[2 * N + n]) * (a[6 * N + n] + b[3 * N + n])));
    c[7 * N + n] =
        -1 * invdet *
        (((a[0 * N + n] + b[0 * N + n]) * (a[7 * N + n] + b[4 * N + n])) -
         ((a[2 * N + n] + b[2 * N + n]) * (a[1 * N + n] + b[1 * N + n])));
    c[8 * N + n] =
        invdet *
        (((a[0 * N + n] + b[0 * N + n]) * (a[6 * N + n] + b[3 * N + n])) -
         ((a[1 * N + n] + b[1 * N + n]) * (a[1 * N + n] + b[1 * N + n])));
  }
}
__forceinline__ __device__ void KalmanGain(const MP6x6SF* A, const MP3x3* B,
                                           MP3x6* C) {
  const float* a = (*A).data;  // ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data;  // ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;        // ASSUME_ALIGNED(c, 64);
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[3 * N + n] +
                   a[2 * N + n] * b[6 * N + n];
    c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[4 * N + n] +
                   a[2 * N + n] * b[7 * N + n];
    c[2 * N + n] = a[0 * N + n] * b[2 * N + n] + a[1 * N + n] * b[5 * N + n] +
                   a[2 * N + n] * b[8 * N + n];
    c[3 * N + n] = a[1 * N + n] * b[0 * N + n] + a[6 * N + n] * b[3 * N + n] +
                   a[7 * N + n] * b[6 * N + n];
    c[4 * N + n] = a[1 * N + n] * b[1 * N + n] + a[6 * N + n] * b[4 * N + n] +
                   a[7 * N + n] * b[7 * N + n];
    c[5 * N + n] = a[1 * N + n] * b[2 * N + n] + a[6 * N + n] * b[5 * N + n] +
                   a[7 * N + n] * b[8 * N + n];
    c[6 * N + n] = a[2 * N + n] * b[0 * N + n] + a[7 * N + n] * b[3 * N + n] +
                   a[11 * N + n] * b[6 * N + n];
    c[7 * N + n] = a[2 * N + n] * b[1 * N + n] + a[7 * N + n] * b[4 * N + n] +
                   a[11 * N + n] * b[7 * N + n];
    c[8 * N + n] = a[2 * N + n] * b[2 * N + n] + a[7 * N + n] * b[5 * N + n] +
                   a[11 * N + n] * b[8 * N + n];
    c[9 * N + n] = a[3 * N + n] * b[0 * N + n] + a[8 * N + n] * b[3 * N + n] +
                   a[12 * N + n] * b[6 * N + n];
    c[10 * N + n] = a[3 * N + n] * b[1 * N + n] + a[8 * N + n] * b[4 * N + n] +
                    a[12 * N + n] * b[7 * N + n];
    c[11 * N + n] = a[3 * N + n] * b[2 * N + n] + a[8 * N + n] * b[5 * N + n] +
                    a[12 * N + n] * b[8 * N + n];
    c[12 * N + n] = a[4 * N + n] * b[0 * N + n] + a[9 * N + n] * b[3 * N + n] +
                    a[13 * N + n] * b[6 * N + n];
    c[13 * N + n] = a[4 * N + n] * b[1 * N + n] + a[9 * N + n] * b[4 * N + n] +
                    a[13 * N + n] * b[7 * N + n];
    c[14 * N + n] = a[4 * N + n] * b[2 * N + n] + a[9 * N + n] * b[5 * N + n] +
                    a[13 * N + n] * b[8 * N + n];
    c[15 * N + n] = a[5 * N + n] * b[0 * N + n] + a[10 * N + n] * b[3 * N + n] +
                    a[14 * N + n] * b[6 * N + n];
    c[16 * N + n] = a[5 * N + n] * b[1 * N + n] + a[10 * N + n] * b[4 * N + n] +
                    a[14 * N + n] * b[7 * N + n];
    c[17 * N + n] = a[5 * N + n] * b[2 * N + n] + a[10 * N + n] * b[5 * N + n] +
                    a[14 * N + n] * b[8 * N + n];
  }
}

HOSTDEV inline float hipo(float x, float y) { return std::sqrt(x * x + y * y); }

__device__ void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar,
                             const MP3x3SF* hitErr, const MP3F* msP,
                             MP1F* rotT00, MP1F* rotT01, MP2x2SF* resErr_loc,
                             MP3x6* kGain, MP2F* res_loc, MP6x6SF* newErr) {
  for (size_t it = threadIdx.x; it < bsize; it += blockDim.x) {
    const float r = hipo(x(msP, it), y(msP, it));
    rotT00->data[it] = -(y(msP, it) + y(inPar, it)) / (2 * r);
    rotT01->data[it] = (x(msP, it) + x(inPar, it)) / (2 * r);

    resErr_loc->data[0 * bsize + it] =
        (rotT00->data[it] *
             (trkErr->data[0 * bsize + it] + hitErr->data[0 * bsize + it]) +
         rotT01->data[it] *
             (trkErr->data[1 * bsize + it] + hitErr->data[1 * bsize + it])) *
            rotT00->data[it] +
        (rotT00->data[it] *
             (trkErr->data[1 * bsize + it] + hitErr->data[1 * bsize + it]) +
         rotT01->data[it] *
             (trkErr->data[2 * bsize + it] + hitErr->data[2 * bsize + it])) *
            rotT01->data[it];
    resErr_loc->data[1 * bsize + it] =
        (trkErr->data[3 * bsize + it] + hitErr->data[3 * bsize + it]) *
            rotT00->data[it] +
        (trkErr->data[4 * bsize + it] + hitErr->data[4 * bsize + it]) *
            rotT01->data[it];
    resErr_loc->data[2 * bsize + it] =
        (trkErr->data[5 * bsize + it] + hitErr->data[5 * bsize + it]);
  }

  for (size_t it = threadIdx.x; it < bsize; it += blockDim.x) {
    const double det = (double)resErr_loc->data[0 * bsize + it] *
                           resErr_loc->data[2 * bsize + it] -
                       (double)resErr_loc->data[1 * bsize + it] *
                           resErr_loc->data[1 * bsize + it];
    const float s = 1.f / det;
    const float tmp = s * resErr_loc->data[2 * bsize + it];
    resErr_loc->data[1 * bsize + it] *= -s;
    resErr_loc->data[2 * bsize + it] = s * resErr_loc->data[0 * bsize + it];
    resErr_loc->data[0 * bsize + it] = tmp;
  }

  for (size_t it = threadIdx.x; it < bsize; it += blockDim.x) {
    kGain->data[0 * bsize + it] =
        trkErr->data[0 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[1 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[3 * bsize + it] * resErr_loc->data[1 * bsize + it];
    kGain->data[1 * bsize + it] =
        trkErr->data[0 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[1 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[3 * bsize + it] * resErr_loc->data[2 * bsize + it];
    kGain->data[2 * bsize + it] = 0;
    kGain->data[3 * bsize + it] =
        trkErr->data[1 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[2 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[4 * bsize + it] * resErr_loc->data[1 * bsize + it];
    kGain->data[4 * bsize + it] =
        trkErr->data[1 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[2 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[4 * bsize + it] * resErr_loc->data[2 * bsize + it];
    kGain->data[5 * bsize + it] = 0;
    kGain->data[6 * bsize + it] =
        trkErr->data[3 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[4 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[5 * bsize + it] * resErr_loc->data[1 * bsize + it];
    kGain->data[7 * bsize + it] =
        trkErr->data[3 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[4 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[5 * bsize + it] * resErr_loc->data[2 * bsize + it];
    kGain->data[8 * bsize + it] = 0;
    kGain->data[9 * bsize + it] =
        trkErr->data[6 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[7 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[8 * bsize + it] * resErr_loc->data[1 * bsize + it];
    kGain->data[10 * bsize + it] =
        trkErr->data[6 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[7 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[8 * bsize + it] * resErr_loc->data[2 * bsize + it];
    kGain->data[11 * bsize + it] = 0;
    kGain->data[12 * bsize + it] =
        trkErr->data[10 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[11 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[12 * bsize + it] * resErr_loc->data[1 * bsize + it];
    kGain->data[13 * bsize + it] =
        trkErr->data[10 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[11 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[12 * bsize + it] * resErr_loc->data[2 * bsize + it];
    kGain->data[14 * bsize + it] = 0;
    kGain->data[15 * bsize + it] =
        trkErr->data[15 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[16 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[0 * bsize + it]) +
        trkErr->data[17 * bsize + it] * resErr_loc->data[1 * bsize + it];
    kGain->data[16 * bsize + it] =
        trkErr->data[15 * bsize + it] *
            (rotT00->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[16 * bsize + it] *
            (rotT01->data[it] * resErr_loc->data[1 * bsize + it]) +
        trkErr->data[17 * bsize + it] * resErr_loc->data[2 * bsize + it];
    kGain->data[17 * bsize + it] = 0;
  }

  for (size_t it = threadIdx.x; it < bsize; it += blockDim.x) {
    res_loc->data[0 * bsize + it] =
        rotT00->data[it] * (x(msP, it) - x(inPar, it)) +
        rotT01->data[it] * (y(msP, it) - y(inPar, it));
    res_loc->data[1 * bsize + it] = z(msP, it) - z(inPar, it);

    setx(inPar, it,
         x(inPar, it) +
             kGain->data[0 * bsize + it] * res_loc->data[0 * bsize + it] +
             kGain->data[1 * bsize + it] * res_loc->data[1 * bsize + it]);
    sety(inPar, it,
         y(inPar, it) +
             kGain->data[3 * bsize + it] * res_loc->data[0 * bsize + it] +
             kGain->data[4 * bsize + it] * res_loc->data[1 * bsize + it]);
    setz(inPar, it,
         z(inPar, it) +
             kGain->data[6 * bsize + it] * res_loc->data[0 * bsize + it] +
             kGain->data[7 * bsize + it] * res_loc->data[1 * bsize + it]);
    setipt(inPar, it,
           ipt(inPar, it) +
               kGain->data[9 * bsize + it] * res_loc->data[0 * bsize + it] +
               kGain->data[10 * bsize + it] * res_loc->data[1 * bsize + it]);
    setphi(inPar, it,
           phi(inPar, it) +
               kGain->data[12 * bsize + it] * res_loc->data[0 * bsize + it] +
               kGain->data[13 * bsize + it] * res_loc->data[1 * bsize + it]);
    settheta(inPar, it,
             theta(inPar, it) +
                 kGain->data[15 * bsize + it] * res_loc->data[0 * bsize + it] +
                 kGain->data[16 * bsize + it] * res_loc->data[1 * bsize + it]);
  }

  for (size_t it = threadIdx.x; it < bsize; it += blockDim.x) {
    newErr->data[0 * bsize + it] =
        kGain->data[0 * bsize + it] * rotT00->data[it] *
            trkErr->data[0 * bsize + it] +
        kGain->data[0 * bsize + it] * rotT01->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[1 * bsize + it] * trkErr->data[3 * bsize + it];
    newErr->data[1 * bsize + it] =
        kGain->data[3 * bsize + it] * rotT00->data[it] *
            trkErr->data[0 * bsize + it] +
        kGain->data[3 * bsize + it] * rotT01->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[4 * bsize + it] * trkErr->data[3 * bsize + it];
    newErr->data[2 * bsize + it] =
        kGain->data[3 * bsize + it] * rotT00->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[3 * bsize + it] * rotT01->data[it] *
            trkErr->data[2 * bsize + it] +
        kGain->data[4 * bsize + it] * trkErr->data[4 * bsize + it];
    newErr->data[3 * bsize + it] =
        kGain->data[6 * bsize + it] * rotT00->data[it] *
            trkErr->data[0 * bsize + it] +
        kGain->data[6 * bsize + it] * rotT01->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[7 * bsize + it] * trkErr->data[3 * bsize + it];
    newErr->data[4 * bsize + it] =
        kGain->data[6 * bsize + it] * rotT00->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[6 * bsize + it] * rotT01->data[it] *
            trkErr->data[2 * bsize + it] +
        kGain->data[7 * bsize + it] * trkErr->data[4 * bsize + it];
    newErr->data[5 * bsize + it] =
        kGain->data[6 * bsize + it] * rotT00->data[it] *
            trkErr->data[3 * bsize + it] +
        kGain->data[6 * bsize + it] * rotT01->data[it] *
            trkErr->data[4 * bsize + it] +
        kGain->data[7 * bsize + it] * trkErr->data[5 * bsize + it];
    newErr->data[6 * bsize + it] =
        kGain->data[9 * bsize + it] * rotT00->data[it] *
            trkErr->data[0 * bsize + it] +
        kGain->data[9 * bsize + it] * rotT01->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[10 * bsize + it] * trkErr->data[3 * bsize + it];
    newErr->data[7 * bsize + it] =
        kGain->data[9 * bsize + it] * rotT00->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[9 * bsize + it] * rotT01->data[it] *
            trkErr->data[2 * bsize + it] +
        kGain->data[10 * bsize + it] * trkErr->data[4 * bsize + it];
    newErr->data[8 * bsize + it] =
        kGain->data[9 * bsize + it] * rotT00->data[it] *
            trkErr->data[3 * bsize + it] +
        kGain->data[9 * bsize + it] * rotT01->data[it] *
            trkErr->data[4 * bsize + it] +
        kGain->data[10 * bsize + it] * trkErr->data[5 * bsize + it];
    newErr->data[9 * bsize + it] =
        kGain->data[9 * bsize + it] * rotT00->data[it] *
            trkErr->data[6 * bsize + it] +
        kGain->data[9 * bsize + it] * rotT01->data[it] *
            trkErr->data[7 * bsize + it] +
        kGain->data[10 * bsize + it] * trkErr->data[8 * bsize + it];
    newErr->data[10 * bsize + it] =
        kGain->data[12 * bsize + it] * rotT00->data[it] *
            trkErr->data[0 * bsize + it] +
        kGain->data[12 * bsize + it] * rotT01->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[13 * bsize + it] * trkErr->data[3 * bsize + it];
    newErr->data[11 * bsize + it] =
        kGain->data[12 * bsize + it] * rotT00->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[12 * bsize + it] * rotT01->data[it] *
            trkErr->data[2 * bsize + it] +
        kGain->data[13 * bsize + it] * trkErr->data[4 * bsize + it];
    newErr->data[12 * bsize + it] =
        kGain->data[12 * bsize + it] * rotT00->data[it] *
            trkErr->data[3 * bsize + it] +
        kGain->data[12 * bsize + it] * rotT01->data[it] *
            trkErr->data[4 * bsize + it] +
        kGain->data[13 * bsize + it] * trkErr->data[5 * bsize + it];
    newErr->data[13 * bsize + it] =
        kGain->data[12 * bsize + it] * rotT00->data[it] *
            trkErr->data[6 * bsize + it] +
        kGain->data[12 * bsize + it] * rotT01->data[it] *
            trkErr->data[7 * bsize + it] +
        kGain->data[13 * bsize + it] * trkErr->data[8 * bsize + it];
    newErr->data[14 * bsize + it] =
        kGain->data[12 * bsize + it] * rotT00->data[it] *
            trkErr->data[10 * bsize + it] +
        kGain->data[12 * bsize + it] * rotT01->data[it] *
            trkErr->data[11 * bsize + it] +
        kGain->data[13 * bsize + it] * trkErr->data[12 * bsize + it];
    newErr->data[15 * bsize + it] =
        kGain->data[15 * bsize + it] * rotT00->data[it] *
            trkErr->data[0 * bsize + it] +
        kGain->data[15 * bsize + it] * rotT01->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[16 * bsize + it] * trkErr->data[3 * bsize + it];
    newErr->data[16 * bsize + it] =
        kGain->data[15 * bsize + it] * rotT00->data[it] *
            trkErr->data[1 * bsize + it] +
        kGain->data[15 * bsize + it] * rotT01->data[it] *
            trkErr->data[2 * bsize + it] +
        kGain->data[16 * bsize + it] * trkErr->data[4 * bsize + it];
    newErr->data[17 * bsize + it] =
        kGain->data[15 * bsize + it] * rotT00->data[it] *
            trkErr->data[3 * bsize + it] +
        kGain->data[15 * bsize + it] * rotT01->data[it] *
            trkErr->data[4 * bsize + it] +
        kGain->data[16 * bsize + it] * trkErr->data[5 * bsize + it];
    newErr->data[18 * bsize + it] =
        kGain->data[15 * bsize + it] * rotT00->data[it] *
            trkErr->data[6 * bsize + it] +
        kGain->data[15 * bsize + it] * rotT01->data[it] *
            trkErr->data[7 * bsize + it] +
        kGain->data[16 * bsize + it] * trkErr->data[8 * bsize + it];
    newErr->data[19 * bsize + it] =
        kGain->data[15 * bsize + it] * rotT00->data[it] *
            trkErr->data[10 * bsize + it] +
        kGain->data[15 * bsize + it] * rotT01->data[it] *
            trkErr->data[11 * bsize + it] +
        kGain->data[16 * bsize + it] * trkErr->data[12 * bsize + it];
    newErr->data[20 * bsize + it] =
        kGain->data[15 * bsize + it] * rotT00->data[it] *
            trkErr->data[15 * bsize + it] +
        kGain->data[15 * bsize + it] * rotT01->data[it] *
            trkErr->data[16 * bsize + it] +
        kGain->data[16 * bsize + it] * trkErr->data[17 * bsize + it];

    newErr->data[0 * bsize + it] =
        trkErr->data[0 * bsize + it] - newErr->data[0 * bsize + it];
    newErr->data[1 * bsize + it] =
        trkErr->data[1 * bsize + it] - newErr->data[1 * bsize + it];
    newErr->data[2 * bsize + it] =
        trkErr->data[2 * bsize + it] - newErr->data[2 * bsize + it];
    newErr->data[3 * bsize + it] =
        trkErr->data[3 * bsize + it] - newErr->data[3 * bsize + it];
    newErr->data[4 * bsize + it] =
        trkErr->data[4 * bsize + it] - newErr->data[4 * bsize + it];
    newErr->data[5 * bsize + it] =
        trkErr->data[5 * bsize + it] - newErr->data[5 * bsize + it];
    newErr->data[6 * bsize + it] =
        trkErr->data[6 * bsize + it] - newErr->data[6 * bsize + it];
    newErr->data[7 * bsize + it] =
        trkErr->data[7 * bsize + it] - newErr->data[7 * bsize + it];
    newErr->data[8 * bsize + it] =
        trkErr->data[8 * bsize + it] - newErr->data[8 * bsize + it];
    newErr->data[9 * bsize + it] =
        trkErr->data[9 * bsize + it] - newErr->data[9 * bsize + it];
    newErr->data[10 * bsize + it] =
        trkErr->data[10 * bsize + it] - newErr->data[10 * bsize + it];
    newErr->data[11 * bsize + it] =
        trkErr->data[11 * bsize + it] - newErr->data[11 * bsize + it];
    newErr->data[12 * bsize + it] =
        trkErr->data[12 * bsize + it] - newErr->data[12 * bsize + it];
    newErr->data[13 * bsize + it] =
        trkErr->data[13 * bsize + it] - newErr->data[13 * bsize + it];
    newErr->data[14 * bsize + it] =
        trkErr->data[14 * bsize + it] - newErr->data[14 * bsize + it];
    newErr->data[15 * bsize + it] =
        trkErr->data[15 * bsize + it] - newErr->data[15 * bsize + it];
    newErr->data[16 * bsize + it] =
        trkErr->data[16 * bsize + it] - newErr->data[16 * bsize + it];
    newErr->data[17 * bsize + it] =
        trkErr->data[17 * bsize + it] - newErr->data[17 * bsize + it];
    newErr->data[18 * bsize + it] =
        trkErr->data[18 * bsize + it] - newErr->data[18 * bsize + it];
    newErr->data[19 * bsize + it] =
        trkErr->data[19 * bsize + it] - newErr->data[19 * bsize + it];
    newErr->data[20 * bsize + it] =
        trkErr->data[20 * bsize + it] - newErr->data[20 * bsize + it];
  }

  /*
  MPlexLH K;           // kalman gain, fixme should be L2
  KalmanHTG(rotT00, rotT01, resErr_loc, tempHH); // intermediate term to get
  kalman gain (H^T*G) KalmanGain(psErr, tempHH, K);

  MPlexHV res_glo;   //position residual in global coordinates
  SubtractFirst3(msPar, psPar, res_glo);
  MPlex2V res_loc;   //position residual in local coordinates
  RotateResidulsOnTangentPlane(rotT00,rotT01,res_glo,res_loc);

  //    Chi2Similarity(res_loc, resErr_loc, outChi2);

  MultResidualsAdd(K, psPar, res_loc, outPar);
  MPlexLL tempLL;
  squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
  KHMult(K, rotT00, rotT01, tempLL);
  KHC(tempLL, psErr, outErr);
  outErr.Subtract(psErr, outErr);
  */

  trkErr = newErr;
}

HOSTDEV inline void sincos4(const float x, float& sin, float& cos) {
  // Had this writen with explicit division by factorial.
  // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

  const float x2 = x * x;
  cos = 1.f - 0.5f * x2 + 0.04166667f * x2 * x2;
  sin = x - 0.16666667f * x * x2;
}

constexpr float kfact = 100 / 3.8;
constexpr int Niter = 5;
__device__ void propagateToR(const MP6x6SF* inErr, const MP6F* inPar,
                             const MP1I* inChg, const MP3F* msP,
                             MP6x6SF* outErr, MP6F* outPar, MP6x6F* errorProp,
                             MP6x6F* temp) {
  // MP6x6F errorProp, temp;
  for (size_t it = threadIdx.x; it < bsize; it += blockDim.x) {
    // initialize erroProp to identity matrix
    for (size_t i = 0; i < 6; ++i)
      errorProp->data[bsize * PosInMtrx(i, i, 6) + it] = 1.f;

    float r0 = hipo(x(inPar, it), y(inPar, it));
    const float k = q(inChg, it) * kfact;
    const float r = hipo(x(msP, it), y(msP, it));

    const float xin = x(inPar, it);
    const float yin = y(inPar, it);
    const float iptin = ipt(inPar, it);
    const float phiin = phi(inPar, it);
    const float thetain = theta(inPar, it);

    // initialize outPar to inPar
    setx(outPar, it, xin);
    sety(outPar, it, yin);
    setz(outPar, it, z(inPar, it));
    setipt(outPar, it, iptin);
    setphi(outPar, it, phiin);
    settheta(outPar, it, thetain);

    const float kinv = 1.f / k;
    const float pt = 1.f / iptin;

    float D = 0., cosa = 0., sina = 0., id = 0.;
    // no trig approx here, phi can be large
    float cosPorT = std::cos(phiin), sinPorT = std::sin(phiin);
    float pxin = cosPorT * pt;
    float pyin = sinPorT * pt;

    // derivatives initialized to value for first iteration, i.e. distance =
    // r-r0in
    float dDdx = r0 > 0.f ? -xin / r0 : 0.f;
    float dDdy = r0 > 0.f ? -yin / r0 : 0.f;
    float dDdipt = 0.;
    float dDdphi = 0.;

    for (int i = 0; i < Niter; ++i) {
      // compute distance and path for the current iteration
      r0 = hipo(x(outPar, it), y(outPar, it));
      id = (r - r0);
      D += id;
      sincos4(id * iptin * kinv, sina, cosa);

      // update derivatives on total distance
      if (i + 1 != Niter) {
        const float xtmp = x(outPar, it);
        const float ytmp = y(outPar, it);
        const float oor0 =
            (r0 > 0.f && std::abs(r - r0) < 0.0001f) ? 1.f / r0 : 0.f;

        const float dadipt = id * kinv;

        const float dadx = -xtmp * iptin * kinv * oor0;
        const float dady = -ytmp * iptin * kinv * oor0;

        const float pxca = pxin * cosa;
        const float pxsa = pxin * sina;
        const float pyca = pyin * cosa;
        const float pysa = pyin * sina;

        float tmp = k * dadx;
        dDdx -=
            (xtmp * (1.f + tmp * (pxca - pysa)) + ytmp * tmp * (pyca + pxsa)) *
            oor0;
        tmp = k * dady;
        dDdy -=
            (xtmp * tmp * (pxca - pysa) + ytmp * (1.f + tmp * (pyca + pxsa))) *
            oor0;
        // now r0 depends on ipt and phi as well
        tmp = dadipt * iptin;
        dDdipt -= k *
                  (xtmp * (pxca * tmp - pysa * tmp - pyca - pxsa + pyin) +
                   ytmp * (pyca * tmp + pxsa * tmp - pysa + pxca - pxin)) *
                  pt * oor0;
        dDdphi += k *
                  (xtmp * (pysa - pxin + pxca) - ytmp * (pxsa - pyin + pyca)) *
                  oor0;
      }

      // update parameters
      setx(outPar, it, x(outPar, it) + k * (pxin * sina - pyin * (1.f - cosa)));
      sety(outPar, it, y(outPar, it) + k * (pyin * sina + pxin * (1.f - cosa)));
      const float pxinold = pxin;  // copy before overwriting
      pxin = pxin * cosa - pyin * sina;
      pyin = pyin * cosa + pxinold * sina;
    }

    const float alpha = D * iptin * kinv;
    const float dadx = dDdx * iptin * kinv;
    const float dady = dDdy * iptin * kinv;
    const float dadipt = (iptin * dDdipt + D) * kinv;
    const float dadphi = dDdphi * iptin * kinv;

    sincos4(alpha, sina, cosa);

    errorProp->data[bsize * PosInMtrx(0, 0, 6) + it] =
        1.f + k * dadx * (cosPorT * cosa - sinPorT * sina) * pt;
    errorProp->data[bsize * PosInMtrx(0, 1, 6) + it] =
        k * dady * (cosPorT * cosa - sinPorT * sina) * pt;
    errorProp->data[bsize * PosInMtrx(0, 2, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(0, 3, 6) + it] =
        k *
        (cosPorT * (iptin * dadipt * cosa - sina) +
         sinPorT * ((1.f - cosa) - iptin * dadipt * sina)) *
        pt * pt;
    errorProp->data[bsize * PosInMtrx(0, 4, 6) + it] =
        k *
        (cosPorT * dadphi * cosa - sinPorT * dadphi * sina - sinPorT * sina +
         cosPorT * cosa - cosPorT) *
        pt;
    errorProp->data[bsize * PosInMtrx(0, 5, 6) + it] = 0.f;

    errorProp->data[bsize * PosInMtrx(1, 0, 6) + it] =
        k * dadx * (sinPorT * cosa + cosPorT * sina) * pt;
    errorProp->data[bsize * PosInMtrx(1, 1, 6) + it] =
        1.f + k * dady * (sinPorT * cosa + cosPorT * sina) * pt;
    errorProp->data[bsize * PosInMtrx(1, 2, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(1, 3, 6) + it] =
        k *
        (sinPorT * (iptin * dadipt * cosa - sina) +
         cosPorT * (iptin * dadipt * sina - (1.f - cosa))) *
        pt * pt;
    errorProp->data[bsize * PosInMtrx(1, 4, 6) + it] =
        k *
        (sinPorT * dadphi * cosa + cosPorT * dadphi * sina + sinPorT * cosa +
         cosPorT * sina - sinPorT) *
        pt;
    errorProp->data[bsize * PosInMtrx(1, 5, 6) + it] = 0.f;

    // no trig approx here, theta can be large
    cosPorT = std::cos(thetain);
    sinPorT = std::sin(thetain);
    // redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f / sinPorT;

    setz(outPar, it, z(inPar, it) + k * alpha * cosPorT * pt * sinPorT);

    errorProp->data[bsize * PosInMtrx(2, 0, 6) + it] =
        k * cosPorT * dadx * pt * sinPorT;
    errorProp->data[bsize * PosInMtrx(2, 1, 6) + it] =
        k * cosPorT * dady * pt * sinPorT;
    errorProp->data[bsize * PosInMtrx(2, 2, 6) + it] = 1.f;
    errorProp->data[bsize * PosInMtrx(2, 3, 6) + it] =
        k * cosPorT * (iptin * dadipt - alpha) * pt * pt * sinPorT;
    errorProp->data[bsize * PosInMtrx(2, 4, 6) + it] =
        k * dadphi * cosPorT * pt * sinPorT;
    errorProp->data[bsize * PosInMtrx(2, 5, 6) + it] =
        -k * alpha * pt * sinPorT * sinPorT;

    setipt(outPar, it, iptin);

    errorProp->data[bsize * PosInMtrx(3, 0, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(3, 1, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(3, 2, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(3, 3, 6) + it] = 1.f;
    errorProp->data[bsize * PosInMtrx(3, 4, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(3, 5, 6) + it] = 0.f;

    setphi(outPar, it, phi(inPar, it) + alpha);

    errorProp->data[bsize * PosInMtrx(4, 0, 6) + it] = dadx;
    errorProp->data[bsize * PosInMtrx(4, 1, 6) + it] = dady;
    errorProp->data[bsize * PosInMtrx(4, 2, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(4, 3, 6) + it] = dadipt;
    errorProp->data[bsize * PosInMtrx(4, 4, 6) + it] = 1.f + dadphi;
    errorProp->data[bsize * PosInMtrx(4, 5, 6) + it] = 0.f;

    settheta(outPar, it, thetain);

    errorProp->data[bsize * PosInMtrx(5, 0, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(5, 1, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(5, 2, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(5, 3, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(5, 4, 6) + it] = 0.f;
    errorProp->data[bsize * PosInMtrx(5, 5, 6) + it] = 1.f;
  }

  MultHelixProp(errorProp, inErr, temp);
  MultHelixPropTransp(errorProp, temp, outErr);
}

__device__ __constant__ int ie_range = (int)nevts / num_streams;
//__global__ void GPUsequence(MPTRK* trk, MPHIT* hit, MPTRK* outtrk, MP6x6SF*
//newErr,MP6x6F* errorProp , const int stream){
__global__ void GPUsequence(MPTRK* trk, MPHIT* hit, MPTRK* outtrk,
                            const int stream) {
  ///*__shared__*/ struct MP6x6F errorProp, temp; // shared memory here causes a
  ///race condition. Probably move to inside the p2z function? i forgot why I
  ///did it this way to begin with. maybe to make it shared?

  __shared__ struct MP6x6F errorProp, temp;
  __shared__ struct MP1F rotT00, rotT01;
  __shared__ struct MP2x2SF resErr_loc;
  __shared__ struct MP3x6 kGain;
  __shared__ struct MP2F res_loc;
  __shared__ struct MP6x6SF newErr;

  const int end = (stream < num_streams) ? nb * nevts / num_streams
                                         :       // for "full" streams
                      nb * nevts % num_streams;  // possible remainder

  for (size_t ti = blockIdx.x; ti < end; ti += gridDim.x) {
    int ie = ti / nb;
    int ib = ti % nb;
    const MPTRK* btracks = bTk(trk, ie, ib);
    MPTRK* obtracks = bTk(outtrk, ie, ib);
    for (int layer = 0; layer < nlayer; ++layer) {
      const MPHIT* bhits = bHit(hit, ie, ib, layer);
      propagateToR(&(*btracks).cov, &(*btracks).par, &(*btracks).q,
                   &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par,
                   &(errorProp), &temp);
      KalmanUpdate(&(*obtracks).cov, &(*obtracks).par, &(*bhits).cov,
                   &(*bhits).pos, &rotT00, &rotT01, &resErr_loc, &kGain,
                   &res_loc, &(newErr));
    }
    // if((index)%100==0 ) printf("index = %i ,(block,grid)=(%i,%i), track =
    // (%.3f)\n ", index,blockDim.x,gridDim.x,&(*btracks).par.data[8]);
  }
}

//
// backend to run p2r code as a service. The code is modified
// from the identity backend example.
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                    \
  do {                                                                 \
    if ((RESPONSES)[IDX] != nullptr) {                                 \
      TRITONSERVER_Error* err__ = (X);                                 \
      if (err__ != nullptr) {                                          \
        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(                       \
                         (RESPONSES)[IDX],                             \
                         TRITONSERVER_RESPONSE_COMPLETE_FINAL, err__), \
                     "failed to send error response");                 \
        (RESPONSES)[IDX] = nullptr;                                    \
        TRITONSERVER_ErrorDelete(err__);                               \
      }                                                                \
    }                                                                  \
  } while (false)

#define CK_CUDA_THROW_(x)                                                      \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + " \n");        \
    }                                                                          \
  } while (0)

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Block the thread for seconds specified in 'creation_delay_sec' parameter.
  // This function is used for testing.
  TRITONSERVER_Error* CreationDelay();

  // Get input data entry map
  std::map<std::string, size_t> GetInputmap() { return input_map_; }

 private:
  ModelState(TRITONSERVER_Server* triton_server,
             TRITONBACKEND_Model* triton_model, const char* name,
             const uint64_t version, common::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  common::TritonJson::Value model_config_;

  bool supports_batching_initialized_;
  bool supports_batching_;

  std::map<std::string, size_t> input_map_{{"INPUT_ATRK_PAR", 0},
                                           {"INPUT_ATRK_COV", 1},
                                           {"INPUT_ATRK_Q", 2},
                                           {"INPUT_AHIT_POS", 3},
                                           {"INPUT_AHIT_COV", 4}};
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(triton_server, triton_model, model_name,
                          model_version, std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(TRITONSERVER_Server* triton_server,
                       TRITONBACKEND_Model* triton_model, const char* name,
                       const uint64_t version,
                       common::TritonJson::Value&& model_config)
    : triton_server_(triton_server),
      triton_model_(triton_model),
      name_(name),
      version_(version),
      model_config_(std::move(model_config)),
      supports_batching_initialized_(false),
      supports_batching_(false) {}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 5 input (Par, Cov, and Q for Trk, POS and COV for HIT) and 1
  // output (dummy).
  RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 5, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected 5 input, got ") +
                            std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(outputs.ArraySize() == 1,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected 1 output, got ") +
                            std::to_string(outputs.ArraySize()));

  for (int i = 0; i < 5; i++) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));
    // Checkout input data_type and dims
    std::string input_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));

    std::vector<int64_t> input_shape;
    RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));

    std::string input_name;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name));
    RETURN_ERROR_IF_FALSE(
        GetInputmap().count(input_name) > 0, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("input name is not among the expected: ") + input_name);

    if (input_name == "INPUT_ATRK_PAR") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected INPUT_ATRK_PAR input datatype as TYPE_FP32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 6, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 6, got ") +
                                backend::ShapeToString(input_shape));
    }

    if (input_name == "INPUT_ATRK_COV") {
      RETURN_ERROR_IF_FALSE(
          (input_dtype == "TYPE_FP32"), TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected INPUT_ATRK_COV input datatype as TYPE_FP32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 21,
                            TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 21, got ") +
                                backend::ShapeToString(input_shape));
    }

    if (input_name == "INPUT_ATRK_Q") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string("expected ROWINDEX input datatype as TYPE_INT32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 1, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 1, got ") +
                                backend::ShapeToString(input_shape));
    }

    if (input_name == "INPUT_AHIT_POS") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string("expected ROWINDEX input datatype as TYPE_FP32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 3, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 3, got ") +
                                backend::ShapeToString(input_shape));
    }

    if (input_name == "INPUT_AHIT_COV") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string("expected ROWINDEX input datatype as TYPE_FP32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 6, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 6, got ") +
                                backend::ShapeToString(input_shape));
    }
  }

  common::TritonJson::Value output;
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));
  std::string output_dtype;
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  RETURN_ERROR_IF_FALSE(
      output_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected  output datatype as TYPE_FP32, got ") +
          output_dtype);

  //  output must have 1 shape
  std::vector<int64_t> output_shape;
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(output_shape[0] == 1, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected  output shape equal 1, got ") +
                            backend::ShapeToString(output_shape));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Passed Model Configuration Checks").c_str()));

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  ~ModelInstanceState();

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance() {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // initialize some trk and hit parameters
  TRITONSERVER_Error* Init(std::vector<float>& vtrk_par, std::vector<float>& vtrk_cov, std::vector<int32_t>& vtrk_q, std::vector<float>& vhit_pos, std::vector<float>& vhit_co);

  // execute the p2r
  float ProcessRequest(TRITONBACKEND_Request* request);

 private:
  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance,
                     const char* name,
                     const TRITONSERVER_InstanceGroupKind kind,
                     const int32_t device_id);

  MPTRK* prepareTracks(ATRK inputtrk);
  MPHIT* prepareHits(AHIT inputhit);

  int chunkSize(int s);
  void transferAsyncTrk(int s);
  void transferAsyncHit(int s);
  void transfer_backAsync(int s);
  double doWork(const char* msg, int nIters);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  int stream_chunk_;
  int stream_remainder_;
  int stream_range_;

  cudaStream_t streams_[8];

  MPTRK* trk_;
  MPHIT* hit_;
  MPTRK* outtrk_;

  // device pointers
  MPTRK* trk_dev_;
  MPHIT* hit_dev_;
  MPTRK* outtrk_dev_;

  struct timeval timecheck_;
  long setup_start_, setup_stop_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              std::string("ModelInstanceState::Create").c_str());
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(model_state, triton_model_instance,
                                  instance_name, instance_kind, instance_id);
  return nullptr;  // success
}

ModelInstanceState::~ModelInstanceState() {
  printf("calling destructor\n");
  cudaFreeHost(trk_);
  cudaFreeHost(hit_);
  cudaFreeHost(outtrk_);
  cudaFree(trk_dev_);
  cudaFree(hit_dev_);
  cudaFree(outtrk_dev_);
  printf("destructor done\n");
}

TRITONSERVER_Error* ModelInstanceState::Init(std::vector<float>& vtrk_par, std::vector<float>& vtrk_cov, std::vector<int32_t>& vtrk_q, std::vector<float>& vhit_pos, std::vector<float>& vhit_cov) {
  ATRK inputtrk {
    {vtrk_par[0], vtrk_par[1], vtrk_par[2], vtrk_par[3], vtrk_par[4], vtrk_par[5]},
    {vtrk_cov[0], vtrk_cov[1], vtrk_cov[2], vtrk_cov[3], vtrk_cov[4], vtrk_cov[5], vtrk_cov[6], 
     vtrk_cov[7], vtrk_cov[8], vtrk_cov[9], vtrk_cov[10], vtrk_cov[11], vtrk_cov[12], vtrk_cov[13], 
     vtrk_cov[14], vtrk_cov[15], vtrk_cov[16], vtrk_cov[17], vtrk_cov[18], vtrk_cov[19], vtrk_cov[20]},
     1
  };
  AHIT inputhit = {
    {vhit_pos[0], vhit_pos[1], vhit_pos[2]},
    {vhit_cov[0], vhit_cov[1], vhit_cov[2], vhit_cov[3], vhit_cov[4], vhit_cov[5]}
  };
  /*
  ATRK inputtrk = {
      {-12.806846618652344, -7.723824977874756, 38.13014221191406,
       0.23732035065189902, -2.613372802734375, 0.35594117641448975},
      {6.290299552347278e-07,  4.1375109560704004e-08, 7.526661534029699e-07,
       2.0973730840978533e-07, 1.5431574240665213e-07, 9.626245400795597e-08,
       -2.804026640189443e-06, 6.219111130687595e-06,  2.649119409845118e-07,
       0.00253512163402557,    -2.419662877381737e-07, 4.3124190760040646e-07,
       3.1068903991780678e-09, 0.000923913115050627,   0.00040678296006807003,
       -7.755406890332818e-07, 1.68539375883925e-06,   6.676875566525437e-08,
       0.0008420574605423793,  7.356584799406111e-05,  0.0002306247719158348},
      1};

  AHIT inputhit = {
      {-20.7824649810791, -12.24150276184082, 57.8067626953125},
      {2.545517190810642e-06, -2.6680759219743777e-06, 2.8030024168401724e-06,
       0.00014160551654640585, 0.00012282167153898627, 11.385087966918945}};
  */

  {
    std::stringstream ss;
    ss << "track in pos: x=" << inputtrk.par[0] << ", y=" << inputtrk.par[1]
       << ", z=" << inputtrk.par[2] << ", r="
       << sqrtf(inputtrk.par[0] * inputtrk.par[0] +
                inputtrk.par[1] * inputtrk.par[1]);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "track in cov: xx=" << inputtrk.cov[SymOffsets66(PosInMtrx(0, 0, 6))]
       << ", yy=" << inputtrk.cov[SymOffsets66(PosInMtrx(1, 1, 6))]
       << ", zz=" << inputtrk.cov[SymOffsets66(PosInMtrx(2, 2, 6))];
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "hit in pos: x=" << inputhit.pos[0] << ", y=" << inputhit.pos[1]
       << ", z=" << inputhit.pos[2] << ", r="
       << sqrtf(inputhit.pos[0] * inputhit.pos[0] +
                inputhit.pos[1] * inputhit.pos[1]);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "produce nevts=" << nevts << " ntrks=" << ntrks
       << " smearing by=" << smear;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "NITER=" << NITER;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }

  gettimeofday(&timecheck_, NULL);
  setup_start_ =
      (long)timecheck_.tv_sec * 1000 + (long)timecheck_.tv_usec / 1000;

  trk_ = prepareTracks(inputtrk);
  hit_ = prepareHits(inputhit);

  cudaMallocManaged((void**)&outtrk_, nevts * nb * sizeof(MPTRK));

  cudaMalloc((MPTRK**)&trk_dev_, nevts * nb * sizeof(MPTRK));
  cudaMalloc((MPHIT**)&hit_dev_, nlayer * nevts * nb * sizeof(MPHIT));
  cudaMalloc((MPTRK**)&outtrk_dev_, nevts * nb * sizeof(MPTRK));

  stream_chunk_ = ((int)(nevts * nb / num_streams));
  stream_remainder_ = ((int)((nevts * nb) % num_streams));
  if (stream_remainder_ == 0) {
    stream_range_ = num_streams;
  } else {
    stream_range_ = num_streams + 1;
  }

  for (int s = 0; s < stream_range_; s++) {
    cudaStreamCreate(&streams_[s]);
  }

  gettimeofday(&timecheck_, NULL);
  setup_stop_ =
      (long)timecheck_.tv_sec * 1000 + (long)timecheck_.tv_usec / 1000;

  printf("done preparing!\n");

  printf("Number of struct MPTRK trk[] = %d\n", nevts * nb);
  printf("Number of struct MPTRK outtrk[] = %d\n", nevts * nb);
  printf("Number of struct struct MPHIT hit[] = %d\n", nevts * nb);

  printf("Size of struct MPTRK trk[] = %ld\n",
         nevts * nb * sizeof(struct MPTRK));
  printf("Size of struct MPTRK outtrk[] = %ld\n",
         nevts * nb * sizeof(struct MPTRK));
  printf("Size of struct struct MPHIT hit[] = %ld\n",
         nlayer * nevts * nb * sizeof(struct MPHIT));

  return nullptr;
}

float ModelInstanceState::ProcessRequest(TRITONBACKEND_Request* request) {
  doWork("Warming up", NWARMUP);
  auto wall_time = doWork("Launching", NITER);
  printf("setup time time=%f (s)\n", (setup_stop_ - setup_start_) * 0.001);
  printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n",
         nevts * ntrks * int(NITER), wall_time,
         wall_time / (nevts * ntrks * int(NITER)));
  printf("formatted %i %i %i %i %i %f 0 %f %i\n", int(NITER), nevts, ntrks,
         bsize, nb, wall_time, (setup_stop_ - setup_start_) * 0.001, nthreads);

  float avgx = 0, avgy = 0, avgz = 0, avgr = 0;
  float avgpt = 0, avgphi = 0, avgtheta = 0;
  float avgdx = 0, avgdy = 0, avgdz = 0, avgdr = 0;
  for (size_t ie = 0; ie < nevts; ++ie) {
    for (size_t it = 0; it < ntrks; ++it) {
      float x_ = x(outtrk_, ie, it);
      float y_ = y(outtrk_, ie, it);
      float z_ = z(outtrk_, ie, it);
      float r_ = sqrtf(x_ * x_ + y_ * y_);
      float pt_ = 1. / ipt(outtrk_, ie, it);
      float phi_ = phi(outtrk_, ie, it);
      float theta_ = theta(outtrk_, ie, it);
      avgpt += pt_;
      avgphi += phi_;
      avgtheta += theta_;
      avgx += x_;
      avgy += y_;
      avgz += z_;
      avgr += r_;
      float hx_ = x(hit_, ie, it);
      float hy_ = y(hit_, ie, it);
      float hz_ = z(hit_, ie, it);
      float hr_ = sqrtf(hx_ * hx_ + hy_ * hy_);
      avgdx += (x_ - hx_) / x_;
      avgdy += (y_ - hy_) / y_;
      avgdz += (z_ - hz_) / z_;
      avgdr += (r_ - hr_) / r_;
      // if((it+ie*ntrks)%10==0) printf("iTrk = %i,  track
      // (x,y,z,r)=(%.3f,%.3f,%.3f,%.3f) \n", it+ie*ntrks, x_,y_,z_,r_);
    }
  }
  avgpt = avgpt / float(nevts * ntrks);
  avgphi = avgphi / float(nevts * ntrks);
  avgtheta = avgtheta / float(nevts * ntrks);
  avgx = avgx / float(nevts * ntrks);
  avgy = avgy / float(nevts * ntrks);
  avgz = avgz / float(nevts * ntrks);
  avgr = avgr / float(nevts * ntrks);
  avgdx = avgdx / float(nevts * ntrks);
  avgdy = avgdy / float(nevts * ntrks);
  avgdz = avgdz / float(nevts * ntrks);
  avgdr = avgdr / float(nevts * ntrks);

  float stdx = 0, stdy = 0, stdz = 0, stdr = 0;
  float stddx = 0, stddy = 0, stddz = 0, stddr = 0;
  for (size_t ie = 0; ie < nevts; ++ie) {
    for (size_t it = 0; it < ntrks; ++it) {
      float x_ = x(outtrk_, ie, it);
      float y_ = y(outtrk_, ie, it);
      float z_ = z(outtrk_, ie, it);
      float r_ = sqrtf(x_ * x_ + y_ * y_);
      stdx += (x_ - avgx) * (x_ - avgx);
      stdy += (y_ - avgy) * (y_ - avgy);
      stdz += (z_ - avgz) * (z_ - avgz);
      stdr += (r_ - avgr) * (r_ - avgr);
      float hx_ = x(hit_, ie, it);
      float hy_ = y(hit_, ie, it);
      float hz_ = z(hit_, ie, it);
      float hr_ = sqrtf(hx_ * hx_ + hy_ * hy_);
      stddx += ((x_ - hx_) / x_ - avgdx) * ((x_ - hx_) / x_ - avgdx);
      stddy += ((y_ - hy_) / y_ - avgdy) * ((y_ - hy_) / y_ - avgdy);
      stddz += ((z_ - hz_) / z_ - avgdz) * ((z_ - hz_) / z_ - avgdz);
      stddr += ((r_ - hr_) / r_ - avgdr) * ((r_ - hr_) / r_ - avgdr);
    }
  }

  stdx = sqrtf(stdx / float(nevts * ntrks));
  stdy = sqrtf(stdy / float(nevts * ntrks));
  stdz = sqrtf(stdz / float(nevts * ntrks));
  stdr = sqrtf(stdr / float(nevts * ntrks));
  stddx = sqrtf(stddx / float(nevts * ntrks));
  stddy = sqrtf(stddy / float(nevts * ntrks));
  stddz = sqrtf(stddz / float(nevts * ntrks));
  stddr = sqrtf(stddr / float(nevts * ntrks));

  printf("track x avg=%f std/avg=%f\n", avgx, fabs(stdx / avgx));
  printf("track y avg=%f std/avg=%f\n", avgy, fabs(stdy / avgy));
  printf("track z avg=%f std/avg=%f\n", avgz, fabs(stdz / avgz));
  printf("track r avg=%f std/avg=%f\n", avgr, fabs(stdr / avgz));
  printf("track dx/x avg=%f std=%f\n", avgdx, stddx);
  printf("track dy/y avg=%f std=%f\n", avgdy, stddy);
  printf("track dz/z avg=%f std=%f\n", avgdz, stddz);
  printf("track dr/r avg=%f std=%f\n", avgdr, stddr);
  printf("track pt avg=%f\n", avgpt);
  printf("track phi avg=%f\n", avgphi);
  printf("track theta avg=%f\n", avgtheta);

  return avgpt;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state),
      triton_model_instance_(triton_model_instance),
      name_(name),
      kind_(kind),
      device_id_(device_id) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton Model Instance Initialization on device ") +
               std::to_string(device_id))
                  .c_str());
  cudaError_t cuerr = cudaSetDevice(device_id);
  if (cuerr != cudaSuccess) {
    std::cerr << "failed to set CUDA device to " << device_id << ": "
              << cudaGetErrorString(cuerr);
  }
}

MPTRK* ModelInstanceState::prepareTracks(ATRK inputtrk) {
  MPTRK* result;
  cudaMallocHost((void**)&result, nevts * nb * sizeof(MPTRK));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie = 0; ie < nevts; ++ie) {
    for (size_t ib = 0; ib < nb; ++ib) {
      for (size_t it = 0; it < bsize; ++it) {
        // par
        for (size_t ip = 0; ip < 6; ++ip) {
          result[ib + nb * ie].par.data[it + ip * bsize] =
              (1 + smear * randn(0, 1)) * inputtrk.par[ip];
        }
        // cov
        for (size_t ip = 0; ip < 21; ++ip) {
          result[ib + nb * ie].cov.data[it + ip * bsize] =
              (1 + smear * randn(0, 1)) * inputtrk.cov[ip];
        }
        // q
        result[ib + nb * ie].q.data[it] =
            inputtrk.q -
            2 * ceil(-0.5 + (float)rand() / RAND_MAX);  // fixme check
        // if((ib + nb*ie)%10==0 ) printf("prep trk index = %i ,track = (%.3f)\n
        // ", ib+nb*ie);
      }
    }
  }
  return result;
}

MPHIT* ModelInstanceState::prepareHits(AHIT inputhit) {
  MPHIT* result;  // fixme, align?
  cudaMallocHost((void**)&result, nlayer * nevts * nb * sizeof(MPHIT));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay = 0; lay < nlayer; ++lay) {
    for (size_t ie = 0; ie < nevts; ++ie) {
      for (size_t ib = 0; ib < nb; ++ib) {
        for (size_t it = 0; it < bsize; ++it) {
          // pos
          for (size_t ip = 0; ip < 3; ++ip) {
            result[lay + nlayer * (ib + nb * ie)].pos.data[it + ip * bsize] =
                (1 + smear * randn(0, 1)) * inputhit.pos[ip];
          }
          // cov
          for (size_t ip = 0; ip < 6; ++ip) {
            result[lay + nlayer * (ib + nb * ie)].cov.data[it + ip * bsize] =
                (1 + smear * randn(0, 1)) * inputhit.cov[ip];
          }
        }
      }
    }
  }
  return result;
}

int ModelInstanceState::chunkSize(int s) {
  return s < num_streams ? stream_chunk_ : stream_remainder_;
}

void ModelInstanceState::transferAsyncTrk(int s) {
  CK_CUDA_THROW_(cudaMemcpyAsync(
      (trk_dev_ + s * stream_chunk_), (trk_ + s * stream_chunk_),
      chunkSize(s) * sizeof(MPTRK), cudaMemcpyHostToDevice, streams_[s]));
}

void ModelInstanceState::transferAsyncHit(int s) {
  CK_CUDA_THROW_(cudaMemcpyAsync((hit_dev_ + s * stream_chunk_),
                                 (hit_ + s * stream_chunk_),
                                 chunkSize(s) * nlayer * sizeof(MPHIT),
                                 cudaMemcpyHostToDevice, streams_[s]));
}

void ModelInstanceState::transfer_backAsync(int s) {
  CK_CUDA_THROW_(cudaMemcpyAsync(
      (outtrk_ + s * stream_chunk_), (outtrk_dev_ + s * stream_chunk_),
      chunkSize(s) * sizeof(MPTRK), cudaMemcpyDeviceToHost, streams_[s]));
}

double ModelInstanceState::doWork(const char* msg, int nIters) {
  dim3 grid(blockspergrid, 1, 1);
  dim3 block(threadsperblockx, 1, 1);

  double wall_time = 0;

#ifdef MEASURE_H2D_TRANSFER
  for (int itr = 0; itr < nIters; itr++) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < num_streams; s++) {
      transferAsyncTrk(s);
      transferAsyncHit(s);
    }

    for (int s = 0; s < num_streams; s++) {
      GPUsequence<<<grid, block, 0, streams_[s]>>>(
          (trk_dev_ + s * stream_chunk_), (hit_dev_ + s * stream_chunk_),
          (outtrk_dev_ + s * stream_chunk_), s);
    }

#ifdef MEASURE_D2H_TRANSFER
    for (int s = 0; s < num_streams; s++) {
      transfer_backAsync(s);
    }
#endif  // MEASURE_D2H_TRANSFER
    cudaDeviceSynchronize();
    auto wall_stop = std::chrono::high_resolution_clock::now();
    wall_time += static_cast<double>(
                     std::chrono::duration_cast<std::chrono::microseconds>(
                         wall_stop - wall_start)
                         .count()) /
                 1e6;
  }
#else  // not MEASURE_H2D_TRANSFER
  for (int s = 0; s < num_streams; s++) {
    transferAsyncTrk(s);
    transferAsyncHit(s);
  }
  cudaDeviceSynchronize();
  for (int itr = 0; itr < nIters; itr++) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < num_streams; s++) {
      GPUsequence<<<grid, block, 0, streams_[s]>>>(
          (trk_dev_ + s * stream_chunk_), (hit_dev_ + s * stream_chunk_),
          (outtrk_dev_ + s * stream_chunk_), s);
    }

#ifdef MEASURE_D2H_TRANSFER
    for (int s = 0; s < num_streams; s++) {
      transfer_backAsync(s);
    }
#endif  // MEASURE_D2H_TRANSFER
    cudaDeviceSynchronize();
    auto wall_stop = std::chrono::high_resolution_clock::now();
    wall_time += static_cast<double>(
                     std::chrono::duration_cast<std::chrono::microseconds>(
                         wall_stop - wall_start)
                         .count()) /
                 1e6;
  }
#endif  // MEASURE_H2D_TRANSFER

#ifndef MEASURE_D2H_TRANSFER
  for (int s = 0; s < num_streams; s++) {
    transfer_backAsync(s);
  }
  cudaDeviceSynchronize();
#endif

  return wall_time;
}

extern "C" {

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  const char* cname;

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // requires the GPU instance
  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'p2r' backend only supports GPU instances"));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("model instance ") + instance_state->Name() +
               ", executing " + std::to_string(request_count) + " requests")
                  .c_str());

  // This backend does not support models that support batching, so
  // 'request_count' should always be 1.
  RETURN_ERROR_IF_FALSE(
      request_count <= 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("repeat backend does not support batched request execution"));

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(responses, r,
                             TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    // Triton ensures that there is only a single input since that is
    // what is specified in the model configuration, so normally there
    // would be no reason to check it but we do here to demonstrate the
    // API.
    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an
    // error message and move on to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());

    const char* input_name;
    for (uint32_t i = 0; i < 5; ++i) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInputName(request, i /* index */, &input_name));
      RETURN_ERROR_IF_FALSE(
          instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("expected input name as INPUT_ATRK_PAR, INPUT_ATRK_COV, "
                      "INPUT_ATRK_Q, INPUT_AHIT_POS, and INPUT_AHIT_COV in "
                      "request, but got ") +
              input_name);
    }

    std::vector<float> v_trk_par(6);
    size_t trk_par_byte_size = sizeof(float) * 6;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "INPUT_ATRK_PAR", reinterpret_cast<char*>(&(v_trk_par[0])), &trk_par_byte_size));

    std::vector<float> v_trk_cov(21);
    size_t trk_cov_byte_size = sizeof(float) * 21;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "INPUT_ATRK_COV", reinterpret_cast<char*>(&(v_trk_cov[0])), &trk_cov_byte_size));

    std::vector<int32_t> v_trk_q(1);
    size_t trk_q_byte_size = sizeof(int32_t);
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "INPUT_ATRK_Q", reinterpret_cast<char*>(&(v_trk_q[0])), &trk_q_byte_size));

    std::vector<float> v_hit_pos(3);
    size_t hit_pos_byte_size = sizeof(float) * 3;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "INPUT_AHIT_POS", reinterpret_cast<char*>(&(v_hit_pos[0])), &hit_pos_byte_size));

    std::vector<float> v_hit_cov(6);
    size_t hit_cov_byte_size = sizeof(float) * 6;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "INPUT_AHIT_COV", reinterpret_cast<char*>(&(v_hit_cov[0])), &hit_cov_byte_size));

    std::cout << "v_trk_par " ;
    for (auto& val: v_trk_par)
        std::cout << val << " ";
    std::cout << std::endl <<"v_trk_cov " ;
    for (auto& val: v_trk_cov)
        std::cout << val << " ";
    std::cout << std::endl << "v_trk_q";
    for (auto& val: v_trk_q)
        std::cout << val << " ";
    std::cout << std::endl << "v hit pos ";
    for (auto& val: v_hit_pos)
        std::cout << val << " ";
    std::cout << std::endl << "v hit cov ";
    for (auto& val: v_hit_cov)
        std::cout << val << " ";

    // We also validated that the model configuration specifies only a
    // single output, but the request is not required to request any
    // output at all so we only produce an output if requested.
    const char* requested_output_name = nullptr;
    if (requested_output_count > 0) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, 0 /* index */,
                                          &requested_output_name));

      // prepare output
     TRITONBACKEND_Response* response = responses[r];
      TRITONBACKEND_Output* output;
      const int64_t out_putshape=1;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, requested_output_name, TRITONSERVER_TYPE_FP32,
              &out_putshape, 1));

      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, sizeof(float), &output_memory_type,
              &output_memory_type_id));

      instance_state->Init(v_trk_par, v_trk_cov, v_trk_q, v_hit_pos, v_hit_cov);
      float avgpt = instance_state->ProcessRequest(request);

      // dummy output: avgpt
      memcpy(output_buffer, &avgpt, sizeof(float));
    }

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request, true /* success */,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  std::cout << "finished " << std::endl;

  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}
}
}  // namespace p2r
}  // namespace backend
}  // namespace triton
