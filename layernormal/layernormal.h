#ifndef LAYER_NORMAL_H_
#define LAYER_NORMAL_H_
template <const int NUM_THREADS>
void lanuch_kernel(float *x, float *y, float gamma, const int N, const int K);
#endif