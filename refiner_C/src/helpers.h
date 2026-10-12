#ifndef HELPERS_H
#define HELPERS_H

inline __device__ double f_double(float x, float bin, float sigmam1){
    return exp(-(x - bin) * (x - bin) * 0.5 * sigmam1 * sigmam1);
}

inline __device__ float f_float(float x, float bin, float sigmam1){
    return expf(-(x - bin) * (x - bin) * 0.5 * sigmam1 * sigmam1);
}

inline __device__ float fp(float x, float bin, float sigmam1){
    return - (x - bin) * sigmam1 * sigmam1 * expf(-(x - bin) * (x - bin) * 0.5 * sigmam1 * sigmam1);
}

#endif