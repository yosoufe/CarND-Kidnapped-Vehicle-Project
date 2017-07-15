#ifndef PARTICLE_FILTER_CUH_
#define PARTICLE_FILTER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "particle_filter.h"

//#ifdef __cplusplus
//extern "C" {
//#endif

__global__
void setup_random_generation(curandState_t *state);

__global__
void initParticlesGPU(struct Particle* prtcl,
											double x,
											double y,
											double theta,
											double std[]);

__global__
void updateParticlesGPU(struct Particle* prtcl,
												double delta_t,
												double std_pos[],
												double velocity,
												double yaw_rate);

//#ifdef __cplusplus
//}; // extern "C"
//#endif

#endif /* PARTICLE_FILTER_CUH_ */
