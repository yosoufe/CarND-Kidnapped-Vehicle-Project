#ifndef PARTICLE_FILTER_CUH_
#define PARTICLE_FILTER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

//#ifdef __cplusplus
//extern "C" {
//#endif


#include "particle_filter.h"

__global__
void updateParticlesGPU(Particle* particle,
			double delta_t,
			double std_pos[],
			double velocity,
			double yaw_rate);

//#ifdef __cplusplus
//}; // extern "C"
//#endif

#endif /* PARTICLE_FILTER_CUH_ */
