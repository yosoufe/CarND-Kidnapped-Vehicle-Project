
#include"particle_filter.cuh"


__global__
void setup_random_generation(curandState_t *state){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < NUM_PARTICLES){
		//blockIdx
	}
}

__global__
void initParticlesGPU(struct Particle* prtcl,
											double x,
											double y,
											double theta,
											double std[]){

}


__global__
void updateParticlesGPU(struct Particle* prtcl,
												double delta_t,
												double std_pos[],
												double velocity,
												double yaw_rate){

}
