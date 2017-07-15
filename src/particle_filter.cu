
#include"particle_filter.cuh"

__global__
void updateParticlesGPU(Particle* particle,
			double delta_t,
			double std_pos[],
			double velocity,
			double yaw_rate){

}
