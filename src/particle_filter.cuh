/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"


#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_THRDS_IN_BLCK 1024
#define NUM_PARTICLES 100

struct Particle {
	int id;
	double x;
	double y;
	double theta;
	double weight;
};



class ParticleFilter {
	

	// Flag, if filter is initialized
	bool is_initialized;
	
	// Vector of weights of all particles
	double *weights;
	
public:
	Particle *h_d_particles;
	// Number of particles to draw
	int num_particles;

	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param M Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter();

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */
	void dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations);
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations,
			Map map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	bool initialized() const {
		return is_initialized;
	}

private:

	curandState_t* randStates;
	double* gps_std;
	double* lmk_std;
	dim3 dimBlock;
	dim3 dimGrid;
	cudaError_t ierrAsync;
	cudaError_t ierrSync;
	Map::single_landmark_s *lnd_mrks;
};

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
											double std[],
											curandState_t* rState); // random generator's state

__global__
void updateParticlesGPU(struct Particle* prtcl,
												double delta_t,
												double *std_pos,
												double velocity,
												double yaw_rate,
												curandState_t* rState); // random generator's state

__global__
void updateWeightsGPU(struct Particle* prtcl,
											double sensor_range,
											double std_landmark[],
											struct LandmarkObs *observations,
											const int observations_size,
											struct Map::single_landmark_s *lnd_mrks,
											const int lnd_mrks_size,
											curandState_t* rState);

//#ifdef __cplusplus
//} // extern "C"
//#endif



#endif /* PARTICLE_FILTER_H_ */
