/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.cuh"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = NUM_PARTICLES;
	weights = new double [num_particles];
//	double std_x = std[0];
//	double std_y = std[1];
//	double std_psi = std[2];

#ifndef WITH_GPU
	// This line creates a normal (Gaussian) distribution.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_psi(theta, std_psi);

	default_random_engine gen;

	for (int i = 0; i<num_particles ; i++){
		Particle prtcl={i,
										dist_x(gen),
										dist_y(gen),
										dist_psi(gen)}; // TODO: Shall I define the rest using other functions??
		particles.push_back(prtcl);
		weights[i]=1;
	}

	is_initialized = true;
#endif //WITH_GPU

	// initilise number of threads and blocks
	dimBlock = dim3(NUM_THRDS_IN_BLCK , 1 ,1);
	dimGrid = dim3(NUM_PARTICLES/NUM_THRDS_IN_BLCK+1 , 1 ,1);

	// allocate memory in GPU for particles
	cudaMallocManaged(&h_d_particles, num_particles * sizeof(Particle));
	// allocate memory in GPU for random number creation state objects.
	cudaMallocManaged(&randStates, num_particles * sizeof(curandState_t));
	// allocate memory in GPU for standard deviation of gps noise.
	cudaMallocManaged(&gps_std, 3 * sizeof(double));
	// allocate memory in GPU for standard deviation of landmark measurement noise.
	cudaMallocManaged(&lmk_std, 2 * sizeof(double));
//	// allocate memory in GPU for time difference.
//	cudaMallocManaged(&delta_t, 1 * sizeof(double));
//	// allocate memory in GPU for time velocity.
//	cudaMallocManaged(&velocity, 1 * sizeof(double));
//	// allocate memory in GPU for time yaw rete.
//	cudaMallocManaged(&yaw_rate, 1 * sizeof(double));

	// initialise the random generation states:
	setup_random_generation<<<dimGrid,dimBlock>>>(randStates);
	cudaDeviceSynchronize();

	// each thread should initilise each particle in random distribution. (cuRAND library)
	// http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html


	gps_std[0] = std[0];
	gps_std[1] = std[1];
	gps_std[2] = std[2];

	for (int i = 0; i< NUM_PARTICLES; i+=20){
		cout<<"Particle: " <<h_d_particles[i].id << ", x: " << h_d_particles[i].x <<
					", y: " << h_d_particles[i].y << ", theta: "<< h_d_particles[i].theta << endl;
	}

	// initilise the particles on gpu
	initParticlesGPU<<<dimGrid,dimBlock>>>(h_d_particles,
																				 x,
																				 y,
																				 theta,
																				 gps_std,
																				 randStates);
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize();

	if (ierrSync != cudaSuccess) { printf("Sync error: %s in init\n", cudaGetErrorString(ierrSync)); }
	if (ierrAsync != cudaSuccess) { printf("Async error: %s  in init\n", cudaGetErrorString(ierrAsync)); }

	for (int i = 0; i< 3; i++){
		cout<<"====================="<< endl;
	}

	for (int i = 0; i< NUM_PARTICLES; i+=20){
		cout<<"Particle: " <<h_d_particles[i].id << ", x: " << h_d_particles[i].x <<
					", y: " << h_d_particles[i].y << ", theta: "<< h_d_particles[i].theta << endl;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// This line creates a normal (Gaussian) distribution.

#ifndef WITH_GPU
	default_random_engine gen;
	double x,y,theta;
	for (auto it = particles.begin(); it < particles.end(); it++){
		Particle& prtcl = *it;
		x = prtcl.x + velocity * cos(prtcl.theta) * delta_t;
		y = prtcl.y + velocity * sin(prtcl.theta) * delta_t;
		theta = prtcl.theta + yaw_rate*delta_t;

		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_psi(theta, std_pos[2]);

		prtcl.x = dist_x(gen);
		prtcl.y = dist_y(gen);
		prtcl.theta = dist_psi(gen);
	}
#endif //WITH_GPU
	gps_std[0] = std_pos[0];
	gps_std[1] = std_pos[1];
	gps_std[2] = std_pos[2];

	updateParticlesGPU<<<dimGrid,dimBlock>>>(h_d_particles,
																					 delta_t,
																					 gps_std,
																					 velocity,
																					 yaw_rate,
																					 randStates);
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize();

	if (ierrSync != cudaSuccess) { printf("Sync error: %s in predict\n", cudaGetErrorString(ierrSync)); }
	if (ierrAsync != cudaSuccess) { printf("Async error: %s in predict\n", cudaGetErrorString(ierrAsync)); }

	for (int i = 0; i< 3; i++){
		cout<<"====================="<< endl;
	}

	for (int i = 0; i< NUM_PARTICLES; i+=20){
		cout<<"Particle: " <<h_d_particles[i].id << ", x: " << h_d_particles[i].x <<
					", y: " << h_d_particles[i].y << ", theta: "<< h_d_particles[i].theta << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for ( auto it_obsrv = observations.begin(); it_obsrv < observations.end(); it_obsrv++ ){
		LandmarkObs& obsv = *it_obsrv;
		double final_dist = INFINITY; // final distance
		for( auto it_prd = predicted.begin(); it_prd < predicted.end(); it_prd++ ){
			LandmarkObs& prd = *it_prd;
			double dist = sqrt ( pow( obsv.x - prd.x ,2) + pow( obsv.y - prd.y ,2));
			if (dist < final_dist ){
				final_dist = dist;
				obsv.id = it_prd - predicted.begin();// index of the associated predicted point in its vector; prd.id
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
																	 std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	LandmarkObs *observations_arr;
	int observations_size = observations.size();
	observations_arr = new LandmarkObs[observations_size];
	// allocate memory
	cudaMallocManaged(&observations_arr, observations_size * sizeof(LandmarkObs));
	std::copy(observations_arr.begin(),observations_arr.end(),observations_arr);
	//observations_arr = observations.data();
	//memcpy(observations_arr,observations.data(),observations_size * sizeof(LandmarkObs));
	//for (int i=0;i<observations_size;i++)
	//	observations_arr[i] = observations[i];

	bool static land_marks_are_copied = false;
	static int lnd_mrks_size;
	if (!land_marks_are_copied){
		land_marks_are_copied = true;
		lnd_mrks_size = map_landmarks.landmark_list.size();
		lnd_mrks = new Map::single_landmark_s[lnd_mrks_size];
		cudaMallocManaged(&lnd_mrks, lnd_mrks_size * sizeof(Map::single_landmark_s));
		//std::copy(map_landmarks.landmark_list.begin(),map_landmarks.landmark_list.end(),lnd_mrks);
		//lnd_mrks = map_landmarks.landmark_list.data();
		//memcpy(lnd_mrks,map_landmarks.landmark_list.data(),lnd_mrks_size * sizeof(Map::single_landmark_s));
		for (int i=0;i<lnd_mrks_size;i++)
				lnd_mrks[i] = map_landmarks.landmark_list[i];
	}


	lmk_std[0] = std_landmark[0];
	lmk_std[1] = std_landmark[1];

	// allocate memory for observations_arr, lnd_mrks (no need to copy, const, but now copying)

	updateWeightsGPU<<<dimGrid,dimBlock>>>(h_d_particles,
																				 sensor_range,
																				 lmk_std,
																				 observations_arr,
																				 observations_size,
																				 lnd_mrks,
																				 lnd_mrks_size,
																				 randStates);
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize();

	if (ierrSync != cudaSuccess) { printf("Sync error: %s in update\n", cudaGetErrorString(ierrSync)); }
	if (ierrAsync != cudaSuccess) { printf("Async error: %s  in update\n", cudaGetErrorString(ierrAsync)); }
	cudaFree(observations_arr);

	//normalise the weights??

	//what about re sampling

#ifndef WITH_GPU
	// predict measurements to all landmarks within sensor range for each particle (get predicted landmark measurements)
	double sum_weigths = 0;
	for (auto it_prtcl = particles.begin(); it_prtcl< particles.end(); it_prtcl++){
		std::vector<int> associations_id; std::vector<double> sense_x; std::vector<double> sense_y;

		Particle& prtcl = *it_prtcl;
		std::vector<LandmarkObs> predicted;
		for (auto it_lndMrk = map_landmarks.landmark_list.begin();
				 it_lndMrk<map_landmarks.landmark_list.end(); it_lndMrk++){
			Map::single_landmark_s& lndMrk = *it_lndMrk;

			double dist = sqrt( pow( lndMrk.x_f - prtcl.x ,2) + pow( lndMrk.y_f - prtcl.y ,2));
			if (dist < sensor_range){

				// transform landmark to particle coordinate
				LandmarkObs pred_lm; // = glob2particle(lndMrk,prtcl)
				pred_lm.x = cos(prtcl.theta)*((double)lndMrk.x_f - prtcl.x) + sin(prtcl.theta)*((double)lndMrk.y_f - prtcl.y);
				pred_lm.y = cos(prtcl.theta)*((double)lndMrk.y_f - prtcl.y) + sin(prtcl.theta)*(prtcl.x - (double)lndMrk.x_f);
				pred_lm.id = lndMrk.id_i;

				predicted.push_back(pred_lm);
			}
		}

		// use dataAssociation to associate sensor measurements to map landmarks
		dataAssociation(predicted,observations);

		// update weight of each particle and normalize them
		prtcl.weight=1;
		for (auto obsvt_it = observations.begin(); obsvt_it < observations.end(); obsvt_it++){
			LandmarkObs& obsv = *obsvt_it;
			double delta_x = predicted[obsv.id].x - obsv.x;
			double delta_y = predicted[obsv.id].y - obsv.y;
			double argu = pow(delta_x*std_landmark[0],2) + pow(delta_y*std_landmark[1],2);
			prtcl.weight *= exp(-0.5 * argu)/2/M_PI/std_landmark[0]/std_landmark[1];
		}
		weights[it_prtcl - particles.begin()] = prtcl.weight;
		sum_weigths += prtcl.weight;
	}

	// normalize the weights
	for (auto it_prtcl = particles.begin(); it_prtcl< particles.end(); it_prtcl++){
		Particle& prtcl = *it_prtcl;
		prtcl.weight /= sum_weigths;
		weights[it_prtcl - particles.begin()] /= sum_weigths;
	}
#endif
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<double> weights_vect(weights, weights + num_particles);

	discrete_distribution<int> resampler(weights_vect.begin(),weights_vect.end());
	vector<Particle> new_particles;
	default_random_engine generator;
	for (int i=0; i< num_particles ; i++) {
		int number = resampler(generator);
		new_particles.push_back(particles[number]);
	}
	particles = new_particles;
}

ParticleFilter::~ParticleFilter(){
	cudaFree(h_d_particles);
	cudaFree(randStates);
	cudaFree(gps_std);
	cudaFree(lmk_std);
	cudaFree(lnd_mrks);
}

__global__
void setup_random_generation(curandState_t *state){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < NUM_PARTICLES){
		curand_init(0,idx,0,&state[idx]);
	}
}

__global__
void initParticlesGPU(struct Particle* prtcl,
											double x,
											double y,
											double theta,
											double std[],
											curandState_t* rState){ // random generator's state
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < NUM_PARTICLES){

		double noisy_x = x + curand_normal_double(&rState[idx]) * std[0];
		double noisy_y = y + curand_normal_double(&rState[idx]) * std[1];
		double noisy_theta = theta + curand_normal_double(&rState[idx]) * std[2];

		prtcl[idx].id = idx;
		prtcl[idx].x = noisy_x;
		prtcl[idx].y = noisy_y;
		prtcl[idx].theta = noisy_theta;
	}
}


__global__
void updateParticlesGPU(struct Particle* prtcl,
												double delta_t,
												double std_pos[],
												double velocity,
												double yaw_rate,
												curandState_t* rState){ // random generator's state
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < NUM_PARTICLES){
		double x = prtcl[idx].x + velocity * cos(prtcl[idx].theta) * delta_t;
		double y = prtcl[idx].y + velocity * sin(prtcl[idx].theta) * delta_t;
		double theta = prtcl[idx].theta + yaw_rate * delta_t;

		prtcl[idx].x = x + curand_normal_double(&rState[idx]) * std_pos[0];
		prtcl[idx].y = y + curand_normal_double(&rState[idx]) * std_pos[1];
		prtcl[idx].theta = theta + curand_normal_double(&rState[idx]) * std_pos[2];
	}
}


__global__
void updateWeightsGPU(struct Particle* prtcl,
											double sensor_range,
											double *std_landmark,
											struct LandmarkObs *observations,
											const int observations_size,
											struct Map::single_landmark_s *lnd_mrks,
											const int lnd_mrks_size,
											curandState_t* rState){ // random generator's state
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < NUM_PARTICLES){
		// copy to local memory for faster operation.
		double p_x = prtcl[idx].x;
		double p_y = prtcl[idx].y;
		double p_theta = prtcl[idx].theta;
		//double p_id = prtcl[idx].id;
		double p_weight = 1;
		double *dist = new double[lnd_mrks_size];

		// count how many of the landmarks are in the particle sensor range
		int n_obs_in_range = 0;
		for (int i=0; i<lnd_mrks_size;i++){
			dist[i] = sqrt( pow( lnd_mrks[i].x_f - p_x,2) + pow( lnd_mrks[i].y_f - p_y ,2));
			if (dist[i] < sensor_range)
				n_obs_in_range++;
		}

		// initialise the predicted measurements for land marks
		// and transfer the measurements into particle coordinate
		struct LandmarkObs *predicted = new struct LandmarkObs[n_obs_in_range];
		int j=0;
		for (int i=0; i<lnd_mrks_size;i++){
			if (dist[i] < sensor_range){
				predicted[j].id = lnd_mrks[i].id_i;

				// transfer landmark position into particle coordinate
				predicted[j].x = cos(p_theta)*((double)lnd_mrks[i].x_f - p_x) + sin(p_theta)*((double)lnd_mrks[i].y_f - p_y);
				predicted[j].y = cos(p_theta)*((double)lnd_mrks[i].y_f - p_y) + sin(p_theta)*(p_x - (double)lnd_mrks[i].x_f);
				j++;
			}
		}

		// assign for each observation, the closest prediction
		// for each observation finr the closest predicted landmark
		for (int i_obs = 0; i_obs<observations_size; i_obs++){ // loop over observations
			double final_dist = INFINITY; // final distance
			for(int i_prd = 0; i_prd<n_obs_in_range; i_prd++){// loop over predictions
				double dist_p_o = sqrt ( pow( observations[i_obs].x - predicted[i_prd].x ,2) +
																 pow( observations[i_obs].y - predicted[i_prd].y ,2));
				if (dist_p_o < final_dist){
					final_dist = dist_p_o;
					observations[i_obs].id = i_prd; //index of associated prediction in `predicted` array
				}
			}
		}


		// loop over observations and
		for (int i_obs=0; i_obs<observations_size; i_obs++){
			double delta_x = predicted[observations[i_obs].id].x - observations[i_obs].x;
			double delta_y = predicted[observations[i_obs].id].y - observations[i_obs].y;
			double argu = pow(delta_x*std_landmark[0],2) + pow(delta_y*std_landmark[1],2);
			p_weight *= exp(-0.5 * argu)/2/M_PI/std_landmark[0]/std_landmark[1];
		}
		prtcl[idx].weight = p_weight;
	}
}
