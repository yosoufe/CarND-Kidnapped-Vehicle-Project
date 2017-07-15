/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = NUM_PARTICLES;
	weights = new double [num_particles];
	double std_x = std[0];
	double std_y = std[1];
	double std_psi = std[2];

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
#ifdef WITH_GPU
	// allocate memory in GPU for particles
	cudaMallocManaged(&h_d_particles, num_particles * sizeof(Particle));
	// allocate memory in GPU for random number creation state objects.
	cudaMallocManaged(&randStates, num_particles * sizeof(curandState_t));

	// initialise the random generation states:
	dim3 dimBlock(NUM_THRDS_IN_BLCK , 1 ,1);
	dim3 dimGrid (NUM_PARTICLES/NUM_THRDS_IN_BLCK+1 , 1 ,1);
	setup_random_generation<<<dimGrid,dimBlock>>>(randStates);


	// each thread should initilise each particle in random distribution. (cuRAND library)
	// http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
	// CURAND_ORDERING_QUASI_DEFAULT


#endif // WITH_GPU
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// This line creates a normal (Gaussian) distribution.

	default_random_engine gen;
	double x,y,theta;

#ifndef WITH_GPU
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
#ifdef WITH_GPU


	// normal random number generation in cuda cuRAND
	//

#endif // WITH_GPU

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
#ifdef WITH_GPU
	cudaFree(h_d_particles);
	cudaFree(randStates);
#endif // WITH_GPU
}
