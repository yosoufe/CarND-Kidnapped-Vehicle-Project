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
	num_particles = 1000;
	double std_x = std[0];
	double std_y = std[1];
	double std_psi = std[2];

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
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// This line creates a normal (Gaussian) distribution.

	default_random_engine gen;
	double x,y,theta;

	for (auto it = particles.begin(); it < particles.end(); it++){
		Particle prtcl = *it;
		x = prtcl.x + velocity * cos(prtcl.theta);
		y = prtcl.y + velocity * sin(prtcl.theta);
		theta = prtcl.theta + yaw_rate*delta_t;

		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_psi(theta, std_pos[2]);

		prtcl.x = dist_x(gen);
		prtcl.y = dist_y(gen);
		prtcl.theta = dist_psi(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for ( auto it_obsrv = observations.begin(); it_obsrv < observations.end(); it_obsrv++ ){
		LandmarkObs obsv = *it_obsrv;
		double final_dist = INFINITY; // final distance
		for( auto it_prd = predicted.begin(); it_prd < predicted.end(); it_prd++ ){
			LandmarkObs prd = *it_prd;
			double dist = sqrt ( pow( obsv.x - prd.x ,2) + pow( obsv.y - prd.y ,2));
			if (dist < final_dist ){
				final_dist = dist;
				obsv.id = prd.id;
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
	for (auto it_prtcl = particles.begin(); it_prtcl< particles.end(); it_prtcl++){
		Particle prtcl = *it_prtcl;
		std::vector<LandmarkObs> predicted;
		for (auto it_lndMrk = map_landmarks.landmark_list.begin();
				 it_lndMrk<map_landmarks.landmark_list.end(); it_lndMrk++){
			Map::single_landmark_s lndMrk = *it_lndMrk;

			double dist = sqrt( pow( lndMrk.x_f - prtcl.x ,2) + pow( lndMrk.y_f - prtcl.y ,2));
			if (dist < sensor_range){
				prtcl.sense_x.push_back(lndMrk.x_f); // - prtcl.x
				prtcl.sense_y.push_back(lndMrk.y_f); // - prtcl.y
				prtcl.associations.push_back(lndMrk.id_i);

				// transform observation to the world coordinate given the car at each particle position
				LandmarkObs pred_lm = glob2particle(lndMrk,prtcl);
				predicted.push_back(pred_lm);
			}
		}

		// use dataAssociation to associate sensor measurements to map landmarks
		dataAssociation(predicted,observations);

		// TODO update weight of each particle and normalize them

	}
}

LandmarkObs ParticleFilter::glob2particle(Map::single_landmark_s g_lnmk, Particle particle){
	LandmarkObs lndMkObs;
	Eigen::MatrixXd R_c2g(2,2);
	Eigen::MatrixXd t_c2g(2,1);
	Eigen::MatrixXd lndMk_g(2,1); lndMk_g << g_lnmk.x_f,g_lnmk.y_f;
	R_c2g << cos(particle.theta) , -sin(particle.theta) , sin(particle.theta), cos(particle.theta);
	t_c2g << particle.x, particle.y;
	Eigen::MatrixXd obs = R_c2g.transpose() * lndMk_g - R_c2g.transpose()*t_c2g;
	lndMkObs.id=g_lnmk.id_i;
	lndMkObs.x = obs(0,0);
	lndMkObs.y = obs(1,0);
	return lndMkObs;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
