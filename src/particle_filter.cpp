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
  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  particles = vector<Particle>(num_particles);
  weights = vector<double>(num_particles, 1.0);

  for(int i = 0; i < num_particles; i++) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;

  normal_distribution<double> x_noise(0, std_pos[0]);
  normal_distribution<double> y_noise(0, std_pos[1]);
  normal_distribution<double> theta_noise(0, std_pos[2]);

  for(auto &particle: particles) {
    double theta = particle.theta;
    if(fabs(yaw_rate) < 0.0001){
      particle.x += velocity*delta_t*cos(theta);
      particle.y += velocity*delta_t*sin(theta);
    } else {
      particle.x += velocity*(sin(theta + yaw_rate*delta_t) - sin(theta))/yaw_rate;
      particle.y += velocity*(cos(theta) - cos(theta + yaw_rate*delta_t))/yaw_rate;
      particle.theta += yaw_rate*delta_t;
    }
    particle.x += x_noise(gen);
    particle.y += y_noise(gen);
    particle.theta += theta_noise(gen);
  }
}
  
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations){

  const double BIG_NUMBER =numeric_limits<double>::max();

  for (int i = 0; i < observations.size(); i++) {

    int current_j;
    double current_smallest_error = BIG_NUMBER;

    for (int j = 0; j < predicted.size(); j++) {
      const double error = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (error < current_smallest_error) {
        current_j = j;
        current_smallest_error = error;
      }
    }
    observations[i].id = current_j;
  }
}
  
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs> &observations, const Map &map_landmarks) {

  // constants used later for calculating the new weights
  const double stdx = std_landmark[0];
  const double stdy = std_landmark[1];
  const double na = 0.5 / (stdx * stdx);
  const double nb = 0.5 / (stdy * stdy);
  const double d = sqrt( 2.0 * M_PI * stdx * stdy);

  for (int  i = 0; i < num_particles; i++) {

    const double px = particles[i].x;
    const double py = particles[i].y;
    const double ptheta = particles[i].theta;

    vector<LandmarkObs> map_observations;
    for (auto const &observation: observations){
      const int oid = observation.id;
      const double ox = observation.x;
      const double oy = observation.y;
      LandmarkObs map_observation = {
        oid,
        px + ox * cos(ptheta) - oy * sin(ptheta),
        py + oy * cos(ptheta) + ox * sin(ptheta)
      };
      map_observations.push_back(map_observation);
    }

    vector<LandmarkObs> landmarks_in_range;
    for (auto const &landmark: map_landmarks.landmark_list) {
      const int mid = landmark.id_i;
      const double mx = landmark.x_f;
      const double my = landmark.y_f;

      if (dist(px, py, mx, my) < sensor_range) {
        LandmarkObs landmark_in_range = {
          mid, mx, my
        };
        landmarks_in_range.push_back(landmark_in_range);
      }
    }

    dataAssociation(landmarks_in_range, map_observations);

    double w = 1.0;

    for (int j = 0; j < map_observations.size(); j++){

      const int oid = map_observations[j].id;
      const double ox = map_observations[j].x;
      const double oy = map_observations[j].y;

      const double predicted_x = landmarks_in_range[oid].x;
      const double predicted_y = landmarks_in_range[oid].y;

      const double dx = ox - predicted_x;
      const double dy = oy - predicted_y;

      const double a = na * dx * dx;
      const double b = nb * dy * dy;
      const double r = exp(-(a + b)) / d;
      w *= r;
    }
    particles[i].weight = w;
    weights[i] = w;
  }
}

void ParticleFilter::resample(){
  default_random_engine gen;
  discrete_distribution<int> index(weights.begin(), weights.end());

  vector<Particle> resampled_particles;  

  for (int i = 0; i < num_particles; i++) {
    resampled_particles.push_back(particles[index(gen)]);
  }

  particles = resampled_particles;
}
  
Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
  const std::vector<double>& sense_x, const std::vector<double>& sense_y) {

  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best){
	vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best){
	vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
