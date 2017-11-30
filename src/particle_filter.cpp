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

  for (int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;

  normal_distribution<double> x_noise(0, std_pos[0]);
  normal_distribution<double> y_noise(0, std_pos[1]);
  normal_distribution<double> theta_noise(0, std_pos[2]);

  for(auto &particle: particles) {
    double theta = particle.theta;
    if(fabs(yaw_rate) < 0.00001){
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

vector<pair<LandmarkObs, LandmarkObs>> ParticleFilter::dataAssociation(
    const std::vector<LandmarkObs> &predicted, const std::vector<LandmarkObs> &observations) {
  vector<pair<LandmarkObs, LandmarkObs>> data;
  for (auto &observation: observations) {
    double min_dist = numeric_limits<double>::max();
    LandmarkObs p;
    for (auto const &landmark: predicted){
      double m_dist = dist(observation.x, observation.y, landmark.x, landmark.y);
      if(m_dist < min_dist){
        p = landmark;
        min_dist = m_dist;
      }
    }
    data.push_back(make_pair(observation, p));
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const vector<LandmarkObs> &observations, const Map &map_landmarks) {

  double s_x = std_landmark[0], s_y = std_landmark[1];
  for(auto &particle: particles) {

    vector<int> associations_ids;
    vector<double> sense_x;
    vector<double> sense_y;

    // Filter landmarks out of range
    vector<LandmarkObs> landmarks;
    for (auto const &landmark: map_landmarks.landmark_list) {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        landmarks.push_back(LandmarkObs{ landmark.id_i, landmark.x_f, landmark.y_f });
      }
    }

    // Transform observations to map coordiantes
    vector<LandmarkObs> measures;
    for (auto const &observation: observations) {
      double x = cos(particle.theta)*observation.x - sin(particle.theta)*observation.y + particle.x;
      double y = sin(particle.theta)*observation.x + cos(particle.theta)*observation.y + particle.y;
      measures.push_back(LandmarkObs{ observation.id, x, y });
    }

    vector<pair<LandmarkObs, LandmarkObs>> associations = dataAssociation(landmarks, measures);

    particle.weight = 1.0;
    for (auto const &association: associations) {
      LandmarkObs measure = get<0>(association);
      LandmarkObs landmark = get<1>(association);
      double obs_w = 1.0/(2.0*M_PI*s_x*s_y) * exp(-0.5*(pow(measure.x - landmark.x, 2)/pow(s_x, 2) + pow(measure.y - landmark.y, 2)/pow(s_y, 2)));
      particle.weight *= obs_w;
      associations_ids.push_back(landmark.id);
      sense_x.push_back(measure.x);
      sense_y.push_back(measure.y);
    }

    SetAssociations(particle, associations_ids, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;

  vector<double> weights(num_particles);
  for(int i = 0; i < num_particles; ++i){
    weights[i] = particles[i].weight;
  }

  discrete_distribution<> d(weights.begin(), weights.end());

  vector<Particle> new_particles = vector<Particle>(num_particles);

  for (int i = 0; i < num_particles; ++i) {
    new_particles[i] = particles[d(gen)];
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
