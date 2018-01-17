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
#include <iomanip>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> X{x,std[0]},Y{y,std[1]},Theta{theta,std[2]};
  num_particles = 10;
  for(int i=0;i<num_particles;i++) {
    Particle p;
    p.id = i;
    p.x = X(gen);
    p.y = Y(gen);
    p.theta = Theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], double velocity, double yaw_rate) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> X{0,std_pos[0]},Y{0,std_pos[1]},Theta{0,std_pos[2]};
  if(fabs(yaw_rate)>0.0001) {
    double c = velocity/yaw_rate;
    for(Particle& p : particles) {
      double theta_new = p.theta + dt * yaw_rate;
      p.x += c*(sin(theta_new) - sin(p.theta))+X(gen);
      p.y += c*(cos(p.theta) - cos(theta_new))+Y(gen);
      p.theta = theta_new+Theta(gen);
    }
  } else {
    double c = velocity*dt;
    for(Particle& p : particles) {
      p.x += c*cos(p.theta)+X(gen);
      p.y += c*sin(p.theta)+Y(gen);
      p.theta += Theta(gen);
    }
  }
}

double sqr(double x) {
  return x*x;
}
double normal_pdf(double x,double mu,double variance) {
  double ret = exp(-sqr(x-mu)/variance);
  return ret;
}

void dumpParticles(const ParticleFilter& pf,int sid) {
  std::ostringstream stringStream;
  stringStream << "particles_"<<std::setfill('0')<<std::setw(5)<<sid;
  std::ofstream fout(stringStream.str());
  fout<<
    "particle_id x y theta weight num_landmark_obs lid0 sx0 sy0 lid1 sx1 sy1 lid2 sx2 sy2"
    " lid3 sx3 sy3 lid4 sx4 sy4 lid5 sx5 sy5 lid6 sx6 sy6 lid7 sx7 sy7 lid8 sx8 sy8 lid9 sx9 sy9 lid10 sx10 sy10 lid11 sx11 sy11 lid12 sx12 sy12"<<std::endl;
  for(const Particle& p : pf.particles) {
    fout<<p.id<<" "<<p.x<<" "<<p.y<<" "<<p.theta<<" "<<p.weight<<" "<<p.associations.size();
    for(int i=0;i<p.associations.size();i++) {
      fout<<" "<<p.associations[i]<<" "<<p.sense_x[i]<<" "<<p.sense_y[i];
    }
    for(int i=p.associations.size();i<13;i++) {
      fout<<" -1 0.0 0.0";
    }
    fout<<std::endl;
  }
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
				   const std::vector<LandmarkObs> &observations, Map &map_landmarks) {
  static int updateid = 0;
  std::cout<<" updating weights ..." <<std::endl;
  double vx = std_landmark[0]*std_landmark[0];
  double vy = std_landmark[1]*std_landmark[1];
  countsMatchLandmark.clear();
  int breakCount = 0;
  for(Particle& p: particles) {
    double ct = cos(p.theta);
    double st = sin(p.theta);
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    p.weight = 1.0;
    int landmark_id = 0;
    for(const LandmarkObs& l :observations) {
      landmark_id++;
      double x = p.x + l.x*ct - l.y*st;
      double y = p.y + l.x*st + l.y*ct;
      double distSqrd;
      int lid;
      std::tie(lid,distSqrd) = matchLandmarkNaive(x,y,map_landmarks);
      if(lid == -1) { 
	p.weight = 0.0;
	breakCount+=1;
	break;
      } else {
	Map::single_landmark_s& lm = map_landmarks.landmark_list[lid];
	std::cout<<"pid : "<<p.id<<" landmark_id : "<<landmark_id<<" lid : "<<lid<<" p.weight : "<<p.weight<<std::endl;
	std::cout<<" x : "<<x<<" lm.x_f : "<<lm.x_f<<" y : "<<y<<" lm.y_f : "<<lm.y_f<<" distSqrd : "<<distSqrd<<std::endl;
	double xprob = normal_pdf(x,lm.x_f,vx);
	double yprob = normal_pdf(y,lm.y_f,vx);
	std::cout<<" xprob : "<<xprob <<" yprob : "<<yprob <<std::endl;
	p.weight *= (xprob*yprob);	
	p.associations.push_back(lid+1);
	p.sense_x.push_back(x);
	p.sense_y.push_back(y);
      }
    }
  }
  dumpParticles(*this,updateid++);
  std::cout<<"num_particles : "<<num_particles<<" breakCount : "<<breakCount<<std::endl;
  std::cout<<" finished updating weights ..." <<std::endl;
}

void ParticleFilter::resample() {
  std::random_device rd;
  std::mt19937 gen(rd());
  static std::vector<double> weights;
  static std::vector<Particle> newParticles;

  weights.clear();
  newParticles.clear();
  for(Particle& p: particles) {
    std::cout<<"p.id : "<<p.id<<" x : "<<p.x<<" y : "<<p.y<<" theta : "<<p.theta<<" weight : "<<p.weight<<" log weight : "<<log(p.weight)<<std::endl;
    weights.push_back(p.weight);
  }
  
  std::discrete_distribution<> d(weights.begin(),weights.end());
  for(int i=0;i<num_particles;i++) {
    newParticles.push_back(particles[d(gen)]);
  }
  
  particles.swap(newParticles);
  for(Particle& p: particles) {
    std::cout<<"p.id : "<<p.id<<" x : "<<p.x<<" y : "<<p.y<<" theta : "<<p.theta<<" weight : "<<p.weight<<" log weight : "<<log(p.weight)<<std::endl;
  }
  int id = 0;
  for(Particle& p:particles) {
    p.id = id;
    id++;
  }
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
