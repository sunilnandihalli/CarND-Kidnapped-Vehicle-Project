/*
 * map.h
 *
 *  Created on: Dec 12, 2016
 *      Author: mufferm
 */

#ifndef MAP_H_
#define MAP_H_
#include <unordered_map>
#include <map>
#include <list>
class Map {
 public:
  struct single_landmark_s{
    int id_i ; 
    float x_f; 
    float y_f; 
  };
  std::vector<single_landmark_s> landmark_list ;
  std::map<std::tuple<int,int>,std::list<int>> landmark_hash;
  
};



#endif /* MAP_H_ */
