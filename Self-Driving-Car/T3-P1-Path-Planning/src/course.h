//
// Created by Andrii Cherniak on 10/2/17.
//

#ifndef PATH_PLANNING_COURSE_H
#define PATH_PLANNING_COURSE_H

#include <vector>
#include <tuple>
#include "JMT.h"
#include <iostream>
#include "waypoints.h"

#define TARGET_SPEED_MPH     48

#define TARGET_SPEED_MPS     0.44704 * TARGET_SPEED_MPH

#define PATH_PLAN_INCREMENT  0.02
#define KEEP_LANE_HORIZON    50
#define CHANGE_LANE_HORIZON  170

#define LANE_WIDTH           4.0
#define MAX_ACCELERATION     16.0
#define MAX_DECELERATION     -10.0
#define CAR_DISTANCE         5
#define CHANGE_LANE_TIME     3

using namespace std;

tuple<double, double> sense_car_in_front(vector<vector<double>> sensor_fusion, int lane, double car_s);

tuple<double, double> sense_car_behind(vector<vector<double>> sensor_fusion, int lane, double car_s);


int decision(jmt_state car_state_s, jmt_state snapshot_state, int lane, vector<vector<double>> sensor_fusion);

double to_change_lane(jmt_state car_state_s, jmt_state snapshot_state, int from_lane, int to_lane,
                      vector<vector<double>> sensor_fusion);

tuple<vector<jmt_state>, vector<jmt_state>>
generate_path(jmt_state car_state_s, jmt_state car_state_d, jmt_state snapshot_s, int lane,
              vector<vector<double>> sensor_fusion, double max_future_speed);

double get_lane_d(int lane);


#endif //PATH_PLANNING_COURSE_H
