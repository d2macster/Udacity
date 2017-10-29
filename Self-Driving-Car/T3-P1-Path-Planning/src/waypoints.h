//
// Created by Andrii Cherniak on 9/26/17.
//

#ifndef PATH_PLANNING_WAYPOINTS_H
#define PATH_PLANNING_WAYPOINTS_H

#include <fstream>
#include <math.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "json.hpp"
#include "spline.h"
#include <tuple>

#define MAX_S 6945.554
#define RESUMPLED_WAYPOINTS_SIZE 200000


using namespace std;
tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> get_waypoints(string map_file);

#endif //PATH_PLANNING_WAYPOINTS_H
