//
// Created by Andrii Cherniak on 9/28/17.
//

#ifndef PATH_PLANNING_JMT_H
#define PATH_PLANNING_JMT_H

#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include <vector>
#include <cmath>
#include <iostream>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct jmt_state {
    double s;
    double v;
    double a;
};

vector<double> JMT(vector<double> start, vector<double> end, double T);

vector<jmt_state> jmt_path(vector<double> cc, double dT, double T);

#endif //PATH_PLANNING_JMT_H
