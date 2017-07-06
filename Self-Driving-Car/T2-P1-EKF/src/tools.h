//
// Created by Andrii Cherniak on 4/22/17.
//

#ifndef CARND_HW1_TOOLS_H
#define CARND_HW1_TOOLS_H

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include "measurement.h"

void load_measurements(std::string path,
                       std::vector<Measurement> &measurements,
                       std::vector<Measurement> &ground_truth);

void save_results(std::string path,
                  std::vector<Eigen::VectorXd> estimations_v,
                  std::vector<Eigen::VectorXd> ground_truth_v,
                  std::vector<Measurement> measurements);

Eigen::VectorXd
CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

#endif //CARND_HW1_TOOLS_H

