//
// Created by Andrii Cherniak on 4/22/17.
//

#ifndef TOOLS_H_
#define TOOLS_H_

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

void save_results(std::string path, std::vector<double> NIS);

Eigen::VectorXd
CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

#endif