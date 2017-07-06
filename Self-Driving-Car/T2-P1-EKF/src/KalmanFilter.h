//
// Created by Andrii Cherniak on 4/25/17.
//

#ifndef CARND_HW1_KALMANFILTER_H
#define CARND_HW1_KALMANFILTER_H

#include "Eigen/Dense"
#include <cmath>
#include <iostream>


class KalmanFilter {
public:
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd R_;

    KalmanFilter() {}

    ~KalmanFilter() {}


    void Predict();

    void UpdateLidar(const Eigen::VectorXd &z);

    void UpdateRadar(const Eigen::VectorXd &z);

    void Update(const Eigen::VectorXd &y);
};


#endif //CARND_HW1_KALMANFILTER_H
