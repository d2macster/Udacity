//
// Created by Andrii Cherniak on 4/25/17.
//

#ifndef CARND_HW1_FUSIONEKF_H
#define CARND_HW1_FUSIONEKF_H

#include "measurement.h"
#include "KalmanFilter.h"

class FusionEKF{
    KalmanFilter ekf;
    bool isInitialized = false;
    long previous_timestamp;

    Eigen::VectorXd x_polar_;
    Eigen::VectorXd x_cartesian_;
    Eigen::MatrixXd R_laser_;
    Eigen::MatrixXd R_radar_;
    Eigen::MatrixXd H_laser_;
    Eigen::MatrixXd Hj_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd P_;
public:
    FusionEKF();
    ~FusionEKF(){}
    void process_measurement(const Measurement &m);
    Eigen::VectorXd getX();
};

#endif //CARND_HW1_FUSIONEKF_H
