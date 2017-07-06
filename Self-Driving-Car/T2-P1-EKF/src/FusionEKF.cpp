//
// Created by Andrii Cherniak on 4/25/17.
//

#include "FusionEKF.h"
#include <iostream>

Eigen::VectorXd FusionEKF::getX() {
    return ekf.x_;
}

FusionEKF::FusionEKF() {
    x_cartesian_ = Eigen::VectorXd(4);
    x_cartesian_ << 1, 1, 1, 1;

    R_radar_ = Eigen::MatrixXd(3, 3);
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    R_laser_ = Eigen::MatrixXd(2, 2);
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    H_laser_ = Eigen::MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    P_ = Eigen::MatrixXd(4, 4);
    P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

    F_ = Eigen::MatrixXd(4, 4);
    Q_ = Eigen::MatrixXd(4, 4);
    Hj_ = Eigen::MatrixXd(3, 4);

}

void FusionEKF::process_measurement(const Measurement &m) {

    if (!isInitialized) {
        previous_timestamp = m.timestamp;
        if (m.sensor_type == SensorType::LASER) {
            x_cartesian_ << m.measurements[0], m.measurements[1], 0, 0;
        } else if (m.sensor_type == SensorType::RADAR) {
            x_cartesian_ << m.measurements[0] * std::cos(m.measurements[1]),
                    m.measurements[0] * std::sin(m.measurements[1]), 0, 0;
        }

        ekf.x_ = x_cartesian_;
        ekf.P_ = P_;

        isInitialized = true;
        return;
    }


    double dt = (m.timestamp - previous_timestamp) / 1000000.0;    //dt - expressed in seconds
    previous_timestamp = m.timestamp;

    F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;
    ekf.F_ = F_;

    const double noise_ax = 9, noise_ay = 9;
    double dt2, dt3, dt4;
    dt2 = dt * dt;
    dt3 = dt2 * dt;
    dt4 = dt3 * dt;

    Q_ << noise_ax * dt4 / 4, 0, noise_ax * dt3 / 2, 0,
            0, noise_ay * dt4 / 4, 0, noise_ay * dt3 / 2,
            noise_ax * dt3 / 2, 0, noise_ax * dt2, 0,
            0, noise_ay * dt3 / 2, 0, noise_ay * dt2;

    ekf.Q_ = Q_;

    if (m.sensor_type == SensorType::LASER) {
        ekf.H_ = H_laser_;
        ekf.R_ = R_laser_;
        ekf.Predict();
        ekf.UpdateLidar(m.measurements);

    } else if (m.sensor_type == SensorType::RADAR) {
        double px = ekf.x_(0);
        double py = ekf.x_(1);
        double vx = ekf.x_(2);
        double vy = ekf.x_(3);

        double c1 = px * px + py * py;
        double c2 = sqrt(c1);
        double c3 = (c1 * c2);

        if (fabs(c1) < 0.0001) {
            std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
        }
        
        Hj_ << (px / c2), (py / c2), 0, 0,
                -(py / c1), (px / c1), 0, 0,
                py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;
        ekf.H_ = Hj_;

        ekf.R_ = R_radar_;

        ekf.Predict();

        ekf.UpdateRadar(m.measurements);

    }
}

