//
// Created by Andrii Cherniak on 4/25/17.
//

#include "KalmanFilter.h"

void KalmanFilter::Predict() {
    x_ = F_ * x_;

    Eigen::MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateLidar(const Eigen::VectorXd &z) {
    Eigen::VectorXd y = z - H_ * x_;

    Update(y);
}

void KalmanFilter::Update(const Eigen::VectorXd &y) {
    Eigen::MatrixXd Ht = H_.transpose();
    Eigen::MatrixXd S = H_ * P_ * Ht + R_;
    Eigen::MatrixXd Si = S.inverse();
    Eigen::MatrixXd PHt = P_ * Ht;
    Eigen::MatrixXd K = PHt * Si;

    x_ = x_ + (K * y);
    long x_size = x_.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateRadar(const Eigen::VectorXd &z) {
    double rho = std::sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    double phi = std::atan2(x_(1), x_(0));

    double rho_dot;
    if (fabs(rho) < 0.001) {
        rho_dot = 0;
    } else {
        rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
    }
    Eigen::VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;
    Eigen::VectorXd y = z - z_pred;
    if (y(1) > M_PI)
        y(1) -= 2*M_PI;
    if (y(1) < -M_PI)
        y(1) += 2*M_PI;

    Update(y);
}
