#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <random>

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    use_laser_ = true;
    use_radar_ = true;

    x_ = Eigen::VectorXd(5);

    P_ = Eigen::MatrixXd(5, 5);

    // Process noise
    std_a_ = 1.3;
    std_yawdd_ = 0.7;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    is_initialized_ = false;
    previos_timestamp_ = 0;

    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_x_;

    P_ = Eigen::MatrixXd(n_x_, n_x_);
    P_ <<
       1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

    Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);

    weights_ = Eigen::VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(Measurement m) {

    if (!is_initialized_) {
        x_ << 1, 1, 1, 1, 1;
        double px = 0, py = 0, v = 0, phi = 0, phi_dot = 0, rho = 0;
        previos_timestamp_ = m.timestamp;
        if (m.sensor_type == SensorType::LASER) {
            px = m.measurements(0);
            py = m.measurements(1);
        } else if (m.sensor_type == SensorType::RADAR) {
            rho = m.measurements(0);
            phi = m.measurements(1);
            px = rho * cos(phi);
            py = rho * sin(phi);
        }
        x_ << px, py, v, phi, phi_dot;
        is_initialized_ = true;
        return;
    }

    if (m.sensor_type == SensorType::LASER && use_laser_) {
        Prediction((m.timestamp - previos_timestamp_) / 1000000.0);
        previos_timestamp_ = m.timestamp;
        UpdateLidar(m);
    }
    if (m.sensor_type == SensorType::RADAR && use_radar_) {
        Prediction((m.timestamp - previos_timestamp_) / 1000000.0);
        previos_timestamp_ = m.timestamp;
        UpdateRadar(m);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    Eigen::VectorXd x_aug = Eigen::VectorXd(n_aug_);
    Eigen::MatrixXd P_aug = Eigen::MatrixXd(n_aug_, n_aug_);
    Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    Eigen::MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }


    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.01) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }


    // set weights
    double weight_0 = lambda_ / (lambda_ + n_aug_);
    weights_(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_);
        weights_(i) = weight;
    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }
    x_(3) = fmod(x_(3), 2. * M_PI);
    while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
    while (x_(3) < -M_PI) x_(3) += 2. * M_PI;

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = fmod(x_diff(3), 2. * M_PI);
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(Measurement m) {

    int n_z = 2;
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);

    Eigen::VectorXd z_out = Eigen::VectorXd(n_z);
    Eigen::MatrixXd S_out = Eigen::MatrixXd(n_z, n_z);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //mean predicted measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }


    //measurement covariance matrix S
    Eigen::MatrixXd S = Eigen::MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }


    //add measurement noise covariance matrix
    Eigen::MatrixXd R = Eigen::MatrixXd(n_z, n_z);
    R << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;
    S = S + R;

    //create matrix for cross correlation Tc
    Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, n_z);

    Eigen::VectorXd z = Eigen::VectorXd(n_z);
    z << m.measurements(0), m.measurements(1);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

        // state difference
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    Eigen::MatrixXd K = Tc * S.inverse();

    //residual
    Eigen::VectorXd z_diff = m.measurements - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(Measurement m) {

    int n_z = 3;
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);

    Eigen::VectorXd z_out = Eigen::VectorXd(3);
    Eigen::MatrixXd S_out = Eigen::MatrixXd(3, 3);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                        //r
        double phi;
        if (fabs(p_x) < 0.00001 && fabs(p_y) < 0.00001)
            phi = 0.0;
        else phi = atan2(p_y, p_x);
        Zsig(1, i) = phi;                                //phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);   //r_dot
    }

    //mean predicted measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }


    //measurement covariance matrix S
    Eigen::MatrixXd S = Eigen::MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }


    //add measurement noise covariance matrix
    Eigen::MatrixXd R = Eigen::MatrixXd(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
    S = S + R;

    //create matrix for cross correlation Tc
    Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, n_z);

    Eigen::VectorXd z = Eigen::VectorXd(n_z);
    z << m.measurements(0), m.measurements(1), m.measurements(2);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = fmod(z_diff(1), 2. * M_PI);
        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = fmod(x_diff(3), 2. * M_PI);
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    Eigen::MatrixXd K = Tc * S.inverse();

    //residual
    Eigen::VectorXd z_diff = m.measurements - z_pred;

    //angle normalization
    z_diff(1) = fmod(z_diff(1), 2. * M_PI);
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
    while (x_(3) < -M_PI) x_(3) += 2. * M_PI;
    P_ = P_ - K * S * K.transpose();

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
