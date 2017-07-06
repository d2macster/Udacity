//
// Created by Andrii Cherniak on 4/22/17.
//
#include "tools.h"

std::string sensor_type;
double x_measured, y_measured;
double rho_measured, phi_measured, rhodot_measured;
long timestamp;
double x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth;


void load_radar(std::string line, std::vector<Measurement> &measurements,
                std::vector<Measurement> &ground_truth) {
//    sensor_type, rho_measured, phi_measured, rhodot_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth.

    std::istringstream iss(line);
    iss >> sensor_type;
    iss >> rho_measured;
    iss >> phi_measured;
    iss >> rhodot_measured;
    iss >> timestamp;
    iss >> x_groundtruth;
    iss >> y_groundtruth;
    iss >> vx_groundtruth;
    iss >> vy_groundtruth;

    Eigen::VectorXd v_measurement = Eigen::VectorXd(3);
    Eigen::VectorXd v_truth = Eigen::VectorXd(4);

    v_measurement << rho_measured, phi_measured, rhodot_measured;
    v_truth << x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth;

    Measurement m1 = Measurement(timestamp, SensorType::RADAR, v_measurement);
    Measurement m2 = Measurement(timestamp, SensorType::RADAR, v_truth);

    measurements.push_back(m1);
    ground_truth.push_back(m2);

}

void load_lidar(std::string line, std::vector<Measurement> &measurements,
                std::vector<Measurement> &ground_truth) {
//    sensor_type, x_measured, y_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth.

    std::istringstream iss(line);
    iss >> sensor_type;
    iss >> x_measured;
    iss >> y_measured;
    iss >> timestamp;
    iss >> x_groundtruth;
    iss >> y_groundtruth;
    iss >> vx_groundtruth;
    iss >> vy_groundtruth;

    Eigen::VectorXd v_measurement = Eigen::VectorXd(2);
    Eigen::VectorXd v_truth = Eigen::VectorXd(4);

    v_measurement << x_measured, y_measured;
    v_truth << x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth;


    Measurement m1 = Measurement(timestamp, SensorType::LASER, v_measurement);
    Measurement m2 = Measurement(timestamp, SensorType::LASER, v_truth);

    measurements.push_back(m1);
    ground_truth.push_back(m2);

}

void load_measurements(std::string path,
                       std::vector<Measurement> &measurements,
                       std::vector<Measurement> &ground_truth) {
    std::string line;
    std::ifstream infile(path);


    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> sensor_type;
        if (sensor_type == "L") load_lidar(line, measurements, ground_truth);
        if (sensor_type == "R") load_radar(line, measurements, ground_truth);
    }
    infile.close();
}

void save_results(std::string path,
                  std::vector<Eigen::VectorXd> estimations_v,
                  std::vector<Eigen::VectorXd> ground_truth_v,
                  std::vector<Measurement> measurements) {
    std::ofstream myfile;
    myfile.open(path);
    for (int i = 0; i < estimations_v.size(); i++) {
        myfile << estimations_v[i][0] << "\t"
               << estimations_v[i][1] << "\t"
               << estimations_v[i][2] << "\t"
               << estimations_v[i][3] << "\t";
        if (measurements[i].sensor_type == SensorType::LASER) {
            myfile << measurements[i].measurements[0] << "\t"
                   << measurements[i].measurements[1] << "\t";
        }
        if (measurements[i].sensor_type == SensorType::RADAR) {
            double rho = measurements[i].measurements[0];
            double phi = measurements[i].measurements[1];
            myfile << rho * std::cos(phi) << "\t"
                   << rho * std::sin(phi) << "\t";

        }
        myfile << ground_truth_v[i][0] << "\t"
               << ground_truth_v[i][1] << "\t"
               << ground_truth_v[i][2] << "\t"
               << ground_truth_v[i][3] << "\t"

               << std::endl;
    }
    myfile.close();
}

Eigen::VectorXd
CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth) {
    Eigen::VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() != ground_truth.size()
        || estimations.size() == 0) {
        std::cout << "Invalid estimation or ground_truth data" << std::endl;
        return rmse;
    }

    //accumulate squared residuals
    for (unsigned int i = 0; i < estimations.size(); ++i) {

        Eigen::VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse / estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}


