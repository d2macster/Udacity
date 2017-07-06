#include "tools.h"
#include "ukf.h"

int main() {
    std::string data_file = "../data/obj_pose-laser-radar-synthetic-input.txt";
    std::string NIS_file = "../data/NIS.txt";

    std::vector<Measurement> measurements = std::vector<Measurement>();
    std::vector<Measurement> truth = std::vector<Measurement>();

    std::vector<Eigen::VectorXd> estimations = std::vector<Eigen::VectorXd>();
    std::vector<Eigen::VectorXd> truth_v = std::vector<Eigen::VectorXd>();

    std::vector<double> NIS = std::vector<double>();


    load_measurements(data_file, measurements, truth);

    UKF ukf = UKF();
    double px, py, v, phi, vx, vy;

    for (int i = 0; i < measurements.size(); i++) {
        ukf.ProcessMeasurement(measurements[i]);
        Eigen::VectorXd e = Eigen::VectorXd(4);
        px = ukf.x_(0);
        py = ukf.x_(1);
        v = ukf.x_(2);
        phi = ukf.x_(3);
        vx = v * cos(phi);
        vy = v * sin(phi);
        e << px, py, vx, vy;
        estimations.push_back(e);
        truth_v.push_back(truth[i].measurements);

        if (measurements[i].sensor_type == SensorType::LASER) NIS.push_back(ukf.NIS_laser_);
        if (measurements[i].sensor_type == SensorType::RADAR) NIS.push_back(ukf.NIS_radar_);
    }

    std::cout << "RMSE \n" << CalculateRMSE(estimations, truth_v) << std::endl;

    save_results(NIS_file, NIS);

    return 0;
}