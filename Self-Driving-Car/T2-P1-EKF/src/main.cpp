#include <iostream>
#include "tools.h"
#include "FusionEKF.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Expected execution format : "
                  << argv[0] << " path/to/input.txt path/to/output.txt"
                  << std::endl;
    }
    assert((argc == 3) && "provided parameters error");
    std::vector<Measurement> measurements = std::vector<Measurement>();
    std::vector<Measurement> ground_truth = std::vector<Measurement>();

    load_measurements(argv[1], measurements, ground_truth);

    FusionEKF fusionEKF;

    std::vector<Eigen::VectorXd> estimations_v = std::vector<Eigen::VectorXd>();
    std::vector<Eigen::VectorXd> ground_truth_v = std::vector<Eigen::VectorXd>();

    for (int i = 0; i < measurements.size(); i++) {
        fusionEKF.process_measurement(measurements[i]);

        estimations_v.push_back(fusionEKF.getX());
        ground_truth_v.push_back(ground_truth[i].measurements);
    }

    save_results(argv[2], estimations_v, ground_truth_v, measurements);

    std::cout << "RMSE" << std::endl << CalculateRMSE(estimations_v, ground_truth_v) << std::endl;

    return 0;
}