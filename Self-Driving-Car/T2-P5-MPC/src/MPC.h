#ifndef MPC_H
#define MPC_H

#include <vector>
#include <chrono>
#include "Eigen-3.3/Eigen/Core"

typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::ratio<1> > second_;

//const int N = 10;
//const double dt = 0.2;
//const double lag = 0.1;
//
//// 40 mph -> converted to m/s
//const double ref_v = 40.0 * 0.447;
//const double ref_cte = 0.0;
//const double ref_psi = 0;
//
//const double w_cte = 1.0;
//const double w_epsi = 10.0;
//const double w_v = 1.0;
//
//const double w_delta = 500.0;
//const double w_a = 1.0;
//
//const double w_deltadot = 30.0;
//const double w_adot = 1.0;
//
////converting to radians
//const double steering_limit = 20 * M_PI / 180;

const int N = 13;
const double dt = 0.15;
const double lag = 0.1;

// 40 mph -> converted to m/s
const double ref_v = 60.0 * 0.447;
const double ref_cte = 0.0;
const double ref_psi = 0;

const double w_cte = 1.0;
const double w_epsi = 10.0;
const double w_v = 1.0;

const double w_delta = 500.0;
const double w_a = 1.0;

const double w_deltadot = 80.0;
const double w_adot = 1.0;

//converting to radians
const double steering_limit = 20 * M_PI / 180;

using namespace std;

class MPC {
public:
    MPC();

    virtual ~MPC();

    // Solve the model given an initial state and polynomial coefficients.
    // Return the first actuatotions.
    vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */
