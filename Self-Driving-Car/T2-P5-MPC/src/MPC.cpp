#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
public:
    // Fitted polynomial coefficients
    Eigen::VectorXd coeffs;

    FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

    typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;
    // `fg` is a vector containing the cost and constraints.
    // `vars` is a vector containing the variable values (state & actuators).

    void operator()(ADvector &fg, const ADvector &vars) {
        // The cost is stored is the first element of `fg`.
        // Any additions to the cost should be added to `fg[0]`.
        fg[0] = 0;

        // minimizing cost regarding reference state
        for (int i = 0; i < N; i++) {
            fg[0] += w_cte * CppAD::pow(vars[cte_start + i] - ref_cte, 2);
            fg[0] += w_epsi * CppAD::pow(vars[epsi_start + i] - ref_psi, 2);
            fg[0] += w_v * CppAD::pow(vars[v_start + i] - ref_v, 2);
        }

        // minimizing use of actuators
        for (int i = 0; i < N - 1; i++) {
            fg[0] += w_delta * CppAD::pow(vars[delta_start + i], 2);
            fg[0] += w_a * CppAD::pow(vars[a_start + i], 2);

        }

        // Minimize the value gap between sequential actuations.
        for (int t = 0; t < N - 2; t++) {
            fg[0] += w_deltadot * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
            fg[0] += w_adot * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
        }

        fg[1 + x_start] = vars[x_start];
        fg[1 + y_start] = vars[y_start];
        fg[1 + psi_start] = vars[psi_start];
        fg[1 + v_start] = vars[v_start];
        fg[1 + cte_start] = vars[cte_start];
        fg[1 + epsi_start] = vars[epsi_start];

        for (int t = 0; t < N - 1; t++) {
            // state at  time t
            AD<double> x0 = vars[x_start + t];
            AD<double> y0 = vars[y_start + t];
            AD<double> v0 = vars[v_start + t];
            AD<double> psi0 = vars[psi_start + t];
            AD<double> cte0 = vars[cte_start + t];
            AD<double> epsi0 = vars[epsi_start + t];

            // state at time  t + 1
            AD<double> x1 = vars[x_start + t + 1];
            AD<double> y1 = vars[y_start + t + 1];
            AD<double> v1 = vars[v_start + t + 1];
            AD<double> psi1 = vars[psi_start + t + 1];
            AD<double> cte1 = vars[cte_start + t + 1];
            AD<double> epsi1 = vars[epsi_start + t + 1];

            // Only consider the actuation at time t.
            AD<double> delta0 = vars[delta_start + t];
            AD<double> a0 = vars[a_start + t];

            AD<double> f0 = 0.0;
            for (int i = 0; i < coeffs.size(); i++) {
                f0 += coeffs[i] * CppAD::pow(x0, i);
            }

            AD<double> psides0 = 0.0;
            for (int i = 1; i < coeffs.size(); i++) {
                psides0 += i * coeffs[i] * CppAD::pow(x0, i - 1);
            }

            // here we work with time t + 1

            fg[2 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
            fg[2 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
            fg[2 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
            fg[2 + v_start + t] = v1 - (v0 + a0 * dt);
            fg[2 + cte_start + t] =
                    cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
            fg[2 + epsi_start + t] =
                    epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);

        }
    }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}

MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
    bool ok = true;
    typedef CPPAD_TESTVECTOR(double) Dvector;

    size_t i;

    std::chrono::time_point<clock_> t1 = clock_::now();

    double x = state[0];
    double y = state[1];
    double psi = state[2];
    double v = state[3];
    double cte = state[4];
    double epsi = state[5];

    size_t n_vars = N * 6 + (N - 1) * 2;
    size_t n_constraints = N * 6;

    Dvector vars(n_vars);
    for (i = 0; i < n_vars; i++) {
        vars[i] = 0.0;
    }

    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);

    // Set all non-actuators upper and lowerlimits
    // to the max negative and positive values, e.g. x, y, psi, v, cte, epsi
    for (i = 0; i < delta_start; i++) {
        vars_lowerbound[i] = -1.0e19;
        vars_upperbound[i] = 1.0e19;
    }

    // The upper and lower limits of delta are set to -25 and 25 degrees (values in radians).
    for (i = delta_start; i < a_start; i++) {
        vars_lowerbound[i] = -steering_limit;
        vars_upperbound[i] = steering_limit;
    }

    // Acceleration/decceleration upper and lower limits.
    for (i = a_start; i < n_vars; i++) {
        vars_lowerbound[i] = -1.0;
        vars_upperbound[i] = 1.0;
    }

    // Lower and upper limits for the constraints
    // Should be 0 besides initial state.
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);

    for (i = 0; i < n_constraints; i++) {
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }

    constraints_lowerbound[x_start] = x;
    constraints_lowerbound[y_start] = y;
    constraints_lowerbound[psi_start] = psi;
    constraints_lowerbound[v_start] = v;
    constraints_lowerbound[cte_start] = cte;
    constraints_lowerbound[epsi_start] = epsi;

    constraints_upperbound[x_start] = x;
    constraints_upperbound[y_start] = y;
    constraints_upperbound[psi_start] = psi;
    constraints_upperbound[v_start] = v;
    constraints_upperbound[cte_start] = cte;
    constraints_upperbound[epsi_start] = epsi;

    // object that computes objective and constraints
    FG_eval fg_eval(coeffs);

    //
    // NOTE: You don't have to worry about these options
    //
    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          0.5\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
            constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

    // Cost
    auto cost = solution.obj_value;
    std::cout << "Cost " << cost << std::endl;

    auto sol_x = solution.x;

    vector<double> result;

    double  mcp_processing_time = std::chrono::duration_cast<second_> (clock_::now() - t1).count();
    double total_lag = lag + mcp_processing_time;
    int ind_start = std::min(int(std::round(total_lag / dt)), N - 1);

    double steering = sol_x[delta_start + ind_start];
    double throttle = sol_x[a_start + ind_start];

    // steering ange and acceleration at initial time
    result.push_back(steering);
    result.push_back(throttle);

    for (i = 1; i < N; i++) {
        result.push_back(solution.x[x_start + i]);
    }

    for (i = 1; i < N; i++) {
        result.push_back(solution.x[y_start + i]);

    }


    return result;
}
