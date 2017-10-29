//
// Created by Andrii Cherniak on 9/28/17.
//

#include "JMT.h"

vector<double> JMT(vector<double> start, vector<double> end, double T) {
    MatrixXd A = MatrixXd(3, 3);
    VectorXd b = VectorXd(3);
    VectorXd x = VectorXd(3);

    double t = T;
    double t2 = t * t;
    double t3 = t * t2;
    double t4 = t * t3;
    double t5 = t * t4;

    A << t3, t4, t5,
            3 * t2, 4 * t3, 5 * t4,
            6 * t, 12 * t2, 20 * t3;

    b << end[0] - (start[0] + start[1] * t + 0.5 * start[2] * t2),
            end[1] - (start[1] + start[2] * t),
            end[2] - start[2];

    x = A.inverse() * b;

    vector<double> cc = {start[0], start[1], 0.5 * start[2], x[0], x[1], x[2]};
    return cc;
}

vector<jmt_state> jmt_path(vector<double> cc, double dT, double T) {
    vector<jmt_state> path;
    double t2, t3, t4, t5;
    double max_v = 0.0;


    for (double t = dT; t < T + dT; t += dT) {
        t2 = t * t;
        t3 = t * t2;
        t4 = t * t3;
        t5 = t * t4;

        jmt_state state;
        state.s = cc[0] + cc[1] * t + cc[2] * t2 + cc[3] * t3 + cc[4] * t4 + cc[5] * t5;
        state.v = cc[1] + cc[2] * t + 3 * cc[3] * t2 + 4 * cc[4] * t3 + 5 * cc[5] * t4;
        state.a = cc[2] + 6 * cc[3] * t + 12 * cc[4] * t2 + 20 * cc[5] * t3;
        max_v = fmax(max_v, state.v);
        path.push_back(state);
    }

//    if (max_v > 48.0*0.44704)
//    std::cout << max_v << std::endl;
    return path;
}


