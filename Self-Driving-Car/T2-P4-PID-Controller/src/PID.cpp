#include <iostream>
#include "PID.h"

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    this->cum_error = 0.0;
}

void PID::UpdateError(double cte) {
    p_error = cte;
    d_error = cte - prev_cte;
    i_error += cte;

    control_signal = - (Kp * p_error + Ki * i_error + Kd * d_error);
    control_signal = fmax(-1.0, control_signal);
    control_signal = fmin(1.0, control_signal);
    prev_cte = cte;
    cum_error = fmax(cum_error, fabs(cte));
}

double PID::TotalError() {
    return cum_error;
}

double PID::getControl() {
    return control_signal;
}

