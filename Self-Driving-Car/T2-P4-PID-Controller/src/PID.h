#ifndef PID_H
#define PID_H

#include <cmath>

class PID {
    /*
    * Errors
    */
    double p_error;
    double i_error;
    double d_error;
    double prev_cte;
    double control_signal;
    double cum_error;

    /*
    * Coefficients
    */
    double Kp;
    double Ki;
    double Kd;
public:

    /*
    * Constructor
    */
    PID();

    /*
    * Destructor.
    */
    virtual ~PID();

    /*
    * Initialize PID.
    */
    void Init(double Kp, double Ki, double Kd);

    /*
    * Update the PID error variables given cross track error.
    */
    void UpdateError(double cte);

    /*
    * Calculate the total PID error.
    */
    double TotalError();

    double getControl();
};

#endif /* PID_H */
