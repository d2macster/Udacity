//
// Created by Andrii Cherniak on 4/22/17.
//

#ifndef MEASUREMENT_H_
#define MEASUREMENT_H_

#include "Eigen/Dense"

enum SensorType {
    LASER,
    RADAR
};

inline const char* SensorToString(SensorType st)
{
    switch (st)
    {
        case LASER:   return "LASER";
        case RADAR:   return "RADAR";
        default:      return "[Unknown]";
    }
}

class Measurement {
public:
    long timestamp;

    SensorType sensor_type;

    Eigen::VectorXd measurements;

    Measurement(long ts, SensorType st, Eigen::VectorXd ms) {
        timestamp = ts;
        sensor_type = st;
        measurements = ms;
    }

    virtual ~Measurement() {

    }
};

#endif