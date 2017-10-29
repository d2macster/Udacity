//
// Created by Andrii Cherniak on 10/2/17.
//

#include "course.h"

tuple<double, double> sense_car_behind(vector<vector<double>> sensor_fusion, int lane, double car_s) {
    bool sensed_car = false;
    double sensed_car_diff = MAX_S;
    double sensed_car_speed = 0;

    car_s = fmod(car_s, MAX_S);

    for (auto ss : sensor_fusion) {
        double ss_car_vx = ss[3];
        double ss_car_vy = ss[4];
        double ss_car_s = fmod(ss[5], MAX_S);
        double ss_car_d = ss[6];
        double ss_car_v = sqrt(ss_car_vx * ss_car_vx + ss_car_vy * ss_car_vy);

        double diff = ss_car_s - car_s;
        if (diff > 0) {
            diff -= MAX_S;
        }


        if (diff <= 0 &&
            ss_car_d >= 4.0 * lane &&
            ss_car_d <= 4.0 * (lane + 1)) {

            sensed_car = true;
            if (fabs(diff) < sensed_car_diff) {
                sensed_car_diff = fabs(diff);
                sensed_car_speed = ss_car_v;
            }
        }
    }
    if (!sensed_car) {
        sensed_car_diff = MAX_S;
        sensed_car_speed = TARGET_SPEED_MPS;
    }
    return make_tuple(sensed_car_diff, sensed_car_speed);
}

tuple<double, double> sense_car_in_front(vector<vector<double>> sensor_fusion, int lane, double car_s) {
    bool sensed_car = false;
    double sensed_car_diff = MAX_S;
    double sensed_car_speed = 0;

    car_s = fmod(car_s, MAX_S);

    for (auto ss : sensor_fusion) {
        double ss_car_vx = ss[3];
        double ss_car_vy = ss[4];
        double ss_car_s = fmod(ss[5], MAX_S);
        double ss_car_d = ss[6];
        double ss_car_v = sqrt(ss_car_vx * ss_car_vx + ss_car_vy * ss_car_vy);

        double diff = ss_car_s - car_s;
        if (diff < 0) {
            diff += MAX_S;
        }


        if (diff >= 0 &&
            ss_car_d >= 4.0 * lane &&
            ss_car_d <= 4.0 * (lane + 1)) {

            sensed_car = true;
            if (diff < sensed_car_diff) {
                sensed_car_diff = diff;
                sensed_car_speed = ss_car_v;
            }
        }
    }
    if (!sensed_car) {
        sensed_car_diff = MAX_S;
        sensed_car_speed = TARGET_SPEED_MPS;
    }
    return make_tuple(sensed_car_diff, sensed_car_speed);
}

double to_change_lane(jmt_state car_state_s, jmt_state snapshot_state, int from_lane, int to_lane,
                      vector<vector<double>> sensor_fusion) {
    double sensed_car_diff, sensed_car_speed, sensed_car_diff_s, sensed_car_speed_s, sensed_car_diff_behind, sensed_car_speed_behind;


    if (to_lane >= 0 && to_lane <= 2) {

        tie(sensed_car_diff, sensed_car_speed) =
                sense_car_in_front(sensor_fusion, to_lane, car_state_s.s);

        tie(sensed_car_diff_s, sensed_car_speed_s) =
                sense_car_in_front(sensor_fusion, to_lane, snapshot_state.s);

        tie(sensed_car_diff_behind, sensed_car_speed_behind) =
                sense_car_behind(sensor_fusion, to_lane, snapshot_state.s);

        double reward = sensed_car_diff;
        double safety_zone_front =
                sensed_car_speed * CHANGE_LANE_TIME + sensed_car_diff - car_state_s.v * CHANGE_LANE_TIME;
        double safety_zone_front_s =
                sensed_car_speed_s * CHANGE_LANE_TIME + sensed_car_diff_s - car_state_s.v * CHANGE_LANE_TIME;
        double safety_zone_back =
                car_state_s.v * CHANGE_LANE_TIME + sensed_car_diff_behind - sensed_car_speed_behind * CHANGE_LANE_TIME;

        if (
                safety_zone_front >= CAR_DISTANCE &&
                safety_zone_front_s >= CAR_DISTANCE &&
                safety_zone_back >= CAR_DISTANCE &&
                sensed_car_diff_s >= CAR_DISTANCE &&
                sensed_car_diff_behind >= CAR_DISTANCE)
            return reward;
    }

    return 0.0;
}


int decision(jmt_state car_state_s, jmt_state snapshot_state, int lane, vector<vector<double>> sensor_fusion) {
    double sensed_car_diff, sensed_car_speed;

    if (car_state_s.v < 5) return 0;

    tie(sensed_car_diff, sensed_car_speed) = sense_car_in_front(sensor_fusion, lane,
                                                                car_state_s.s);
    int decision = 0;

    double change_left_reward = to_change_lane(car_state_s, snapshot_state, lane, lane - 1, sensor_fusion);
    double change_right_reward = to_change_lane(car_state_s, snapshot_state, lane, lane + 1, sensor_fusion);

    double best_reward = sensed_car_diff;
    if (change_left_reward > best_reward && change_left_reward > 1.1 * sensed_car_diff) {
        best_reward = change_left_reward;
        decision = -1;
    }
    if (change_right_reward > best_reward && change_right_reward > 1.1 * sensed_car_diff) {
        decision = 1;
    }

    return decision;
}

tuple<vector<jmt_state>, vector<jmt_state>>
generate_path(jmt_state car_state_s, jmt_state car_state_d, jmt_state snapshot_s,
              int lane, vector<vector<double>> sensor_fusion, double max_future_speed) {
    double path_t = PATH_PLAN_INCREMENT * KEEP_LANE_HORIZON;
    double max_lane_speed = TARGET_SPEED_MPS;
    max_future_speed = fmax(max_future_speed, snapshot_s.v);

    if (max_future_speed > TARGET_SPEED_MPS) {
        max_lane_speed -= 2 * (max_future_speed - TARGET_SPEED_MPS);
    }

    vector<double> jmt_start_s = {car_state_s.s, car_state_s.v, car_state_s.a};
    double max_a = MAX_ACCELERATION;
//    // slow start to prevent too much jerk
    if (car_state_s.v < 2.0) max_a = 6;
    if (car_state_s.v < 1.0) max_a = 4;
    if (car_state_s.v < 0.5) max_a = 2;

    // constant acceleration case
    double max_speed = car_state_s.v + path_t * max_a;
    double delta_s = 0.5 * (car_state_s.v + max_speed) * path_t;
    max_a *= 1.0 * (1.0 - max_speed / max_lane_speed);

    // reached max allowed speed - taking care
    if (max_speed > max_lane_speed) {
        max_speed = max_lane_speed;
        max_a = 0;
        delta_s = 0.5 * (car_state_s.v + max_speed) * path_t;
    }
    if (delta_s > max_lane_speed * path_t) {
        delta_s = max_lane_speed * path_t;
        max_a = 0;
    }

    double sensed_car_diff, sensed_car_speed, sensed_car_diff_s, sensed_car_speed_s;
    tie(sensed_car_diff, sensed_car_speed) = sense_car_in_front(sensor_fusion, lane, car_state_s.s);
    tie(sensed_car_diff_s, sensed_car_speed_s) = sense_car_in_front(sensor_fusion, lane,
                                                                    snapshot_s.s);

    if (sensed_car_diff_s < sensed_car_diff) {
        sensed_car_diff = sensed_car_diff_s;
        sensed_car_speed = sensed_car_speed_s;
    }


    double collision_point = sensed_car_diff - CAR_DISTANCE;
    // lets prevent collision
    if (delta_s > collision_point) {
        // first lets try to adjust our speed to prevent
        // hitting a car yet dont slam
        // too much on breaks
        max_speed = 2 * collision_point / path_t - car_state_s.v;
        max_a = (max_speed - car_state_s.v) / path_t;
        delta_s = 0.5 * (car_state_s.v + max_speed) * path_t;

        // nothing helps, need to apply maximum breaking power and hope for the best
        if (max_speed < 0 | max_a < MAX_DECELERATION) {
            max_a = MAX_DECELERATION;
            max_speed = car_state_s.v + MAX_DECELERATION * path_t;
            delta_s = 0.5 * (car_state_s.v + max_speed) * path_t;

            if (max_speed < 0.0) {
                max_speed = 0.0;
                max_a = 0.0;
                delta_s = 0.5 * car_state_s.v * car_state_s.v / fabs(MAX_DECELERATION);
            }
        }
    }

    vector<double> jmt_end_s = {car_state_s.s + delta_s, max_speed, max_a};

    vector<double> jmt_cc_s = JMT(jmt_start_s, jmt_end_s, path_t);
    vector<jmt_state> jmt_v_s = jmt_path(jmt_cc_s, PATH_PLAN_INCREMENT, path_t);

    vector<double> jmt_start_d = {car_state_d.s, car_state_d.v, car_state_d.a};


    double lane_d = get_lane_d(lane);
    double target_d = car_state_d.s;
    double target_d_v = 0.0;
    double delta_d = LANE_WIDTH / CHANGE_LANE_HORIZON;
    double delta_d_v = LANE_WIDTH / (CHANGE_LANE_HORIZON * PATH_PLAN_INCREMENT);

    if (target_d > lane_d) {
        target_d -= delta_d * KEEP_LANE_HORIZON;
        target_d_v = -delta_d_v;
        if (target_d < lane_d) {
            target_d = lane_d;
            target_d_v = 0.0;
        }

    } else if (target_d < lane_d) {
        target_d += delta_d * KEEP_LANE_HORIZON;
        target_d_v = delta_d_v;
        if (target_d > lane_d) {
            target_d = lane_d;
            target_d_v = 0.0;
        }
    } else {
        target_d = lane_d;
    }

    double d_change_time = CHANGE_LANE_HORIZON;

    vector<double> jmt_end_d = {target_d, target_d_v, 0.0};
    vector<double> jmt_cc_d = JMT(jmt_start_d, jmt_end_d, path_t);
    vector<jmt_state> jmt_v_d = jmt_path(jmt_cc_d, PATH_PLAN_INCREMENT, d_change_time);

    return make_tuple(jmt_v_s, jmt_v_d);
};

double get_lane_d(int lane) {
    double lane_d = 0.5 * LANE_WIDTH + lane * LANE_WIDTH;
    if (lane == 2) lane_d -= 0.3;
    return lane_d;
}