//
// Created by Andrii Cherniak on 9/26/17.
//
// Load up map values for waypoint's x,y,s and d normalized normal vectors
#include "waypoints.h"

tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> get_waypoints(string map_file) {
    // Load up map values for waypoint's x,y,s and d normalized normal vectors
    vector<double> mw_x_i, mw_y_i, mw_s_i, mw_dx_i, mw_dy_i;
    vector<double> mwa_x_i, mwa_y_i, mwa_s_i, mwa_dx_i, mwa_dy_i;


    ifstream in_map_(map_file.c_str(), ifstream::in);

    string line;
    while (getline(in_map_, line)) {
        istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        mw_x_i.push_back(x);
        mw_y_i.push_back(y);
        mw_s_i.push_back(s);
        mw_dx_i.push_back(d_x);
        mw_dy_i.push_back(d_y);
    }
    for (int i = int(mw_x_i.size()) - 30; i < mw_x_i.size(); i++) {
        mwa_x_i.push_back(mw_x_i[i]);
        mwa_y_i.push_back(mw_y_i[i]);
        mwa_s_i.push_back(mw_s_i[i] - MAX_S);
        mwa_dx_i.push_back(mw_dx_i[i]);
        mwa_dy_i.push_back(mw_dy_i[i]);
    }
    for (int i = 0; i < mw_x_i.size(); i++) {
        mwa_x_i.push_back(mw_x_i[i]);
        mwa_y_i.push_back(mw_y_i[i]);
        mwa_s_i.push_back(mw_s_i[i]);
        mwa_dx_i.push_back(mw_dx_i[i]);
        mwa_dy_i.push_back(mw_dy_i[i]);
    }
    for (int i = 0; i < 30; i++) {
        mwa_x_i.push_back(mw_x_i[i]);
        mwa_y_i.push_back(mw_y_i[i]);
        mwa_s_i.push_back(mw_s_i[i] + MAX_S);
        mwa_dx_i.push_back(mw_dx_i[i]);
        mwa_dy_i.push_back(mw_dy_i[i]);
    }


    vector<double> waypoint_spline_t = {};
    int map_waypoints_size = int(mwa_x_i.size());
    for (int i = 0; i < map_waypoints_size; i++) {
        double t = (double) i / (double) map_waypoints_size;
        waypoint_spline_t.push_back(t);
    }


    tk::spline s_wp_x, s_wp_y, s_wp_s, s_wp_dx, s_wp_dy;

    s_wp_x.set_points(waypoint_spline_t, mwa_x_i);
    s_wp_y.set_points(waypoint_spline_t, mwa_y_i);
    s_wp_s.set_points(waypoint_spline_t, mwa_s_i);
    s_wp_dx.set_points(waypoint_spline_t, mwa_dx_i);
    s_wp_dy.set_points(waypoint_spline_t, mwa_dy_i);

    vector<double> map_waypoints_x, map_waypoints_y, map_waypoints_s;
    vector<double> map_waypoints_dx, map_waypoints_dy;

    for (int i = 0; i < RESUMPLED_WAYPOINTS_SIZE; i++) {
        double s = (double) i / (double) RESUMPLED_WAYPOINTS_SIZE;
        map_waypoints_x.push_back(s_wp_x(s));
        map_waypoints_y.push_back(s_wp_y(s));
        map_waypoints_s.push_back(s_wp_s(s));
        map_waypoints_dx.push_back(s_wp_dx(s));
        map_waypoints_dy.push_back(s_wp_dy(s));
    }

    return make_tuple(map_waypoints_x, map_waypoints_y, map_waypoints_s, map_waypoints_dx, map_waypoints_dy);
}
