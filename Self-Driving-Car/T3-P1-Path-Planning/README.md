# Path Planning Project
This is my C++ implementation of Path Planning project for Self-Driving Car Engineer Nanodegree Program.
The task seem to appear pretty straightforward for a human to relate and understand: 

1. to drive safely along a circular race track
2. drive at least 4.32 miles (one full circle) without an accident or any driving rules violation
3. do not violate rules of physics e.g. accelerate faster than the car possibly can
4. stay in lane unless changing lanes
5. when possible (and safe), change lanes to pass slow moving traffic

However these seemingly simple conditions became a very interesting challenge 
for implementation and a few very special cases to handle.


## Frenet coordinates and map wavepoints
The original repository contains two helper functions to convert 
cartesian coordinates into frenet coordinates and back. The very first task 
was to understand if we need to switch between both or not. 

During the first trial i discovered that car moves with jumps, which were
 associated with non-regularities
in map waypoint coordinates. To solve the problem, I applied spline smoothing
to the original waypoints and generated ~ 100K points. 
I also created a loop with the map coordinates by adding a few map points from the start
which have shifted Frenet `S` coordinate by `MAX_S` - length of the track. Otherwise
i was getting discontinuity in motion.

This solved the problem with uneven car movement when performing frenet -> cartesian transformation.
Thus all computations were performed using frenet coordinates internally,
and converting to cartesian only at the last stage of placing the car on the track.

Helper function `get_waypoints` which does spline transformation, is implemented in `waypoints.cpp`.
To increase code readability, i moved all coordinate transformation-related
functions , including `getFrenet` and `getXY` in `cartesian_frenet.cpp` file.

## Path planner logic
All helper functions are implemented in `course.cpp`. There are two types of actions which we can take:
* stay in lane
* change lane

Given this is a real - world model, preference for changing lane goes to 
change lane to the left, to prevent annoying real-life situations where a car 
in speeding in the right-most lane and causes problems for other cars trying to enter or exit the highway.

### Dealing with rapid lane change / stop
After running multiple simulations I discovered that other cars may behave close to 
bad LA drives which try to cut you off by changing lane very close to the car, or 
slow down rapidly. To prevent an accident we have to re-evaluate previous
path plan and adjust if necessary, e.g. discard some / most of the points and update the path.
At the same time, to achieve smooth driving, we need to feed ~ 50 points into the car simulator.
Thus I take only last 10 points from the previous path, and the rest points re-compute and update the path.
Since internally we work in `s, d frenet` coordinates, i keep internal history 
of those coordinates in two queues , for s and d coordinates `line 41, 94-120 ` in `main.cpp` 
### Stay in lane
The logic boils down to a few steps:
 * increase car speed `v = a*t`
 * make sure `v <= max v (50 mph)`
 * compute new s and d coordinates. when we stay in lane, `d = 2 + 4*lane`
 * detect if there are any cars in front of us, and how far they are
 * if within the path plan we will approach car too close, then make sure it will not happen
 via adjusting car speed ~ speed of car in front, and s coordinate should  never go beyond car in front of us
 
Special cases to remember:
 * We work in `m/s` instead of `mph` units, thus need to convert
 * during test runs i discovered that the same speed limit (in frenet) 
 generates slightly different speed in cartesian coordinates, depending which lane we are in.
 to prevent going above `50mph` i added special logic which reduces target speed as a function of 
 which lane we are in `lines 166-168 in course.cpp`
 * when driving in right-most lane a few times i got an error `outside lane` despite setting
 d coordinate to the middle of the lane. i had to add a special treatment for this
 in `lines 185-187 in course.cpp`
 
 The logic is being implemented in `generate_path` function in  `course.cpp`
### Change lane
I have imlemented rule-based approach to decide if we should change. The primarily reason
was ease of implementation and debugging. I admit this might not be the best 
path planner and I definitely observed sub-optimal decision making.
However I also observed some erratic behavior from other cars, and not sure 
it is very easy to to address optimality with path planning within the 
timeline for this project.

* if the car in front is too far (will take more than 
9 seconds to reach at max speed ), then do not change lanes. we have enough time to accelerate to the
target speed
* otherwise check if shifting to the left lane is more beneficial, if not , check the 
right lane. The logic is implemented in  a helper function `to_change_lane` in `course.cpp`
* when considering lane change, we are looking at the speed, and distance
to the car in front of us and behind us, and only if safe to do so (car behind will not hit us during lane change), and we have 
enough distance between cars for this change, we will perform change.

### Jerk minimizing trajectory
Once we determined the immediate target coordinates for our path, 
i used jerk minimizing trajectory planner to compute the actual 
`s, d` path. The implementation of this helper function is in `JMT.cpp`
