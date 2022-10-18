To run MPC code:
1. Run carla.launch 

``` bash
roslaunch carla_config carla.launch
```

2. Run your lanelet_model from pp_env_model
``` bash
roslaunch pp_env_model lanelet_and_model.launch
```

3. Open vnc, run gui (optional)
``` bash
roslaunch carla_manual_control carla_manual_control.launch
```

4. Run pp_rviz_publisher
``` bash
roslaunch pp_rviz_publisher pp_viz.launch
```

5. Launch MPC Spline and MPC controller inside Local Planning Container
``` bash
roslaunch pp_feedback mpc.launch
```