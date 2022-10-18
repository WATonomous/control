#!/usr/bin/env python3

import time
from typing import Optional
import rospy
import numpy as np

from path_planning_msgs.msg import (
    MPCOutput,
    ReferenceSpline,
    ReferenceLine,
)
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from embedded_msgs.msg import DesiredOutput
from casadi import *
from std_msgs.msg import Float32

from solver import ParallelSolver
from logger import Logger

from utils import (
    obs_from_occ,
    pub_obs,
    pub_states,
    pub_ref_sphere,
    Config,
    yaw_from_odom,
    fwd_vel_from_odom,
    ref_path,
    path_param_from_xy,
    NumericParameters,
)
from occupancy import OccupancyCluster


class Controller(object):
    def __init__(self, config: Config):
        self.config = config

        self.ref: Optional[ReferenceSpline] = None
        self.u = None
        self.x = None
        self._T_old = None

        self.imu = None

        # Sets up NLP
        self.solver = ParallelSolver(config)
        self.logger = Logger()
        self._init_pub_sub()

        self.occupancy_cluster = OccupancyCluster(self._marker_pub, config)

    def _init_pub_sub(self):
        self._marker_pub = rospy.Publisher("mpc_markers", Marker, queue_size=10)
        self.param_viz = rospy.Publisher(
            "/path_planning/mpc/param_viz", Float32, queue_size=10
        )
        self.carla_pub = rospy.Publisher("mpc_output", MPCOutput, queue_size=10)
        self.can_pub = rospy.Publisher(
            "feedback_desired_output", DesiredOutput, queue_size=10
        )
        rospy.Subscriber("/navsat/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber("/imu/data", Imu, self.imu_callback, queue_size=1)
        rospy.Subscriber(
            "/path_planning/mpc/reference",
            ReferenceSpline,
            self.ref_callback,
            queue_size=1,
        )
        rospy.Timer(
            rospy.Duration(1 / self.config.interpolation_rate),
            self.interpolationCallBack,
            oneshot=False,
        )
        rospy.on_shutdown(self.solver.terminate)

    def imu_callback(self, imu: Imu):
        self.imu = imu

    def odom_callback(self, odom: Odometry):
        self.odom = odom
        start_time = time.time()

        # If we don't have a reference to follow yet we cannot do anything
        if self.ref is None:
            # As per MPC, we publish the first control action
            accel_min = self.config.body.bounds.lon_accel_l
            self.publish_carla(accel=accel_min, steer=0)
            self.publish_can(accel=accel_min, steer=0, curr_vel=0)
            return

        # Assume that our current steering angle is what we commanded previously
        measured_steer = self.u[0, 1].__float__() if self.u is not None else 0
        pos = odom.pose.pose.position
        yaw = yaw_from_odom(odom)
        vel = odom.twist.twist.linear

        # Get path parameter from current position
        measured_path_param = path_param_from_xy(self.ref.ref_a, self.ref.ref_b, pos.x, pos.y)
        ref_x, ref_y, ref_yaw = ref_path(self.ref.ref_a, self.ref.ref_b, measured_path_param)

        # Place a marker at reference point we are starting at
        pub_ref_sphere(ref_x, ref_y, self._marker_pub)

        # According to MPC we keep applying the first control action for the whole sample period it was planned for
        if (
            self._T_old is not None
            and rospy.get_time() - self._T_old <= self.config.opt.T
        ):
            return

        self._T_old = rospy.get_time()

        lon_accel = self.imu.linear_acceleration.x if self.imu is not None else 0
        p = NumericParameters(
            x0=[
                pos.x,
                pos.y,
                yaw,
                vel.x,
                fwd_vel_from_odom(odom),
                measured_path_param,
                measured_steer,
            ],
            center_a=list(self.ref.ref_a),
            center_b=list(self.ref.ref_b),
            left_a=list(self.ref.left_a),
            left_b=list(self.ref.left_b),
            right_a=list(self.ref.right_a),
            right_b=list(self.ref.right_b),
            lon_accel=lon_accel,
            obs_x=self.occupancy_cluster.obs_x,
            obs_y=self.occupancy_cluster.obs_y,
            obs_rx=self.occupancy_cluster.obs_rx,
            obs_ry=self.occupancy_cluster.obs_ry,
            obs_theta=self.occupancy_cluster.obs_theta,
        )
        sol = self.solver(p)
        # Cut out control variables and reshape them into matrix
        state_n = self.solver.vehicle_state_n
        con_n = self.solver.control_n
        N = self.config.opt.N
        self.u = reshape(sol[state_n * (N + 1) :].T, con_n, N).T
        self.x = reshape(sol[: state_n * (N + 1)].T, state_n, N + 1).T
        # As per MPC, we publish the first control action
        u0_accel = self.u[0, 0].__float__()
        u0_steer = self.u[0, 1].__float__()
        self.publish_carla(u0_accel, u0_steer)
        self.logger.log_cmd_accel(u0_accel)
        self.logger.log_cmd_steer(u0_steer)
        self.publish_can(u0_accel, u0_steer, fwd_vel_from_odom(odom))
        self.logger.log_react_time(time.time() - start_time)
        self.param_viz.publish(measured_path_param)
        # Publish markers for the predicted states
        pub_states(self.x, self._marker_pub)

        # Log some metrics
        self.logger.log_ref_pose(ref_x, ref_y, ref_yaw)
        self.logger.log_err_yaw(abs(yaw - ref_yaw))
        self.logger.log_err_x(abs(pos.x - ref_x))
        self.logger.log_err_y(abs(pos.y - ref_y))
        self.logger.log_err_vel(
            abs(fwd_vel_from_odom(odom) - self.config.body.desired_fwd_vel)
        )
        self.logger.log_err_lat(((pos.x - ref_x) ** 2 + (pos.y - ref_y) ** 2) ** 0.5)
        self.logger.publish_stats()

    # Get's called at highest clock rate possible, interpolates between first and second MPC control actions
    #   note that this is a relaxation of the receding horizon design of MPC
    def interpolationCallBack(self, _):
        if self.u is None:
            return
        seconds = rospy.get_time()
        currDeltaT = seconds - self._T_old
        T = self.config.opt.T
        accel = (currDeltaT / T) * (
            self.u[1, 0].__float__() - self.u[0, 0].__float__()
        ) + self.u[0, 0].__float__()
        steer = (currDeltaT / T) * (
            self.u[1, 1].__float__() - self.u[0, 1].__float__()
        ) + self.u[0, 1].__float__()
        self.publish_carla(accel, steer)
        self.logger.log_cmd_accel(accel)
        self.logger.log_cmd_steer(steer)
        self.publish_can(accel, steer, fwd_vel_from_odom(self.odom))
        self.logger.publish_stats()

    def ref_callback(self, ref: ReferenceSpline):
        if len(list(ref.ref_a)) == 0:
            self.ref = None
        else:
            self.ref = ref

    def publish_carla(self, accel, steer):
        msg_out = MPCOutput()
        msg_out.accel = accel
        msg_out.steer = steer
        self.carla_pub.publish(msg_out)

    def publish_can(self, accel, steer, curr_vel):
        can = self.config.can
        AeroCons = 0.5 * can.rho * can.A * can.Cd
        RollCons = can.m * can.g * can.Cr
        if curr_vel < 1 and curr_vel > -1.1:
            RollRes = RollCons * curr_vel
        else:
            RollRes = RollCons
        resForce = AeroCons * curr_vel * curr_vel + RollRes
        accForce = accel * can.m
        torque = (can.r_wheel / can.GearRatio) * (accForce + resForce)
        can_out = DesiredOutput()
        can_out.torque = torque
        can_out.acceleration = accel
        can_out.theta = steer / np.pi * 180
        can_out.state = 1
        self.can_pub.publish(can_out)
        self.logger.log_theta(can_out.theta)
        self.logger.log_torque(can_out.torque)
