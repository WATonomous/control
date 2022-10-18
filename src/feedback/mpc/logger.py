import rospy

from path_planning_msgs.msg import MPCStats
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from utils import quaternion_from_yaw, fwd_vel_from_odom, fwd_vel_from_odom


class Logger(object):
    def __init__(self):
        self.stat_msg = MPCStats()
        self.stats_pub = rospy.Publisher("mpc_stats", MPCStats, queue_size=10)
        rospy.Subscriber("/navsat/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber("/imu/data", Imu, self.imu_callback, queue_size=1)
        self.imu = None

        self.gas_viz_pub = rospy.Publisher(
            "/path_planning/mpc/gas_viz", Float32, queue_size=10
        )
        self.brake_viz_pub = rospy.Publisher(
            "/path_planning/mpc/brake_viz", Float32, queue_size=10
        )
        self.torque_viz_pub = rospy.Publisher(
            "/path_planning/mpc/torque_viz", Float32, queue_size=10
        )
        self.accel_deccel_viz_pub = rospy.Publisher(
            "/path_planning/mpc/accel_deccel_viz", Float32, queue_size=10
        )
        self.accel_viz_pub = rospy.Publisher(
            "/path_planning/mpc/accel_viz", Float32, queue_size=10
        )
        self.deccel_viz_pub = rospy.Publisher(
            "/path_planning/mpc/deccel_viz", Float32, queue_size=10
        )
        self.theta_left_viz = rospy.Publisher(
            "/path_planning/mpc/theta_left_viz", Float32, queue_size=10
        )
        self.theta_right_viz = rospy.Publisher(
            "/path_planning/mpc/theta_right_viz", Float32, queue_size=10
        )
        self.theta_viz = rospy.Publisher(
            "/path_planning/mpc/theta_viz", Float32, queue_size=10
        )
        self.solution_time_viz = rospy.Publisher(
            "/path_planning/mpc/solution_time_viz", Float32, queue_size=10
        )

    def log_theta(self, theta: float):
        self.theta_viz.publish(theta)
        self.theta_right_viz.publish(-1 * min(0, theta))
        self.theta_left_viz.publish(max(0, theta))

    def log_torque(self, torque: float):
        self.brake_viz_pub.publish(-1 * min(0, torque))
        self.gas_viz_pub.publish(max(0, torque))
        self.torque_viz_pub.publish(torque)

    def log_react_time(self, react_time: float):
        self.solution_time_viz.publish(react_time)
        self.stat_msg.react_time = react_time

    def log_ref_pose(self, x: float, y: float, yaw: float):
        ref_pose = Pose()
        ref_pose.position.x = x
        ref_pose.position.y = y
        ref_pose.orientation = quaternion_from_yaw(yaw)
        self.stat_msg.ref_pose_list.append(ref_pose)

    def log_err_yaw(self, err_yaw: float):
        self.stat_msg.err_yaw = err_yaw

    def log_err_x(self, err_x: float):
        self.stat_msg.err_x = err_x

    def log_err_y(self, err_y: float):
        self.stat_msg.err_y = err_y

    def log_err_vel(self, err_vel: float):
        self.stat_msg.err_vel = err_vel

    def log_err_lat(self, err_lat: float):
        self.stat_msg.err_lat = err_lat

    def log_cmd_accel(self, cmd_accel):
        self.stat_msg.cmd_accel = cmd_accel
        self.deccel_viz_pub.publish(-1 * min(0, cmd_accel))
        self.accel_viz_pub.publish(max(0, cmd_accel))
        self.accel_deccel_viz_pub.publish(cmd_accel)

    def log_cmd_steer(self, cmd_steer):
        self.stat_msg.cmd_steer = cmd_steer

    def odom_callback(self, msg: Odometry):
        self.stat_msg.act_pose_list.append(msg.pose.pose)
        vel = msg.twist.twist.linear
        self.stat_msg.act_lon_vel = vel.x
        self.stat_msg.act_lat_accel = vel.y
        self.stat_msg.act_fwd_vel = fwd_vel_from_odom(msg)

    def imu_callback(self, msg: Imu):
        if self.imu is None:
            self.imu = msg
            return
        accel = msg.linear_acceleration
        self.stat_msg.act_lon_accel = accel.x
        self.stat_msg.act_lat_accel = accel.y
        fwd_accel = (accel.x ** 2 + accel.y ** 2) ** 0.5
        self.stat_msg.act_fwd_accel = fwd_accel

        prev_accel = self.imu.linear_acceleration
        prev_fwd_accel = (prev_accel.x ** 2 + prev_accel.y ** 2) ** 0.5
        time_delta = (msg.header.stamp - self.imu.header.stamp).to_sec()
        if not (time_delta > 0):
            raise ValueError("In logging IMU timestamp did not increase")
        self.stat_msg.act_lon_jerk = (accel.x - prev_accel.x) / time_delta
        self.stat_msg.act_lat_jerk = (accel.y - prev_accel.y) / time_delta
        self.stat_msg.act_fwd_jerk = (fwd_accel - prev_fwd_accel) / time_delta

        self.imu = msg

    def publish_stats(self):
        self.stat_msg.header.stamp = rospy.Time.now()
        self.stats_pub.publish(self.stat_msg)
