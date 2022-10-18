from typing import List, Tuple
import rospy

from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    Point,
    Quaternion,
    TransformStamped,
    PointStamped,
)
from tf2_geometry_msgs import do_transform_point
import casadi as ca
from nav_msgs.msg import OccupancyGrid
import numpy as np


class Config(object):
    """
    Converts a dictionary to a nested python object, see https://stackoverflow.com/a/1305682/17778993
    """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Config(b) if isinstance(b, dict) else b)


class NumericParameters(object):

    X0 = (0, 7)

    def __init__(
        self,
        x0: List[float],
        center_a: List[float],
        center_b: List[float],
        left_a: List[float],
        left_b: List[float],
        right_a: List[float],
        right_b: List[float],
        lon_accel: float,
        obs_x: List[float],
        obs_y: List[float],
        obs_rx: List[float],
        obs_ry: List[float],
        obs_theta: List[float],
    ):
        self.x0 = x0
        self.center_a = center_a
        self.center_b = center_b
        self.left_a = left_a
        self.left_b = left_b
        self.right_a = right_a
        self.right_b = right_b
        self.lon_accel = lon_accel
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.obs_rx = obs_rx
        self.obs_ry = obs_ry
        self.obs_theta = obs_theta

        self.init = None

    @property
    def P(self):
        return np.array(
            self.x0
            + self.center_a
            + self.center_b
            + self.left_a
            + self.left_b
            + self.right_a
            + self.right_b
            + [self.lon_accel]
            + self.obs_x
            + self.obs_y
            + self.obs_rx
            + self.obs_ry
            + self.obs_theta
        )


class SymbolicParameters(object):
    def __init__(self, config: Config):
        spline_order = config.spline.order
        num_obs = config.num_obs

        self.x0_s, self.x0_e = NumericParameters.X0

        self.center_a_s = self.x0_e
        self.center_a_e = self.center_a_s + spline_order

        self.center_b_s = self.center_a_e
        self.center_b_e = self.center_b_s + spline_order

        self.left_a_s = self.center_b_e
        self.left_a_e = self.left_a_s + spline_order

        self.left_b_s = self.left_a_e
        self.left_b_e = self.left_b_s + spline_order

        self.right_a_s = self.left_b_e
        self.right_a_e = self.right_a_s + spline_order

        self.right_b_s = self.right_a_e
        self.right_b_e = self.right_b_s + spline_order

        self.lon_accel_e = self.right_b_e

        self.obs_x_s = self.lon_accel_e + 1
        self.obs_x_e = self.obs_x_s + num_obs

        self.obs_y_s = self.obs_x_e
        self.obs_y_e = self.obs_y_s + num_obs

        self.obs_rx_s = self.obs_y_e
        self.obs_rx_e = self.obs_rx_s + num_obs

        self.obs_ry_s = self.obs_rx_e
        self.obs_ry_e = self.obs_ry_s + num_obs

        self.obs_theta_s = self.obs_ry_e
        self.obs_theta_e = self.obs_theta_s + num_obs

        self.sx = ca.SX.sym("P", self.obs_theta_e, 1)


    @property
    def x0(self):
        return self.sx[self.x0_s:self.x0_e]

    @property
    def center_a(self):
        return self.sx[self.center_a_s:self.center_a_e]

    @property
    def center_b(self):
        return self.sx[self.center_b_s:self.center_b_e]

    @property
    def left_a(self):
        return self.sx[self.left_a_s:self.left_a_e]

    @property
    def left_b(self):
        return self.sx[self.left_b_s:self.left_b_e]

    @property
    def right_a(self):
        return self.sx[self.right_a_s:self.right_a_e]

    @property
    def right_b(self):
        return self.sx[self.right_b_s:self.right_b_e]

    @property
    def lon_accel(self):
        return self.sx[self.lon_accel_e]

    @property
    def obs_x(self):
        return self.sx[self.obs_x_s:self.obs_x_e]

    @property
    def obs_y(self):
        return self.sx[self.obs_y_s:self.obs_y_e]


    @property
    def obs_rx(self):
        return self.sx[self.obs_rx_s:self.obs_rx_e]


    @property
    def obs_ry(self):
        return self.sx[self.obs_ry_s:self.obs_ry_e]


    @property
    def obs_theta(self):
        return self.sx[self.obs_theta_s:self.obs_theta_e]



def obs_from_occ(occ_msg: OccupancyGrid) -> Tuple[List[float], List[float]]:
    tf = TransformStamped()
    tf.transform.rotation = occ_msg.info.origin.orientation
    tf.transform.translation.x = occ_msg.info.origin.position.x
    tf.transform.translation.y = occ_msg.info.origin.position.y
    tf.transform.translation.z = occ_msg.info.origin.position.z
    tf.header.frame_id = "odom"
    tf.child_frame_id = "grid_link"

    def cell_to_odom(w, h):
        grid_ps = PointStamped()
        grid_ps.header.frame_id = "grid_link"
        grid_ps.point.x = w * occ_msg.info.resolution + occ_msg.info.resolution / 2
        grid_ps.point.y = h * occ_msg.info.resolution + occ_msg.info.resolution / 2
        return do_transform_point(grid_ps, tf)

    obs_x = []
    obs_y = []
    for width in range(50, occ_msg.info.width):
        for height in range(occ_msg.info.height):
            if occ_msg.data[height * occ_msg.info.width + width] > 0:
                odom_ps = cell_to_odom(width, height)
                obs_x.append(odom_ps.point.x)
                obs_y.append(odom_ps.point.y)
    return obs_x, obs_y


def pub_obs(obs_x: List[float], obs_y: List[float], pub: rospy.Publisher):
    if len(obs_x) == 0:
        return
    marker = Marker()
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 1
    marker.ns = "mpc_obs"
    marker.header.frame_id = "odom"
    marker.color.r = 1.0
    marker.color.a = 1.0
    for ob_x, ob_y in zip(obs_x, obs_y):
        point = Point()
        point.x = ob_x
        point.y = ob_y
        marker.points.append(point)
    pub.publish(marker)


def pub_states(x: ca.DM, pub: rospy.Publisher):
    marker = Marker()
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = 0.4
    marker.scale.y = 0.4
    marker.scale.z = 0.4
    marker.ns = "mpc_predicted"
    marker.header.frame_id = "odom"
    marker.pose.position.z = 1
    marker.color.r = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    for n in range(x.size(1)):
        point = Point()
        point.x = x[n, 0]
        point.y = x[n, 1]
        marker.points.append(point)
    pub.publish(marker)


# Helper function that publishes sphere indicating where the reference point we're starting at is
def pub_ref_sphere(x: float, y: float, pub: rospy.Publisher):
    pt = Point()
    pt.x = x
    pt.y = y
    marker = Marker()
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE
    marker.ns = "path_param_ref"
    marker.header.frame_id = "odom"
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.pose.position = pt
    marker.pose.position.z += 0.5
    marker.color.a = 0.65
    marker.color.r = 1
    marker.color.g = float(215 / 256)
    pub.publish(marker)


# Helper function that creates a quaternion from a yaw
def quaternion_from_yaw(yaw: float):
    ret = Quaternion()
    quaternion = quaternion_from_euler(0, 0, yaw)
    ret.x = quaternion[0]
    ret.y = quaternion[1]
    ret.z = quaternion[2]
    ret.w = quaternion[3]
    return ret


def yaw_from_quaternion(quat: Quaternion):
    return euler_from_quaternion(
        [
            quat.x,
            quat.y,
            quat.z,
            quat.w,
        ]
    )[2]


def yaw_from_odom(odom: Odometry):
    return yaw_from_quaternion(odom.pose.pose.orientation)


def fwd_vel_from_odom(odom: Odometry):
    vel = odom.twist.twist.linear
    return (vel.x ** 2 + vel.y ** 2) ** 0.5


def ref_path(a: ca.SX, b: ca.SX, u: ca.SX):
    """
    Evaluates a spline defined by a and b at location u
    """
    order = a.shape[0] if isinstance(a, ca.SX) else len(a)
    xSpline = sum([a[o] * u ** o for o in range(order)])
    ySpline = sum([b[o] * u ** o for o in range(order)])

    dxSpline = sum([o * a[o] * u ** (o - 1) for o in range(1, order)])
    dySpline = sum([o * b[o] * u ** (o - 1) for o in range(1, order)])
    yawSpline = ca.atan2(dySpline, dxSpline)
    return xSpline, ySpline, yawSpline


def path_param_from_xy(a, b, x, y):
    """
    Splits [0,1] spline parameter range into 100 sections and returns index that is closest to passed in x,y
    """
    if len(a) < 1:
        return 0
    min_dist = np.inf
    min_k = 0
    for k in np.linspace(0, 1, 100):
        [sx, sy, _] = ref_path(a, b, k)
        dist = (x - sx) ** 2 + (y - sy) ** 2
        if dist < min_dist:
            min_dist = dist
            min_k = k
    return min_k


def mvee(points, tol=0.001):
    """
    Function to solve the minimum volume enclosing ellipsis problem (https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python)
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
    c = np.dot(u, points)
    A = (
        np.linalg.inv(
            np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(c, c)
        )
        / d
    )
    return A, c


def _transform(occ_msg):
    tf = TransformStamped()
    tf.transform.rotation = occ_msg.info.origin.orientation
    tf.transform.translation.x = occ_msg.info.origin.position.x
    tf.transform.translation.y = occ_msg.info.origin.position.y
    tf.transform.translation.z = occ_msg.info.origin.position.z
    tf.header.frame_id = "odom"
    tf.child_frame_id = "grid_link"
    return tf


# Functon to transfer cells from occ grid to odom frame
def cell_to_odom_point(occ_msg, w: int, h: int):
    grid_ps = PointStamped()
    grid_ps.header.frame_id = "grid_link"
    grid_ps.point.x = h * occ_msg.info.resolution + occ_msg.info.resolution / 2
    grid_ps.point.y = w * occ_msg.info.resolution + occ_msg.info.resolution / 2
    tf = _transform(occ_msg)
    odom_p = do_transform_point(grid_ps, tf)
    return np.array([odom_p.point.x, odom_p.point.y])


def cells_to_points(occ_msg, cells: np.array) -> np.array:
    points = np.zeros_like(cells, dtype=float)
    for i, cell in enumerate(cells):
        points[i] = cell_to_odom_point(occ_msg, cell[0], cell[1])
    return points


def center_to_outer_points(occ_msg: OccupancyGrid, points):
    yaw = yaw_from_quaternion(occ_msg.info.origin.orientation)
    outer_points = np.zeros((4 * points.shape[0], 2))
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rad = occ_msg.info.resolution / 2
    tl = np.matmul(rot, np.array([rad, rad]))
    tr = np.matmul(rot, np.array([rad, -rad]))
    bl = np.matmul(rot, np.array([-rad, rad]))
    br = np.matmul(rot, np.array([-rad, -rad]))
    for i, point in enumerate(points):
        start_i = i * 4
        outer_points[start_i + 0] = point + tl
        outer_points[start_i + 1] = point + tr
        outer_points[start_i + 2] = point + bl
        outer_points[start_i + 3] = point + br
    return outer_points


def pub_ellipses(marker_pub, obs_x, obs_y, obs_rx, obs_ry, obs_theta):
    """
    Helper function that publishes obs which the MPC formulation knows about (post-filtering)
    """
    if len(obs_x) == 0:
        return
    for i in range(len(obs_x)):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.ns = "%s %d" % ("mpc_obs", i)
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Time(1.0)
        marker.pose.position.x = obs_x[i]
        marker.pose.position.y = obs_y[i]
        marker.pose.position.z = 0
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = quaternion_from_euler(0, 0, obs_theta[i])
        marker.scale.x = 2 * obs_rx[i]
        marker.scale.y = 2 * obs_ry[i]
        marker.scale.z = 0.2
        marker.color.a = 0.75
        marker.color.r = 1.0
        marker_pub.publish(marker)


def fill_occ_grid(occ_msg: OccupancyGrid):
    """
    Returns: A HxW np.array with 1s at occupied cells
    and 0s else where
    """
    w, h = occ_msg.info.width, occ_msg.info.height
    grid = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            if occ_msg.data[r * w + c] > 0:
                grid[r][c] = 1
    return grid
