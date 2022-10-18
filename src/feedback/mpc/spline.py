#!/usr/bin/env python3
from typing import List
import numpy as np
from utils import Config
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from casadi import SX, DM, nlpsol, inf, vertcat
from std_msgs.msg import ColorRGBA
from path_planning_msgs.msg import ReferenceLine, ReferenceSpline


class SplineFinder:
    def __init__(self, config: Config):
        self.order = config.spline.order
        self.ref_pub = rospy.Publisher(
            f"/path_planning/mpc/reference", ReferenceSpline, queue_size=1
        )
        self.viz_pub = rospy.Publisher(
            f"/path_planning/mpc/reference_viz", Marker, queue_size=10
        )
        self.ref_sub = rospy.Subscriber(
            "/path_planning/ref_line",
            ReferenceLine,
            self.ref_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

    def gen_spline(self, fit: List[Point]):
        a = SX.sym("a", self.order, 1)
        b = SX.sym("b", self.order, 1)
        obj = 0  # Objective function (summation)
        for u, p in zip(np.linspace(0, 1, len(fit)), fit):
            obj += (sum([a[o] * u ** o for o in range(self.order)]) - p.x) ** 2
            obj += (sum([b[o] * u ** o for o in range(self.order)]) - p.y) ** 2

        nlp_prob = {
            "f": obj,
            "x": vertcat(a, b),
            "g": SX([]),
            "p": SX([]),
        }
        # Solver options
        opts = {
            "ipopt.max_iter": 500,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-10,
            "ipopt.acceptable_obj_change_tol": 1e-8,
        }
        solver = nlpsol("solver", "ipopt", nlp_prob, opts)
        # Constraints initialization:
        sol = solver(
            x0=DM.ones(self.order * 2),
            lbx=-inf * DM.ones(self.order * 2),
            ubx=inf * DM.ones(self.order * 2),
            lbg=[],
            ubg=[],
            p=[],
        )
        a_sol = sol["x"][0:self.order]
        a_sol = [a_sol[i].__float__() for i in range(self.order)]
        b_sol = sol["x"][self.order:self.order * 2]
        b_sol = [b_sol[i].__float__() for i in range(self.order)]
        return a_sol, b_sol


    def vis_spline(self, a, b, c, id):
        marker = Marker()
        marker.ns = f"spline_points_{id}"
        marker.id = id
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.header.frame_id = "odom"
        size = 0.2
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size
        marker.pose.position.z += 0.5
        marker.color = c
        marker.lifetime = rospy.Duration(5)
        for u in np.linspace(0, 1, num=100):
            pt = Point()
            pt.x = sum([a_o * u ** o for o, a_o in enumerate(a)])
            pt.y = sum([b_o * u ** o for o, b_o in enumerate(b)])
            marker.points.append(pt)

        self.viz_pub.publish(marker)

    def ref_callback(self, ref_msg: ReferenceLine):
        ref = ref_msg.ref_line
        left = ref_msg.left_bound
        right = ref_msg.right_bound
        s = ReferenceSpline()
        if len(ref) == 0:
            self.ref_pub.publish(s)
            return
        s.ref_a, s.ref_b = self.gen_spline(ref)
        s.left_a, s.left_b = self.gen_spline(left)
        s.right_a, s.right_b = self.gen_spline(right)
        self.ref_pub.publish(s)
        c = ColorRGBA()
        c.a = 1
        c.b = 1
        self.vis_spline(s.ref_a, s.ref_b, c, 1)
        c.r = 1
        self.vis_spline(s.left_a, s.left_b, c, 2)
        c.r = 0
        c.g = 1
        self.vis_spline(s.right_a, s.right_b, c, 3)

def main():
    rospy.init_node("mpc_spline", log_level=rospy.INFO)
    rospy.loginfo("mpc_spline started")
    config = Config(rospy.get_param("/mpc"))
    SplineFinder(config)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
