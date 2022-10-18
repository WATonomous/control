#!/usr/bin/env python3

import rospy
from controller import Controller
from utils import Config


if __name__ == "__main__":
    rospy.init_node("mpc_main", log_level=rospy.DEBUG)
    config = Config(rospy.get_param("/mpc"))
    Controller(config)
    rospy.loginfo("mpc_main started")
    rospy.spin()
