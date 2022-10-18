#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
from itertools import product

from nav_msgs.msg import OccupancyGrid
import rospy
from utils import (
    Config,
    mvee,
    cells_to_points,
    center_to_outer_points,
    pub_ellipses,
    fill_occ_grid,
)


# Class to execute the obstacle callback by clustering the obstacles, enclosing them by an ellipsis and publishing them to the ROS network
class OccupancyCluster:
    def __init__(self, marker_pub, config: Config):
        self._marker_pub = marker_pub
        self.obs_x = []
        self.obs_y = []
        self.obs_rx = []
        self.obs_ry = []
        self.obs_theta = []
        self.rad = config.occ_cluster.neighbor_rad

        rospy.Subscriber(
            "/path_planning/filtered_occ", OccupancyGrid, self.callback, queue_size=1
        )

    def regionGrow(self, r: int, c: int) -> np.array:
        """
        Grows a connected region (where connectivity is determined by self.rad)
        See https://en.wikipedia.org/wiki/Region_growing

        Returns: [N, 2] where N is the number of cells in the cluster
        """
        seedList = []
        cluster_r = [r]
        cluster_c = [c]
        seedList.append((r, c))
        # Generates a radius pattern for neighbor selection, e.g. for rad=1:
        # (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
        neighbors = [
            c
            for c in product(range(-self.rad, self.rad + 1), repeat=2)
            if c[0] != 0 or c[1] != 0
        ]
        while len(seedList) > 0:
            currentCell = seedList.pop(0)
            self.visited[currentCell[0]][currentCell[1]] = 1
            for neigh in neighbors:
                tmp_r = currentCell[0] + neigh[0]
                tmp_c = currentCell[1] + neigh[1]
                if (
                    tmp_r < 0
                    or tmp_c < 0
                    or tmp_r >= self.grid_height
                    or tmp_c >= self.grid_width
                ):
                    continue
                if self.grid[tmp_r][tmp_c] == 1 and self.visited[tmp_r][tmp_c] == 0:
                    seedList.append((tmp_r, tmp_c))
                    cluster_r.append(tmp_r)
                    cluster_c.append(tmp_c)
                    self.visited[tmp_r][tmp_c] = 1
        return np.array([cluster_r, cluster_c]).T

    def callback(self, occ: OccupancyGrid):
        self.occ_msg = occ
        self.grid_width = self.occ_msg.info.width
        self.grid_height = self.occ_msg.info.height
        self.grid = fill_occ_grid(occ)
        self.visited = np.zeros((self.grid_height, self.grid_width))

        obs_x = []
        obs_y = []
        obs_rx = []
        obs_ry = []
        obs_theta = []
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                # if not already concidered in any cluster, then build new cluster with this cell as initial seed
                if self.visited[r][c] == 0 and self.grid[r][c] == 1:
                    cluster_points = cells_to_points(
                        self.occ_msg, self.regionGrow(r, c)
                    )
                    outer_points = center_to_outer_points(self.occ_msg, cluster_points)
                    A, centroid = mvee(outer_points)
                    _, D, _ = la.svd(A)
                    rx, ry = 1.0 / np.sqrt(D)

                    # angle in rad, counterclockwise, (-pi/2, pi/2)
                    theta = 0.5 * np.arctan2((2 * A[0, 1]), (A[0, 0] - A[1, 1]))

                    obs_x.append(centroid[0])
                    obs_y.append(centroid[1])
                    obs_rx.append(rx)
                    obs_ry.append(ry)
                    obs_theta.append(theta)

        self.obs_x = obs_x
        self.obs_y = obs_y
        self.obs_rx = obs_rx
        self.obs_ry = obs_ry
        self.obs_theta = obs_theta
        pub_ellipses(self._marker_pub, obs_x, obs_y, obs_rx, obs_ry, obs_theta)
