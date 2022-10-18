#!/usr/bin/env python3

from copy import deepcopy
from functools import partial
import multiprocessing as mp
import multiprocessing.connection as mp_conn
from casadi import *
from utils import Config, NumericParameters, SymbolicParameters, ref_path
from typing import Callable, List, Tuple


class Solver(object):
    def __init__(self, config: Config):
        self.construct_solver(config)

    def construct_solver(self, config: Config):
        num_obs = config.num_obs

        # +++ Optimal Control Problem Formulation +++

        # Vehicle State Vector

        # Vehicle position along fixed frame x-axis
        vehicle_state_x = SX.sym("vehicle_state_x")
        # Vehicle position along fixed frame y-axis
        vehicle_state_y = SX.sym("vehicle_state_y")
        # Vehicle orientation angle, RHS off x-axis
        vehicle_state_theta = SX.sym("vehicle_state_theta")
        # Virtual state that represents path parameter
        vehicle_state_pthprm = SX.sym("vehicle_state_pthprm")
        # Vehicle speed (forward velocity)
        vehicle_state_vel = SX.sym("vehicle_state_vel")
        # Vehicle steering angle (virtual central wheel)
        vehicle_state_steer = SX.sym("vehicle_state_steer")
        # Vehicle body frame longitudinal velocity
        vehicle_state_lon_vel = SX.sym("vehicle_state_lon_vel")
        # Vehicle body frame lateral velocity
        vehicle_state_lat_accel = SX.sym("vehicle_state_lat_accel")

        vehicle_state_vector = vertcat(
            vehicle_state_x,
            vehicle_state_y,
            vehicle_state_theta,
            vehicle_state_lon_vel,
            vehicle_state_vel,
            vehicle_state_pthprm,
            vehicle_state_steer,
        )
        vehicle_state_x_i = 0
        vehicle_state_y_i = 1
        vehicle_state_theta_i = 2
        vehicle_state_lon_vel_i = 3
        vehicle_state_vel_i = 4
        vehicle_state_pthprm_i = 5
        vehicle_state_steer_i = 6

        vehicle_state_n = SX.size(vehicle_state_vector)
        self.vehicle_state_n = vehicle_state_n[0]

        # Control Vector

        # System Control - Forward Speed
        control_accel = SX.sym("_control_accel")
        # System Control - Steering angle
        control_steer = SX.sym("control_steer")
        # System Control - Virtual Input
        control_vir = SX.sym("control_vir")

        control_vector = vertcat(control_accel, control_steer, control_vir)

        control_accel_i = 0
        control_steer_i = 1
        control_vir_i = 2

        control_n = SX.size(control_vector)
        self.control_n = control_n[0]

        # System Model

        # Formulas for kinematic bicycle model, see https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357

        # Second parameter to atan is fraction of CoG to back of vehicle
        # Currently assumed to be at center of vehicle
        beta = atan2(tan(control_steer), 2)
        inertial_x_velocity = vehicle_state_vel * cos(beta + vehicle_state_theta)
        inertial_y_velocity = vehicle_state_vel * sin(beta + vehicle_state_theta)
        yaw_rate = (vehicle_state_vel * cos(beta) * tan(control_steer)) / config.body.L
        lon_accel = control_accel
        accel = control_accel

        # The right hand side of the dynamics equation, i.e. the derivative of the state vector
        rhs = vertcat(
            inertial_x_velocity,  # Derivative of _vehicle_state_x
            inertial_y_velocity,  # Derivative of _vehicle_state_y
            yaw_rate,  # Derivative of _vehicle_state_theta
            lon_accel,  # Derivative of _vehicle_state_lon_vel
            accel,  # Derivative of _vehicle_state_vel
            control_vir,  # Derivative of _vehicle_state_pthprm
            control_steer,  # NOT Derivative of _vehicle_state_steer, steering is set directly
        )

        # Nonlinear mapping Function f(x,u)
        f = Function("f", [vehicle_state_vector, control_vector], [rhs])

        # +++ NLP Formulation +++

        # NLP Variables - See Mohamed W. Mehrez playlist for details: https://www.youtube.com/watch?v=RrnkPrcpyEA&list=PLK8squHT_Uzej3UCUHjtOtm5X7pMFSgAL

        # Control decision variables NLP

        U = SX.sym("U", control_n[0], config.opt.N)
        # State decision variables of NLP
        X = SX.sym("X", vehicle_state_n[0], (config.opt.N + 1))
        # Parameters of NLP, see definition below
        P = SymbolicParameters(config)

        obj = 0  # Objective function that we will be building up
        g = SX([])  # Functional constraints vector that we will be adding to
        lbg = []  # Decision variables lower bounds
        ubg = []  # Decision variables upper bounds

        # Current steering angle
        steer_old = P.x0[vehicle_state_steer_i]
        reference_speed = config.body.desired_fwd_vel
        ref_spline_a = P.center_a
        ref_spline_b = P.center_b
        left_spline_a = P.left_a
        left_spline_b = P.left_b
        right_spline_a = P.right_a
        right_spline_b = P.right_b
        # Current longitudinal accel
        lon_accel_old = P.lon_accel
        # x-axis positions of obstacles
        x_obs = P.obs_x
        # y-axis positions of obstacles
        y_obs = P.obs_y
        rx_obs = P.obs_rx
        ry_obs = P.obs_ry
        theta_obs = P.obs_theta

        st = X[:, 0]  # initial state
        # Enforce that initial state decisions variables are equal to the vehicle's measured state passed in the parameters
        g = vertcat(g, st[:] - P.x0)  # initial condition constraints
        lbg = vertcat(lbg, DM.zeros(vehicle_state_n[0], 1))
        ubg = vertcat(ubg, DM.zeros(vehicle_state_n[0], 1))

        for k in range(config.opt.N):
            # State decisions variables at step k in the precition horizon
            st = X[:, k]
            # Control decisions variables at step k in the precition horizon
            con = U[:, k]

            # Get symbolic representation of minimum distance from obstacle parameters
            min_dist = 100000
            for i in range(num_obs):
                # get rotation matrix back from ellipsis angle, here implemented is the inverse is here
                rot_theta = blockcat(
                    [
                        [np.cos(theta_obs[i]), np.sin(theta_obs[i])],
                        [-np.sin(theta_obs[i]), np.cos(theta_obs[i])],
                    ]
                )

                footprint = config.body.footprint
                # ego representation as 3 circles
                center_1 = vertcat(
                    st[0] + footprint.circle_placement * cos(st[2]),
                    st[1] + footprint.circle_placement * sin(st[2]),
                )
                center_2 = vertcat(st[0], st[1])
                center_3 = vertcat(
                    st[0] - footprint.circle_placement * cos(st[2]),
                    st[1] - footprint.circle_placement * sin(st[2]),
                )

                # rotate ego and obstacle pos so that ellipsis axis are allignt with x and y axis to calculate if ego center is in ellipsis
                ego_pos_rot_1 = mtimes(rot_theta, center_1)
                ego_pos_rot_2 = mtimes(rot_theta, center_2)
                ego_pos_rot_3 = mtimes(rot_theta, center_3)
                ellipsis_pos = vertcat(x_obs[i], y_obs[i])
                obs_pos_rot = mtimes(rot_theta, ellipsis_pos)

                # add safetydistance to axis', to take care of the fact, that just ego center is concidered
                safetyDistance = footprint.circle_radius + footprint.safety
                rx_axis = rx_obs[i] + safetyDistance
                ry_axis = ry_obs[i] + safetyDistance

                # calculate if circle centerpoints are in ellipsis plus radius
                c1 = (ego_pos_rot_1[0] - obs_pos_rot[0]) ** 2 / rx_axis ** 2 + (
                    ego_pos_rot_1[1] - obs_pos_rot[1]
                ) ** 2 / ry_axis ** 2
                c2 = (ego_pos_rot_2[0] - obs_pos_rot[0]) ** 2 / rx_axis ** 2 + (
                    ego_pos_rot_2[1] - obs_pos_rot[1]
                ) ** 2 / ry_axis ** 2
                c3 = (ego_pos_rot_3[0] - obs_pos_rot[0]) ** 2 / rx_axis ** 2 + (
                    ego_pos_rot_3[1] - obs_pos_rot[1]
                ) ** 2 / ry_axis ** 2

                c = fmin(c1, fmin(c2, c3))
                min_dist = fmin(min_dist, c)

            # Enforce that ego center is outside all obstacles
            g = vertcat(g, min_dist)
            lbg = vertcat(lbg, 1)
            ubg = vertcat(ubg, inf)

            # Get symbolic representation of reference point
            u = st[vehicle_state_pthprm_i]

            bound_c = config.boundary
            # Create halfplanes via the vectors that point from path_param along
            # the bound to the ego position. Enforce that the ego position is inside
            # those halfplanes
            if bound_c.strat == "halfplane":
                st_x = st[vehicle_state_x_i]
                st_y = st[vehicle_state_y_i]
                x0, y0, _ = ref_path(
                    left_spline_a, left_spline_b, st[vehicle_state_pthprm_i]
                )
                x1, y1, _ = ref_path(
                    right_spline_a, right_spline_b, st[vehicle_state_pthprm_i]
                )
                left_const = (
                    (x1 - x0) * st_x
                    - (x1 - x0) * x0
                    + (y1 - y0) * st_y
                    - (y1 - y0) * y0
                )
                x1, y1, _ = ref_path(
                    left_spline_a, left_spline_b, st[vehicle_state_pthprm_i]
                )
                x0, y0, _ = ref_path(
                    right_spline_a, right_spline_b, st[vehicle_state_pthprm_i]
                )
                right_const = (
                    (x1 - x0) * st_x
                    - (x1 - x0) * x0
                    + (y1 - y0) * st_y
                    - (y1 - y0) * y0
                )
                g = vertcat(g, left_const)
                lbg = vertcat(lbg, 0)
                ubg = vertcat(ubg, inf)
                g = vertcat(g, right_const)
                lbg = vertcat(lbg, 0)
                ubg = vertcat(ubg, inf)

            # Get a directional vector for each bound using a small lookahead
            # and enforce that the ego is on the correct side of that vector
            elif bound_c.strat == "side":
                la = bound_c.side.lookahead
                pad = bound_c.side.padding
                left_x, left_y, _ = ref_path(P.left_a, P.left_b, u)
                left_x1, left_y1, _ = ref_path(P.left_a, P.left_b, u + la)
                left_side = (st[0] - left_x) * (left_y1 - left_y) - (st[1] - left_y) * (
                    left_x1 - left_x
                )
                right_x, right_y, _ = ref_path(P.right_a, P.right_b, u)
                right_x1, right_y1, _ = ref_path(P.right_a, P.right_b, u + la)
                right_side = (st[0] - right_x) * (right_y1 - right_y) - (
                    st[1] - right_y
                ) * (right_x1 - right_x)
                g = vertcat(g, left_side)
                lbg = vertcat(lbg, pad)
                ubg = vertcat(ubg, inf)
                g = vertcat(g, right_side)
                lbg = vertcat(lbg, -inf)
                ubg = vertcat(ubg, -pad)

            elif bound_c.strat != "none":
                raise ValueError(f"Unrecognized boundary.strat '{bound_c.strat}'")

            x_ref, y_ref, yaw_ref = ref_path(P.center_a, P.center_b, u)
            # No idea what this calcuation is and we don't currently use it. Somehow characterized yaw error
            objYaw = (
                1
                - sin(st[vehicle_state_theta_i]) * sin(yaw_ref)
                - cos(st[vehicle_state_theta_i]) * cos(yaw_ref)
            )
            # Squared distance to reference point
            objPos = (x_ref - st[0]) ** 2 + (y_ref - st[1]) ** 2
            # Squared parametric distance to route completion
            objPathParam = (st[vehicle_state_pthprm_i] - 1) ** 2
            # Squared velocity error
            objRefVel = (st[vehicle_state_vel_i] - reference_speed) ** 2

            # Calculations for higher order body frame position derivatives as shown in https://www.sciencedirect.com/science/article/pii/S2405896319304185
            # Note that these metrics needs to be maintained as per SAE comfortable driving guidelines
            lon_jerk = (con[control_accel_i] - lon_accel_old) / config.opt.T
            lon_accel_old = con[control_accel_i]
            lat_accel = (
                (st[vehicle_state_lon_vel_i] ** 2) * con[control_steer_i]
            ) / config.body.L
            steer_delta = (con[control_steer_i] - steer_old) / config.opt.T
            lat_jerk = (
                2 * con[control_accel_i] * con[control_steer_i]
                + st[vehicle_state_lon_vel_i] * steer_delta
            ) * (st[vehicle_state_lon_vel_i] / config.body.L)
            objLatJerk = lat_jerk ** 2

            # side_pen = if_else(
            #     side < 0, 2 ** (sqrt(objPos) - 10), 2 ** (sqrt(objPos) - 1)
            # )
            # These 3 lines of code take up most of the time tuning weights, find out how to automate this process?
            w_obs = config.opt.weights.obstacle
            obj_obs = (
                w_obs.pos * objPos
                + w_obs.path_param * objPathParam
                + w_obs.ref_vel * objRefVel
                # How we penalize the state if it is decided to be close to an obstacle
                #   note that we add a repulsive field term around the obstacle, see plot here: https://www.wolframalpha.com/input/?i=plot+1%2F%28%281%2F4%29*x%29+from+0+to+5
                + w_obs.obs_dist * (1 / min_dist)
                + w_obs.lat_jerk * objLatJerk
                # + side_pen
            )
            w_free = config.opt.weights.free
            obj_free = (
                w_free.pos * objPos
                + w_free.path_param * objPathParam
                + w_free.ref_vel * objRefVel
                + w_free.lat_jerk * objLatJerk
            )
            non_stop_obj = if_else(
                # Switch how we penailize the state in step k depending on whether or not it is decided to be close to an obstacle
                min_dist < config.opt.obstacle_switch_dist,
                obj_obs,
                # How we penalize the state if it is not close to an obstacle
                obj_free,
                True,
            )

            # If the state is 70% done with the local route, stop by penalizing our speed and travel forward
            w_stop = config.opt.weights.stop
            obj_stop = (
                w_stop.pos * objPos
                + fabs(st[vehicle_state_vel_i])
                + con[control_accel_i]
            )
            obj += if_else(
                st[vehicle_state_pthprm_i] > config.opt.stop_switch_frac,
                obj_stop,
                non_stop_obj,
            )
            # self._obj += non_stop_obj
            # end_speed = if_else(self._st[self._vehicle_state_pthprm_i] > 0.9, self._st[self._vehicle_state_vel_i], 0)
            # self._g = vertcat(self._g, end_speed)
            # self._lbg = vertcat(self._lbg, 0)
            # self._ubg = vertcat(self._ubg, 0)

            # Bounds on body frame metrics as specififed by SAE
            bounds = config.body.bounds
            g = vertcat(g, lat_accel)
            lbg = vertcat(lbg, DM(bounds.lat_accel_l))
            ubg = vertcat(ubg, DM(bounds.lat_accel_u))
            g = vertcat(g, lon_jerk)
            lbg = vertcat(lbg, DM(bounds.lon_jerk_l))
            ubg = vertcat(ubg, DM(bounds.lon_jerk_u))

            # Calculate the next step state by solving the dynamics function ODE using RK4: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
            k1 = f(st, con)
            k2 = f(st + k1 * config.opt.T / 2, con)
            k3 = f(st + k2 * config.opt.T / 2, con)
            k4 = f(st + k3 * config.opt.T, con)
            gradientRK4 = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Note that the steering state component is set directly, so ODE is only solved for the other components
            st_next_RK4 = (
                st[0:vehicle_state_steer_i]
                + config.opt.T * gradientRK4[0:vehicle_state_steer_i]
            )
            # And the steering state is taken directly from the f function
            st_next_RK4 = vertcat(st_next_RK4, k1[vehicle_state_steer_i])
            # Now, we just enforce that the next state adheres to the dynaimcs by enforcing that their difference is zero
            #   This programming of the dynamics is called multiple shooting
            st_next = X[:, k + 1]
            g = vertcat(g, st_next - st_next_RK4)
            lbg = vertcat(lbg, DM.zeros(vehicle_state_n[0], 1))
            ubg = vertcat(ubg, DM.zeros(vehicle_state_n[0], 1))

        # Now we populate the lower and upper bounds for every decision variable (state and control decision variables)
        N = config.opt.N
        lbx = DM.zeros(vehicle_state_n[0] * (N + 1) + control_n[0] * N, 1)
        ubx = DM.zeros(vehicle_state_n[0] * (N + 1) + control_n[0] * N, 1)

        # constraint on X position
        end_of_state_vars = vehicle_state_n[0] * (N + 1)
        lbx[vehicle_state_x_i : end_of_state_vars : vehicle_state_n[0], 0] = -inf
        ubx[vehicle_state_x_i : end_of_state_vars : vehicle_state_n[0], 0] = inf

        # constraint on Y position
        lbx[vehicle_state_y_i : end_of_state_vars : vehicle_state_n[0], 0] = -inf
        ubx[vehicle_state_y_i : end_of_state_vars : vehicle_state_n[0], 0] = inf

        # constraint on yaw angle
        lbx[
            vehicle_state_theta_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = -inf
        ubx[
            vehicle_state_theta_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = inf

        # constraint on lon velocity (state)
        lbx[
            vehicle_state_lon_vel_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = bounds.vel_l
        ubx[
            vehicle_state_lon_vel_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = bounds.vel_u

        # constraint on fwd velocity (state)
        lbx[
            vehicle_state_vel_i : end_of_state_vars : vehicle_state_n[0], 0
        ] = bounds.vel_l
        ubx[
            vehicle_state_vel_i : end_of_state_vars : vehicle_state_n[0], 0
        ] = bounds.vel_u

        # constraint on virtual state
        lbx[
            vehicle_state_pthprm_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = 0
        ubx[
            vehicle_state_pthprm_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = 1

        # constraint on steering angle (state) (0->a,5)
        lbx[
            vehicle_state_steer_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = bounds.steer_l
        ubx[
            vehicle_state_steer_i : end_of_state_vars : vehicle_state_n[0],
            0,
        ] = bounds.steer_u

        # constraint on accel input (a->end)
        lbx[
            end_of_state_vars
            + control_accel_i : vehicle_state_n[0] * (N + 1)
            + control_n[0] * N : control_n[0],
            0,
        ] = bounds.lon_accel_l
        ubx[
            end_of_state_vars
            + control_accel_i : vehicle_state_n[0] * (N + 1)
            + control_n[0] * N : control_n[0],
            0,
        ] = bounds.lon_accel_u

        # constraint on steering input (a->end)
        lbx[
            end_of_state_vars
            + control_steer_i : vehicle_state_n[0] * (N + 1)
            + control_n[0] * N : control_n[0],
            0,
        ] = bounds.steer_l
        ubx[
            end_of_state_vars
            + control_steer_i : vehicle_state_n[0] * (N + 1)
            + control_n[0] * N : control_n[0],
            0,
        ] = bounds.steer_u

        # constraint on virtual input (a->end)
        lbx[
            end_of_state_vars
            + control_vir_i : vehicle_state_n[0] * (N + 1)
            + control_n[0] * N : control_n[0],
            0,
        ] = 0
        ubx[
            end_of_state_vars
            + control_vir_i : vehicle_state_n[0] * (N + 1)
            + control_n[0] * N : control_n[0],
            0,
        ] = 1

        # Lastly, we just pack up everything we setup into a NLP, see https://web.casadi.org/docs/#nonlinear-programming

        # Make the decision variables a column vector
        opt_variables = reshape(X, vehicle_state_n[0] * (N + 1), 1)
        opt_variables = vertcat(opt_variables, reshape(U, control_n[0] * N, 1))

        nlp_prob = {
            "f": obj,
            "x": opt_variables,
            "g": g,
            "p": P.sx,
        }

        # Solver options
        opts = {
            "ipopt.max_iter": config.opt.max_iters,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        self.solver = partial(
            nlpsol("solver", "ipopt", nlp_prob, opts),
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
        )

    def __call__(self, init, p):
        return self.solver(x0=init, p=p)


class SolverContainer(object):
    """
    Constructs a solver for each possible number of obstacles
    """

    def __init__(self, config: Config):
        self._solvers = {}
        self.max_num_obs = config.max_num_obs
        for num_obs in range(self.max_num_obs + 1):
            num_obs_config = deepcopy(config)
            num_obs_config.num_obs = num_obs
            self._solvers[num_obs] = Solver(num_obs_config)

    def __call__(self, p: NumericParameters):
        num_obs = len(p.obs_x)
        if num_obs > self.max_num_obs:
            raise ValueError(f"Observed {num_obs} obstacles, max is {self.max_num_obs}")
        return self._solvers[num_obs](init=p.init, p=p.P)


class SolverProc(mp.Process):
    TERM = "term"

    def __init__(self, solver: Callable, conn: mp_conn.Connection):
        super(SolverProc, self).__init__()
        self.conn = conn
        self.solver = solver

    def run(self):
        while True:
            job = self.conn.recv()
            if job == self.TERM:
                self.conn.send(self.TERM)
                return
            # TODO: Add profiling for this call
            self.conn.send(self.solver(job))


class ParallelSolver(object):
    """
    Runs multiple solvers in parallel and synchronizes the best
        solution between them
    """

    def __init__(self, config: Config):
        config.opt.max_iters = config.opt.online_max_iters
        self._online_solver = SolverContainer(config)

        self.N = config.opt.N
        self.vehicle_state_n = self._online_solver._solvers[0].vehicle_state_n
        self.control_n = self._online_solver._solvers[0].control_n

        self._exploring = not config.parallel.active
        self.explore_sol = None
        self.explore_copy = config.parallel.copy_warm
        self.online_sol = None

        self.prev_yaw = 0

        explore_config = deepcopy(config)
        explore_config.opt.max_iters = config.opt.explore_max_iters
        self.conn, conn = mp.Pipe()
        self.proc = SolverProc(
            partial(
                SolverContainer(explore_config),
            ),
            conn,
        )
        self.proc.start()

    def _get_init(self, yaw: float) -> Tuple[DM, DM]:
        """
        Checks to see if the exploring solver has a solution. If it does,
            and solution is better than the current online solution, 
            returns the exploring solution. Otherwise, returns the online solution.

        Params: 
            - yaw: Current yaw angle of the ego

        Returns:
            - x_warm: State decision variables to initialize the iterative solver with
            - u_warm: Control decision variables to initialize the iterative solver with
        """
        if self.conn.poll():
            self.explore_sol = self.conn.recv()
            self._exploring = False
        if self.online_sol is None:
            return DM.zeros(self.vehicle_state_n * (self.N + 1)), DM.zeros(
                self.control_n * self.N
            )
        if self.explore_sol is None or self.online_sol["f"] < self.explore_sol["f"]:
            sol = self.online_sol
        else:
            sol = self.explore_sol
            self.explore_sol = None

        N = self.N
        x = sol["x"][: self.vehicle_state_n * (N + 1)]
        x_warm = DM.zeros(self.vehicle_state_n * (self.N + 1))
        # Set first N states based on moving prev solution forward by 1 step
        x_warm[: self.vehicle_state_n * N] = x[
            self.vehicle_state_n : self.vehicle_state_n * (N + 1)
        ]
        # Set last state to last state of prev solution
        x_warm[self.vehicle_state_n * N :] = x[self.vehicle_state_n * N :]

        # If odom switched yaw sign, we also need to switch initial
        # optimization variable assignments for yaw, or else solver
        # will have a hard time numerically
        if abs(yaw - self.prev_yaw) > 6:
            for n in range(N + 1):
                x_warm[n * self.vehicle_state_n + 2] *= -1

        
        u = sol["x"][self.vehicle_state_n * (N + 1) :]
        u_warm = DM.zeros(self.control_n * self.N)
        # Set first N-1 controls based on moving prev solution forward by 1 step
        u_warm[: self.control_n * (N - 1)] = u[self.control_n : self.control_n * N]
        # Set last control to last control of prev solution
        u_warm[self.control_n * (N - 1) :] = u[self.control_n * (N - 1) :]

        self.prev_yaw = yaw

        return x_warm, u_warm

    def terminate(self):
        self.conn.send(SolverProc.TERM)
        while self.conn.recv() != SolverProc.TERM:
            continue
        self.proc.terminate()

    def __call__(self, p: NumericParameters):
        x_warm, u_warm = self._get_init(yaw=p.x0[2])
        if not self._exploring:
            self._exploring = True
            x_warm_explore = deepcopy(x_warm)
            u_warm_explore = deepcopy(u_warm)
            x_warm_explore[
                self.explore_copy
                * self.vehicle_state_n : self.vehicle_state_n
                * (self.N + 1)
            ] = 0
            u_warm_explore[
                self.explore_copy * self.control_n : self.control_n * self.N
            ] = 0
            explore_p = deepcopy(p)
            explore_p.init = vertcat(x_warm_explore, u_warm_explore)
            self.conn.send(explore_p)

        p.init = vertcat(x_warm, u_warm)
        self.online_sol = self._online_solver(p)
        return self.online_sol["x"]
