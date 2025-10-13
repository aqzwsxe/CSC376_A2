import copy
from os import error
import random

import roboticstoolbox as rtb
from spatialmath import SE3, SO3, Twist3
from math import pi
import numpy as np
import pandas as pd

# Panda parameters from https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
robot = rtb.models.DH.Panda()
# robot = rtb.models.DH.Panda()
# robot.tool = SE3()

def matrix_log_se3(T):
    return T.logm()

def matrix_exp_se3(S):
    return SE3.Exp(S)

def getHomeConfiguration(robot: rtb.DHRobot) -> SE3:
    original_tool = robot.tool
    robot.tool = SE3()
    M = robot.fkine(robot.qz)
    robot.tool = original_tool # Restore the tool definition
    return M


def getSpaceTwists(robot: rtb.DHRobot) -> np.ndarray:
    S_vectors = []

    T_prev = SE3()

    for i, link in enumerate(robot.links):
        R_prev = T_prev.R
        omega = R_prev @ np.array([0, 0, 1])

        q = T_prev.t

        v = -np.cross(omega, q)

        twist_vector = np.concatenate((v, omega))
        S_vectors.append(twist_vector)
        T_link = link.A(0)
        T_prev = T_prev * T_link
    return np.array(S_vectors).T


def forwardKinematicsPoESpace(q):
    q = np.asarray(q)
    if q.shape != (7,):
        raise ValueError("q must be a 7-element list or array.")

    M = getHomeConfiguration(robot)

    S_matrix = getSpaceTwists(robot)
    S = [Twist3(S_matrix[:, i]) for i in range(7)]

    T = M.copy()  # Initialize with M

    for i in range(6, -1, -1):
        # The exponential map: expm(hat(Si) * qi)
        exp_Si_qi = S[i].exp(q[i])

        # Left multiplication: T = exp_Si_qi * T
        T = exp_Si_qi * T

    return T

def forwardKinematicsPoEBody(q):
    q = np.asarray(q)
    if q.shape != (7,):
        raise ValueError("q must be a 7-element list or array.")

    M = getHomeConfiguration(robot)
    S_matrix = getSpaceTwists(robot)
    M_inv = M.inv().A

    B_matrix = []
    for i in range(7):
        # Adjoint transformation of the twist S_i
        Ad_M_inv = SE3(M_inv).Ad()
        B_i = Ad_M_inv @ S_matrix[:, i]
        B_matrix.append(B_i)
    B_matrix = np.array(B_matrix).T
    B = [Twist3(B_matrix[:, i]) for i in range(7)]
    T = M.copy()  # Initialize with M
    for i in range(7):
        exp_Bi_qi = B[i].exp(q[i])
        T = T * exp_Bi_qi

    return T


def evalManufacturingTolerances():
    print("\n--- 3. Manufacturing Tolerances Analysis ---")
    robot_ref = rtb.models.DH.Panda()
    robot_ref.tool = SE3()  # Remove tool offset
    modified_links = copy.deepcopy(robot_ref.links)

    d1_mod = 0.383
    alpha3_mod = 1.5807

    modified_links[0].d = d1_mod
    modified_links[2].alpha = alpha3_mod

    panda_mod = rtb.DHRobot(modified_links, name='Panda_Mod')
    panda_mod.tool = SE3()  # Remove tool offset

    N_samples = 1000
    pos_errors_mag = []
    rpy_errors_mag = []

    qlims = robot_ref.qlim

    if qlims.shape == (2, 7):
        qlims = qlims.T  # Convert 2x7 (min/max rows) to 7x2 (joint rows)
    if qlims.shape[0] != 7:
        raise ValueError(f"Joint limits must have 7 rows, but shape is {qlims.shape}")

    for _ in range(N_samples):
        q = np.array([random.uniform(qlims[i, 0], qlims[i, 1]) for i in range(7)])
        T_ref = robot_ref.fkine(q)
        T_mod = panda_mod.fkine(q)

        T_err = T_ref.inv() * T_mod

        # Positional Error (Magnitude)
        delta_p = T_err.t
        pos_error_mag = np.linalg.norm(delta_p)
        pos_errors_mag.append(pos_error_mag)

        # Orientation Error (Angle magnitude in radians)
        R_err = T_err.R
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
        rpy_errors_mag.append(angle)

    pos_errors_mag = np.array(pos_errors_mag)
    rpy_errors_mag = np.array(rpy_errors_mag)

    print(f"Sampling {N_samples} random joint configurations...")

    ## Positional Error (Meters)
    print("\n## Positional Error (m) Analysis (Magnitude)")
    print(f"  - Mean Error: {np.mean(pos_errors_mag):.4e} m")
    print(f"  - Standard Deviation: {np.std(pos_errors_mag):.4e} m")
    print(f"  - Maximum Error: {np.max(pos_errors_mag):.4e} m")

    ## Orientation Error (Radians)
    print("\n## Orientation Error (rad) Analysis (Rotation Angle Magnitude)")
    print(f"  - Mean Error: {np.mean(rpy_errors_mag):.4e} rad")
    print(f"  - Standard Deviation: {np.std(rpy_errors_mag):.4e} rad")
    print(f"  - Maximum Error: {np.max(rpy_errors_mag):.4e} rad")

    return 0



# Load the robot model
robot = rtb.models.DH.Panda()
robot.tool = SE3() # we remove the tool offset as we are only interesed in the FK to the end-effector

# Q1.1
poe = forwardKinematicsPoESpace(robot.qz)
print("\n FK POE space", poe)

# Q1.2
poe_b = forwardKinematicsPoEBody(robot.qz)
print("\n FK POE body", poe_b)

# You can compare your implementation of the forward kinematics with the one from prtb
# test with multiple valid robot configurations
dhres = robot.fkine(robot.qz)
print("\n FK DH",dhres)

# Q1.3
evalManufacturingTolerances()