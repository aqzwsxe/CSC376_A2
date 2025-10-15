import copy
from os import error
import random

import roboticstoolbox as rtb
from spatialmath import SE3, SO3, Twist3
from math import pi
import numpy as np

# Panda DH parameters from https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
robot = rtb.models.DH.Panda()
# robot = rtb.models.DH.Panda()
# robot.tool = SE3()

def matrix_log_se3(T):
    """Compute the matrix logarithm [S] of the SE(3) matrix T (i.e., the se(3) element)."""
    return T.logm()


def matrix_exp_se3(S):
    """Compute the matrix exponential T = Exp([S]) of the se(3) element [S]."""
    # SE3.Exp takes the 6-vector representation of the twist S
    return SE3.Exp(S)


def getHomeConfiguration(robot: rtb.DHRobot) -> SE3:
    """
    Calculates the home configuration M, which is the end-effector pose
    when all joint angles q are zero (q=qz).
    """
    # Temporarily remove the tool offset to get the pose of the last DH frame
    original_tool = robot.tool
    robot.tool = SE3()
    M = robot.fkine(robot.qz)
    robot.tool = original_tool  # Restore the tool definition
    return M


def getSpaceTwists(robot: rtb.DHRobot) -> np.ndarray:
    S_vectors = []
    T_prev = SE3()  # T_prev is the transformation T_{i-1} (from {s} to joint i-1)

    for i, link in enumerate(robot.links):
        # 1. Rotation (R_prev) and Translation (q) of the i-th joint frame in {s}
        R_prev = T_prev.R
        q = T_prev.t

        # 2. Rotation axis omega: For revolute joints, it's the z-axis of the current joint frame,
        # transformed to the space frame {s}.
        omega = R_prev @ np.array([0, 0, 1])

        # 3. Linear velocity component v: v = -omega x q
        v = -np.cross(omega, q)

        # 4. Concatenate to form the 6-vector twist S_i
        twist_vector = np.concatenate((v, omega))
        S_vectors.append(twist_vector)

        # 5. Update T_prev: T_{i} = T_{i-1} * A_i(0)
        T_link = link.A(0)  # A_i(0) is the link transformation for q_i = 0
        T_prev = T_prev * T_link

    return np.array(S_vectors).T  # Returns 6x7 matrix of space twists


def forwardKinematicsPoESpace(q):
    q = np.asarray(q)
    if q.shape != (7,):
        raise ValueError("q must be a 7-element list or array.")

    M = getHomeConfiguration(robot)

    S_matrix = getSpaceTwists(robot)

    # Convert 6-vector columns to spatialmath Twist3 objects
    S = [Twist3(S_matrix[:, i]) for i in range(7)]

    T = M.copy()  # Initialize T with M (the right-most term in the formula)

    # Iterate backwards (from n to 1) to apply the exponentials from left to right
    # T_new = Exp([S_i]q_i) * T_old
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
    M_inv = M.inv().A  # Inverse of the home configuration matrix

    # Calculate Body Twists Bi = Ad_M_inv(Si)
    B_matrix = []
    for i in range(7):
        # Adjoint transformation of M_inv: Ad_M_inv
        Ad_M_inv = SE3(M_inv).Ad()
        # Bi = Ad_M_inv @ Si
        B_i = Ad_M_inv @ S_matrix[:, i]
        B_matrix.append(B_i)

    B_matrix = np.array(B_matrix).T  # 6x7 matrix of body twists
    # Convert 6-vector columns to spatialmath Twist3 objects
    B = [Twist3(B_matrix[:, i]) for i in range(7)]

    T = M.copy()  # Initialize T with M (the left-most term in the formula)

    # Iterate forwards (from 1 to n) to apply the exponentials from left to right
    # T_new = T_old * Exp([B_i]q_i)
    for i in range(7):
        exp_Bi_qi = B[i].exp(q[i])

        # Right multiplication: T = T * exp_Bi_qi
        T = T * exp_Bi_qi

    return T


def evalManufacturingTolerances():
    print("\n--- 3. Manufacturing Tolerances Analysis ---")

    # 1. Define Reference Robot (Ideal DH parameters)
    robot_ref = rtb.models.DH.Panda()
    robot_ref.tool = SE3()  # Remove tool offset for clean FK

    # 2. Define Modified Robot (Simulated Manufacturing Error)
    modified_links = copy.deepcopy(robot_ref.links)

    # Error in d1 (Link 1, prismatic offset)
    d1_mod = 0.383  # Default d1 is 0.333. Error: 50mm
    # Error in alpha3 (Link 3, link twist)
    alpha3_mod = 1.5807  # Default alpha3 is pi/2 ~ 1.5708. Error: ~0.0099 rad

    # Apply modifications to the copied links
    modified_links[0].d = d1_mod
    modified_links[2].alpha = alpha3_mod

    panda_mod = rtb.DHRobot(modified_links, name='Panda_Mod')
    panda_mod.tool = SE3()  # Remove tool offset

    # 3. Sampling Setup
    N_samples = 1000
    pos_errors_mag = []  # To store |delta_p|
    rpy_errors_mag = []  # To store rotation angle magnitude |theta|
    qlims = robot_ref.qlim  # Get joint limits

    # Ensure qlims is in the correct 7x2 format (min/max for each joint)
    if qlims.shape == (2, 7):
        qlims = qlims.T
    if qlims.shape[0] != 7:
        raise ValueError(f"Joint limits must have 7 rows, but shape is {qlims.shape}")

    # 4. Error Calculation Loop
    for _ in range(N_samples):
        # Generate a random joint configuration q within limits
        q = np.array([random.uniform(qlims[i, 0], qlims[i, 1]) for i in range(7)])

        # Compute FK for both models
        T_ref = robot_ref.fkine(q)
        T_mod = panda_mod.fkine(q)

        # Compute the relative error transformation: T_err = T_ref_inv * T_mod
        T_err = T_ref.inv() * T_mod

        # Positional Error (Magnitude)
        delta_p = T_err.t
        pos_error_mag = np.linalg.norm(delta_p)
        pos_errors_mag.append(pos_error_mag)

        # Orientation Error (Angle magnitude in radians)
        # Angle of rotation is calculated from the trace of the error rotation matrix R_err
        R_err = T_err.R
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
        rpy_errors_mag.append(angle)

    # 5. Statistical Analysis and Output
    pos_errors_mag = np.array(pos_errors_mag)
    rpy_errors_mag = np.array(rpy_errors_mag)

    print(f"Sampling {N_samples} random joint configurations...")

    # Positional Error (Meters)
    print("\n## Positional Error (m) Analysis (Magnitude)")
    print(f"  - Mean Error: {np.mean(pos_errors_mag):.4e} m")
    print(f"  - Standard Deviation: {np.std(pos_errors_mag):.4e} m")
    print(f"  - Maximum Error: {np.max(pos_errors_mag):.4e} m")

    # Orientation Error (Radians)
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