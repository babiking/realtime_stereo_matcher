import os
import copy
import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from collections import namedtuple


class BezierCurve3D:
    def __init__(self, control_points):
        self.control_points = control_points

    def binomial_coefficient(self, n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    def evaluate_once(self, t):
        n = len(self.control_points) - 1
        result = np.zeros(3, dtype=np.float32)

        for i in range(n + 1):
            result += (
                self.binomial_coefficient(n, i)
                * (1 - t) ** (n - i)
                * t**i
                * self.control_points[i]
            )
        return result

    def evaluate_all(self, num_splits):
        result = []
        for i in range(num_splits + 1):
            t = i / num_splits
            result.append(self.evaluate_once(t))
        return np.array(result, dtype=np.float32)


def draw_multi_world_lanes(lanes, is_3d=True, revert=False, save_file=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    for i, lane in enumerate(lanes):
        if is_3d:
            ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], label=f"Lane-{i}")
        else:
            ax.plot(lane[:, 0], lane[:, 1], label=f"Lane-{i}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if is_3d:
        ax.set_zlabel("Z")
    ax.set_title(("3D" if is_3d else "2D") + " Bezier curves for multiple lanes")
    ax.grid(True)
    ax.legend()

    if revert:
        # ax.invert_xaxis()
        ax.invert_yaxis()

    if save_file is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)


def get_transform_from_pose(pose):
    rot_mat = R.from_euler(
        "ZYX", [pose.yaw, pose.pitch, pose.roll], degrees=True
    ).as_matrix()

    t_vec = np.array([pose.x, pose.y, pose.z], dtype=np.float32)

    tf_mat = np.eye(4, dtype=np.float32)
    tf_mat[:3, :3] = rot_mat
    tf_mat[:3, 3] = t_vec
    return tf_mat


def get_projection_from_camera(cam_params):
    return np.array(
        [
            [cam_params.fx, 0.0, cam_params.cx],
            [0.0, cam_params.fy, cam_params.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main():
    np.random.seed(2)

    work_path = os.path.dirname(os.path.abspath(__file__))

    Camera = namedtuple("Camera", ["fx", "fy", "cx", "cy", "width", "height"])

    Pose = namedtuple("Pose", ["roll", "pitch", "yaw", "x", "y", "z"])

    # 480P medium-focal-length camera
    cam_params = Camera(
        fx=390.00, fy=390.00, cx=640.00, cy=360.00, width=1280, height=720
    )
    # camera located at (0m, 0m, 2.26m) with tiny euler angle offsets
    base_pose = Pose(roll=0.0, pitch=0.0, yaw=0.0, x=0, y=0, z=2.26)

    xmin = 1.0
    xmax = 30.0

    num_ctrl_pnts = 4
    ctrl_xs = np.linspace(xmin, xmax, num_ctrl_pnts, dtype=np.float32)[:, np.newaxis]
    ctrl_ys = (
        np.random.random(size=[num_ctrl_pnts, 1]).astype(np.float32) * 2.0 - 1.0
    ) * 10.0
    ctrl_zs = np.random.random(size=[num_ctrl_pnts, 1]).astype(np.float32) * 0.5
    ctrl_pnts = np.concatenate([ctrl_xs, ctrl_ys, ctrl_zs], axis=1)

    bezier_curve = BezierCurve3D(ctrl_pnts)
    num_splits_per_lane = 100
    bezier_lane = bezier_curve.evaluate_all(num_splits_per_lane)

    num_lanes = 5
    world_lanes = []
    for i in range(num_lanes):
        lane = copy.deepcopy(bezier_lane)
        lane[:, 1] += np.random.random() * i * 4.8
        lane[:, 2] += np.random.normal(0.0, 1.0) * 0.5
        world_lanes.append(lane)

    view_y = world_lanes[num_lanes // 2][:, 1].mean()
    for i in range(num_lanes):
        world_lanes[i][:, 1] -= view_y

    draw_multi_world_lanes(
        world_lanes,
        True,
        False,
        os.path.join(work_path, "result", "bezier_lanes_3d.png"),
    )
    draw_multi_world_lanes(
        world_lanes,
        False,
        False,
        os.path.join(work_path, "result", "bezier_lanes_2d.png"),
    )

    # base-link: X-forward, Y-left, Z-up
    # camera-link: X-right, Y-down, Z-forward
    tf_base_cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    tf_base_world = get_transform_from_pose(base_pose)
    tf_world_cam = tf_base_cam @ np.linalg.inv(tf_base_world)

    yaw_noise, pitch_noise, roll_noise = np.random.random(size=[3]) * 6.0
    noise_pose = Pose(
        roll=roll_noise, pitch=pitch_noise, yaw=yaw_noise, x=0.0, y=0.0, z=0.0
    )
    tf_noise_base = get_transform_from_pose(noise_pose)
    tf_noise_world = tf_noise_base @ tf_base_world
    tf_world_noise = tf_base_cam @ np.linalg.inv(tf_noise_world)

    cam_proj = get_projection_from_camera(cam_params)

    h_mat = np.eye(3, dtype=np.float32)
    h_mat[:3, :2] = tf_world_cam[:3, :2]
    h_mat[:3, 2] = tf_world_cam[:3, 3]
    h_mat = cam_proj @ h_mat
    h_mat = np.linalg.inv(h_mat)

    cam_pixs = []
    cam_reprojs = []
    cam_reprojs_denoised = []
    for lane in world_lanes:
        lane[:, 2] = 0.0

        lane = np.concatenate(
            (lane, np.ones([lane.shape[0], 1], dtype=np.float32)), axis=1
        )
        cam_lane_3d = (lane @ tf_world_noise.T)[:, :3]
        cam_lane_2d = cam_lane_3d @ cam_proj.T
        cam_lane_2d /= cam_lane_2d[:, 2:3]
        cam_pixs.append(cam_lane_2d)

        cam_reproj = cam_lane_2d @ h_mat.T
        cam_reproj /= cam_reproj[:, 2:3]
        cam_reprojs.append(cam_reproj)

        h_mat_noise, status = cv.findHomography(
            srcPoints=cam_lane_2d[:, :2], dstPoints=lane[:, :2]
        )
        cam_reproj_denoise = cam_lane_2d @ h_mat_noise.T
        cam_reproj_denoise /= cam_reproj_denoise[:, 2:3]
        cam_reprojs_denoised.append(cam_reproj_denoise)

    draw_multi_world_lanes(
        cam_pixs,
        False,
        True,
        os.path.join(work_path, "result", "camera_pixels_2d.png"),
    )
    draw_multi_world_lanes(
        cam_reprojs,
        False,
        False,
        os.path.join(work_path, "result", "camera_reprojections_2d.png"),
    )
    draw_multi_world_lanes(
        cam_reprojs_denoised,
        False,
        False,
        os.path.join(work_path, "result", "camera_reprojections_denoised_2d.png"),
    )


if __name__ == "__main__":
    main()