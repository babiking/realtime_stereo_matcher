import os
import copy
import math
import numpy as np
from matplotlib import pyplot as plt
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


def draw_multi_world_lanes(lanes, is_3d=True, save_file=None):
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
    ax.set_title(
        ("3D" if is_3d else "2D Bird-Eye") + " Bezier curves for multiple lanes"
    )
    ax.grid(True)
    ax.legend()

    if save_file is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)


def main():
    work_path = os.path.dirname(os.path.abspath(__file__))

    Camera = namedtuple("Camera", ["fx", "fy", "cx", "cy", "width", "height"])

    Pose = namedtuple("Pose", ["roll", "pitch", "yaw", "x", "y", "z"])

    # 480P medium-focal-length camera
    cam_params = Camera(
        fx=390.00, fy=390.00, cx=320.00, cy=240.00, width=640, height=480
    )
    # camera located at (0m, 0m, 2.26m) with tiny euler angle offsets
    cam_pose = Pose(roll=0.5, pitch=2.0, yaw=1.0, x=0, y=0, z=2.26)

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
        world_lanes, True, os.path.join(work_path, "result", "bezier_lanes_3d.png")
    )
    draw_multi_world_lanes(
        world_lanes, False, os.path.join(work_path, "result", "bezier_lanes_2d.png")
    )


if __name__ == "__main__":
    main()
