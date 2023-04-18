from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple
import numpy as np

from shapely.geometry import LineString, Point, Polygon
from .pose_config import config

Img = Sequence[Sequence[int]]


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

class Keypoint:
    x: int
    y: int
    score: float
    isNone: bool
    def __init__(self, x: int, y: int, score: float = 0):
        self.x = int(x)
        self.y = int(y)
        self.score = score
        self.isNone = False
    def __str__(self) -> str:
        return "Kp ({},{})".format(self.x, self.y)
    def __repr__(self) -> str:
        return "Kp ({},{})".format(self.x, self.y)

def p(point: Point) -> Tuple[int, int]:
    return (int(point.x), int(point.y))

def k(keypoint: Keypoint) -> Tuple[int, int]:
    return (int(keypoint.x), int(keypoint.y))

class NoneKeypoint(Keypoint):
    x: int
    y: int
    isNone: bool
    score: int
    def __init__(self):
        self.x = -1
        self.y = -1
        self.score = 0
        self.isNone = True
    def __str__(self) -> str:
        return "N Kp"
    def __repr__(self) -> str:
        return "N Kp"


@dataclass
class Pose:
    keypoints: Sequence[Keypoint]

Poses = Sequence[Pose]


def hrnet_to_compoelem_poses(pose_data) -> Poses:
    poses: Poses = []
    # for entry in pose_data["pose_entries"]:
    for ik, entry in enumerate(pose_data):
        # print(f"here: {entry}")
        entry = entry.tolist()
        keypoints: Sequence[Keypoint] = []
        if entry != -1 and ik != 18 and ik != 17:
            try:
                # print(f"here2: {entry}")
                # print(f"here3: {entry[0:2]}")
                y, x = entry[0:2]
                keypoint = Keypoint(int(x), int(y))
                keypoints.append(keypoint)
            except IndexError:
                print('could not add keypoint')
        else:
            keypoints.append(NoneKeypoint())
        pose = Pose(keypoints)
        poses.append(pose)
    return poses


class PoseTriangle:
    shape: Polygon
    top: Keypoint
    left: Keypoint
    right: Keypoint
    def __init__(self, top: Keypoint, left: Keypoint, right: Keypoint):
        self.top = top
        self.left = left
        self.right = right
        self.shape = Polygon([[top.x, top.y], [left.x, left.y], [right.x, right.y]])

class PoseLine:
    line: LineString
    top: Point
    bottom: Point
    def __init__(self, top:Point, bottom: Point):
        self.top = top
        self.bottom = bottom
        self.line = LineString([[top.x, top.y], [bottom.x, bottom.y]])


def get_pose_triangle(pose: Pose) -> PoseTriangle:
    pose_keypoints = np.array(pose.keypoints, dtype=Keypoint)
    print(pose_keypoints)

    # Partition the keypoint list in three sequences corresponding to left, right, top triangle corner points
    left_keypoint_selection: Sequence[Keypoint] = pose_keypoints[
        config["pose_abstraction"]["keypoint_list"]["left"]].tolist()
    right_keypoint_selection: Sequence[Keypoint] = pose_keypoints[
        config["pose_abstraction"]["keypoint_list"]["right"]].tolist()
    top_keypoint_selection: Sequence[Keypoint] = pose_keypoints[
        config["pose_abstraction"]["keypoint_list"]["top"]].tolist()

    # Select first keypoint of each partition witch is not None
    left_keypoints = list(filter(lambda kp: not kp.isNone, left_keypoint_selection))
    right_keypoints = list(filter(lambda kp: not kp.isNone, right_keypoint_selection))
    top_keypoints = list(filter(lambda kp: not kp.isNone, top_keypoint_selection))
    if (len(top_keypoints) == 0):
        raise ValueError('missing valid top keypoints for triangle calculation!')
    if (len(left_keypoints) == 0):
        raise ValueError('missing valid left keypoints for triangle calculation!')
        # raise AssertionError('missing valid left keypoints for triangle calculation!')
    if (len(right_keypoints) == 0):
        raise ValueError('missing valid right keypoints for triangle calculation!')
        # raise AssertionError('missing valid right keypoints for triangle calculation!')

    return PoseTriangle(top_keypoints[0], right_keypoints[0], left_keypoints[0])

def get_pose_triangles(poses: Poses) -> Sequence[PoseTriangle]:
    pose_triangles: Sequence[PoseTriangle] = []
    for pose in poses:
        try:
            triangle = get_pose_triangle(pose)
            pose_triangles.append(triangle)
        except ValueError as e:
            # print(e)
            pass
    return pose_triangles


def get_pose_line(triangle: PoseTriangle) -> PoseLine:
    bottom_line = LineString([[triangle.left.x, triangle.left.y], [triangle.right.x, triangle.right.y]])
    top_point = Point(triangle.top.x, triangle.top.y)
    bottom_point = Point(bottom_line.centroid)
    return PoseLine(top_point, bottom_point)


def get_pose_abstraction(pose: Pose) -> PoseLine:
    # try:
    triangle = get_pose_triangle(pose)
    poseline = get_pose_line(triangle)
    return poseline


def get_pose_lines_without_fallback(poses: Poses) -> Sequence[PoseLine]:
    pose_lines: Sequence[PoseLine] = []
    for pose in poses:
        try:
            pose_abstraction = get_pose_abstraction(pose)
            pose_lines.append(pose_abstraction)
        except ValueError as e:
            # print(e)
            pass
    return pose_lines

def get_pose_lines(poses: Poses, fallback=False) -> Sequence[PoseLine]:
    return get_pose_lines_without_fallback(poses)
    # return get_pose_lines_with_fallback(poses) if fallback else get_pose_lines_without_fallback(poses)


#DUMPYARD
# # FIXes: missing pose line for virgin and child / maria
# def get_fallback_pose_line(pose: Pose) -> PoseLine:
#     pose_keypoints = np.array(pose.keypoints, dtype=Keypoint)
#     left_eye_kp: Keypoint = pose_keypoints[config["pose_abstraction"]["fallback"]["left_eye_kp"]]
#     right_eye_kp: Keypoint = pose_keypoints[config["pose_abstraction"]["fallback"]["right_eye_kp"]]
#     left_shoulder_kp: Keypoint = pose_keypoints[config["pose_abstraction"]["fallback"]["left_shoulder_kp"]]
#     right_shoulder_kp: Keypoint = pose_keypoints[config["pose_abstraction"]["fallback"]["right_shoulder_kp"]]
#     neck_kp: Keypoint = pose_keypoints[config["pose_abstraction"]["fallback"]["neck_kp"]]
#     nose_kp: Keypoint = pose_keypoints[config["pose_abstraction"]["fallback"]["nose_kp"]]
#     if neck_kp.isNone:
#         if left_shoulder_kp.isNone or right_shoulder_kp.isNone:
#             neck_kp = Keypoint(int((left_shoulder_kp.x + right_shoulder_kp.x) / 2),
#                                int((left_shoulder_kp.y + right_shoulder_kp.y) / 2),
#                                (left_shoulder_kp.score + right_shoulder_kp.score) / 2)
#         else:
#             raise AssertionError('missing valid keypoint 1 (neck) and left/right shoulder for pose line fallback!')
#     if nose_kp.isNone:
#         if left_eye_kp.isNone or right_eye_kp.isNone:
#             nose_kp = Keypoint(int((left_eye_kp.x + right_eye_kp.x) / 2), int((left_eye_kp.y + right_eye_kp.y) / 2),
#                                (left_eye_kp.score + right_eye_kp.score) / 2)
#         else:
#             raise AssertionError('missing valid keypoint 0 (nose) and left/right eye for pose line fallback!')
#
#     top_keypoint_selection: Sequence[Keypoint] = pose_keypoints[
#         config["pose_abstraction"]["keypoint_list"]["top"]].tolist()
#     top_keypoints = list(filter(lambda kp: not kp.isNone, top_keypoint_selection))
#     if len(top_keypoints) == 0:
#         if nose_kp.isNone:
#             raise ValueError('missing valid top keypoints and neck_kp for pose line fallback!!')
#         else:
#             top_point = Point(*k(nose_kp))  # use midpoint calculated from above
#     else:
#         top_point = Point(*k(top_keypoints[0]))
#
#     nose_point = np.array(k(nose_kp))
#     neck_point = np.array(k(neck_kp))
#     fallback_length = np.linalg.norm(nose_point - neck_point) * config["pose_abstraction"]["fallback"][
#         "length_scale_factor"]
#
#     bottom_point = Point(neck_point[0], (neck_point[1] + fallback_length))
#     return PoseLine(top_point, bottom_point)

# def get_pose_lines_with_fallback(poses: Poses) -> Sequence[PoseLine]:
#     global poses_counter
#     global normal_pose_line_counter
#     global fallback_line_counter
#     pose_lines: Sequence[PoseLine] = []
#     for pose in poses:
#         poses_counter += 1
#         try:
#             pose_abstraction = get_pose_abstraction(pose)
#             pose_lines.append(pose_abstraction)
#             normal_pose_line_counter += 1
#         except ValueError as e:
#             try:
#                 pose_lines.append(get_fallback_pose_line(pose))
#                 fallback_line_counter += 1
#             except AssertionError as e2:
#                 print("fallback err", e2)
#     # print(
#     #     "poses_counter", poses_counter,
#     #     "normal_pose_line_counter", normal_pose_line_counter,
#     #     "fallback_line_counter", fallback_line_counter,
#     # )
#     return pose_lines
