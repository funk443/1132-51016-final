# Copyright © 2025 CToID <funk443@yahoo.com.tw>
#
# This program is free software. It comes without any warranty, to the
# extent permitted by applicable law. You can redistribute it and/or
# modify it under the terms of the Do What The Fuck You Want To Public
# License, Version 2, as published by Sam Hocevar. See the COPYING file
# for more details.

from argparse import ArgumentParser
import math
import time

from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf

import numpy as np
from numpy.typing import NDArray

import cv2 as cv

argparser = ArgumentParser()
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument(
    "--input-image",
    action="append",
    type=str,
    metavar="PATH",
)
input_group.add_argument(
    "--stream-url",
    action="store",
    type=str,
    metavar="URL",
)
argparser.add_argument(
    "--model",
    action="store",
    default="./static/single-lightning.3.tflite",
    type=str,
    metavar="PATH",
)

INPUT_SIZE = 192

# fmt: off
KEYPOINT_NAMES = [
    "nose",
    "left_eye",      "right_eye",
    "left_ear",      "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]
KEYPOINT_CONNECTIONS = [
    (0, 1),   (0, 2),   # nose - eyes
    (1, 3),   (2, 4),   # eyes - ears
    (5, 6),             # left/right shoulders
    (3, 5),   (4, 6),   # ears - shoulders
    (5, 7),   (6, 8),   # shoulders - elbows
    (7, 9),   (8, 10),  # elbows - wrists
    (11, 12),           # left/right hips
    (5, 11),  (6, 12),  # shoulders - hips
    (11, 13), (12, 14), # hips - knees
    (13, 15), (14, 16), # knees - ankles
]
# fmt: on

PositionDict = dict[str, tuple[float, float]]


def detect_pose(
    raw_image: str | np.ndarray, interpreter: Interpreter
) -> NDArray:
    if isinstance(raw_image, str):
        image = tf.io.read_file(raw_image)
        image = tf.io.decode_jpeg(raw_image)
    elif isinstance(raw_image, np.ndarray):
        image = tf.convert_to_tensor(raw_image)

    # Pyrefly spitting out nonsense about this perfectly fine function call.
    # type: ignore
    image = tf.expand_dims(image, axis=0)

    # Pyrefly spitting out nonsense about this perfectly fine function call.
    # type: ignore
    image = tf.image.resize_with_pad(image, INPUT_SIZE, INPUT_SIZE)

    interpreter.allocate_tensors()
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]["index"])


def calculate_positions(
    keypoints: NDArray,
    image_shape: tuple[int, int, int],
) -> tuple[PositionDict, PositionDict]:
    image_y_size, image_x_size, _ = image_shape
    normalized_positions: PositionDict = {}
    actual_positions: PositionDict = {}
    for i, data in enumerate(keypoints[0][0]):
        keypoint_name = KEYPOINT_NAMES[i]
        y, x, _ = data
        normalized_positions[keypoint_name] = (x, y)
        actual_positions[keypoint_name] = (
            int(round(image_x_size * x)),
            int(round(image_y_size * y)),
        )

    return normalized_positions, actual_positions


def calculate_angles(positions: PositionDict) -> tuple[float, float]:
    def calculate_length(a: str, b: str) -> float:
        return (
            (positions[a][0] - positions[b][0]) ** 2
            + (positions[a][1] - positions[b][1]) ** 2
        ) ** 0.5

    def calculate_angle_ac(a: float, b: float, c: float) -> float:
        return math.acos((a**2 + c**2 - b**2) / (2 * a * c)) * 180 / math.pi

    neck_length = calculate_length("left_shoulder", "left_ear")
    back_length = calculate_length("left_shoulder", "left_hip")
    lap_length = calculate_length("left_hip", "left_knee")

    neck_angle = calculate_angle_ac(
        back_length, calculate_length("left_ear", "left_hip"), neck_length
    )
    back_angle = calculate_angle_ac(
        lap_length, calculate_length("left_shoulder", "left_knee"), back_length
    )

    return neck_angle, back_angle


def draw_skeleton(base: NDArray, actual_positions: PositionDict) -> None:
    for point in actual_positions.values():
        cv.circle(base, point, 10, (0, 0, 255), -1)

    for begin, end in KEYPOINT_CONNECTIONS:
        begin_name = KEYPOINT_NAMES[begin]
        end_name = KEYPOINT_NAMES[end]
        begin_point = actual_positions[begin_name]
        end_point = actual_positions[end_name]
        cv.line(base, begin_point, end_point, (0, 0, 255), 5)


if __name__ == "__main__":
    argv = argparser.parse_args()

    interpreter = Interpreter(model_path=argv.model)
    if argv.input_image is not None:
        for image_path in argv.input_image:
            image = cv.imread(image_path)
            image_rgb = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

            keypoints = detect_pose(image_rgb, interpreter)
            _, actual_positions = calculate_positions(keypoints, image.shape)

            draw_skeleton(image, actual_positions)
            cv.imshow(image_path, image)
            cv.waitKey(0)

    elif argv.stream_url is not None:
        cap = None
        while cap is None:
            cap = cv.VideoCapture(argv.stream_url)
            time.sleep(1)

        if not cap.isOpened():
            print("ERROR: Failed to open video stream.")
            cap.release()
            exit(1)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints = detect_pose(frame, interpreter)
                _, actual_positions = calculate_positions(
                    keypoints, frame.shape
                )
                neck_angle, back_angle = calculate_angles(actual_positions)

                draw_skeleton(frame, actual_positions)
                cv.imshow("HHH", frame)
                if cv.waitKey(1) == ord('q'):
                    break

        finally:
            cap.release()
            cv.destroyAllWindows()
