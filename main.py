# Copyright © 2025 CToID <funk443@yahoo.com.tw>
#
# This program is free software. It comes without any warranty, to the
# extent permitted by applicable law. You can redistribute it and/or
# modify it under the terms of the Do What The Fuck You Want To Public
# License, Version 2, as published by Sam Hocevar. See the COPYING file
# for more details.

from argparse import ArgumentParser

from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cv2 as cv

argparser = ArgumentParser()
argparser.add_argument("--input",
    action="append",
    type=str,
    required=True,
    metavar="PATH",
)
argparser.add_argument("--model",
    action="store",
    default="./static/single-lightning.3.tflite",
    type=str,
    metavar="PATH",
)

INPUT_SIZE = 192

# fmt: off
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
KEYPOINT_CONNECTIONS = [
    (0, 1),   (0, 2),   # nose - eyes
    (1, 3),   (2, 4),   # eyes - ears
    (3, 5),   (4, 6),   # ears - shoulders
    (5, 7),   (6, 8),   # shoulders - elbows
    (7, 9),   (8, 10),  # elbows - wrists
    (5, 11),  (6, 12),  # shoulders - hips
    (11, 13), (12, 14), # hips - knees
    (13, 15), (14, 16), # knees - ankles
]
# fmt: on

PositionDict = dict[str, tuple[float, float]]


def detect_pose_static_image(image_path: str, interpreter: Interpreter) -> NDArray:
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)

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
        actual_positions[keypoint_name] = (image_x_size * x, image_y_size * y)

    return normalized_positions, actual_positions


if __name__ == "__main__":
    argv = argparser.parse_args()

    interpreter = Interpreter(model_path=argv.model)
    for image_path in argv.input:
        keypoints = detect_pose_static_image(image_path, interpreter)
        plot_img = cv.imread(image_path)
        plot_img_rgb = cv.cvtColor(plot_img, cv.COLOR_BGR2RGB)
        _, actual_positions = calculate_positions(keypoints, plot_img_rgb.shape)

        plt.imshow(plot_img_rgb)
        plt.plot(
            [x for x, _ in actual_positions.values()],
            [y for _, y in actual_positions.values()],
            ".r",
        )
        for begin, end in KEYPOINT_CONNECTIONS:
            begin_x, begin_y = actual_positions[KEYPOINT_NAMES[begin]]
            end_x, end_y = actual_positions[KEYPOINT_NAMES[end]]
            plt.plot([begin_x, end_x], [begin_y, end_y], color="red")

        plt.show()
