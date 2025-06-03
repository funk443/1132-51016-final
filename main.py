# Copyright © 2025 CToID <funk443@yahoo.com.tw>
#
# This program is free software. It comes without any warranty, to the
# extent permitted by applicable law. You can redistribute it and/or
# modify it under the terms of the Do What The Fuck You Want To Public
# License, Version 2, as published by Sam Hocevar. See the COPYING file
# for more details.

import sys

from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

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


assert len(sys.argv) > 1
INPUT_SIZE = 192
IMAGE_PATH = sys.argv[1]
MODEL_PATH = "single-lightning.tflite"

image = tf.io.read_file(IMAGE_PATH)
image = tf.compat.v1.image.decode_jpeg(image)

# Pyrefly spitting out nonsense about this perfectly fine function call.
# type: ignore
image = tf.expand_dims(image, axis=0)

# Pyrefly spitting out nonsense about this perfectly fine function call.
# type: ignore
image = tf.image.resize_with_pad(image, INPUT_SIZE, INPUT_SIZE)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_image = tf.cast(image, dtype=tf.float32)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
interpreter.invoke()

keypoints = interpreter.get_tensor(output_details[0]["index"])
normalized_locations: dict[str, tuple[float, float]] = {}
for i, data in enumerate(keypoints[0][0]):
    keypoint_name = KEYPOINT_NAMES[i]
    y, x, _ = data
    normalized_locations[keypoint_name] = (x, y)

plot_img = cv.imread(IMAGE_PATH)
plot_img_rgb = cv.cvtColor(plot_img, cv.COLOR_BGR2RGB)
plot_y_size, plot_x_size, _ = plot_img.shape
actual_locations: dict[str, tuple[float, float]] = {}
for name, (ratio_x, ratio_y) in normalized_locations.items():
    actual_locations[name] = (plot_x_size * ratio_x, plot_y_size * ratio_y)

plt.imshow(plot_img_rgb)
plt.plot(
    [x for x, _ in actual_locations.values()],
    [y for _, y in actual_locations.values()],
    ".",
)
plt.show()
