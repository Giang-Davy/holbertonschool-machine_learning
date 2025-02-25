#!/usr/bin/env python3
"""Yolo class for object detection"""

import numpy as np
from tensorflow import keras as K

class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class for object detection.

        Args:
            model_path (str): path to the Darknet Keras model.
            classes_path (str): path to the file containing class names.
            class_t (float): box score threshold for filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (np.ndarray): array of anchor boxes of shape (outputs
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the Darknet model.

        Args:
            outputs (list): list of numpy.ndarrays containing the predictions from the Darknet model for a single image.
            image_size (np.ndarray): array containing the image’s original size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box_xy = 1 / (1 + np.exp(-output[..., :2]))  # sigmoid(tx), sigmoid(ty)
            box_wh = np.exp(output[..., 2:4]) * self.anchors[i]  # pw * exp(tw), ph * exp(th)
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))  # sigmoid(box_confidence)
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))  # sigmoid(class_probs)

            col = np.tile(np.arange(0, grid_width), grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height).reshape(-1, 1), grid_width)

            col = col.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)

            box_xy += np.concatenate((col, row), axis=-1)
            box_xy /= (grid_width, grid_height)
            box_wh /= (self.model.input.shape[1], self.model.input.shape[2])

            box_xy -= (box_wh / 2)
            box_xy1 = box_xy * (image_width, image_height)
            box_xy2 = (box_xy + box_wh) * (image_width, image_height)
            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
