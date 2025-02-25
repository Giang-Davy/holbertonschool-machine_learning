#!/usr/bin/env python3
"""Yolo class for object detection"""

from tensorflow import keras as K
import numpy as np

class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class for object detection.

        Args:
            model_path (str): path to the Darknet Keras model.
            classes_path (str): path to the file containing class names.
            class_t (float): box score threshold for filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (np.ndarray): array of anchor boxes of shape (outputs, 3, 2)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the Darknet model for a single image.

        Args:
            outputs (list): list of numpy.ndarrays containing predictions from the model.
            image_size (numpy.ndarray): image’s original size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            box_xy = self.sigmoid(output[..., :2])
            box_wh = np.exp(output[..., 2:4])
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_probs = self.sigmoid(output[..., 5:])

            anchors = self.anchors[:anchor_boxes]

            grid_x = np.tile(np.arange(grid_width).reshape(1, grid_width, 1), (grid_height, 1, anchor_boxes))
            grid_y = np.tile(np.arange(grid_height).reshape(grid_height, 1, 1), (1, grid_width, anchor_boxes))
            box_xy += np.stack([grid_x, grid_y], axis=-1)
            box_xy /= [grid_width, grid_height]

            box_wh *= anchors / [image_width, image_height]

            x1 = (box_xy[..., 0] - box_wh[..., 0] / 2) * image_width
            y1 = (box_xy[..., 1] - box_wh[..., 1] / 2) * image_height
            x2 = (box_xy[..., 0] + box_wh[..., 0] / 2) * image_width
            y2 = (box_xy[..., 1] + box_wh[..., 1] / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_probs)

        return (boxes, box_confidences, box_class_probs)
