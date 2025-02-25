#!/usr/bin/env python3
"""Yolo class for object detection"""


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
