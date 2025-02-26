#!/usr/bin/env python3
"""Yolo class for object detection"""


import numpy as np
from tensorflow import keras as K


class Yolo:
    """yolo-v3 model to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class for object detection.

        Args:
            model_path (str): path to the Darknet Keras model.
            classes_path (str): path to the file containinges.
            class_t (float): box score threshold for filtering.
            nms_t (float): IOU threshold for non-max sup
            anchors (np.ndarray): array of anchor boxes ofts
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
            outputs (list): list of numpy.ndarrays coimage.
            image_size (np.ndarray): array containin, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box_xy = 1 / (1 + np.exp(-output[..., :2]))
            box_wh = np.exp(output[..., 2:4]) * self.anchors[i]
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            col = np.tile(
                np.arange(0, grid_width), grid_height).reshape(-1, grid_width)
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """boîte de filtre"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_class_prob in zip(boxes,
                                                 box_confidences,
                                                 box_class_probs):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            # Calculer les scores des boîtes
            box_score = box_conf * box_class_prob

            # Trouver les classes et les scores maximaux pour chaque boîte
            box_class = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            # Appliquer un masque de filtrage basé sur un seuil de score
            filtering_mask = box_class_score >= 0.6

            # Filtrer les boîtes, les classes et les scores
            filtered_boxes.append(box[filtering_mask])
            box_classes.append(box_class[filtering_mask])
            box_scores.append(box_class_score[filtering_mask])

        # Convertir les listes en numpy.ndarrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Applique la suppression non maximale"""
        indices = np.argsort(box_scores)[::-1]
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        while len(indices) > 0:
            i = indices[0]
            box_predictions.append(filtered_boxes[i])
            predicted_box_classes.append(box_classes[i])
            predicted_box_scores.append(box_scores[i])

            x1 = np.maximum(filtered_boxes[i][0], filtered_boxes[indices[1:]][:, 0])
            y1 = np.maximum(filtered_boxes[i][1], filtered_boxes[indices[1:]][:, 1])
            x2 = np.minimum(filtered_boxes[i][2], filtered_boxes[indices[1:]][:, 2])
            y2 = np.minimum(filtered_boxes[i][3], filtered_boxes[indices[1:]][:, 3])

            inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            box1_area = (filtered_boxes[i][2] - filtered_boxes[i][0]) * (filtered_boxes[i][3] - filtered_boxes[i][1])
            box2_area = (filtered_boxes[indices[1:]][:, 2] - filtered_boxes[indices[1:]][:, 0]) * (filtered_boxes[indices[1:]][:, 3] - filtered_boxes[indices[1:]][:, 1])
            iou = inter_area / (box1_area + box2_area - inter_area)

            indices = indices[1:][iou < 0.5]

        return np.array(box_predictions), np.array(predicted_box_classes), np.array(predicted_box_scores)
