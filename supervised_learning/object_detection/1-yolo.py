#!/usr/bin/env python3
"""Yolo class for object detection"""


from tensorflow import keras as K
import tensorflow as tf


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
        """sortie"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i in range(len(outputs)):
            output = outputs[i]
            grid_height, grid_width, anchor_boxes, _ = output.shape

            for k in range(grid_height):
                for j in range(grid_width):
                    for a in range(anchor_boxes):
                        box = output[k, j, a]
                        
                        # Récupère les coordonnées
                        t_x, t_y, t_w, t_h = box[:4]
                        
                        # Calcule les coordonnées réelles (x1, y1, x2, y2)
                        x1 = (t_x - t_w / 2) * image_size[1]  # image_width
                        y1 = (t_y - t_h / 2) * image_size[0]  # image_height
                        x2 = (t_x + t_w / 2) * image_size[1]  # image_width
                        y2 = (t_y + t_h / 2) * image_size[0]  # image_height
                        
                        # Ajoute les coordonnées dans la liste
                        boxes.append([x1, y1, x2, y2])
                        box_confidences.append(box[4])
                        box_class_probs.append(box[5:])

        # Applique la suppression non maximale (NMS)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        box_confidences = tf.convert_to_tensor(box_confidences, dtype=tf.float32)

        # NMS : Suppression des boîtes en fonction de l'IOU
        nms_indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=box_confidences,
            max_output_size=50,
            iou_threshold=self.nms_t
        )

        # Filtre les boîtes et les scores avec les indices après NMS
        boxes = tf.gather(boxes, nms_indices)
        box_confidences = tf.gather(box_confidences, nms_indices)
        box_class_probs = tf.gather(box_class_probs, nms_indices)

        return boxes, box_confidences, box_class_probs
