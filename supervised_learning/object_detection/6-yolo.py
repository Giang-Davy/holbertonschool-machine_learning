#!/usr/bin/env python3
"""Yolo class for object detection"""


import numpy as np
from tensorflow import keras as K
import glob
import cv2
import os


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
        """
        Non suppression
        """
        def iou(b1, b2):
            """Calculer l'intersection sur l'union"""
            (x1, y1, x2, y2) = b1
            (x1bis, y1bis, x2bis, y2bis) = b2

            xi1 = max(x1, x1bis)
            y1i = max(y1, y1bis)
            xi2 = min(x2, x2bis)
            yi2 = min(y2, y2bis)

            width = max(0, xi2 - xi1)
            height = max(0, yi2 - y1i)

            inter_area = width * height

            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2bis - x1bis) * (y2bis - y1bis)

            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area

        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_box_scores = box_scores[cls_mask]

            sorted_indices = np.argsort(cls_box_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_box_scores = cls_box_scores[sorted_indices]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[0])

                if len(cls_boxes) == 1:
                    break

                ious = np.array([
                    iou(cls_boxes[0], box) for box in cls_boxes[1:]])
                cls_boxes = cls_boxes[1:][ious < self.nms_t]
                cls_box_scores = cls_box_scores[1:][ious < self.nms_t]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """charger une image """
        image_paths = glob.glob(f"{folder_path}/*")
        images = []

        for path in image_paths:
            image = cv2.imread(path)
            if image is not None:
                images.append(image)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images"""
        processed_images = []
        original_images = []

        for image in images:
            if image is not None:
                original = image.shape[:2]
                resized_image = cv2.resize(
                    image,
                    (self.model.input.shape[1], self.model.input.shape[2]),
                    interpolation=cv2.INTER_CUBIC
                )
                normalized_image = resized_image / 255.0
                processed_images.append(normalized_image)
                original_images.append(original)
        return np.array(processed_images), np.array(original_images)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """montrer les boîtes"""
        for i in range(len(boxes)):
            # Définir les coordonnées de la boîte
            start_point = (int(boxes[i][0]), int(boxes[i][1]))
            end_point = (int(boxes[i][2]), int(boxes[i][3]))
            
            # Dessiner la boîte en bleu
            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 255)
            thickness = 1
            text = f"{self.class_names[box_classes[i]]} {box_scores[i]:.2f}"
            linetype = cv2.LINE_AA
            position = (int(boxes[i][0]), int(boxes[i][1]) - 5)
            cv2.putText(image, text, position, font, font_scale, color, thickness, linetype)

        if not os.path.exists("detections"):
            os.makedirs("detections")

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            save_path = os.path.join("detections", file_name)
            try:
                cv2.imwrite(save_path, image)
                print(f"Image saved to {save_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
        else:
            print("Image not saved")

        cv2.destroyAllWindows()
