#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K
import numpy as np
import cv2


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):

        self.model = K.models.load_model(model_path)
        self.class_names = open(classes_path).read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def detect_objects(self, image, input_size=(416, 416)):
        """
        :param image: Image à détecter
        :param input_size: Taille de l'image d'entrée
        :return: Boîtes de détection avec les classes et scores
        """
        # Prétraiter l'image (redimensionner et normaliser)
        image_resized = cv2.resize(image, input_size) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        # Faire une prédiction
        predictions = self.model.predict(image_resized)

        # Appliquer la suppression non-maximale (NMS) ici si nécessaire
        # Pour l'instant, retourner les prédictions brutes
        return predictions
