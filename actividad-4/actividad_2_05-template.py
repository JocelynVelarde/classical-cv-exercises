import time
import sys
import math
import numpy as np
import cv2

class TrafficLightDetection:
    def __init__(self):
        self.cameraWidth = 320
        self.cameraHeight = 240

    def detect_state(self, image):
        """
        Detecta el estado del semáforo y lo reporta como texto.
        :return: String con alguno de los siguientes contenidos: green, yellow, red, none
        """
        state = "none"

        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Máscaras para cada color del semáforo
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_red = mask_red1 + mask_red2

        mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

        mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))

        # Contar píxeles de cada color
        red_count = cv2.countNonZero(mask_red)
        yellow_count = cv2.countNonZero(mask_yellow)
        green_count = cv2.countNonZero(mask_green)

        # El color con más píxeles es el estado
        counts = {"red": red_count, "yellow": yellow_count, "green": green_count}
        max_color = max(counts, key=counts.get)

        # Umbral mínimo para evitar falsos positivos
        if counts[max_color] >= 50:
            state = max_color

        return state