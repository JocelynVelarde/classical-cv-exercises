import time
import sys
import math
import numpy as np
import cv2

class CenterLineDetector:
    def __init__(self):
        self.cameraWidth = 320
        self.cameraHeight = 240

    def detect_center_line(self, image):
        """
        Detecta la línea central de la pista en el 1/4 inferior de la imagen.
        :param image: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) en coordenadas de la imagen original, None si no se detecta.
        """
        h = self.cameraHeight
        w = self.cameraWidth

        # 1. Usar solo el 1/4 inferior de la imagen
        quarter = image[3 * h // 4 :, :]

        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(quarter, cv2.COLOR_BGR2GRAY)

        # 3. Suavizado Gaussiano para reducir ruido 
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # 4. Binarizar con Otsu para detectar la línea clara 
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Encontrar contornos 
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 6. Tomar el contorno más grande 
        biggest = max(contours, key=cv2.contourArea)

        # 7. Calcular centroide con momentos 
        M = cv2.moments(biggest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + 3 * h // 4 

        best_candidate = (cx, cy)
        return best_candidate
