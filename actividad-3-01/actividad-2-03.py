import time
import sys
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class OrangeDetector:
    def __init__(self):
        """
        Inicializa la conexion con CoppeliaSim y configura el sensor de vision.
        """
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.sensor1Handle = self.sim.getObject('/Vision_sensor')

    def capture_image(self):
        """
        Captura una imagen desde el sensor de vision de CoppeliaSim.
        :return: Imagen en formato OpenCV (BGR) o None si no se pudo capturar.
        """
        try:
            img, res = self.sim.getVisionSensorImg(self.sensor1Handle)

            if img is not None and res is not None:
                image = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))

                image = cv2.flip(image, 0)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                return image_bgr
        except Exception as e:
            print(f"Error al capturar la imagen: {e}")

        return None

    def detect_orange_object(self, img):
        """
        Detecta el objeto naranja en la imagen y devuelve las coordenadas del mejor candidato.
        :param img: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) del mejor candidato o None si no se detecta.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_orange = np.array([5, 150, 50])
        upper_orange = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        max_area = 0

        MIN_AREA = 10
        MAX_AREA = 12000

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            if area > max_area:
                max_area = area

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_candidate = (cx, cy)

        return best_candidate

    def run(self):
        """
        Ejecuta el detector en un bucle continuo.
        """
        while True:
            img = self.capture_image()

            if img is not None:
                best_candidate = self.detect_orange_object(img)

                if best_candidate:
                    print(f"Mejor centroide detectado en: {best_candidate}")
                else:
                    print("No se detectó ningun objeto naranja.")

            time.sleep(0.05)

if __name__ == "__main__":
    detector = OrangeDetector()
    detector.run()