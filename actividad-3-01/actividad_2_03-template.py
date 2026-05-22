import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class OrangeDetector:
    def __init__(self):
        """
        Inicializa la conexión con CoppeliaSim y configura el sensor de visión.
        """
        # Conexión con CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # Obtener el handle del sensor de visión
        self.sensor1Handle = self.sim.getObject('/Vision_sensor')

    def capture_image(self):
        """
        Captura una imagen desde el sensor de visión de CoppeliaSim.
        :return: Imagen en formato OpenCV (BGR) o None si no se pudo capturar.
        """
        img, resX, resY = self.sim.getVisionSensorCharImage(self.sensor1Handle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(img, 0)  
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def detect_orange_object(self, img):
        """
        Detecta el objeto naranja en la imagen y devuelve las coordenadas del mejor candidato.
        :param img: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) del mejor candidato o None si no se detecta.
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Rango para color naranja
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])

        # Crear máscara
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Tomar el contorno más grande
        biggest = max(contours, key=cv2.contourArea)

        # Calcular centroide con momentos
        M = cv2.moments(biggest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        best_candidate = (cx, cy)

        return best_candidate

    def run(self):
        """
        Ejecuta el detector en un bucle continuo.
        """
        while True:
            # Capturar la imagen desde el sensor de visión
            img = self.capture_image()

            if img is not None:
                # Detectar el objeto naranja y obtener el mejor candidato
                best_candidate = self.detect_orange_object(img)

                if best_candidate:
                    print(f"Mejor centroide detectado en: {best_candidate}")
                else:
                    print("No se detectó ningún objeto naranja.")

if __name__ == "__main__":
    detector = OrangeDetector()
    detector.run()