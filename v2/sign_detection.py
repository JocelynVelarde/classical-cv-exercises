"""
==============================================================================
 ACTIVIDAD 2.6 - DETECCION DE "STOP" (v3 - ANTI-WORKERS)
==============================================================================
 Problema detectado: el WORKERS (triangulo con borde rojo) se confunde
 con el STOP porque AMBOS tienen rojo.

 SOLUCION: agregar un filtro por "fill ratio" de rojo:
   - STOP     -> el rojo cubre ~60-95% del bounding box (fondo rojo solido)
   - WORKERS  -> el rojo cubre solo ~15-40% (solo el borde del triangulo)

 Asi SIN CAMBIAR el pipeline (Template Matching + Interseccion de Histogramas
 + Laplaciano) solo le agregamos una caracteristica geometrica que separa
 las 2 señales rojas.

 NUEVO filtro: RED_FILL_RATIO_MIN = 0.50
   Descarta ROIs donde menos del 50% de los pixeles son rojos
   -> esto elimina el WORKERS.
==============================================================================
"""

import cv2
import numpy as np
import threading
import time
import os

import grpc
import te3002b_pb2
import te3002b_pb2_grpc
import google.protobuf.empty_pb2


# -----------------------------------------------------------------------------
# HIPERPARAMETROS
# -----------------------------------------------------------------------------
TEMPLATE_PATH    = "./templates/stop.png"
BLUR_THRESHOLD   = 20.0
TM_THRESHOLD     = 0.35
HIST_THRESHOLD   = 0.15
SCALES           = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MIN_ROI_AREA     = 100

# >>> NUEVO: fill ratio minimo de rojo dentro del bounding box
# STOP: ~60-95% (fondo rojo), WORKERS: ~15-40% (solo borde)
RED_FILL_RATIO_MIN = 0.50

# Aspect ratio mas estricto: el STOP es casi cuadrado
ASPECT_MIN = 0.70
ASPECT_MAX = 1.40

RED_LOWER_1 = np.array([0,   40, 40])
RED_UPPER_1 = np.array([15,  255, 255])
RED_LOWER_2 = np.array([160, 40, 40])
RED_UPPER_2 = np.array([179, 255, 255])


# =============================================================================
# PIPELINE
# =============================================================================

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_red_rois(frame_bgr):
    """
    Devuelve (rois_filtradas, mask).
    Cada ROI pasa: area minima, aspect ratio, Y FILL RATIO (clave anti-WORKERS).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1) | \
           cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    H, W = frame_bgr.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area_box = w * h
        if area_box < MIN_ROI_AREA:
            continue

        aspect = w / float(h)
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue

        # >>> FILTRO CLAVE: porcentaje de rojo en el bounding box
        roi_mask = mask[y:y+h, x:x+w]
        red_pixels = cv2.countNonZero(roi_mask)
        fill_ratio = red_pixels / float(area_box)
        if fill_ratio < RED_FILL_RATIO_MIN:
            continue  # descarta WORKERS (solo borde rojo)

        pad = 8
        xp = max(0, x - pad); yp = max(0, y - pad)
        wp = min(W - xp, w + 2*pad); hp = min(H - yp, h + 2*pad)
        rois.append((xp, yp, wp, hp, fill_ratio))
    return rois, mask


def template_match_multiscale(roi_gray, template_gray):
    best = 0.0
    th, tw = template_gray.shape[:2]
    rh, rw = roi_gray.shape[:2]
    for s in SCALES:
        nw, nh = int(tw * s), int(th * s)
        if nw < 10 or nh < 10 or nw >= rw or nh >= rh:
            continue
        t = cv2.resize(template_gray, (nw, nh))
        res = cv2.matchTemplate(roi_gray, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best:
            best = max_val
    return best


def hist_intersection(roi_bgr, template_bgr):
    hsv1 = cv2.cvtColor(roi_bgr,      cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0], None, [180], [0, 180])
    h2 = cv2.calcHist([hsv2], [0], None, [180], [0, 180])
    cv2.normalize(h1, h1, alpha=1, beta=0, norm_type=cv2.NORM_L1)
    cv2.normalize(h2, h2, alpha=1, beta=0, norm_type=cv2.NORM_L1)
    return float(np.sum(np.minimum(h1, h2)))


def detect_stop_debug(frame, template_bgr, template_gray):
    info = {'blur_var': 0.0, 'mask': None, 'rois': [],
            'roi_scores': [], 'detection': None}
    if frame is None:
        return info

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    info['blur_var'] = variance_of_laplacian(gray)
    if info['blur_var'] < BLUR_THRESHOLD:
        return info

    rois, mask = get_red_rois(frame)
    info['mask'] = mask
    info['rois'] = rois
    if not rois:
        return info

    best_det = None
    best_combined = 0.0
    for (x, y, w, h, fill) in rois:
        roi_bgr  = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue

        tm = template_match_multiscale(roi_gray, template_gray)
        hi = hist_intersection(roi_bgr, template_bgr)
        passed = tm >= TM_THRESHOLD and hi >= HIST_THRESHOLD
        info['roi_scores'].append(((x, y, w, h), tm, hi, fill, passed))

        if passed:
            combined = 0.6 * tm + 0.4 * hi
            if combined > best_combined:
                best_combined = combined
                best_det = (x, y, w, h, combined)

    info['detection'] = best_det
    return info


# =============================================================================
# NODO DEL SIMULADOR
# =============================================================================
class SimRobotNode():
    def __init__(self):
        self._addr = "127.0.0.1"
        self.channel = grpc.insecure_channel(self._addr + ':7072')
        self.stub = te3002b_pb2_grpc.TE3002BSimStub(self.channel)

        self.cv_image = None
        self.result = None
        self.datacmd = te3002b_pb2.CommandData()
        self.dataconfig = te3002b_pb2.ConfigurationData()
        self.twist = [0, 0, 0, 0, 0, 0]
        self.running = True
        self.timer_delta = 0.025
        self.running_time = 0.0

        if not os.path.exists(TEMPLATE_PATH):
            raise FileNotFoundError(f"Falta: {TEMPLATE_PATH}")
        tpl = cv2.imread(TEMPLATE_PATH)
        tpl = cv2.resize(tpl, (64, 64))
        self.template_bgr  = tpl
        self.template_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        print(f"[OK] Template STOP cargado desde {TEMPLATE_PATH}")

        self.video_writer = None

    def _init_video_writer(self, frame):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                'deteccion_stop_v3.mp4', fourcc, 20.0, (w, h))
            print("[OK] Grabando: deteccion_stop_v3.mp4")

    def callback(self):
        self.dataconfig.resetRobot = True
        self.dataconfig.mode = 2
        self.dataconfig.cameraWidth = 360
        self.dataconfig.cameraHeight = 240
        self.dataconfig.resetCamera = False
        self.dataconfig.scene = 2026
        self.dataconfig.cameraLinear.x = 0
        self.dataconfig.cameraLinear.y = 0
        self.dataconfig.cameraLinear.z = 0
        self.dataconfig.cameraAngular.x = 0
        self.dataconfig.cameraAngular.y = 0
        self.dataconfig.cameraAngular.z = 0
        req = google.protobuf.empty_pb2.Empty()
        self.twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
        self.stub.SetConfiguration(self.dataconfig)
        self.dataconfig.resetRobot = False
        time.sleep(0.25)
        self.stub.SetConfiguration(self.dataconfig)

        while self.running:
            self.result = self.stub.GetImageFrame(req)
            img_buffer = np.frombuffer(self.result.data, np.uint8)
            img_in = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            img = img_in
            if img_in is not None:
                img = self.add_noise_to_image(img_in, 3)

            new_dim = (320, 240)
            self.cv_image = cv2.resize(img, new_dim, interpolation=cv2.INTER_LANCZOS4)

            info = detect_stop_debug(self.cv_image,
                                     self.template_bgr, self.template_gray)

            vis = self.cv_image.copy()
            # Dibuja TODAS las ROIs que pasaron fill_ratio (amarillo/rojo)
            for (roi, tm, hi, passed) in info['roi_scores']:
                x, y, w, h = roi
                color = (0, 0, 255) if passed else (0, 255, 255)
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 1)
                cv2.putText(vis, f"tm={tm:.2f} hi={hi:.2f}",
                            (x, max(10, y-3)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, color, 1)

            if info['detection']:
                x, y, w, h, score = info['detection']
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vis, f"STOP {score:.2f}", (x, max(12, y-12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(vis, "EN FRENTE: STOP", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"   >>> STOP! score={score:.2f}")
            else:
                status = f"blur={info['blur_var']:.0f} rois={len(info['rois'])}"
                cv2.putText(vis, status, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if info['mask'] is not None:
                cv2.imshow('Mascara ROJO',
                           cv2.cvtColor(info['mask'], cv2.COLOR_GRAY2BGR))

            self.twist = [0.01, 0.0, 0.0, 0.0, 0.0, -0.001]
            self.datacmd.linear.x  = self.twist[0]
            self.datacmd.linear.y  = self.twist[1]
            self.datacmd.linear.z  = self.twist[2]
            self.datacmd.angular.x = self.twist[3]
            self.datacmd.angular.y = self.twist[4]
            self.datacmd.angular.z = self.twist[5]
            self.result = self.stub.SetCommand(self.datacmd)

            self._init_video_writer(vis)
            self.video_writer.write(vis)
            cv2.imshow('STOP Detector v3', vis)
            cv2.waitKey(1)

            time.sleep(self.timer_delta - 0.001)
            self.running_time += self.timer_delta

    def add_noise_to_image(self, image, kernel_s, noise_level=5):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        noise = np.random.randint(-noise_level, noise_level + 1, v.shape, dtype='int16')
        v_noisy = v.astype('int16') + noise
        v_noisy = np.clip(v_noisy, 0, 255).astype('uint8')
        hsv_noisy = cv2.merge([h, s, v_noisy])
        noisy_image = cv2.cvtColor(hsv_noisy, cv2.COLOR_HSV2BGR)
        noisy_image = cv2.GaussianBlur(noisy_image, (kernel_s, kernel_s), 0)
        alpha = 0.55
        beta = 55
        return cv2.convertScaleAbs(noisy_image, alpha=alpha, beta=beta)


def main(args=None):
    robot_node = SimRobotNode()
    thread = threading.Thread(target=robot_node.callback)
    thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        robot_node.running = False
        thread.join()
        if robot_node.video_writer is not None:
            robot_node.video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()