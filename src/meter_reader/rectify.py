"""Rectificación del visor: rotarlo para que quede horizontal y recortarlo.

Se usa en dos lugares:
  - al armar el dataset de dígitos (el visor se deriva de los dígitos anotados);
  - en el pipeline, a partir de la máscara predicha por la segmentación.

El visor se aproxima con el rectángulo rotado mínimo que contiene al polígono.
Se elige la rotación más chica que deja su lado largo horizontal, de modo que un
visor apenas inclinado conserve el orden de lectura (izquierda a derecha). Un
visor invertido (180°) no se distingue geométricamente: se resuelve en la
etapa de lectura.
"""

import cv2
import numpy as np


def rectify_matrix(polygon: np.ndarray, pad: float = 0.1) -> tuple[np.ndarray, tuple[int, int]]:
    """Calcula la transformación afín que endereza el visor.

    Args:
        polygon: puntos del visor en píxeles, forma (N, 2).
        pad: margen relativo agregado a cada lado del visor.

    Returns:
        La matriz afín 2x3 y el tamaño (ancho, alto) del recorte resultante.
    """
    box = cv2.boxPoints(cv2.minAreaRect(np.asarray(polygon, dtype=np.float32)))
    edges = [box[(i + 1) % 4] - box[i] for i in range(2)]
    lengths = [float(np.linalg.norm(e)) for e in edges]
    long_i = int(np.argmax(lengths))
    direction = edges[long_i] / lengths[long_i]
    # Se orienta el eje largo hacia la derecha: mínima rotación, se conserva el orden de lectura.
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    cos, sin = direction

    width = round(lengths[long_i] * (1 + 2 * pad))
    height = round(lengths[1 - long_i] * (1 + 2 * pad))

    # Rotación que lleva "direction" al eje x (coordenadas de imagen, y hacia abajo).
    rotation = np.array([[cos, sin], [-sin, cos]])
    center = box.mean(axis=0)
    translation = np.array([width / 2, height / 2]) - rotation @ center
    return np.hstack([rotation, translation[:, None]]), (width, height)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Aplica la transformación afín a puntos (N, 2)."""
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def bbox_crop(image: np.ndarray, polygon: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Recorte por la caja alineada a los ejes del visor, sin enderezar (línea de base del
    experimento rectificar contra recortar). Usa el mismo margen relativo que rectify."""
    x0, y0 = np.asarray(polygon, dtype=np.float64).min(axis=0)
    x1, y1 = np.asarray(polygon, dtype=np.float64).max(axis=0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    height, width = image.shape[:2]
    left, top = max(0, int(round(x0 - dx))), max(0, int(round(y0 - dy)))
    right, bottom = min(width, int(round(x1 + dx))), min(height, int(round(y1 + dy)))
    return image[top:bottom, left:right].copy()


def rectify(image: np.ndarray, polygon: np.ndarray, pad: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve el recorte del visor enderezado y la matriz usada."""
    matrix, (width, height) = rectify_matrix(polygon, pad)
    crop = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix
