"""Clasificador del modelo de medidor: red, preprocesamiento y carga de pesos.

Lo usan scripts/train_meter_model.py para entrenar y el pipeline (load_classifier) para
predecir. La entrada es el visor con su entorno armado por layout.model_input. torch y
torchvision se importan dentro de las funciones: el resto del paquete no los necesita.
"""

from pathlib import Path

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # normalización de ImageNet (RGB)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ARCHS = ("mobilenet_v3_small", "mobilenet_v3_large", "resnet18")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """CNN de torchvision con la última capa reemplazada por una de num_classes salidas."""
    import torch
    from torchvision import models

    if arch.startswith("mobilenet_v3"):
        net = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        net = models.resnet18(weights="DEFAULT" if pretrained else None)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"arquitectura desconocida: {arch}")
    return net


def to_tensor(images: list[np.ndarray]):
    """Lista de imágenes BGR uint8 → tensor normalizado (N, 3, H, W)."""
    import torch

    rgb = np.stack([img[:, :, ::-1] for img in images]).astype(np.float32) / 255
    return torch.from_numpy((rgb - MEAN) / STD).permute(0, 3, 1, 2).contiguous()


def predict_probs(net, images: list[np.ndarray], device="cpu") -> np.ndarray:
    """Probabilidades por clase, promediando la imagen y su giro de 180° (el visor puede venir invertido)."""
    import torch

    net.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(images), 32):
            x = to_tensor(images[start:start + 32]).to(device)
            p = net(x).softmax(1) + net(torch.flip(x, dims=(2, 3))).softmax(1)
            probs.append((p / 2).cpu().numpy())
    return np.concatenate(probs)


def load_classifier(weights: str | Path, device="cpu"):
    """Clasificador listo para el pipeline: visor con entorno → (clase, probabilidad)."""
    import torch

    checkpoint = torch.load(weights, map_location="cpu")
    classes = checkpoint["classes"]
    net = build_model(checkpoint["arch"], len(classes), pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device).eval()

    def classify(image: np.ndarray) -> tuple[str, float]:
        probs = predict_probs(net, [image], device)[0]
        best = int(probs.argmax())
        return classes[best], float(probs[best])

    return classify
