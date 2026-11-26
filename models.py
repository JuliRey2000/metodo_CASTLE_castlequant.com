"""
models.py - Modelos en PyTorch
==============================
El modelo lineal que predice el retorno futuro a partir de los retornos anteriores.

Proyecto: Castle Quant · Método CASTLE (A · Alfa)
Adaptado de la Mentoría Quant y del curso "Build a Quant Trading System" de MemLabs (licencia MIT).
"""

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)  # Una sola salida (predicción del retorno)

    def forward(self, x):
        return self.linear(x)


def load_model(path: str = "model_weights.pth") -> LinearModel:
    """Carga un modelo lineal guardado; el número de variables se lee del propio archivo."""
    state = torch.load(path, weights_only=True)     # weights_only=True: solo se cargan los pesos, nunca código
    model = LinearModel(state["linear.weight"].shape[1])
    model.load_state_dict(state)
    model.eval()
    return model
