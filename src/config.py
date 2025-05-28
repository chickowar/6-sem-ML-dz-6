from dataclasses import dataclass

@dataclass
class Config:
    # Обучение
    num_epochs: int = 5000
    batch_size: int = 128
    lr: float = 1e-4
    # Архитектура
    z_dim: int = 1
    hidden_dim: int = 64
    # Диапазон x для гамма-функции
    x_min: float = 1.0
    x_max: float = 3.0
    # Устройство
    device: str = 'cpu'
    # Отладка и визуализация
    print_every: int = 500
    plot: bool = True