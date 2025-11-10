import numpy as np

def nll_score(y_true, y_pred):
    """
    Вычисляет отрицательное логарифмическое правдоподобие (кросс-энтропию).

    Аргументы:
        y_true (np.ndarray): Истинные бинарные метки (0 или 1).
        y_pred (np.ndarray): Предсказанные вероятности.

    Возвращает:
        float: Средний балл NLL.
    """
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def brier_score(y_true, y_pred):
    """
    Вычисляет оценку Брайера.

    Аргументы:
        y_true (np.ndarray): Истинные бинарные метки (0 или 1).
        y_pred (np.ndarray): Предсказанные вероятности.

    Возвращает:
        float: Оценка Брайера.
    """
    return np.mean((y_pred - y_true)**2)

def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Вычисляет ожидаемую ошибку калибровки (ECE).

    Аргументы:
        y_true (np.ndarray): Истинные бинарные метки (0 или 1).
        y_pred (np.ndarray): Предсказанные вероятности.
        n_bins (int): Количество интервалов для разделения вероятностей.

    Возвращает:
        float: Оценка ECE.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            confidence_in_bin = np.mean(y_pred[in_bin])
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

    return ece
