import numpy as np

class PointMassEnv:
    """
    Реализует синтетическую среду 1D точечной массы.

    Эта среда имитирует точечную массу, движущуюся к цели.
    Включает два набора динамики:
    1.  'Симулятор': Смещенная модель с низким трением и шумом.
    2.  'Реальная': Модель с более высоким трением и шумом, представляющая реальный мир.
    """
    def __init__(
        self,
        sim_noise_std: float = 0.1,
        sim_friction_factor: float = 1.0,
        real_noise_std: float = 0.5,
        real_friction_factor: float = 0.7,
        tolerance: float = 0.5,
        T: int = 10 # Time steps for the rollout
    ):
        """
        Инициализирует среду с параметрами динамики.

        Аргументы:
            sim_noise_std (float): Стандартное отклонение шума действия симулятора.
            sim_friction_factor (float): Коэффициент трения для симулятора.
            real_noise_std (float): Стандартное отклонение шума действия реального мира.
            real_friction_factor (float): Коэффициент трения для реального мира.
            tolerance (float): Радиус успешного достижения цели.
            T (int): Количество шагов времени в одном эпизоде.
        """
        self.dynamics_params = {
            'simulator': {'noise_std': sim_noise_std, 'friction': sim_friction_factor},
            'real': {'noise_std': real_noise_std, 'friction': real_friction_factor},
        }
        self.tolerance = tolerance
        self.T = T

    def run_episode(self, context: float, mode: str = 'simulator') -> bool:
        """
        Запускает один эпизод (прогон) для заданного контекста и режима.

        Аргументы:
            context (float): Целевое расстояние 'd'.
            mode (str): Либо 'simulator', либо 'real'.

        Возвращает:
            bool: True, если эпизод был успешным, False в противном случае.
        """
        if mode not in self.dynamics_params:
            raise ValueError("Mode must be either 'simulator' or 'real'")

        params = self.dynamics_params[mode]
        noise_std = params['noise_std']
        friction = params['friction']

        # Контроллер: фиксированная скорость для достижения расстояния d за T шагов без трения
        v = context / self.T
        
        position = 0.0
        for _ in range(self.T):
            # Применение динамики
            control_input = friction * v
            noise = np.random.normal(0, noise_std)
            position += control_input + noise

        # Проверка на успех
        success = np.abs(position - context) <= self.tolerance
        return success

    def estimate_success_prob(self, context: float, num_rollouts: int = 200) -> float:
        """
        Оценивает вероятность успеха для заданного контекста с использованием Монте-Карло.
        Используется для вычисления p_s(x) из симулятора.

        Аргументы:
            context (float): Целевое расстояние 'd'.
            num_rollouts (int): Количество эпизодов для оценки.

        Возвращает:
            float: Оценочная вероятность успеха.
        """
        success_count = 0
        for _ in range(num_rollouts):
            if self.run_episode(context, mode='simulator'):
                success_count += 1
        
        return success_count / num_rollouts

    def get_real_outcome(self, context: float) -> int:
        """
        Получает один результат из 'реальной' среды.

        Аргументы:
            context (float): Целевое расстояние 'd'.

        Возвращает:
            int: 1 для успеха, 0 для неудачи.
        """
        return 1 if self.run_episode(context, mode='real') else 0
