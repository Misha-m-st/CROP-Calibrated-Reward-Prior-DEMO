import numpy as np
import pandas as pd
import torch
from environment import PointMassEnv
import os
from learning import CROP_GP_GPyTorch as CROP_GP, PlattScaling, IsotonicRegressionModel, TemperatureScaling, Acquisition
from metrics import nll_score, brier_score, expected_calibration_error
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    """Устанавливает начальное значение для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Начальное значение генератора случайных чисел установлено в {seed}")

def generate_simulator_data():
    """
    Генерирует вероятности успеха симулятора (p_s(x)) для сетки контекстов
    и сохраняет их в CSV-файл.
    """
    print("Генерация вероятностей успеха симулятора...")
    distances = np.linspace(0, 10, 200)
    num_rollouts = 200
    env = PointMassEnv()
    probabilities = [env.estimate_success_prob(d, num_rollouts=num_rollouts) for d in distances]
    df = pd.DataFrame({'context': distances, 'p_simulator': probabilities})
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/simulator_probabilities.csv', index=False)
    print("Данные симулятора сохранены.")

def generate_real_data(num_trials_per_context=5):
    """
    Генерирует набор данных реальных результатов (y_i) для сетки контекстов.
    """
    print("\nГенерация данных реального мира...")
    distances = np.linspace(0, 10, 200)
    env = PointMassEnv()
    results = [{'context': d, 'outcome': env.get_real_outcome(d)} for d in distances for _ in range(num_trials_per_context)]
    df = pd.DataFrame(results)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/real_outcomes.csv', index=False)
    print("Данные реального мира сохранены.")

def run_passive_learning_curve_experiment(num_runs=5, total_budget=200, initial_samples=10, batch_size=10):
    """
    Проводит эксперимент с кривой обучения для всех пассивных моделей
    за несколько запусков и усредняет результаты.
    """
    print(f"\n--- Запуск эксперимента с пассивной кривой обучения ({num_runs} запусков) ---")
    
    sim_df = pd.read_csv('data/raw/simulator_probabilities.csv')
    real_df = pd.read_csv('data/raw/real_outcomes.csv')
    full_dataset = pd.merge(real_df, sim_df, on='context')

    X_full = full_dataset['context'].values
    y_full = full_dataset['outcome'].values
    p_sim_full = full_dataset['p_simulator'].values

    all_model_names = ["CROP-GP", "Platt", "Isotonic", "TempScale"]
    all_metrics = ['NLL', 'Brier', 'ECE']
    # Store history for each run: { 'CROP-GP': [run1_df, run2_df, ...], ... }
    run_histories = {name: [] for name in all_model_names}
    
    trial_counts = [initial_samples + i * batch_size for i in range((total_budget - initial_samples) // batch_size + 1)]

    for run in range(num_runs):
        print(f"\n--- Пассивный запуск {run + 1}/{num_runs} ---")
        set_seed(42 + run)

        # Define fresh models for each run
        models = {
            "CROP-GP": CROP_GP(random_state=42 + run),
            "Platt": PlattScaling(),
            "Isotonic": IsotonicRegressionModel(),
            "TempScale": TemperatureScaling()
        }
        
        performance_history = {name: [] for name in all_model_names}
        
        shuffled_indices = np.random.permutation(len(X_full))
        X_shuffled, y_shuffled, p_sim_shuffled = X_full[shuffled_indices], y_full[shuffled_indices], p_sim_full[shuffled_indices]

        for num_trials in trial_counts:
            X_train, y_train, p_sim_train = X_shuffled[:num_trials], y_shuffled[:num_trials], p_sim_shuffled[:num_trials]
            X_test, y_test, p_sim_test = X_full, y_full, p_sim_full

            for name, model in models.items():
                try:
                    if len(np.unique(y_train)) < 2:
                        print(f"  Пропуск {name} для {num_trials} испытаний (присутствует только один класс).")
                        nll, brier, ece = np.nan, np.nan, np.nan
                    else:
                        model.fit(X_train, y_train, p_sim_train)
                        
                        if name == "CROP-GP":
                            p_eval, _ = model.predict(X_test, p_sim_test)
                        else:
                            p_eval = model.predict(X_test, p_sim_test)
                        
                        nll = nll_score(y_test, p_eval)
                        brier = brier_score(y_test, p_eval)
                        ece = expected_calibration_error(y_test, p_eval)
                except Exception as e:
                    print(f"  Ошибка при подгонке/предсказании {name} для {num_trials} испытаний: {e}")
                    nll, brier, ece = np.nan, np.nan, np.nan
                
                performance_history[name].append({
                    'trials': num_trials,
                    'NLL': nll,
                    'Brier': brier,
                    'ECE': ece
                })

        for name, history in performance_history.items():
            run_histories[name].append(pd.DataFrame(history))

    # Average the results across runs
    averaged_histories = {}
    for name, dfs in run_histories.items():
        # Concatenate all runs, group by 'trials', and compute mean/std
        combined_df = pd.concat(dfs).groupby('trials')
        mean_df = combined_df.mean()
        std_df = combined_df.std()
        mean_df.columns = [f'{col}_mean' for col in mean_df.columns]
        std_df.columns = [f'{col}_std' for col in std_df.columns]
        averaged_histories[name] = mean_df.join(std_df).reset_index()
        
    print("\n--- Эксперимент с пассивной кривой обучения завершен ---")
    return averaged_histories

def run_active_learning_experiment(num_runs=5, total_budget=200, initial_samples=10, batch_size=10):
    """
    Проводит полный эксперимент по активному обучению для CROP-GP за несколько запусков.
    """
    print(f"\n--- Запуск эксперимента с активным обучением для CROP-GP ({num_runs} запусков) ---")
    
    sim_df = pd.read_csv('data/raw/simulator_probabilities.csv')
    test_df = pd.read_csv('data/raw/real_outcomes.csv')
    test_df = pd.merge(test_df, sim_df, on='context')

    X_candidates, p_sim_candidates = sim_df['context'].values, sim_df['p_simulator'].values
    X_test, y_test, p_sim_test = test_df['context'].values, test_df['outcome'].values, test_df['p_simulator'].values
    
    run_histories = []
    
    # Define hyperparameters
    alpha = 2.0
    beta = 0.5
    
    for run in range(num_runs):
        print(f"\n--- Активный запуск {run + 1}/{num_runs} ---")
        set_seed(1337 + run)
        
        env = PointMassEnv()
        
        # Create initial random dataset
        initial_indices = np.random.choice(len(X_candidates), size=initial_samples, replace=False)
        
        # Keep track of used indices to avoid re-sampling
        used_indices = set(initial_indices)
        
        X_train = X_candidates[initial_indices]
        p_sim_train = p_sim_candidates[initial_indices]
        y_train = np.array([env.get_real_outcome(x) for x in X_train])

        crop_model = CROP_GP(random_state=1337 + run)
        acquisition = Acquisition(alpha=alpha, beta=beta) # Tuned beta down
        
        performance_history = []
        num_rounds = (total_budget - initial_samples) // batch_size
        
        for i in range(num_rounds + 1):
            num_trials = len(y_train)
            
            crop_model.fit(X_train, y_train, p_sim_train)
            
            p_eval, _ = crop_model.predict(X_test, p_sim_test)
            nll = nll_score(y_test, p_eval)
            brier = brier_score(y_test, p_eval)
            ece = expected_calibration_error(y_test, p_eval)
            performance_history.append({'trials': num_trials, 'NLL': nll, 'Brier': brier, 'ECE': ece})
            print(f"  Раунд {i+1}, Испытаний: {num_trials}, Тестовый NLL: {nll:.4f}, Брайер: {brier:.4f}, ECE: {ece:.4f}")
            
            if num_trials >= total_budget:
                break
            
            candidate_pool_indices = np.array([i for i in range(len(X_candidates)) if i not in used_indices])
            if len(candidate_pool_indices) == 0:
                print("Все кандидаты были выбраны. Остановка.")
                break
                
            X_candidate_pool = X_candidates[candidate_pool_indices]
            p_sim_candidate_pool = p_sim_candidates[candidate_pool_indices]
            
            _, next_indices_in_pool = acquisition.select_next_batch(crop_model, X_candidate_pool, p_sim_candidate_pool, batch_size)
            
            next_original_indices = candidate_pool_indices[next_indices_in_pool]
            
            next_contexts = X_candidates[next_original_indices]
            next_outcomes = np.array([env.get_real_outcome(x) for x in next_contexts])
            next_p_sim = p_sim_candidates[next_original_indices]
            
            X_train = np.concatenate([X_train, next_contexts])
            y_train = np.concatenate([y_train, next_outcomes])
            p_sim_train = np.concatenate([p_sim_train, next_p_sim])
            used_indices.update(next_original_indices)
            
        run_histories.append(pd.DataFrame(performance_history))

    # Average the results across runs
    combined_df = pd.concat(run_histories).groupby('trials')
    mean_df = combined_df.mean()
    std_df = combined_df.std()
    mean_df.columns = [f'{col}_mean' for col in mean_df.columns]
    std_df.columns = [f'{col}_std' for col in std_df.columns]
    averaged_history = mean_df.join(std_df).reset_index()
    
    print("\n--- Эксперимент с активным обучением завершен ---")
    return averaged_history, alpha, beta

def plot_results(active_history_df, passive_histories_dict, alpha, beta, initial_samples):
    """
    Генерирует и сохраняет график сравнения производительности активного и пассивного обучения
    с доверительными интервалами.
    """
    print("\n--- Генерация графика эффективности выборки ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if active_history_df is not None:
        trials = active_history_df['trials']
        mean = active_history_df['NLL_mean']
        std = active_history_df['NLL_std'].fillna(0)
        ax.plot(trials, mean, marker='o', linestyle='-', label='CROP-GP (Активное)', linewidth=2.5, markersize=8, zorder=10)
        ax.fill_between(trials, mean - std, mean + std, alpha=0.2, zorder=9)
    
    styles = {'Platt': ':', 'Isotonic': '--', 'TempScale': '-.', 'CROP-GP': '-'}
    markers = {'Platt': 'x', 'Isotonic': 's', 'TempScale': '^', 'CROP-GP': 'd'}
    
    if passive_histories_dict is not None:
        for name, df in passive_histories_dict.items():
            label_name = f'{name} (Пассивное)' if name != 'CROP-GP' else 'CROP-GP (Пассивное)'
            trials = df['trials']
            mean = df['NLL_mean']
            std = df['NLL_std'].fillna(0)
            
            ax.plot(trials, mean, marker=markers.get(name), linestyle=styles.get(name), label=label_name)
            ax.fill_between(trials, mean - std, mean + std, alpha=0.15)
            
    hyperparam_text = (
        f'Гиперпараметры (Активное обучение):\n'
        f'  alpha = {alpha}\n'
        f'  beta = {beta}\n'
        f'  initial_samples = {initial_samples}'
    )
    ax.text(0.95, 0.95, hyperparam_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
    ax.set_xlabel("Количество испытаний в реальном мире", fontsize=14)
    ax.set_ylabel("Отрицательное логарифмическое правдоподобие (NLL)", fontsize=14)
    ax.set_title("Сравнение эффективности выборки (усреднено по 5 запускам)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0.3, top=0.8)
    ax.set_xlim(left=0)
    plt.tight_layout()

    plot_path = 'sample_efficiency_comparison_final.png'
    plt.savefig(plot_path)
    print(f"График сохранен в {plot_path}")

def plot_calibration_curves(models, X_test, y_test, p_sim_test):
    """
    Строит и сохраняет калибровочные кривые для всех обученных моделей.
    """
    print("\n--- Генерация калибровочного графика ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot([0, 1], [0, 1], "k:", label="Идеально откалибровано")
    
    for name, model in models.items():
        if not getattr(model, 'is_fitted', False):
            print(f"Пропуск калибровочного графика для необученной модели: {name}")
            continue
            
        if name == "CROP-GP":
            p_eval, _ = model.predict(X_test, p_sim_test)
        else:
            p_eval = model.predict(X_test, p_sim_test)
            
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, p_eval, n_bins=10)
        
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)

    ax.set_xlabel("Средняя предсказанная вероятность", fontsize=14)
    ax.set_ylabel("Доля положительных результатов", fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('Калибровочные кривые (после полного обучения)', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plot_path = 'calibration_curves.png'
    plt.savefig(plot_path)
    print(f"График сохранен в {plot_path}")

def plot_extra_metrics(active_history_df, passive_histories_dict):
    """
    Генерирует и сохраняет графики для оценки Брайера и ECE.
    """
    print("\n--- Генерация графиков оценки Брайера и ECE ---")
    
    metrics_to_plot = ['Brier', 'ECE']
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if active_history_df is not None:
            trials = active_history_df['trials']
            mean = active_history_df[f'{metric}_mean']
            std = active_history_df[f'{metric}_std'].fillna(0)
            ax.plot(trials, mean, marker='o', linestyle='-', label='CROP-GP (Активное)', linewidth=2.5, markersize=8, zorder=10)
            ax.fill_between(trials, mean - std, mean + std, alpha=0.2, zorder=9)
        
        styles = {'Platt': ':', 'Isotonic': '--', 'TempScale': '-.', 'CROP-GP': '-'}
        markers = {'Platt': 'x', 'Isotonic': 's', 'TempScale': '^', 'CROP-GP': 'd'}
        
        if passive_histories_dict is not None:
            for name, df in passive_histories_dict.items():
                label_name = f'{name} (Пассивное)' if name != 'CROP-GP' else 'CROP-GP (Пассивное)'
                trials = df['trials']
                mean = df[f'{metric}_mean']
                std = df[f'{metric}_std'].fillna(0)
                
                ax.plot(trials, mean, marker=markers.get(name), linestyle=styles.get(name), label=label_name)
                ax.fill_between(trials, mean - std, mean + std, alpha=0.15)
        
        ax.set_xlabel("Количество испытаний в реальном мире", fontsize=14)
        ax.set_ylabel(f"Оценка {metric}", fontsize=14)
        ax.set_title(f"Эффективность выборки: {metric} (усреднено по 5 запускам)", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(left=0)
        
        if metric == 'Brier':
            ax.set_ylim(bottom=0.1, top=0.25)
        elif metric == 'ECE':
            ax.set_ylim(bottom=0, top=0.4)

        plt.tight_layout()
        plot_path = f'{metric.lower()}_score_comparison.png'
        plt.savefig(plot_path)
        print(f"График сохранен в {plot_path}")

if __name__ == '__main__':
    print("Генерация новых данных для эксперимента...")
    set_seed(0)
    generate_simulator_data()
    generate_real_data()
    print("Генерация данных завершена.")

    # --- Run experiments ---
    NUM_RUNS = 5
    INITIAL_SAMPLES = 10
    
    # Run active learning for CROP-GP
    active_history, alpha, beta = run_active_learning_experiment(num_runs=NUM_RUNS, initial_samples=INITIAL_SAMPLES)
    
    # Run passive learning curves for all models
    passive_histories = run_passive_learning_curve_experiment(num_runs=NUM_RUNS, initial_samples=INITIAL_SAMPLES)
    
    # --- Plot the results ---
    plot_results(active_history, passive_histories, alpha=alpha, beta=beta, initial_samples=INITIAL_SAMPLES)
    plot_extra_metrics(active_history, passive_histories)

    print("\n--- Обучение финальных моделей для калибровочного графика ---")
    sim_df = pd.read_csv('data/raw/simulator_probabilities.csv')
    real_df = pd.read_csv('data/raw/real_outcomes.csv')
    full_dataset = pd.merge(real_df, sim_df, on='context')
    X_full = full_dataset['context'].values
    y_full = full_dataset['outcome'].values
    p_sim_full = full_dataset['p_simulator'].values
    
    final_models = {
        "CROP-GP": CROP_GP(random_state=42),
        "Platt": PlattScaling(),
        "Isotonic": IsotonicRegressionModel(),
        "TempScale": TemperatureScaling()
    }
    
    for name, model in final_models.items():
        print(f"Обучение {name} на полном наборе данных...")
        train_indices = np.random.choice(len(X_full), size=200, replace=False)
        model.fit(X_full[train_indices], y_full[train_indices], p_sim_full[train_indices])

    plot_calibration_curves(final_models, X_full, y_full, p_sim_full)
