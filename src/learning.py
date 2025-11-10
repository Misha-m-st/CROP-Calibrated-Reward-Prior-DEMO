import numpy as np
import torch
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

# Вспомогательные функции для операций в логит-пространстве
def to_logit(p):
    """Преобразует вероятность в логит."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))

def from_logit(l):
    """Преобразует логит в вероятность."""
    return 1 / (1 + np.exp(-l))


class GPClassificationModel(gpytorch.models.ApproximateGP):
    """
    Модель GP классификации на GPyTorch.
    Моделирует скрытую функцию f(x). Вероятность p(x) = sigmoid(f(x)).
    """
    def __init__(self, train_x, prior_mean_shape):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class CROP_GP_GPyTorch:
    """
    CROP с использованием GPyTorch для надежной GP классификации.
    
    Модель реализует идею CROP: logit(p_real) = logit(p_sim) + g(x),
    где g(x) моделируется гауссовским процессом.
    """
    def __init__(self, random_state=None, training_iterations=50, learning_rate=0.1):
        if random_state:
            torch.manual_seed(random_state)
        
        self.model = None
        self.likelihood = None
        self.is_fitted = False
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        self.train_x_tensor = None

    def fit(self, X_train, y_train, p_sim_train):
        """
        Обучает модель GP на обучающих данных.
        """
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        p_sim_train = torch.from_numpy(p_sim_train).float()
        
        if X_train.ndim == 1:
            X_train = X_train.unsqueeze(-1)

        self.train_x_tensor = X_train
        logit_p_sim = torch.tensor(to_logit(p_sim_train.numpy()), dtype=torch.float)

        self.model = GPClassificationModel(train_x=X_train, prior_mean_shape=logit_p_sim.shape)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            X_train, y_train, logit_p_sim = X_train.cuda(), y_train.cuda(), logit_p_sim.cuda()

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_train.size(0))

        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model(X_train)
            f_dist = gpytorch.distributions.MultivariateNormal(output.mean + logit_p_sim, output.lazy_covariance_matrix)
            loss = -mll(f_dist, y_train)
            loss.backward()
            optimizer.step()
        
        self.is_fitted = True

    def predict(self, X_test, p_sim_test):
        """
        Предсказывает калиброванные вероятности и их неопределенность.
        """
        if not self.is_fitted:
            raise RuntimeError("Модель еще не обучена. Сначала вызовите fit().")

        X_test = torch.from_numpy(X_test).float()
        p_sim_test = torch.from_numpy(p_sim_test).float()
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(-1)
        
        logit_p_sim_test = torch.tensor(to_logit(p_sim_test.numpy()), dtype=torch.float)

        if torch.cuda.is_available():
            X_test, logit_p_sim_test = X_test.cuda(), logit_p_sim_test.cuda()

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            g_dist = self.model(X_test)
            f_mean = g_dist.mean + logit_p_sim_test
            f_var = g_dist.variance
            
            p_calibrated = from_logit(f_mean.cpu().numpy())
            logit_variance = f_var.cpu().numpy()

        return p_calibrated, logit_variance


# Базовые модели

class PlattScaling:
    """
    Масштабирование Платта / Логистическая калибровка.
    """
    def __init__(self):
        self.model = LogisticRegression()
        self.is_fitted = False

    def fit(self, X_train, y_train, p_sim_train):
        logits_sim = to_logit(p_sim_train).reshape(-1, 1)
        self.model.fit(logits_sim, y_train)
        self.is_fitted = True

    def predict(self, X_test, p_sim_test):
        if not self.is_fitted:
            raise RuntimeError("Модель еще не обучена.")
        logits_sim = to_logit(p_sim_test).reshape(-1, 1)
        p_calibrated = self.model.predict_proba(logits_sim)[:, 1]
        return p_calibrated


class IsotonicRegressionModel:
    """
    Изотоническая регрессия для калибровки.
    """
    def __init__(self):
        self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, X_train, y_train, p_sim_train):
        self.model.fit(p_sim_train, y_train)
        self.is_fitted = True

    def predict(self, X_test, p_sim_test):
        if not self.is_fitted:
            raise RuntimeError("Модель еще не обучена.")
        return self.model.predict(p_sim_test)


class TemperatureScaling:
    """
    Температурное масштабирование для калибровки.
    """
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def _nll(self, temp, logits, labels):
        scaled_logits = logits / temp
        p = from_logit(scaled_logits)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    def fit(self, X_train, y_train, p_sim_train):
        logits_sim = to_logit(p_sim_train)
        res = minimize(
            self._nll,
            x0=1.0,
            args=(logits_sim, y_train),
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)]
        )
        self.temperature = res.x[0]
        self.is_fitted = True
        print(f"Оптимальная температура: {self.temperature:.4f}")

    def predict(self, X_test, p_sim_test):
        if not self.is_fitted:
            raise RuntimeError("Модель еще не обучена.")
        logits_sim = to_logit(p_sim_test)
        scaled_logits = logits_sim / self.temperature
        return from_logit(scaled_logits)


class Acquisition:
    """
    Обрабатывает активное получение сэмплов.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def select_next_batch(self, crop_model, X_candidates, p_sim_candidates, batch_size=10):
        if not crop_model.is_fitted:
            print("Внимание: Модель CROP не обучена. Используется случайная выборка.")
            indices = np.random.choice(len(X_candidates), size=batch_size, replace=False)
            return X_candidates[indices], indices

        _, logit_variance = crop_model.predict(X_candidates, p_sim_candidates)

        epsilon = 1e-9
        acq_scores = (p_sim_candidates**self.alpha) * ((logit_variance + epsilon)**self.beta)

        n_candidates = len(X_candidates)
        actual_batch_size = min(batch_size, n_candidates)
        
        top_indices = np.argpartition(acq_scores, -actual_batch_size)[-actual_batch_size:]
        
        top_scores = acq_scores[top_indices]
        sorted_top_indices = top_indices[np.argsort(top_scores)[::-1]]

        print("\n--- Выбор кандидатов для активного обучения ---")
        print(f"Выбрано {actual_batch_size} лучших кандидатов:")
        for i in sorted_top_indices:
            context_val = X_candidates[i].item() if X_candidates[i].size == 1 else X_candidates[i]
            print(f"  Индекс: {i}, Контекст: {context_val:.2f}, Оценка: {acq_scores[i]:.4f} (p_sim={p_sim_candidates[i]:.2f}, var={logit_variance[i]:.2f})")

        return X_candidates[sorted_top_indices], sorted_top_indices
