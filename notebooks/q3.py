# %% [markdown]
# # 3
# ## f
# ### i

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt


def beta_ordinary(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the ordinary least squares estimator.

    Parameters:
    X : np.ndarray
        Input matrix of shape (samples, features + 1) or (n,p).
    Y : np.ndarray
        Output vector of shape (samples,) or (n,).
    Returns:
    beta : np.ndarray
        Estimated coefficients vector of shape (features + 1,) or (p,).

    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta


np.random.seed(1)
beta = np.array([-1.5, 2.0])
input_range = np.linspace(-1, 1, 100)
X = np.vstack([np.ones(100), input_range]).T
y = X @ beta + np.random.normal(0, 0.3, 100)


def calculate_laplacian_loss(parameters, X, y):
    return np.sum(np.abs(y - X @ parameters))


epsilon = y - X @ beta
n = len(epsilon)


# %%
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, color="black", label="Data points")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Data")
plt.legend()
plt.savefig(
    "../reports/figures/3/data_scatter.png", dpi=300, bbox_inches="tight"
)
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.hist(epsilon, bins=20, color="gray", edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Errors")
plt.savefig(
    "../reports/figures/3/errors_histogram.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.bar(range(1, n + 1), epsilon)
plt.xlabel("Index")
plt.ylabel("Error")
plt.title("Values of Errors")
plt.savefig(
    "../reports/figures/3/error_values_by_index.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# %%
def calculate_beta_hat(
    X: np.ndarray, Y: np.ndarray, error_distribution: str
) -> np.ndarray:
    """
    Compute the estimator beta_hat based on the specified error distribution.
    Parameters:
    X : np.ndarray
        Input matrix of shape (samples, features + 1) or (n,p).
    Y : np.ndarray
        Output vector of shape (samples,) or (n,).
    error_distribution : str
        Type of error distribution ('gaussian' or 'laplacian').
    Returns:
    beta_hat : np.ndarray
        Estimated coefficients vector of shape (features + 1,) or (p,).
    """

    np.random.seed(0)

    p = X.shape[1]
    if error_distribution == "gaussian":

        def loss_function(beta):
            return np.sum((Y - X @ beta) ** 2)
    elif error_distribution == "laplacian":

        def loss_function(beta):
            return np.sum(np.abs(Y - X @ beta))
    else:
        raise ValueError("Unsupported error distribution")

    beta_0 = np.random.uniform(size=p)
    beta_hat = scipy.optimize.minimize(loss_function, beta_0)
    return beta_hat["x"]


beta_hat_gaussian = calculate_beta_hat(X, y, "gaussian")
beta_hat_gaussian_old_function = beta_ordinary(X, y)
beta_hat_laplacian = calculate_beta_hat(X, y, "laplacian")

# Print results
print("True Coefficients:", beta)
print("Coefficients (Gaussian with minimize):", beta_hat_gaussian)
print(
    "Coefficients (Gaussian closed form):", beta_hat_gaussian_old_function
)
print("Coefficients (Laplacian):", beta_hat_laplacian)

# %% [markdown]
# ### ii

# %%
yhat_gaussina = X @ beta_hat_gaussian
yhat_laplace = X @ beta_hat_laplacian
yhat_real = X @ beta

plt.figure(figsize=(16, 12))
plt.scatter(X[:, 1], y, color="black", label="Data points")
plt.plot(
    X[:, 1], yhat_gaussina, color="blue", label="Gaussian Fit", linewidth=2
)
plt.plot(
    X[:, 1], yhat_laplace, color="red", label="Laplacian Fit", linewidth=2
)
plt.plot(
    X[:, 1], yhat_real, color="green", label="True Function", linewidth=2
)
plt.title("Regression Fits")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.savefig(
    "../reports/figures/3/regression_fits_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
error_gaussian = np.linalg.norm(beta - beta_hat_gaussian, ord=2)
error_laplacian = np.linalg.norm(beta - beta_hat_laplacian, ord=2)
print("Norma do erro do estimador Gaussian:", error_gaussian)
print("Norma do erro do estimador Laplacian:", error_laplacian)

# %% [markdown]
# ### iii

# %%
# Regenerate data for reproducibility
np.random.seed(1)
beta = np.array([-1.5, 2.0])
input_range = np.linspace(-1, 1, 100)
X = np.vstack([np.ones(100), input_range]).T
y = X @ beta + np.random.normal(0, 0.3, 100)
y[80] = 10

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, color="black", label="Data points")

# %%
# Calculate new estimators
beta_hat_gaussian_outlier = calculate_beta_hat(X, y, "gaussian")
beta_hat_gaussian_old_outlier = beta_ordinary(X, y)
beta_hat_laplacian_outlier = calculate_beta_hat(X, y, "laplacian")

print("True Coefficients:", beta)
print(
    "Coefficients (Gaussian with minimize) with outlier:",
    beta_hat_gaussian_outlier,
)
print(
    "Coefficients (Gaussian closed form) with outlier:",
    beta_hat_gaussian_old_outlier,
)
print("Coefficients (Laplacian) with outlier:", beta_hat_laplacian_outlier)

# %%
error_gaussian_outlier = np.linalg.norm(
    beta - beta_hat_gaussian_outlier, ord=2
)
error_laplacian_outlier = np.linalg.norm(
    beta - beta_hat_laplacian_outlier, ord=2
)
print(
    "L2 Error of estimators with outlier (Gaussian):",
    error_gaussian_outlier,
)
print(
    "L2 Error of estimators with outlier (Laplacian):",
    error_laplacian_outlier,
)

# %%
yhat_gaussina_outlier = X @ beta_hat_gaussian_outlier
yhat_laplace_outlier = X @ beta_hat_laplacian_outlier

plt.figure(figsize=(16, 12))
plt.scatter(X[:, 1], y, color="black", label="Data points")
plt.plot(
    X[:, 1],
    yhat_gaussina_outlier,
    color="blue",
    label="Gaussian Fit with outlier",
    linewidth=2,
)
plt.plot(
    X[:, 1],
    yhat_laplace_outlier,
    color="red",
    label="Laplacian Fit with outlier",
    linewidth=2,
)
plt.plot(
    X[:, 1], yhat_real, color="green", label="True Function", linewidth=2
)
plt.title("Regression Fits")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.savefig(
    "../reports/figures/3/regression_fits_with_outlier.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
