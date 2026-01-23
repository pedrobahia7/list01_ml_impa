# Q5
# %%
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from IPython.display import display
import sys

# Redirect prints to file
original_stdout = sys.stdout

bodyfat = pd.read_csv("../data/bodyfat.csv")
X = bodyfat.drop(columns=["BodyFat", "Density"])
y = bodyfat["BodyFat"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

kf = KFold(n_splits=5, shuffle=True, random_state=10)
cv_fold = np.zeros(len(y_train)).astype(int)

for i, (_, fold_indexes) in enumerate(kf.split(X_train)):
    cv_fold[fold_indexes] = int(i)

# %%
print("X_train:")
display(X_train.head())
display(X_train.shape)

print("\ncv_fold:")
display(cv_fold)

print("\ny_train:")
display(y_train.head())
display(y_train.shape)


# %%
def normalize_data(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the training and testing data using StandardScaler.

    Parameters:
    - X_train (np.ndarray): Training feature data.
    - X_test (np.ndarray): Testing feature data.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Normalized training and testing feature data.
    """
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# %%
X_train_norm, X_test_norm = normalize_data(X_train, X_test)
print("X_train_norm:")
display(pd.DataFrame(X_train_norm, columns=X_train.columns).head())
display(X_train_norm.shape)
print("Standard deviation:", np.std(X_train_norm, axis=0))


# %% [markdown]
# ## b

# %%
import itertools


def best_subset_selection(
    X_train: np.ndarray, Y_train: np.ndarray
) -> dict:
    """
    Perform best subset selection for linear regression. This function evaluates all possible combinations of features
    and selects the best model for each subset size based on R-squared.

    Parameters:
    - X_train (np.ndarray): Training feature data. Shape (n_samples, n_features).
    - Y_train (np.ndarray): Training target data. Shape (n_samples,).

    Returns:
    - dict: A dictionary where each key is the subset size (as a string) and the value is another dictionary containing:
        - 'best_model': The statsmodels OLS regression results object for the best model.
        - 'best_r2': The R-squared value of the best model.
        - 'best_features': A tuple of feature indices used in the best model.


    """

    # Useful dimensions
    n, p = X_train.shape

    # Initialize dictionary to store the best model for each subset size
    best_model_k = {}

    # Create model for no feature
    best_model_k["0"] = {}
    best_model_k["0"]["best_model"] = sm.OLS(
        Y_train, np.ones((len(Y_train), 1))
    ).fit()
    best_model_k["0"]["best_r2"] = best_model_k["0"]["best_model"].rsquared
    best_model_k["0"]["best_features"] = []
    tests_made = [()]

    for k in range(1, p + 1):
        best_r2 = 0
        best_features = []
        best_model = []

        # Create every model with k features (excluding intercept)
        for feature_combination in itertools.combinations(range(p), k):
            X_train_aux = sm.add_constant(X_train[:, feature_combination])
            model = sm.OLS(Y_train, X_train_aux).fit()
            tests_made.append(feature_combination)
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_model = model
                best_features = feature_combination

        # Store the best model for this subset size
        best_model_k[str(k)] = {
            "best_model": best_model,
            "best_r2": best_r2,
            "best_features": list(best_features),
        }

    # Sanity checks
    assert len(tests_made) == 2**p, (
        "Number of feature subsets evaluated does not match expected count."
    )
    assert len(tests_made) == len(set(tests_made)), (
        "Duplicate feature subsets were evaluated."
    )
    assert len(best_model_k.keys()) == p + 1, (
        "Best model dictionary does not contain expected number of entries."
    )
    assert not (
        any(
            [
                best_model_k[k]["best_model"] == []
                for k in best_model_k.keys()
            ]
        )
    ), "Not all best models are valid."

    return best_model_k


# %%
def foward_stepwise_selection(
    X_train: np.ndarray, Y_train: np.ndarray
) -> dict:
    """
    Function to perform forward stepwise selection for linear regression. This function iteratively adds features
    to the model and selects the best model for each subset size based on R-squared. Only one feature is added at each step.


    Parameters:
    - X_train (np.ndarray): Training feature data. Shape (n_samples, n_features).
    - Y_train (np.ndarray): Training target data. Shape (n_samples,).

    Returns:
    - dict: A dictionary where each key is the subset size (as a string) and the value is another dictionary containing:
        - 'best_model': The statsmodels OLS regression results object for the best model.
        - 'best_r2': The R-squared value of the best model.
        - 'best_features': A tuple of feature indices used in the best model.


    """

    # Useful dimensions
    n, p = X_train.shape

    # Initialize dictionary to store the best model for each subset size
    best_model_k = {}

    # Create model for no feature
    best_model_k["0"] = {}
    best_model_k["0"]["best_model"] = sm.OLS(
        Y_train, np.ones((len(Y_train), 1))
    ).fit()
    best_model_k["0"]["best_r2"] = best_model_k["0"]["best_model"].rsquared
    best_model_k["0"]["best_features"] = []

    # Variables to keep track of features
    remaining_features = list(range(p))
    current_features = []

    for k in range(1, p + 1):
        # Initialize variables for this step
        tests_made = []

        best_r2 = 0
        best_features = []
        best_model = []

        # Create every model with k features (excluding intercept)
        for new_feature in remaining_features:
            feature_combination_list = current_features + [new_feature]
            feature_combination_list.sort()

            X_train_aux = sm.add_constant(
                X_train[:, feature_combination_list]
            )
            model = sm.OLS(Y_train, X_train_aux).fit()
            tests_made.append(feature_combination_list)

            # Check if this model is the best so far
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_model = model
                best_features = feature_combination_list

        # Move best feature for k from the remainig_features list to current_features
        remaining_features = list(
            set(remaining_features) - set(best_features)
        )
        remaining_features.sort()
        current_features = list(set(best_features + current_features))
        current_features.sort()

        # Store the best model for this subset size
        best_model_k[str(k)] = {
            "best_model": best_model,
            "best_r2": best_r2,
            "best_features": current_features.copy(),
        }

        assert len(tests_made) == p + 1 - k, (
            "Number of feature subsets evaluated does not match expected count."
        )
        assert len(tests_made) == len(
            set(tuple(sorted(test)) for test in tests_made)
        ), "Duplicate feature subsets were evaluated."
        assert all([len(test) == k for test in tests_made]), (
            "Not all tested features added have size k."
        )
        assert (
            best_model_k[str(k - 1)]["best_features"] in tests_made[i]
            for i in tests_made
        ), "All tests must include previously selected features."

    assert len(best_model_k.keys()) == p + 1, (
        "Best model dictionary does not contain expected number of entries."
    )
    assert not (
        any(
            [
                best_model_k[k]["best_model"] == []
                for k in best_model_k.keys()
            ]
        )
    ), "Not all best models are valid."

    return best_model_k


# %%
def backward_stepwise_selection(
    X_train: np.ndarray, Y_train: np.ndarray
) -> dict:
    """
    Function to perform backward stepwise selection for linear regression. This function iteratively removes features
    from the model and selects the best model for each subset size based on R-squared. Only one feature is removed at each step.


    Parameters:
    - X_train (np.ndarray): Training feature data. Shape (n_samples, n_features).
    - Y_train (np.ndarray): Training target data. Shape (n_samples,).

    Returns:
    - dict: A dictionary where each key is the subset size (as a string) and the value is another dictionary containing:
        - 'best_model': The statsmodels OLS regression results object for the best model.
        - 'best_r2': The R-squared value of the best model.
        - 'best_features': A tuple of feature indices used in the best model.


    """

    # Useful dimensions
    n, p = X_train.shape

    # Initialize dictionary to store the best model for each subset size
    best_model_k = {}
    all_features = list(range(p))

    # Create model for all features
    best_model_k[f"{p}"] = {}
    best_model_k[f"{p}"]["best_model"] = sm.OLS(
        Y_train, sm.add_constant(X_train)
    ).fit()
    best_model_k[f"{p}"]["best_r2"] = best_model_k[f"{p}"][
        "best_model"
    ].rsquared
    best_model_k[f"{p}"]["best_features"] = list(all_features)

    # Variables to keep track of features
    current_features = all_features
    deleted_features = []

    for k in range(1, p + 1):
        # Initialize variables for this step
        tests_made = []

        best_r2 = 0
        best_features = []
        best_model = []

        # Create every model with k features (excluding intercept)
        for new_feature in current_features:
            feature_combination_list = current_features.copy()
            feature_combination_list.remove(new_feature)
            X_train_aux = sm.add_constant(
                X_train[:, feature_combination_list]
            )
            model = sm.OLS(Y_train, X_train_aux).fit()
            tests_made.append(feature_combination_list)

            # Check if this model is the best so far
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_model = model
                best_features = feature_combination_list

        # Update current features and deleted features lists
        deleted_features += list(
            set(current_features) - set(best_features)
        )
        current_features = best_features

        # Store the best model for this subset size
        best_model_k[str(p - k)] = {
            "best_model": best_model,
            "best_r2": best_r2,
            "best_features": current_features.copy(),
        }

        # Sanity checks
        assert len(tests_made) == p - k + 1, (
            "Number of feature subsets evaluated does not match expected count."
        )
        assert len(tests_made) == len(
            set(tuple(sorted(test)) for test in tests_made)
        ), "Duplicate feature subsets were evaluated."
        assert all([len(test) == p - k for test in tests_made]), (
            "Not all tested features added have size p-k."
        )
        assert (
            tests_made[i] in best_model_k[str(p - k + 1)]["best_features"]
            for i in tests_made
        ), "All tests must be included in previously selected features."
        assert len(deleted_features) == k, (
            "Number of deleted features does not match expected count."
        )

    # Create model for no feature
    best_model_k["0"] = {}
    best_model_k["0"]["best_model"] = sm.OLS(
        Y_train, np.ones((len(Y_train), 1))
    ).fit()
    best_model_k["0"]["best_r2"] = best_model_k["0"]["best_model"].rsquared
    best_model_k["0"]["best_features"] = []

    # Sanity checks
    assert len(best_model_k.keys()) == p + 1, (
        "Best model dictionary does not contain expected number of entries."
    )
    assert not (
        any(
            [
                best_model_k[k]["best_model"] == []
                for k in best_model_k.keys()
            ]
        )
    ), "Not all best models are valid."

    best_model_k = dict(
        sorted(best_model_k.items(), key=lambda x: int(x[0]))
    )
    return best_model_k


models_backward = backward_stepwise_selection(
    X_train.values, y_train.values
)


# %%
models_forward = foward_stepwise_selection(X_train.values, y_train.values)
models_best = best_subset_selection(X_train.values, y_train.values)
models_backward = backward_stepwise_selection(
    X_train.values, y_train.values
)


# %%
# Save key outputs to file
with open("../reports/figures/5/q5_outputs.txt", "w") as f:
    sys.stdout = f

    print("=== MODELS COMPARISON ===")
    print("Best models:", models_best)
    print("\nForward models:", models_forward)
    print("\nBackward models:", models_backward)

    print("\n=== VALIDATION ===")
    print("All checks passed successfully.")


sys.stdout = original_stdout


tol = 1e-8
for key in models_best.keys():
    assert (
        models_best[key]["best_r2"] + tol >= models_forward[key]["best_r2"]
    ), (
        "Best subset selection should have R2 greater than or equal to forward stepwise selection."
    )
    assert (
        models_best[key]["best_r2"] + tol
        >= models_backward[key]["best_r2"]
    ), (
        "Best subset selection should have R2 greater than or equal to backward stepwise selection."
    )
# %% [markdown]
# ## c

# %%
keys = [int(x) for x in list(models_best.keys())]
models_best_r2 = [models_best[str(k)]["best_r2"] for k in keys]
models_backward_r2 = [models_backward[str(k)]["best_r2"] for k in keys]
models_forward_r2 = [models_forward[str(k)]["best_r2"] for k in keys]


plt.figure(figsize=(16, 6))
plt.plot(keys, models_best_r2, marker="o", label="Best Subset Selection")
plt.plot(
    keys, models_forward_r2, marker="s", label="Forward Stepwise Selection"
)
plt.plot(
    keys,
    models_backward_r2,
    marker="^",
    label="Backward Stepwise Selection",
)
plt.xlabel("Number of Features")
plt.ylabel("R-squared")
plt.title("Model Selection Methods Comparison")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig(
    "../reports/figures/5/model_selection_methods_r2_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# %% [markdown]
# ## d

# %%
from sklearn.metrics import mean_squared_error

alphas = 10 ** np.linspace(5, -2, 100)

mean_cv_error = {}
for alpha_0 in alphas:
    fold_error = []
    for test_fold in np.unique(cv_fold):
        # Split data into training and testing for the current fold
        x_train_fold = X_train[cv_fold != test_fold]
        y_train_fold = y_train[cv_fold != test_fold]
        x_test_fold = X_train[cv_fold == test_fold]
        y_test_fold = y_train[cv_fold == test_fold]

        # Normalize data
        x_train_fold, x_test_fold = normalize_data(
            x_train_fold, x_test_fold
        )

        # Train and evaluate Lasso model
        model_ = Lasso(alpha=alpha_0).fit(x_train_fold, y_train_fold)
        yhat = model_.predict(x_test_fold)
        fold_error.append(mean_squared_error(yhat, y_test_fold))

    mean_cv_error[alpha_0] = np.mean(fold_error)

# %%
plt.figure(figsize=(10, 6))
plt.plot(list(mean_cv_error.keys()), list(mean_cv_error.values()))
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Mean CV Error (MSE)")
plt.title("Lasso Regression: Mean CV Error vs Alpha")
plt.savefig(
    "../reports/figures/5/lasso_cv_error_vs_alpha.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

best_alpha = min(mean_cv_error, key=mean_cv_error.get)
X_train_norm, X_test_norm = normalize_data(X_train, X_test)
lasso_model = Lasso(alpha=best_alpha).fit(X_train_norm, y_train)

# %%
# Continue writing to output file
with open("../reports/figures/5/q5_outputs.txt", "a") as f:
    sys.stdout = f

    print("\n=== LASSO RESULTS ===")
    print("Best Alpha:", best_alpha)
    print("Minimum Mean CV Error:", mean_cv_error[best_alpha])
    print("Lasso coefficients:", lasso_model.coef_)

sys.stdout = original_stdout


# %% [markdown]
# ## e

# %%
# Evaluate Lasso model on test set
yhat = lasso_model.predict(X_test_norm)
lasso_test_error = mean_squared_error(yhat, y_test)


# Evaluate all models in best subset selection on test set
models_best_error = {}
models_best_best_features = {}
for key in models_best.keys():
    y_hat = models_best[key]["best_model"].predict(
        sm.add_constant(
            X_test.values[:, models_best[key]["best_features"]]
        )
    )
    models_best_error[key] = mean_squared_error(y_hat, y_test)
    models_best_best_features[key] = models_best[key]["best_features"]

# Evalueate all models in forward stepwise selection on test set
models_forward_error = {}
models_forward_best_features = {}
for key in models_forward.keys():
    y_hat = models_forward[key]["best_model"].predict(
        sm.add_constant(
            X_test.values[:, models_forward[key]["best_features"]]
        )
    )
    models_forward_error[key] = mean_squared_error(y_hat, y_test)
    models_forward_best_features[key] = models_forward[key][
        "best_features"
    ]

# Evalueate all models in backward stepwise selection on test set
models_backward_error = {}
for key in models_backward.keys():
    y_hat = models_backward[key]["best_model"].predict(
        sm.add_constant(
            X_test.values[:, models_backward[key]["best_features"]]
        )
    )
    models_backward_error[key] = mean_squared_error(y_hat, y_test)


# %%
plt.figure(figsize=(10, 6))
plt.plot(
    models_backward_error.keys(),
    models_backward_error.values(),
    marker="^",
    label="Backward Stepwise Selection",
)
plt.plot(
    models_forward_error.keys(),
    models_forward_error.values(),
    marker="s",
    label="Forward Stepwise Selection",
)
plt.plot(
    models_best_error.keys(),
    models_best_error.values(),
    marker="o",
    label="Best Subset Selection",
)
plt.plot(
    models_best_error.keys(),
    [lasso_test_error] * len(models_best_error.keys()),
    color="red",
    label="Lasso Regression for best alpha",
    marker="",
    linestyle="--",
)
plt.xlabel("Number of Features")
plt.ylabel("Test Set MSE")
plt.title("Model Selection Methods Test Set Error Comparison")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig(
    "../reports/figures/5/model_selection_test_error_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# Restore stdout
sys.stdout = original_stdout
print("Q5 outputs saved to ../reports/figures/5/q5_outputs.txt")

# %%
# Continue writing to output file
with open("../reports/figures/5/q5_outputs.txt", "a") as f:
    sys.stdout = f

    print("\n=== TEST ERRORS TABLE ===")
    print(
        "Number of Features | Best Subset MSE | Forward Stepwise MSE | Backward Stepwise MSE"
    )
    for k in models_best_error.keys():
        print(
            f"{k:>18} | {models_best_error[k]:>15.4f} | {models_forward_error[k]:>19.4f} | {models_backward_error[k]:>20.4f}"
        )
    print(f"Lasso Regression Test Set MSE: {lasso_test_error:.4f}")

    print("\n=== BEST FEATURES ===")
    best_features = X_train.columns[models_best_best_features["2"]]
    print("Best features for 2-feature model:", best_features)
    print("Model parameters:", models_best["2"]["best_model"].params)

sys.stdout = original_stdout
