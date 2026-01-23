# %% [markdown]
# # 4
#

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn import preprocessing
from IPython.display import display
import sys

# Redirect prints to file
original_stdout = sys.stdout

df = pd.read_csv("../data/soccer.csv")
X = df.drop("target", axis=1)
y = df[["target"]]

X_train, y_train = X.iloc[:2560], y.iloc[:2560]
X_test, y_test = X.iloc[2560:], y.iloc[2560:]

# Save initial dataset info
with open("../reports/figures/4/q4_outputs.txt", "w") as f:
    sys.stdout = f

    print("=== DATASET INFO ===")
    print("Samples of X_train dataset", X_train.shape[0])
    print("Samples of X_test dataset", X_test.shape[0])
    print("Samples of y_train dataset", y_train.shape[0])
    print("Samples of y_test dataset", y_test.shape[0])

# Restore stdout
sys.stdout = original_stdout

# %%
X_train = X_train.drop(["home_team", "away_team"], axis=1)
X_test = X_test.drop(["home_team", "away_team"], axis=1)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

display(pd.DataFrame(X_train).head())
display(pd.DataFrame(X_test).head())

# %% [markdown]
# ## b

# %%
models_to_test = [LDA, QDA, LR, NB, kNN]
results_dict = {}
for model_type in models_to_test:
    model_name = model_type.__name__
    params = {}
    if model_type in [LDA, QDA]:
        params.update({"store_covariance": True})
    results_dict[model_name] = {}
    cls = model_type(**params)
    cls.fit(X_train, y_train.values.ravel())

    # save model for later analysis
    results_dict[model_name]["in_sample_predictions"] = cls.predict(
        X_train
    )
    results_dict[model_name]["test_predictions"] = cls.predict(X_test)
    results_dict[model_name]["model"] = cls

    # train model with no ravel for the sake of comparison
    cls2 = model_type(**params)
    cls2.fit(X_train, y_train)
    results_dict[model_name]["in_sample_predictions_no_ravel"] = (
        cls2.predict(X_train)
    )
    results_dict[model_name]["test_predictions_no_ravel"] = cls2.predict(
        X_test
    )
    results_dict[model_name]["model_no_ravel"] = cls2

# %%
# Check if predictions are the same for both ravel and no ravel models
with open("../reports/figures/4/q4_outputs.txt", "a") as f:
    sys.stdout = f

    print("\n=== MODEL PREDICTIONS COMPARISON ===")
    for model in models_to_test:
        print(model.__name__)
        print(
            "In-sample:",
            (
                results_dict[model.__name__]["in_sample_predictions"]
                == results_dict[model.__name__][
                    "in_sample_predictions_no_ravel"
                ]
            ).all(),
        )
        print(
            "Test predictions:",
            (
                results_dict[model.__name__]["test_predictions"]
                == results_dict[model.__name__][
                    "test_predictions_no_ravel"
                ]
            ).all(),
        )
        print("\n")

sys.stdout = original_stdout

# %%
# Check the coefficients of all models
with open("../reports/figures/4/q4_outputs.txt", "a") as f:
    sys.stdout = f

    print("\n=== MODEL COEFFICIENTS ===")
    for model_name in models_to_test:
        print(model_name.__name__)
        print("\n")
        model = results_dict[model_name.__name__]["model"]

        features = [
            "coef_",
            "intercept_",
            "theta_",
            "var_",
            "covariance_",
            "means_",
        ]
        for feature in features:
            if hasattr(model, feature):
                print(f"- {feature}:", getattr(model, feature), "\n")
        print("\n", "-" * 100, "\n")

sys.stdout = original_stdout

# What is the covariance matrix learned by LDA and QDA?
# What model NBGaussian corresponds to?

# %%
from sklearn.metrics import ConfusionMatrixDisplay

with open("../reports/figures/4/q4_outputs.txt", "a") as f:
    sys.stdout = f
    print("\n=== CONFUSION MATRICES ===")

sys.stdout = original_stdout

for model_name in models_to_test:
    model_type_name = model_name.__name__
    in_sample_predictions = results_dict[model_type_name][
        "in_sample_predictions"
    ]
    test_predictions = results_dict[model_type_name]["test_predictions"]

    with open("../reports/figures/4/q4_outputs.txt", "a") as f:
        sys.stdout = f
        print(f"Confusion Matrix for {model_type_name}:")

    sys.stdout = original_stdout
    ConfusionMatrixDisplay.from_predictions(y_train, in_sample_predictions)
    plt.title("In-sample prediction for " + model_type_name)
    plt.savefig(
        f"../reports/figures/4/confusion_matrix_insample_{model_type_name.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()
    ConfusionMatrixDisplay.from_predictions(y_test, test_predictions)
    plt.title("Test prediction for " + model_type_name)
    plt.savefig(
        f"../reports/figures/4/confusion_matrix_test_{model_type_name.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    with open("../reports/figures/4/q4_outputs.txt", "a") as f:
        sys.stdout = f
        print("\n", "-" * 100, "\n")

    sys.stdout = original_stdout

# %% [markdown]
# ## c
#

# %%
plot_markers = ["o", "s", "^", "D", "x"]
plot_colors = ["b", "g", "r", "c", "m"]
plt.figure(figsize=(10, 8))
i = 0

for model_name in models_to_test:
    model_type_name = model_name.__name__
    in_sample_predictions = results_dict[model_type_name][
        "in_sample_predictions"
    ]
    test_predictions = results_dict[model_type_name]["test_predictions"]
    train_error = np.mean(in_sample_predictions != y_train.values.ravel())
    test_error = np.mean(test_predictions != y_test.values.ravel())
    plt.scatter(
        train_error,
        test_error,
        label=model_type_name,
        color=plot_colors[i],
        marker=plot_markers[i],
    )
    i += 1

plt.title("Erros para diferentes modelos")
plt.xlabel("Erro de Treinamento")
plt.ylabel("Erro de Teste")
# move legend outside the plot
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig(
    "../reports/figures/4/models_error_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# %% [markdown]
# ### d

# %%
# use numbers as markers for kNN from 1 to 10
plot_colors = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",
    "orange",
    "purple",
    "brown",
    "pink",
]

plt.figure(figsize=(10, 8))
for k in range(1, 11):
    model = kNN(n_neighbors=k)
    model.fit(X_train, y_train.values.ravel())
    in_sample_predictions_k = model.predict(X_train)
    test_predictions_k = model.predict(X_test)
    train_error = np.mean(
        in_sample_predictions_k != y_train.values.ravel()
    )
    test_error = np.mean(test_predictions_k != y_test.values.ravel())
    plt.scatter(
        train_error,
        test_error,
        label=f"K = {k}",
        color=plot_colors[k - 1],
        marker=f"${k}$",
    )


plt.title("Erros para kNN com diferentes números de vizinhos (K) ")
plt.xlabel("Erro de Treinamento")
plt.ylabel("Erro de Teste")
# move legend outside the plot
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig(
    "../reports/figures/4/knn_error_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# Restore stdout
sys.stdout = original_stdout
print("Q4 outputs saved to ../reports/figures/4/q4_outputs.txt")
