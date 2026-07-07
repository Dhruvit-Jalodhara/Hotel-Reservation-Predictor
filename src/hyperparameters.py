# Hyperparameter grids for all classification models.

lr_params = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [10, 1, 0.1, 0.01],
    "solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
    "class_weight": ["balanced"],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "random_state": [42],
    "n_jobs": [-1]
}

linearsvc_params = {
    "loss": ["squared_hinge", "hinge"],
    "class_weight": ["balanced"],
    "penalty": ["l1", "l2"],
    "C": [10, 1, 0.1, 0.01],
    "dual": [False],
    "random_state": [42]
}

knn_params = {
    "n_neighbors": [5, 7, 9, 10, 12, 13, 15],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "weights": ["uniform", "distance"],
    "n_jobs": [-1]
}

dt_params = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [1, 2, 3, 4, 5],
    "max_features": [None, "sqrt", "log2"],
    "class_weight": ["balanced"],
    "random_state": [42]
}

rf_params = {
    "max_depth": [5, 8, 10, 15, None],
    "max_features": [5, 7, 8, "sqrt", "log2", None],
    "min_samples_split": [2, 8, 15, 20],
    "n_estimators": [100, 200, 500, 1000],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight": ["balanced"],
    "random_state": [42],
    "n_jobs": [-1]
}

ada_params = {
    "n_estimators": [20, 40, 50, 60, 70, 80, 90, 100],
    "learning_rate": [1, 0.5, 0.1, 0.055, 0.05, 0.01],
    "random_state": [42]
}

gb_params = {
    "loss": ["log_loss", "exponential"],
    "learning_rate": [1, 0.5, 0.1, 0.055, 0.05, 0.01],
    "criterion": ["friedman_mse", "squared_error"],
    "max_features": [None, "sqrt", "log2"],
    "n_estimators": [20, 40, 50, 60, 70, 80, 90, 100],
    "subsample": [0.7, 0.8, 1.0],
    "max_depth": [1, 2, 3, 4, 5],
    "random_state": [42]
}

xgb_params = {
    "learning_rate": [1, 0.5, 0.1, 0.055, 0.05, 0.01],
    "max_depth": [5, 8, 12, 20, 30],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.8, 1.0]
}

cat_params = {
    "learning_rate": [1, 0.5, 0.1, 0.055, 0.05, 0.01],
    "auto_class_weights": ["Balanced"],
    "verbose": [False],
    "random_state": [42]
}


MODEL_PARAMS = {
    "Logistic Regression": lr_params,
    "Linear Support Vector Classifier": linearsvc_params,
    "Gaussian NB": {},
    "KNN Classifier": knn_params,
    "Decision Tree Classifier": dt_params,
    "Random Forest Classifier": rf_params,
    "AdaBoost Classifier": ada_params,
    "Gradient Boosting Classifier": gb_params,
    "XGBoost Classifier": xgb_params,
    "CatBoost Classifier": cat_params
}