"""Model construction: the model alone (`build_model`) and preprocessing + model (`build_pipeline`)."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def adaboost(max_depth: int = 1, **params):
    """
    AdaBoost over decision trees of depth `max_depth` (1 = stumps); `params` go to
    AdaBoostClassifier. The trees' depth is tuned as `model__estimator__max_depth`.
    """
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    A forest of oblique trees: each tree is fit on a bootstrap sample, on `n_components`
    random sparse linear combinations of the features (a new set per tree), so its splits are
    hyperplanes. `tree_params` go to every tree (`max_features` defaults to "sqrt"). Tuned as
    `model__estimator__tree__<param>` and `model__estimator__project__<param>`.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    """Builds the model named by `model_config["type"]` with `model_config["params"]`."""
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models, fitted on the training
    rows only: a random forest and gradient boosting vote for features with above-average
    importance, an L1-penalised logistic regression for features it doesn't shrink to zero.
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):
        return self.support_


def build_pipeline(preprocessing_config: dict, model_config: dict, selection_config: dict = None,
                   memory=None) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing, feature selection
    (if `selection_config["enabled"]`) and model as one Pipeline, so every learned step is
    re-fit on the training part of each fold. Step names address hyperparameters from outside
    (`model__max_depth`, `select__min_votes`). `memory` (a folder) caches the fitted
    preprocessing and selection steps: a fit on the same rows with the same settings is
    reused instead of repeated, so tuning the model doesn't refit them in every trial.
    """
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", VoteSelector(min_votes=selection_config.get("min_votes", 2),
                                             random_state=selection_config.get("random_state"))))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)