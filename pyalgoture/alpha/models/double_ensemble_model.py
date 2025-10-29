from typing import Optional, Union, cast

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from ...utils.logger import get_logger
from ..dataset import AlphaDataset
from ..datasets.utils.utility import Segment
from ..model import AlphaModel

logger = get_logger()


class DoubleEnsembleModel(AlphaModel):
    """Double Ensemble Model with sample reweighting and feature selection"""

    def __init__(
        self,
        base_model: str = "gbm",
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        num_boost_round: int = 100,  # epochs
        early_stopping_rounds: int = 50,
        log_evaluation_period: int = 20,
        # Double Ensemble specific parameters
        num_models: int = 6,
        enable_sr: bool = True,
        enable_fs: bool = True,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        bins_sr: int = 10,
        bins_fs: int = 5,
        decay: float | None = None,
        sample_ratios: list[float] | None = None,
        sub_weights: list[float] | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        base_model : str
            Base model type, currently only 'gbm' (LightGBM) is supported
        learning_rate : float
            Learning rate for LightGBM
        num_leaves : int
            Number of leaf nodes for LightGBM
        num_boost_round : int
            Maximum number of training rounds
        early_stopping_rounds : int
            Number of rounds for early stopping
        log_evaluation_period : int
            Interval rounds for printing training logs
        num_models : int
            Number of sub-models in the ensemble
        enable_sr : bool
            Whether to enable sample reweighting
        enable_fs : bool
            Whether to enable feature selection
        alpha1 : float
            Weight for loss values in sample reweighting
        alpha2 : float
            Weight for loss curve in sample reweighting
        bins_sr : int
            Number of bins for sample reweighting
        bins_fs : int
            Number of bins for feature selection
        decay : float
            Decay factor for sample reweighting
        sample_ratios : list[float]
            Ratios to sample features from each bin in feature selection
        sub_weights : list[float]
            Weights for each sub-model in ensemble prediction
        seed : int
            Random seed
        """
        # Base LightGBM parameters
        self.params: dict = {
            "objective": "mse",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "seed": seed,
        }

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period

        # Double Ensemble specific parameters
        self.base_model: str = base_model
        self.num_models: int = num_models
        self.enable_sr: bool = enable_sr
        self.enable_fs: bool = enable_fs
        self.alpha1: float = alpha1
        self.alpha2: float = alpha2
        self.bins_sr: int = bins_sr
        self.bins_fs: int = bins_fs
        self.decay: float = decay if decay is not None else 1.0

        # Set default values if not provided
        if sample_ratios is None:
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:
            sub_weights = [1] * self.num_models

        # Validate parameters
        if not len(sample_ratios) == bins_fs:
            raise ValueError("The length of sample_ratios should be equal to bins_fs.")
        if not len(sub_weights) == num_models:
            raise ValueError("The length of sub_weights should be equal to num_models.")

        self.sample_ratios: list[float] = sample_ratios
        self.sub_weights: list[float] = sub_weights

        # Initialize model components
        self.ensemble: list[lgb.Booster] = []  # the current ensemble model, a list contains all the sub-models
        self.sub_features: list[pd.Index] = []  # the features for each sub model in the form of pandas.Index

    def _prepare_data(
        self, df_train: pd.DataFrame, df_valid: pd.DataFrame, weights: pd.Series, features: pd.Index
    ) -> tuple[lgb.Dataset, lgb.Dataset]:
        """
        Prepare data for LightGBM training

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with features and labels
        df_valid : pd.DataFrame
            Validation data with features and labels
        weights : pd.Series
            Sample weights for training data
        features : pd.Index
            Features to use for training

        Returns
        -------
        tuple[lgb.Dataset, lgb.Dataset]
            LightGBM datasets for training and validation
        """
        x_train, y_train = df_train[self.feature_names].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid[self.feature_names].loc[:, features], df_valid["label"]
        y_train, y_valid = np.array(y_train), np.array(y_valid)

        dtrain = lgb.Dataset(x_train, label=y_train, weight=weights)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def _train_submodel(
        self, df_train: pd.DataFrame, df_valid: pd.DataFrame, weights: pd.Series, features: pd.Index
    ) -> lgb.Booster:
        """
        Train a single sub-model

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with features and labels
        df_valid : pd.DataFrame
            Validation data with features and labels
        weights : pd.Series
            Sample weights for training data
        features : pd.Index
            Features to use for training

        Returns
        -------
        lgb.Booster
            Trained LightGBM model
        """
        dtrain, dvalid = self._prepare_data(df_train, df_valid, weights, features)
        evals_result: dict = {}

        # Set up callbacks
        callbacks = [
            lgb.log_evaluation(self.log_evaluation_period),
            lgb.record_evaluation(evals_result),
        ]

        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            logger.info("Training with early stopping...")

        # Train the model
        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Extract evaluation results
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

        return model

    def _get_loss(self, label: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        Calculate loss values for given labels and predictions

        Parameters
        ----------
        label : np.ndarray
            Ground truth labels
        pred : np.ndarray
            Model predictions

        Returns
        -------
        np.ndarray
            Loss values for each sample
        """
        # Currently only MSE is supported
        loss: np.ndarray = (label - pred) ** 2
        return loss

    def _retrieve_loss_curve(self, model: lgb.Booster, df_train: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
        """
        Retrieve loss curve during model training

        Parameters
        ----------
        model : lgb.Booster
            Trained LightGBM model
        df_train : pd.DataFrame
            Training data with features and labels
        features : pd.Index
            Features used for training

        Returns
        -------
        pd.DataFrame
            Loss curve for each sample at each training iteration
        """
        if self.base_model == "gbm":
            num_trees = model.num_trees()
            x_train, y_train = df_train[self.feature_names].loc[:, features], df_train["label"]
            y_train = np.array(y_train)

            N = x_train.shape[0]
            loss_curve: pd.DataFrame = pd.DataFrame(np.zeros((N, num_trees)))
            pred_tree = np.zeros(N, dtype=float)

            for i_tree in range(num_trees):
                pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
                loss_curve.iloc[:, i_tree] = self._get_loss(y_train.to_numpy(), pred_tree)
        else:
            raise ValueError(f"Base model '{self.base_model}' is not supported")

        return loss_curve

    def _sample_reweight(self, loss_curve: pd.DataFrame, loss_values: pd.Series, k_th: int) -> pd.Series:
        """
        Reweight samples based on loss curve and loss values

        Parameters
        ----------
        loss_curve : pd.DataFrame
            Loss curve for each sample at each training iteration
        loss_values : pd.Series
            Current ensemble loss on each sample
        k_th : int
            Index of the current sub-model

        Returns
        -------
        pd.Series
            New sample weights
        """
        # Normalize loss_curve and loss_values with ranking
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)

        # Calculate l_start and l_end from loss_curve
        N, T = loss_curve.shape
        part = np.maximum(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # Calculate h-value for each sample
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # Calculate weights
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins")["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))

        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)

        return weights

    def _feature_selection(self, df_train: pd.DataFrame, loss_values: pd.Series) -> pd.Index:
        """
        Select features based on their importance

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with features and labels
        loss_values : pd.Series
            Current ensemble loss on each sample

        Returns
        -------
        pd.Index
            Selected features
        """
        x_train, y_train = df_train[self.feature_names], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)

        # Shuffle specific columns and calculate g-value for each feature
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_train_tmp.loc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)

            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), index=x_train_tmp.index
                    )
                    / M
                )

            loss_feat: np.ndarray = self._get_loss(np.squeeze(y_train.values), pred.values)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
            x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()

        # Handle NaN values
        g["g_value"].replace(np.nan, 0, inplace=True)

        # Divide features into bins_fs bins
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)

        # Randomly sample features from bins to construct the new features
        res_feat: list[str] = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)

        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()

        return pd.Index(set(res_feat))

    def _predict_sub(self, submodel: lgb.Booster, df_data: pd.DataFrame, features: pd.Index) -> pd.Series:
        """
        Make predictions using a sub-model

        Parameters
        ----------
        submodel : lgb.Booster
            Trained sub-model
        df_data : pd.DataFrame
            Data to make predictions on
        features : pd.Index
            Features to use for prediction

        Returns
        -------
        pd.Series
            Predictions
        """
        x_data = df_data[self.feature_names].loc[:, features]
        pred_sub: pd.Series = pd.Series(submodel.predict(x_data.values), index=x_data.index)
        return pred_sub

    def fit(self, dataset: AlphaDataset) -> None:
        """
        Fit the ensemble model

        Parameters
        ----------
        dataset : AlphaDataset
            Dataset containing training and validation data

        Returns
        -------
        None
        """
        # Prepare training and validation data
        df_train = dataset.fetch_learn(Segment.TRAIN)
        df_valid = dataset.fetch_learn(Segment.VALID)

        # Convert polars DataFrames to pandas
        df_train = df_train.to_pandas()
        df_valid = df_valid.to_pandas()

        self.feature_names = df_train.columns[2:-1]

        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train[self.feature_names], df_train["label"]

        # Initialize sample weights
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float), index=x_train.index)

        # Initialize features
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)

        # Train sub-models
        for k in range(self.num_models):
            self.sub_features.append(features)
            logger.info(f"Training sub-model: ({k + 1}/{self.num_models})")

            model_k = self._train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)

            # No further sample re-weight and feature selection needed for the last sub-model
            if k + 1 == self.num_models:
                break

            logger.info("Retrieving loss curve and loss values...")
            loss_curve = self._retrieve_loss_curve(model_k, df_train, features)
            pred_k = self._predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k

            # Calculate ensemble prediction
            pred_ensemble = (pred_sub.iloc[:, : k + 1] * self.sub_weights[0 : k + 1]).sum(axis=1) / np.sum(
                self.sub_weights[0 : k + 1]
            )
            loss_values: pd.Series = pd.Series(self._get_loss(np.squeeze(y_train.values), pred_ensemble.to_numpy()))

            if self.enable_sr:
                logger.info("Sample re-weighting...")
                weights = self._sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                logger.info("Feature selection...")
                features = self._feature_selection(df_train, loss_values)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Make predictions with the ensemble model

        Parameters
        ----------
        dataset : AlphaDataset
            Dataset containing features
        segment : Segment
            Segment to make predictions on

        Returns
        -------
        np.ndarray
            Prediction results

        Raises
        ------
        ValueError
            If the model has not been fitted yet
        """
        if not self.ensemble:
            raise ValueError("Model is not fitted yet!")

        # Prepare data for inference
        x_test = dataset.fetch_infer(segment)
        x_test = x_test.to_pandas()

        # Initialize predictions
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)

        # Aggregate predictions from all sub-models
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            pred += (
                pd.Series(submodel.predict(x_test.loc[:, feat_sub].values), index=x_test.index)
                * self.sub_weights[i_sub]
            )

        # Normalize by sum of weights
        pred = pred / np.sum(self.sub_weights)

        return cast(np.ndarray, pred.values)

    def get_feature_importance(self, importance_type: str = "split") -> pd.Series:
        """
        Get feature importance from all sub-models

        Parameters
        ----------
        importance_type : str
            Type of feature importance, 'split' or 'gain'

        Returns
        -------
        pd.Series
            Feature importance values

        Raises
        ------
        ValueError
            If the model has not been fitted yet
        """
        if not self.ensemble:
            raise ValueError("Model is not fitted yet!")

        # Collect feature importance from all sub-models
        res = []
        for model, weight in zip(self.ensemble, self.sub_weights):
            res.append(
                pd.Series(model.feature_importance(importance_type=importance_type), index=model.feature_name())
                * weight
            )

        # Combine all feature importances
        result: pd.Series = pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)

        return result

    def detail(self) -> None:
        """
        Display model details with feature importance plots

        Returns
        -------
        None
        """
        if not self.ensemble:
            return

        # Plot feature importance for split and gain
        for importance_type in ["split", "gain"]:
            imp = self.get_feature_importance(importance_type=importance_type)

            # Limit to top 50 features
            plt.figure(figsize=(10, 20))
            ax = plt.subplot(111)
            imp[:50].plot(kind="barh", ax=ax)
            plt.title(f"Feature Importance ({importance_type})")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
