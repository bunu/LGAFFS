import numpy as np
from numpy import ravel
from sklearn import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score


class SequentialForwardSelection(SequentialFeatureSelector):

    def fit(self, X, y=None):
        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]
        cloned_estimator = clone(self.estimator)

        current_mask = np.zeros(shape=n_features, dtype=bool)
        new_feature_idx = self._get_best_new_feature(cloned_estimator, X, ravel(y), current_mask)
        current_mask[new_feature_idx] = True
        max_performance = np.mean(cross_val_score(cloned_estimator, X[:, current_mask], y, cv=self.cv,
                                                  scoring=self.scoring, n_jobs=self.n_jobs))
        new_performance = max_performance
        features = 1
        while new_performance >= max_performance and features < n_features:
            max_performance = new_performance
            new_feature_idx = self._get_best_new_feature(cloned_estimator, X, ravel(y), current_mask)
            current_mask[new_feature_idx] = True
            new_performance = np.mean(cross_val_score(cloned_estimator, X[:, current_mask], ravel(y), cv=self.cv,
                                                      scoring=self.scoring, n_jobs=self.n_jobs))
            features += 1
        # If we exit the while loop due to a decrease in performance, we need to remove the last added feature
        if new_performance < max_performance:
            current_mask[new_feature_idx] = False
        self.support_ = current_mask

        return self
