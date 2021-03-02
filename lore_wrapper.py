#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:49:12 2020

@author: francesco
"""

import pandas as pd
import numpy as np
from lore.datamanager import prepare_dataset
from lore.lorem import LOREM
from lore.util import neuclidean
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def compute_medoid(X):
    distance_matrix = pairwise_distances(X, n_jobs=-1)
    medoid_idx = np.argmin(distance_matrix.sum(axis=0))
    return medoid_idx


def plot_exemplars_and_counterexemplars_by_rule(
        rules_dataframes,
        labels=None,
        **kwargs
):
    for rule in rules_dataframes.keys():
        plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
        label = rules_dataframes[rule]["Rule_obj"].cons
        if rule == "rule":
            plt.title(r"$b(\tilde{Z}_{=}^*)$" + " = " + labels[label] if labels
                      else r"$b(\tilde{Z}_{=}^*)$" + " = " + str(label),
                      fontsize=kwargs.get("fontsize", 12))
            for ts in rules_dataframes[rule]["df"]:
                plt.plot(ts, c="#2ca02c", alpha=kwargs.get("alpha", 0.1))
        else:
            plt.title(r"$b(\tilde{Z}_\neq ^*)$" + " = " + labels[label] if labels
                      else r"$b(\tilde{Z}_\neq ^*)$" + " = " + str(label),
                      fontsize=kwargs.get("fontsize", 12))
            for ts in rules_dataframes[rule]["df"]:
                plt.plot(ts, c="#d62728", alpha=kwargs.get("alpha", 0.1))
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
        plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
        plt.show()


class LoreWrapper(object):
    def __init__(self,
                 blackbox,
                 decoder,
                 ):
        self.blackbox = blackbox
        self.decoder = decoder

        self._class_name = None
        self._df = None
        self._feature_names = None
        self._class_values = None
        self._numeric_columns = None
        self._rdf = None
        self._real_feature_names = None
        self._features_map = None
        self._z = None
        self._Z = None
        self._y = None
        self._lorem_explainer = None
        self._lorem_explanation = None
        self._Z_star = None
        self._Z_tilde_star = None
        self._y_star = None

    def _build(self):
        columns = [str(i) for i in range(self._K_encoded.shape[1])]
        df = pd.DataFrame(self._K_encoded, columns=columns)
        df["class"] = self.blackbox.predict(self._K)
        self._class_name = "class"
        (self._df, self._feature_names, self._class_values, self._numeric_columns,
         self._rdf, self._real_feature_names, self._features_map) = prepare_dataset(df, self._class_name)
        return self

    def _decode_predict(self, X):
        """Decode and predict the class labels for the provided latent data.
        Parameters
        ----------
        X : {array-like}
            Latent data. Shape [n_samples, n_features].
        Returns
        -------
        y : array of shape [n_samples]
        """
        X_decoded = self.decoder.predict(X)
        y = self.blackbox.predict(X_decoded)
        return y

    def _decode_predict_proba(self, X):
        """Decode and predict the class probability for the provided latent data.
        Parameters
        ----------
        X : {array-like}
            Latent data. Shape [n_samples, n_features].
        Returns
        -------
        y : array of shape [n_samples, n_classes]
        """
        X_decoded = self.decoder.predict(X)
        y = self.blackbox.predict_proba(X_decoded)
        return y

    def generate_neighborhood(self, z, K, K_encoded, **params):
        self._K = K
        self._K_encoded = K_encoded
        self._build()
        self._generate_neighborhood(z, **params)
        self._tree_rules_extraction()
        self._build_rules_dataframes()
        #self.closest_counterfactual = self._Z_star[np.argmin(cdist(z, self._Z_star))].reshape(1, -1)

        Z = list()
        for z_ in self._Z_star:
            if not np.array_equal(z_, z.ravel()):
                Z.append(z_)
        Z = np.array(Z)
        return Z  #self._Z_star

    def _generate_neighborhood(self, z, **params):
        # generate 2d df of latent space for LOREM method
        # self._preprocess_dataset
        z = z.ravel()
        self._z = z
        self._lorem_explainer = LOREM(
            K=self._K_encoded,
            bb_predict=self._decode_predict,
            bb_predict_proba=self._decode_predict_proba,
            feature_names=self._feature_names,
            class_name=self._class_name,
            class_values=self._class_values,
            numeric_columns=self._numeric_columns,
            features_map=self._features_map,
            neigh_type=params.get("neigh_type", 'geneticp'),
            categorical_use_prob=params.get("categorical_use_prob", True),
            continuous_fun_estimation=params.get("continuous_fun_estimation", False),
            size=params.get("size", 1000),
            samples=params.get("samples", 1000),
            ocr=params.get("ocr", 0.1),
            multi_label=params.get("multi_label", False),
            one_vs_rest=params.get("one_vs_rest", False),
            random_state=params.get("random_state", None),
            verbose=params.get("verbose", False),
            filter_crules=params.get("filter_crules", False),
            ngen=params.get("ngen", 10)
        )

        # neighborhood generation
        Z = self._lorem_explainer.neighgen_fn(z, params.get("samples", 1000))

        if self._lorem_explainer.multi_label:
            # maybe wrong
            y = self._decode_predict(Z)
            Z = np.array([a for a, y in
                          zip(Z, y) if np.sum(y) > 0])

        if self._lorem_explainer.verbose:
            # generated neighborhood blackbox predicted labels
            y = self._decode_predict(Z)
            if not self._lorem_explainer.multi_label:
                neigh_class, neigh_counts = np.unique(y, return_counts=True)
                neigh_class_counts = {self._lorem_explainer.class_values[k]:
                                          v for k, v in zip(neigh_class, neigh_counts)}
            else:
                neigh_counts = np.sum(y, axis=0)
                neigh_class_counts = {self._lorem_explainer.class_values[k]:
                                          v for k, v in enumerate(neigh_counts)}
            print('synthetic neighborhood class counts %s' % neigh_class_counts)
        self._Z = Z
        return Z

    def _weights_calculation(self, use_weights=True, metric=neuclidean):
        if not use_weights:
            weights = None
        else:
            weights = self._lorem_explainer.__calculate_weights__(self._Z, metric)
        return weights

    def _tree_rules_extraction(self):
        weights = self._weights_calculation(use_weights=True, metric=neuclidean)
        # decodes the latent neighborhood
        self._Z_tilde = self.decoder.predict(self._Z)
        self._y = self.blackbox.predict(self._Z_tilde)
        if self._lorem_explainer.one_vs_rest and self._lorem_explainer.multi_label:
            exp = self._lorem_explainer._LOREM__explain_tabular_instance_multiple_tree(
                self._z,
                self._Z,
                self._y,
                weights)
        else:  # binary, multiclass, multilabel all together
            exp = self._lorem_explainer._LOREM__explain_tabular_instance_single_tree(
                self._z,
                self._Z,
                self._y,
                weights)
        self._lorem_explanation = exp

    def _build_rules_dataframes(self, verbose=False):

        # creates a dictionary having as keys ["rule", "crule0", ... , "cruleN"]
        # and as values a dictionary with keys ["Rule_obj", "df"]
        rules_dataframes = dict()
        rules_dataframes["rule"] = {"Rule_obj": self._lorem_explanation.rule, "df": []}

        rules_dataframes_latent = dict()
        rules_dataframes_latent["rule"] = {"Rule_obj": self._lorem_explanation.rule, "df": []}

        for i, counterfactual in enumerate(self._lorem_explanation.crules):
            rules_dataframes["crule" + str(i)] = {"Rule_obj": counterfactual, "df": []}
            rules_dataframes_latent["crule" + str(i)] = {"Rule_obj": counterfactual, "df": []}
        if verbose:
            print("N.RULES = ", 1)
            print("N.COUNTERFACTUAL = ", len(self._lorem_explanation.crules))

        Z_star = []
        Z_tilde_star = []
        for i, latent_instance in enumerate(self._Z):
            for rule in rules_dataframes.keys():
                if self._is_covered(rules_dataframes[rule]["Rule_obj"], latent_instance):
                    decoded_instance = self._Z_tilde[i].ravel()
                    rules_dataframes[rule]["df"].append(decoded_instance)
                    rules_dataframes_latent[rule]["df"].append(latent_instance)
                    Z_star.append(latent_instance)
                    Z_tilde_star.append(decoded_instance)
        self._Z_star = np.array(Z_star)
        self._Z_tilde_star = np.array(Z_tilde_star)[:, :, np.newaxis]
        self._y_star = self.blackbox.predict(self._Z_tilde_star)

        for rule in rules_dataframes.keys():
            rules_dataframes[rule]["df"] = pd.DataFrame(rules_dataframes[rule]["df"]).values
            rules_dataframes_latent[rule]["df"] = pd.DataFrame(rules_dataframes_latent[rule]["df"]).values
            medoid_idx = compute_medoid(rules_dataframes_latent[rule]["df"])
            rules_dataframes_latent[rule]["medoid_idx"] = medoid_idx
            rules_dataframes[rule]["medoid_idx"] = medoid_idx
            if verbose:
                print(rule + ": " + str(len(rules_dataframes[rule]["df"])) + " time series")

        self.rules_dataframes = rules_dataframes
        self.rules_dataframes_latent = rules_dataframes_latent
        return self

    def _is_covered(self, lorem_rule, z_prime):
        # checks if a latent instance satisfy a LOREM_Rule
        xd = self._vector_to_dict(z_prime, self._lorem_explainer.feature_names)
        for p in lorem_rule.premises:
            if p.op == '<=' and xd[p.att] > p.thr:
                return False
            elif p.op == '>' and xd[p.att] <= p.thr:
                return False
        return True

    def _vector_to_dict(self, x, feature_names):
        return {k: v for k, v in zip(feature_names, x)}

    def plot_exemplars_and_counterexemplars_by_rule(self, labels=None, **kwargs):
        plot_exemplars_and_counterexemplars_by_rule(
            self.rules_dataframes,
            labels=labels,
            **kwargs
        )
        return self


if __name__ == "__main__":
    from datasets import build_cbf
    from datasets import build_rnd_blobs
    from joblib import load
    from blackbox_wrapper import BlackboxWrapper
    import keras

    from blackbox_wrapper import BlackboxWrapper
    import keras
    from datasets import build_cbf, build_ecg200
    from variational_autoencoder import load_model
    from neighborhood_generators import NeighborhoodGenerator
    from utils import choose_z
    from saxdt import Saxdt
    from saxlog import Saxlog

    random_state = 0
    np.random.seed(random_state)
    dataset_name = "cbf"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600,
                                                               random_state=random_state)

    knn = load("./trained_models/cbf/cbf_knn.joblib")

    _, _, autoencoder = load_model("./trained_models/cbf/cbf_vae")

    blackbox = BlackboxWrapper(knn, 2, 1)
    encoder = autoencoder.layers[2]
    decoder = autoencoder.layers[3]

    i = 0
    x = X_exp_test[i].ravel().reshape(1, -1, 1)
    z = choose_z(x, encoder, decoder, n=1000, x_label=blackbox.predict(x)[0],
                 blackbox=blackbox, check_label=True, mse=False)
    z_label = blackbox.predict(decoder.predict(z))[0]

    neighborhood_generator = LoreWrapper(
        blackbox, decoder
    )

    neigh_kwargs = {
        "verbose": False, "samples": 100, "ngen": 1, "K": X_exp_test[1:], "K_encoded": encoder.predict(X_exp_test[1:])
    }

    out = neighborhood_generator.generate_neighborhood(z, **neigh_kwargs)
