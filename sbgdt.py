#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:19:29 2019

@author: francesco
"""
from tslearn.shapelets import (ShapeletModel,
                               grabocka_params_to_shapelet_size_dict,
                               GlobalMinPooling1D, LocalSquaredDistanceLayer,
                               GlobalArgminPooling1D)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tree_utils import (NewTree,
                        get_root_leaf_path,
                        get_thresholds_signs,
                        minimumDistance,
                        prune_duplicate_leaves,
                        coverage_score_tree, coverage_score_tree_old,
                        precision_score_tree
                        )
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib
from joblib import dump, load
import warnings
import keras


def save_sbgdt(explainer, file_path):
    """Saves the sbgdt model
    Parameters
    ----------
    explainer : Sbgdt object
    file_path : str, optional (default = "")
        saved file path
        
    Returns
    -------
    None
    
    Notes
    -----
    Saving the model also deletes the models inside the explainer
    """
    explainer._shapelet_model.locator_model_.save(file_path + "_locator.h5")
    explainer._shapelet_model.model_.save(file_path + "_model.h5")
    explainer._shapelet_model.transformer_model_.save(file_path + "_transformer.h5")
    explainer._shapelet_model.locator_model_ = None
    explainer._shapelet_model.model_ = None
    explainer._shapelet_model.transformer_model_ = None
    dump(explainer, file_path + "surrogate.pkl")


def load_sbgdt(file_path, locator=True, model=True, transformer=True):
    """Loades the sbgdt model
    Parameters
    ----------
    file_path : str, optional (default = "")
        loaded file path
    locator : bool, optional (default = True)
        load the locator
    model : bool, optional (default = True)
        load the model
    transformer : bool, optional (default = True)
        load the transformer
        
    Returns
    -------
    Sbgdt object
    
    Notes
    -----
    Loading all the models inside the sbgdt models takes a long time. Depending 
    on the specific task it could be desirable to load only some 
    """
    explainer = load(file_path + "surrogate.pkl")
    if locator:
        explainer._shapelet_model.locator_model_ = keras.models.load_model(
            file_path + "_locator.h5",
            custom_objects={'GlobalMinPooling1D': GlobalMinPooling1D,
                            'LocalSquaredDistanceLayer': LocalSquaredDistanceLayer,
                            'GlobalArgminPooling1D': GlobalArgminPooling1D})
    if model:
        explainer._shapelet_model.model_ = keras.models.load_model(
            file_path + "_model.h5",
            custom_objects={'GlobalMinPooling1D': GlobalMinPooling1D,
                            'LocalSquaredDistanceLayer': LocalSquaredDistanceLayer,
                            'GlobalArgminPooling1D': GlobalArgminPooling1D})
    if transformer:
        explainer._shapelet_model.transformer_model_ = keras.models.load_model(
            file_path + "_transformer.h5",
            custom_objects={'GlobalMinPooling1D': GlobalMinPooling1D,
                            'LocalSquaredDistanceLayer': LocalSquaredDistanceLayer,
                            'GlobalArgminPooling1D': GlobalArgminPooling1D})
    return explainer


def save_reload_sbgdt(explainer, file_path, locator=True, model=True, transformer=True):
    """Saves and reloades the sbgdt model
    Parameters
    ----------
    file_path : str, optional (default = "")
        loaded file path
    locator : bool, optional (default = True)
        load the locator
    model : bool, optional (default = True)
        load the model
    transformer : bool, optional (default = True)
        load the transformer
        
    Returns
    -------
    Sbgdt object
    
    Notes
    -----
    Loading all the models inside the sbgdt models takes a long time. Depending 
    on the specific task it could be desirable to load only some 
    """
    save_sbgdt(explainer, file_path)
    explainer = load_sbgdt(file_path, locator, model, transformer)
    return explainer


def plot_shapelets(explainer, figsize=(10, 3), color="mediumblue", dpi=60):
    """Plots all the model shapelets
    Parameters
    ----------
    explainer : Sbgdt object
    
    Returns
    -------
    None
    """
    shapelets = explainer._shapelet_model.shapelets_.copy()
    ts_length = explainer.X.shape[1]
    ts_max = explainer.X.max()
    ts_min = explainer.X.min()
    for i, shapelet in enumerate(shapelets):
        plt.figure(figsize=figsize)
        plt.gca().set_ylim((ts_min, ts_max))
        plt.gca().set_xlim((0, ts_length))
        plt.plot(shapelet.ravel(), lw=3, color=color)
        plt.axis('off')
        plt.text(len(shapelet.ravel()),
                 (shapelet.ravel().max()) + (ts_max - ts_min) / 10,
                 str(i),
                 c=color
                 )
        plt.show()


def plot_subsequences_grid(subsequence_list, n, m, starting_idx=0, random=False, color="mediumblue", fontsize=12,
                           text_height=0, **kwargs):
    fig, axs = plt.subplots(
        n,
        m,
        figsize=kwargs.get("figsize", (10, 5)),
        dpi=kwargs.get("dpi", 72)
    )
    fig.patch.set_visible(False)
    for i in range(n):
        for j in range(m):
            if random:
                starting_idx = np.random.randint(0, len(subsequence_list))
            axs[i, j].plot(subsequence_list[starting_idx].ravel()-subsequence_list[starting_idx].mean(), lw=3, color=color)
            axs[i, j].set_aspect('equal', adjustable='datalim')
            y_lim = axs[i, j].get_ylim()
            x_lim = axs[i, j].get_xlim()
            #axs[i, j].set_xlim((0, l))
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            axs[i, j].axis('off')
            axs[i, j].text(np.min(x_lim), 0 + text_height, str(starting_idx), fontsize=fontsize,
                           color=color,
                           horizontalalignment='center', verticalalignment='center', weight='bold',
                           path_effects=[patheffects.Stroke(linewidth=3, foreground='white', alpha=0.6),
                                         patheffects.Normal()])
            starting_idx += 1
    plt.tight_layout()
    plt.show()



def plot_binary_heatmap(
        x_label,
        y,
        X_binary,
        figsize=(8, 8),
        dpi=60,
        fontsize=20,
        labelsize=20,
        step=1,
):
    """Plots a heatmap of the contained and not contained shapelet
    Parameters
    ----------
    explainer : Sbgdt object
    exemplars_labels : int
        instance to explain label
    
    Returns
    -------
    self
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = y.copy()
    X = X_binary.copy()
    # 0: no shapelet, 1: shapelet counterfactual, 2: shapelet factual
    X[y == x_label] *= 2
    sorted_by_class_idxs = y.argsort()
    sorted_dataset = X[sorted_by_class_idxs]
    cmap = matplotlib.colors.ListedColormap(['white', "#d62728", '#2ca02c'])
    plt.ylabel("subsequences", fontsize=fontsize)
    plt.xlabel("time series", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.matshow(sorted_dataset.T, cmap=cmap)
    ax.set_yticks(np.arange(0, sorted_dataset.shape[1], step=step))
    ax.set_aspect(0.5 * sorted_dataset.shape[0] / sorted_dataset.shape[1])
    plt.show()


def plot_shapelet_space(
        explainer,
        x,
        x_label,
        figsize=(8, 8),
        dpi=60,
        frameon=True,
        fontsize=12):
    """Plots a 2d representation of the shapelet space
    Parameters
    ----------
    explainer : Sbgdt object
    x : instance to explain
    x_label : instance to explain label
    
    Returns
    -------
    self
    """
    pca_2d = PCA(n_components=2)
    pca_2d.fit(explainer.X_transformed)
    dataset_latent_2dconversion = pca_2d.transform(explainer.X_transformed)
    dataset_latent_2dconversion_labels = explainer.y
    instance_to_explain_shapelet = explainer._shapelet_model.transform(x)
    instance_to_explain_2d = pca_2d.transform(instance_to_explain_shapelet).ravel()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(r"$\hat{\Xi}$", fontsize=fontsize)

    exemplars = np.argwhere(dataset_latent_2dconversion_labels == x_label)
    counterexemplars = np.argwhere(dataset_latent_2dconversion_labels != x_label)

    ax.scatter(dataset_latent_2dconversion[:, 0][exemplars],
               dataset_latent_2dconversion[:, 1][exemplars],
               c="#2ca02c",
               alpha=0.5,
               label=r"$\hat{\Xi}_=$"
               )
    ax.scatter(dataset_latent_2dconversion[:, 0][counterexemplars],
               dataset_latent_2dconversion[:, 1][counterexemplars],
               c="#d62728",
               alpha=0.5,
               label=r"$\hat{\Xi}_\neq$"
               )
    ax.scatter(instance_to_explain_2d[0],
               instance_to_explain_2d[1], label=r"$\hat{\xi}$",
               c="royalblue", marker="X", edgecolors="white",
               s=400)
    ax.legend(fontsize=fontsize, frameon=frameon)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.show()


def plot_graphical_explanation(explainer,
                               x,
                               rule,
                               predicted_locations,
                               title,
                               legend_label,
                               figsize,
                               dpi,
                               fontsize,
                               text_height,
                               labelfontsize,
                               loc,
                               frameon,
                               forced_y_lim=None,
                               return_y_lim=False
                               ):
    """Plots a graphical representation of a rule
    Parameters
    ----------
    explainer : Sbgdt object
    x : array of shape
        instance to explain
    rule : dict
        root-leaf path dictionary
    predicted_location : list
        best shapelets alignments
    [...]
    forced_y_lim : tuple, optional (default = None)
        y_lim value
    return_y_lim : bool, optional (default = None)
        return y_lim for consistent y scale
    
    Returns
    -------
    None or y_lim
    """
    threshold_array = np.full(len(x.ravel()), np.NaN)
    for i, idx_shp in enumerate(rule["features"][:-1]):
        shp = explainer._shapelet_model.shapelets_[idx_shp].ravel()
        threshold_sign = rule["thresholds_signs"][i]
        t0 = predicted_locations[0, idx_shp]
        if threshold_sign == "contained":
            threshold_array[t0:t0 + len(shp)] = 0

    cmap = matplotlib.colors.ListedColormap(["#2ca02c"])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title, fontsize=fontsize)

    ax.plot(x.T, c="royalblue", alpha=0.2, lw=3, label=legend_label)
    for i, idx_shp in enumerate(rule["features"][:-1]):
        shp = explainer._shapelet_model.shapelets_[idx_shp].ravel()
        threshold_sign = rule["thresholds_signs"][i]
        t0 = predicted_locations[0, idx_shp]
        ax.plot(np.arange(t0, t0 + len(shp)), shp,
                linestyle="-" if threshold_sign == "contained" else "--",
                alpha=0.5 if threshold_sign == "contained" else 0.5,
                label=threshold_sign,
                c="#2ca02c" if threshold_sign == "contained" else "#d62728",
                lw=5
                )
        plt.text((t0 + t0 + len(shp) - 2) / 2 if i != 0 else 1 + (t0 + t0 + len(shp)) / 2,
                 shp[int(len(shp) / 2)] + text_height,
                 str(idx_shp),
                 fontsize=fontsize - 2,
                 c="#2ca02c" if threshold_sign == "contained" else "#d62728",
                 horizontalalignment='center', verticalalignment='center', weight='bold',
                 path_effects=[patheffects.Stroke(linewidth=3, foreground='white', alpha=0.6),
                               patheffects.Normal()]
                 )
    # ax.pcolorfast((0, len(threshold_array) - 1),
    #               ax.get_ylim() if forced_y_lim is None else forced_y_lim,
    #               threshold_array[np.newaxis],
    #               cmap=cmap,
    #               alpha=0.2
    #               )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xlabel("time-steps", fontsize=fontsize)
    plt.ylabel("value", fontsize=fontsize)
    plt.legend(by_label.values(),
               by_label.keys(),
               frameon=frameon,
               fontsize=labelfontsize,
               loc=loc)
    if forced_y_lim is not None:
        plt.gca().set_ylim(forced_y_lim)
    if return_y_lim:
        y_lim = plt.gca().get_ylim()
    plt.show()
    if return_y_lim:
        return y_lim


def plot_factual_and_counterfactual(
        explainer,
        x,
        x_label,
        draw_on=None,
        verbose_explanation=True,
        hybrid_explanation=False,
        graphical_explanation=True,
        figsize=(10, 3),
        fontsize=18,
        text_height=0.5,
        c_index=0,
        labelfontsize=12,
        dpi=60,
        loc=None,
        frameon=False
):
    """Plot the explanation for the prediction of an instance
    Parameters
    ----------
    explainer : Sbgdt object
    x : instance to explain (it should be z_tilde)
    x_label : instance to explain label
    draw_on : array of shape (1, n_features), optional (default = None)
        instance to plot for the exemplars (if you want to plot on an instance
        different than z_tilde)
    verbose_explanation : bool, optional (default = True)
        print the factual and counterfactual rule
    hybrid_explanation = bool, optional (default = False)
        plot the verbose explanation plottin also the shapelets
    graphical_explanation = bool, optional (default = True)
        plot the factual and counterfactual rules
    c_index : int, optional (default = 0)
        index of the counterfactual in the leaf to plot
    """

    if len(x.shape) == 3:
        x = x[:, :, 0]

    x_transformed = explainer._shapelet_model.transform(x)
    x_thresholded = 1 * (x_transformed < explainer.tau)

    predicted_locations = explainer._shapelet_model.locate(x)

    if draw_on is not None:
        predicted_locations = explainer._shapelet_model.locate(draw_on)

    dtree = NewTree(explainer.decision_tree)
    dtree.build_tree()
    leave_id = explainer.decision_tree.apply(x_thresholded)[0]

    factual = get_root_leaf_path(dtree.nodes[leave_id])
    factual = get_thresholds_signs(dtree, factual)

    nearest_leaf = minimumDistance(dtree.nodes[0], dtree.nodes[leave_id])[1]

    counterfactual = get_root_leaf_path(dtree.nodes[nearest_leaf])
    counterfactual = get_thresholds_signs(dtree, counterfactual)

    rules_list = [factual, counterfactual]

    if verbose_explanation:
        print("VERBOSE EXPLANATION")
        for i, rule in enumerate(rules_list):
            print()
            print("RULE" if i == 0 else "COUNTERFACTUAL")
            if i == 0:
                print("real class ==", explainer.labels[x_label] if explainer.labels else x_label)
            print("If", end=" ")
            for i, idx_shp in enumerate(rule["features"][:-1]):
                print("shapelet n.", idx_shp, "is", rule["thresholds_signs"][i], end="")
                if i != len(rule["features"][:-1]) - 1:
                    print(", and", end=" ")
                else:
                    print(",", end=" ")
            print("then the class is", rule["labels"][-1] if not explainer.labels \
                else explainer.labels[rule["labels"][-1]])

    if hybrid_explanation:
        print()
        print("HYBRID EXPLANATION")
        for i, rule in enumerate(rules_list):
            print("RULE" if i == 0 else "COUNTERFACTUAL")
            print("If", end=" ")
            for i, idx_shp in enumerate(rule["features"][:-1]):
                plt.figure(figsize=figsize)
                plt.xlim((0, len(x.ravel()) - 1))
                plt.plot(x.T, c="gray", alpha=0)
                print("shapelet n.", idx_shp, "is", rule["thresholds_signs"][i], end="")
                shp = explainer._shapelet_model.shapelets_[idx_shp].ravel()
                plt.plot(shp,
                         c="#2ca02c" if rule["thresholds_signs"][i] == "contained" else "#d62728",
                         linewidth=3
                         )
                plt.axis('off')
                plt.show()
                if i != len(rule["features"][:-1]) - 1:
                    print("and", end=" ")
                else:
                    print("", end="")
            print("then the class is", rule["labels"][-1] if not explainer.labels \
                else explainer.labels[rule["labels"][-1]])
            print()

    if graphical_explanation:
        # factual rule applied to x
        title = ("Factual Rule\n" + r"$p_s\rightarrow$" + " " +
                          explainer.labels[rules_list[0]["labels"][-1]] if explainer.labels
                          else "Factual Rule\n" + r"$p_s\rightarrow$" + " " +
                               str(rules_list[0]["labels"][-1]))
        legend_label = r"$x$"
        y_lim = plot_graphical_explanation(
            explainer,
            x if draw_on is None else draw_on,
            rules_list[0],
            predicted_locations,
            title,
            legend_label,
            figsize,
            dpi,
            fontsize,
            text_height,
            labelfontsize,
            loc,
            frameon,
            return_y_lim=True
        )

        # counterfactual rule applied to x
        title = ("Counterfactual Rule\n" + r"$q_s\rightarrow$" + " " +
                 explainer.labels[rules_list[1]["labels"][-1]] if explainer.labels
                 else "Counterfactual Rule\n" + r"$q_s\rightarrow$" + " " +
                      str(rules_list[1]["labels"][-1]))
        legend_label = r"$x$"
        plot_graphical_explanation(
            explainer,
            x if draw_on is None else draw_on,
            rules_list[1],
            predicted_locations,
            title,
            legend_label,
            figsize,
            dpi,
            fontsize,
            text_height,
            labelfontsize,
            loc,
            frameon,
            forced_y_lim=y_lim
        )

        # counterfactual rule applied to a counterfactual z_tilde
        # get all the leave ids
        leave_ids = explainer.decision_tree.apply(explainer.X_thresholded)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.argwhere(leave_ids == nearest_leaf)
        # choose one counterfactual
        counterfactual_idx = counterfactuals_idxs[c_index]
        counterfactual_ts = explainer.X[counterfactual_idx].reshape(1, -1)
        counterfactual_y = explainer.y[counterfactual_idx][0]
        print("real class ==", explainer.labels[counterfactual_y] if explainer.labels else counterfactual_y)
        predicted_locations = explainer._shapelet_model.locate(counterfactual_ts)
        title = ("Factual Rule for a " + r"$\tilde{z}'$" + "\n" + r"$q_s\rightarrow$" + " " +
                 explainer.labels[rules_list[1]["labels"][-1]] if explainer.labels else
                 "Factual Rule for a " + r"$\tilde{z}'$" + "\n" + r"$q_s\rightarrow$" + " " +
                 str(rules_list[1]["labels"][-1]))
        legend_label = r"$\tilde{z}'$"
        plot_graphical_explanation(explainer,
                                   counterfactual_ts,
                                   rules_list[1],
                                   predicted_locations,
                                   title,
                                   legend_label,
                                   figsize,
                                   dpi,
                                   fontsize,
                                   text_height,
                                   labelfontsize,
                                   loc,
                                   frameon,
                                   forced_y_lim=y_lim
                                   )

def generate_n_shapelets_per_size(ts_length, n_shapelets_per_length=4, min_length=8, start_divider=2, divider_multiplier=2):
    n_shapelets_per_size = dict()
    while int(ts_length / start_divider) >= min_length:
        n_shapelets_per_size[int(ts_length / start_divider)] = n_shapelets_per_length
        start_divider *= divider_multiplier
    return n_shapelets_per_size


class Sbgdt(object):
    def __init__(
            self,
            labels=None,
            random_state=None,
            tau=None,
            tau_quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            shapelet_model_params={
                "l": 0.1,
                "r": 2,
                "optimizer": "sgd",
                "n_shapelets_per_size": "heuristic",
                "weight_regularizer": .01,
                "max_iter": 100},
            decision_tree_grid_search_params={
                'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]},
            prune_duplicate_tree_leaves=True
    ):

        """Shapelet-based global decision tree
        Parameters
        ----------
        labels : array of shape (n_classes,) (default = None)
            classes string labels
        random_state : int or None, optional (default = None)
            The seed of the pseudo random number generator to use when shuffling
            the data.  If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the RandomState
        tau: int, optional (default = None)
            distance threshold that defines contained/not-contained shapelets
        tau_quantiles: array, optional (default = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
            quantiles to test in order to find tau
        shapelet_model_params: dict, optional
            parameters for the ShapeletModel
        decision_tree_grid_search_params: dict, optional
            parameters for the gridsearch to find the best hyperparameters for the DecisionTreeClassifier
        
        
        Attributes
        ----------
        
        """

        self.random_state = random_state
        self.tau = tau
        self.tau_quantiles = tau_quantiles
        self.labels = labels
        self.shapelet_model_params = shapelet_model_params
        self.decision_tree_grid_search_params = decision_tree_grid_search_params
        self.prune_duplicate_tree_leaves = prune_duplicate_tree_leaves

        self.decision_tree = None
        self.decision_tree_explorable = None
        self._shapelet_model = None
        self.X = None
        self.y = None
        self.X_transformed = None
        self.X_thresholded = None
        self._graph = None

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : {array-like}
            Training data. Shape [n_samples, n_features].
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]
        """

        self.X = X
        self.y = y

        n_shapelets_per_size = self.shapelet_model_params.get("n_shapelets_per_size", "heuristic")
        if n_shapelets_per_size == "heuristic":
            n_ts, ts_sz = X.shape[:2]
            n_classes = len(set(y))
            n_shapelets_per_size = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                                         ts_sz=ts_sz,
                                                                         n_classes=n_classes,
                                                                         l=self.shapelet_model_params.get("l", 0.1),
                                                                         r=self.shapelet_model_params.get("r", 2))

        shp_clf = ShapeletModel(n_shapelets_per_size=n_shapelets_per_size,
                                optimizer=self.shapelet_model_params.get("optimizer", "sgd"),
                                weight_regularizer=self.shapelet_model_params.get("weight_regularizer", .01),
                                max_iter=self.shapelet_model_params.get("max_iter", 100),
                                random_state=self.random_state,
                                verbose=self.shapelet_model_params.get("verbose", 0))

        shp_clf.fit(X, y)
        X_transformed = shp_clf.transform(X)
        self.X_transformed = X_transformed

        if self.tau is not None:
            self.X_thresholded = 1 * (self.X_transformed < self.tau)
            clf = DecisionTreeClassifier()
            param_grid = self.decision_tree_grid_search_params
            grid = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=0)
            grid.fit(self.X_thresholded, y)
        else:
            grids = []
            grids_scores = []
            for quantile in self.tau_quantiles:
                _X_thresholded = 1 * (self.X_transformed < (np.quantile(self.X_transformed, quantile)))
                clf = DecisionTreeClassifier()
                param_grid = self.decision_tree_grid_search_params
                grid = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=0)
                grid.fit(_X_thresholded, y)
                grids.append(grid)
                grids_scores.append(grid.best_score_)
            grid = grids[np.argmax(np.array(grids_scores))]
            best_quantile = self.tau_quantiles[np.argmax(np.array(grids_scores))]
            self.tau = np.quantile(self.X_transformed, best_quantile)
            self.X_thresholded = 1 * (self.X_transformed < self.tau)

        clf = DecisionTreeClassifier(**grid.best_params_)
        clf.fit(self.X_thresholded, y)
        if self.prune_duplicate_tree_leaves:
            prune_duplicate_leaves(clf)  # FIXME: does it influence the .tree properties?

        self.decision_tree = clf
        self.decision_tree_explorable = NewTree(clf)
        self.decision_tree_explorable.build_tree()
        self._shapelet_model = shp_clf
        self._build_tree_graph()

        return self

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def predict(self, X):
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : {array-like}
            Test data. Shape [n_samples, n_features].
        Returns
        -------
        y : array of shape [n_samples]
        """

        X_transformed = self._shapelet_model.transform(X)
        X_thresholded = 1 * (X_transformed < self.tau)
        y = self.decision_tree.predict(X_thresholded)
        return y

    def _build_tree_graph(self, out_file=None):
        dot_data = tree.export_graphviz(self.decision_tree, out_file=out_file,
                                        class_names=self.labels,
                                        filled=True, rounded=True,
                                        special_characters=True)
        self._graph = graphviz.Source(dot_data)
        return self

    def coverage_score_old(self, ts, X=None):
        if X is None:
            X = self.X_thresholded.copy()
        else:
            X_transformed = self._shapelet_model.transform(X)
            X_thresholded = 1 * (X_transformed < self.tau)
            X = X_thresholded

        ts_transformed = self._shapelet_model.transform(ts)
        ts_thresholded = 1 * (ts_transformed < self.tau)

        return coverage_score_tree_old(self.decision_tree, X, ts_thresholded)

    def find_leaf_id(self, ts):
        ts_transformed = self._shapelet_model.transform(ts)
        ts_thresholded = 1 * (ts_transformed < self.tau)
        leaf_id = self.decision_tree.apply(ts_thresholded)[0]
        return leaf_id

    def coverage_score(self, leaf_id):
        return coverage_score_tree(self.decision_tree, leaf_id)

    def precision_score(self, leaf_id, y, X=None):
        if X is None:
            X = self.X_thresholded.copy()
        else:
            X_transformed = self._shapelet_model.transform(X)
            X_thresholded = 1 * (X_transformed < self.tau)
            X = X_thresholded
        return precision_score_tree(self.decision_tree, X, y, leaf_id)

    def plot_factual_and_counterfactual(
            self,
            x,
            x_label,
            **kwargs
    ):
        plot_factual_and_counterfactual(self, x, x_label, **kwargs)
        return self

    def plot_binary_heatmap(
            self,
            x_label,
            **kwargs
    ):
        plot_binary_heatmap(x_label, self.y, self.X_thresholded, **kwargs)
        return self

    def plot_subsequences_grid(self, n, m, starting_idx=0, random=False, color="mediumblue", **kwargs):
        plot_subsequences_grid(self._shapelet_model.shapelets_, n, m, starting_idx=starting_idx, random=random,
                               color=color, **kwargs)
        return self


if __name__ == '__main__':
    from pyts.datasets import make_cylinder_bell_funnel
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # warnings.filterwarnings("ignore")

    random_state = 0
    dataset_name = "cbf"

    X, y = make_cylinder_bell_funnel(n_samples=100, random_state=random_state)

    print("DATASET INFO:")
    print("X SHAPE: ", X.shape)
    print("y SHAPE: ", y.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("\nCLASSES BALANCE")
    for i, label in enumerate(unique):
        print(label, ": ", round(counts[i] / sum(counts), 2))

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=random_state)

    print("\nSHAPES:")
    print("TRAINING SET: ", X_train.shape)
    print("TEST SET: ", X_test.shape)
    n_timesteps, n_outputs, n_features = X_train.shape[1], len(np.unique(y)), 1
    print("\nTIMESTEPS: ", n_timesteps)
    print("N. LABELS: ", n_outputs)


    shapelet_model_params = {
                "l": 0.1,
                "r": 2,
                "optimizer": "sgd",
                "n_shapelets_per_size": generate_n_shapelets_per_size(128, divider_multiplier=1.2),  #"heuristic",
                "weight_regularizer": .01,
                "max_iter": 100
    }

    sbgdt_ = Sbgdt(random_state=random_state, shapelet_model_params=shapelet_model_params)
    sbgdt_.fit(X_train[:, :, np.newaxis], y_train)
    print("Test Accuracy: ", accuracy_score(y_test, sbgdt_.predict(X_test[:, :, np.newaxis])))

    idx = 0

    print("\nSHAPELETS")
    # plot_shapelets(sbgdt)

    print("\nSHAPELET SPACE")
    plot_shapelet_space(sbgdt_, X_test[idx].reshape(1, -1), y_test[idx])

    print("\nBINARY SPACE")
    sbgdt_.plot_binary_heatmap(y_test[idx])

    print("\nRULES")
    sbgdt_.plot_factual_and_counterfactual(X_test[idx].reshape(1, -1, 1), y_test[idx])