#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 00:13:49 2020

@author: francesco
"""

from sklearn.metrics import accuracy_score, pairwise_distances
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import warnings
from scipy.stats import wasserstein_distance, norm
import seaborn as sns
import os
import glob


def compute_medoid(X):
    distance_matrix = pairwise_distances(X, n_jobs=-1)
    medoid_idx = np.argmin(distance_matrix.sum(axis=0))
    return X[medoid_idx]


def convert_numpy_to_sktime(X):
    df_dict = dict()
    for ts in X:
        for dimension in range(X.shape[2]):
            if dimension in df_dict.keys():
                df_dict[dimension].append(pd.Series(ts[:, dimension]))
            else:
                df_dict[dimension] = [pd.Series(ts[:, dimension])]
    df = pd.DataFrame(df_dict)
    return df


def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))


def explanation_error(true_importance, pred_importance):
    return np.abs(true_importance - pred_importance).sum() / len(true_importance)


def baseline_error(true_importance):
    ones = np.ones_like(true_importance)
    zeros = np.zeros_like(true_importance)
    baseline = min(explanation_error(true_importance, ones), explanation_error(true_importance, zeros))
    # if baseline == 0:
    #     baseline = 1
    return baseline


def convert_sktime_to_numpy(X):
    np_tss = []
    for ts in X.iloc:
        np_ts = []
        for dimension in range(len(X.columns)):
            np_ts.append(np.array(ts[dimension]).reshape(1, -1, 1))
        np_ts = np.concatenate(np_ts, axis=2)
        np_tss.append(np_ts)
    np_tss = np.concatenate(np_tss)
    return np_tss


def sliding_window_distance(ts, s):
    distances = []
    for i in range(len(ts) - len(s) + 1):
        ts_s = ts[i:i + len(s)]
        dist = np.linalg.norm(s - ts_s)
        distances.append(dist)
    return np.argmin(distances)


def sliding_window_euclidean(ts, s):
    distances = []
    for i in range(len(ts) - len(s) + 1):
        ts_s = ts[i:i + len(s)]
        dist = np.linalg.norm(s - ts_s)
        distances.append(dist)
    return np.min(distances)


def choose_z(x, encoder, decoder, n=1000, x_label=None, blackbox=None, check_label=False, verbose=False, mse=False):
    X = np.repeat(x, n, axis=0)
    Z = encoder.predict(X)
    Z_tilde = decoder.predict(Z)
    if check_label:
        y_tilde = blackbox.predict(Z_tilde)
        y_correct = np.nonzero(y_tilde == x_label)
        if len(Z_tilde[y_correct]) == 0:
            if verbose:
                warnings.warn("No instances with the same label of x found.")
        else:
            Z_tilde = Z_tilde[y_correct]
            Z = Z[y_correct]
    if mse:
        distances = []
        for z_tilde in Z_tilde:
            distances.append(((x - z_tilde) ** 2).sum())
        distances = np.array(distances)
    else:
        distances = cdist(x[:, :, 0], Z_tilde[:, :, 0])
    best_z = Z[np.argmin(distances)]
    return best_z.reshape(1, -1)


def plot_choose_z(x, encoder, decoder, n=100, K=None):
    Z = []
    for i in range(n):
        Z.append(encoder.predict(x).ravel())
    Z = np.array(Z)
    Z_tilde = decoder.predict(Z)
    distances = cdist(x[:, :, 0], Z_tilde[:, :, 0]).ravel()
    plt.scatter(Z[:, 0], Z[:, 1], c=distances, cmap="Greens_r", norm=matplotlib.colors.PowerNorm(gamma=0.1))
    if K is not None:
        plt.scatter(K[:, 0], K[:, 1], c="lightgray")
    plt.show()


def euclidean_norm(Z):
    Z_norm = list()
    for z in Z:
        Z_norm.append(np.linalg.norm(z))
    return np.array(Z_norm)


def norm_distance(Z, distance=wasserstein_distance):
    """Compute the distance between the euclidean norms of the instances in Z
    and instances extracted from a gaussian distribution

    Parameters
    ----------
    Z: array of shape (n_samples, n_features)
        dataset
    distance: string, optional (default=wasserstein_distance)
        type of distance
    Returns
    -------
    distance: int
        distance between the euclidean norms
    """

    Z_norm = euclidean_norm(Z)
    rnd_norm = euclidean_norm(np.random.normal(size=Z.shape))
    return distance(Z_norm, rnd_norm)


def plot_norm_distributions(norm_array_list, labels=None):
    norm_array_df = pd.DataFrame(norm_array_list).T
    norm_array_df.columns = labels
    for column in norm_array_df.columns:
        sns.kdeplot(norm_array_df[column])
    plt.show()


def reconstruction_accuracy(X, encoder, decoder, blackbox, repeat=1, verbose=True):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        y_tilde = blackbox.predict(decoder.predict(encoder.predict(X)))
        accuracy = accuracy_score(y, y_tilde)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


def reconstruction_accuracy_vae(X, encoder, decoder, blackbox, repeat=1, n=100, check_label=True, verbose=True):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        Z = list()
        for x in X:
            if check_label:
                x_label = blackbox.predict(x[np.newaxis, :, :])
            else:
                x_label = None
            z = choose_z(x[np.newaxis, :, :], encoder, decoder, n, x_label, blackbox, check_label, verbose)
            Z.append(z.ravel())
        Z = np.array(Z)
        y_tilde = blackbox.predict(decoder.predict(Z))
        accuracy = accuracy_score(y, y_tilde)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


def exemplars_and_counterexemplars_similarities(x, x_label, X, y):
    """Compute similarities between an instance and exemplars and counterexemplars
    Parameters
    ----------
    x : {array-like}
        Instance. Shape [n_samples, n_features].
    x_label : {integer}
        Instance label.
    X : {array-like}
        Data. Shape [n_samples, n_features].
    y : {array-like} 
        Data labels of shape [n_samples]
    Returns
    -------
    s_exemplars : similarities from the exemplars
    s_counterexemplars : similarities from the counterexemplars
    """
    x = x.ravel().reshape(1, -1)
    exemplar_idxs = np.argwhere(y == x_label).ravel()
    exemplars = X[exemplar_idxs]
    counterexemplar_idxs = np.argwhere(y != x_label).ravel()
    counterexemplars = X[counterexemplar_idxs]
    s_exemplars = 1 / (1 + cdist(exemplars, x))
    s_counterexemplars = 1 / (1 + cdist(counterexemplars, x))

    return s_exemplars.ravel(), s_counterexemplars.ravel()


def plot_usefulness(lasts_df, real_df, dataset_label="", figsize=(7, 3), fontsize=20, dpi=72, alpha=1, **kwargs):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(dataset_label, fontsize=fontsize)
    plt.plot(np.array(lasts_df).mean(axis=0), label="lasts", lw=3, marker="o", c="#01665e", alpha=alpha)
    plt.plot(np.array(real_df).mean(axis=0), label="real", lw=3, marker="o", c="darkgoldenrod", alpha=alpha)
    plt.xticks(ticks=list(range(len(lasts_df.columns))), labels=lasts_df.columns)
    plt.gca().set_ylim((0.3, 1.05))
    plt.yticks(ticks=np.arange(3, 11) / 10)
    plt.ylabel("accuracy", fontsize=fontsize)
    plt.xlabel("nbr (counter)exemplars", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    loc = matplotlib.ticker.MultipleLocator(base=0.1)  # this locator puts ticks at regular intervals
    axes = plt.gca()
    # y_lim = plt.gca().get_ylim()
    axes.yaxis.set_major_locator(loc)
    plt.legend(frameon=False, fontsize=fontsize, loc="lower right", ncol=2, columnspacing=2, handletextpad=0.5)
    plt.show()


def usefulness_scores_real(X, y, n=[1, 2, 4, 8, 16]):
    df = pd.DataFrame()
    for i, x in enumerate(X[:, :, 0]):
        X_wo_x = np.delete(X[:, :, 0], i, axis=0)
        y_wo_x_label = np.delete(y, i)
        df = df.append(pd.DataFrame(usefulness_scores(X_wo_x, y_wo_x_label, x, y[i], n=n), index=[i]))
    return df


def usefulness_scores_lasts(lasts_list, n=[1, 2, 4, 8, 16]):
    df = pd.DataFrame()
    for i, lasts_ in enumerate(lasts_list):
        df = df.append(pd.DataFrame(lasts_.usefulness_scores(n=n), index=[i]))
    return df


def usefulness_scores(X, y, x, x_label, n=[1, 2, 4, 8, 16]):
    """Compute the knn prediction for x using n exemplars and counterexemplars
    Parameters
    ----------
    x_label
    X : array of shape (n_samples, n_features)
        dataset
    y : array of shape (n_samples,)
        dataset labels
    x : array of shape (n_features,)
        instance to benchmark
    n : list, optional (default = [1,2,4,8,16])
        number of instances to extract from every class
    Returns
    -------
    y_by_n : array of shape (len(n),)
        predicted class for x, for every n
    """
    x = x.ravel()
    dfs = dict()
    for key in n:
        dfs[key] = {"X": [], "y": []}
    for unique_label in np.unique(y):
        same_label_idxs = np.argwhere(y == unique_label).ravel()
        same_label_records = X[same_label_idxs]
        for n_record in n:
            random_idxs = np.random.choice(same_label_records.shape[0],
                                           min(n_record, same_label_records.shape[0]),
                                           replace=False).ravel()
            dfs[n_record]["X"].extend(same_label_records[random_idxs])
            dfs[n_record]["y"].extend(np.repeat(unique_label, len(random_idxs)))
    for key in dfs.keys():
        dfs[key]["X"] = np.array(dfs[key]["X"])
        dfs[key]["y"] = np.array(dfs[key]["y"])
    accuracy_by_n = dict()
    for key in dfs.keys():
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(dfs[key]["X"], dfs[key]["y"])
        # y_by_n.append(knn.predict(x.ravel().reshape(1, -1))[0])
        accuracy_by_n[key] = knn.score(x.ravel().reshape(1, -1), [x_label])
    return accuracy_by_n



def triangle_distribution(size, **kwargs):
    coordinates = np.array([[1, 0], [-1, 0], [0, math.sqrt(3)]])
    idxs = np.random.randint(0, 3, size=(size[0],))
    return coordinates[idxs]


def plot_reconstruction(X, encoder, decoder, figsize=(20, 15), n=0):
    X_tilde = decoder.predict(encoder.predict(X))
    g = 1
    plt.figure(figsize=figsize)
    for i in range(n, n + 5):
        # display original
        ax = plt.subplot(5, 1, g)
        g += 1
        plt.plot(X[i], label="real")
        plt.plot(X_tilde[i], label="reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()


def plot_reconstruction_vae(X, encoder, decoder, figsize=(20, 15), n=0):
    Z = list()
    for x in X:
        z = choose_z(x[np.newaxis, :, :], encoder, decoder)
        Z.append(z.ravel())
    Z = np.array(Z)
    X_tilde = decoder.predict(Z)
    g = 1
    plt.figure(figsize=figsize)
    for i in range(n, n + 5):
        # display original
        ax = plt.subplot(5, 1, g)
        g += 1
        plt.plot(X[i], label="real")
        plt.plot(X_tilde[i], label="reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()


def plot_choose_z_latent_space(X, encoder, decoder, blackbox, n=1000, figsize=(20, 15)):
    Z = list()
    for x in X:
        x_label = blackbox.predict(x[np.newaxis, :, :])[0]
        z = choose_z(x[np.newaxis, :, :], encoder, decoder, n, x_label, blackbox)
        Z.append(z.ravel())
    Z = np.array(Z)
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.show()


def plot_labeled_latent_space_matrix(
        Z,
        y,
        **kwargs
):
    Z = list(Z)
    Z = pd.DataFrame(Z)
    pd.plotting.scatter_matrix(Z,
                               c=y,
                               cmap="viridis",
                               diagonal="kde",
                               alpha=1,
                               s=100,
                               figsize=kwargs.get("figsize", (8, 8)))


def probability_density_mean(Z):
    return norm.pdf(Z).mean()


def boxplots_from_dfs(df1, df2, labels=(1, 2)):
    for column in df1.columns:
        array = np.concatenate([df1[column].values[np.newaxis, :], df2[column].values[np.newaxis, :]], axis=0)
        plt.title(column)
        plt.boxplot(array.T)
        plt.xticks([1, 2], labels)
        plt.show()
    return


def boxplots_from_df_list(df_list, labels=None, print_means=True, figsize=(8.0, 6.0)):
    medianprops = dict(linestyle='-', linewidth=3, color='#8c510a')
    meanprops = dict(marker='D', markeredgecolor='#003c30', markerfacecolor='#003c30')
    boxprops = dict(linestyle='-', linewidth=3, color='#01665e')
    whiskerprops = dict(linestyle='-', linewidth=3, color='#01665e')
    capprops = dict(linestyle='-', linewidth=3, color='#01665e')
    if labels is None:
        labels = np.array(range(len(df_list))) + 1
    for column in df_list[0].columns:
        array = df_list[0][column].values[np.newaxis, :]
        if len(df_list) > 1:
            for df in df_list[1:]:
                array = np.concatenate([array, df[column].values[np.newaxis, :]], axis=0)
        # array = array[0]
        plt.figure(figsize=figsize)
        plt.title(column)
        plt.boxplot(array.T, medianprops=medianprops, boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops,
                    meanprops=meanprops, showmeans=False, showfliers=False)
        plt.xticks(list(np.array(range(len(df_list))) + 1), labels)
        plt.show()
    if print_means:
        df_means = means_from_df_list(df_list, labels)
        print(df_means)
    return


def boxplot_from_df(df, figsize=(8.0, 6.0), dpi=72, labels=None, fontsize=18, ylabel=""):
    medianprops = dict(linestyle='-', linewidth=3, color='#8c510a')
    meanprops = dict(marker='D', markeredgecolor='#003c30', markerfacecolor='#003c30')
    boxprops = dict(linestyle='-', linewidth=3, color='#01665e')
    whiskerprops = dict(linestyle='-', linewidth=3, color='#01665e')
    capprops = dict(linestyle='-', linewidth=3, color='#01665e')
    plt.figure(figsize=figsize, dpi=dpi)
    #plt.title("Explanation Error")
    if isinstance(df, list):
        plt.boxplot(df, medianprops=medianprops, boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops,
                    meanprops=meanprops, showmeans=False, showfliers=False)
    else:
        plt.boxplot(df.values, medianprops=medianprops, boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops,
                    meanprops=meanprops, showmeans=False, showfliers=False)
    #plt.axhline(y=1, color='r', linestyle='--')
    plt.xticks(np.array(list(range(len(df.columns)))) + 1, list(df.columns) if labels is None else labels)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.show()



def means_from_df_list(df_list, labels=None):
    df_means = pd.DataFrame()
    for df in df_list:
        df_means = df_means.append(df.mean(), ignore_index=True)
    if labels is not None:
        df_means.index = labels
    return df_means


def read_metrics_csv(folder, plot=True, **kwargs):
    names = sorted([os.path.basename(filename.replace(".csv", "")) for filename in glob.glob(folder + "*.csv")])
    df_list = list()
    for name in names:
        df = pd.read_csv(folder + name + ".csv", sep=";")
        df_list.append(df)
    if plot:
        boxplots_from_df_list(
            df_list,
            labels=names,
            figsize=kwargs.get("figsize", (8.0, 6.0)),
            print_means=kwargs.get("print_means", False)
        )
    df_means = means_from_df_list(df_list, labels=names)
    return df_list, df_means, names