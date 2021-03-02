import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sbgdt import (Sbgdt, plot_factual_and_counterfactual, plot_binary_heatmap,
                   plot_shapelet_space, save_sbgdt, load_sbgdt)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from utils import usefulness_scores, baseline_error
from joblib import load, dump
import seaborn as sns
from scipy.spatial.distance import cdist
from neighborhood_generators import interpolate
from neighborhood_generators import filter_neighborhood
from variational_autoencoder import save_model, load_model
from tree_utils import minimumDistance, get_branch_length
from imblearn.under_sampling import RandomUnderSampler
from tslearn.metrics import dtw



def save_lasts(
        explainer,
        file_path="",
        skip_encoder=False,
        custom_autoencoder=None,
        custom_autoencoder_kwargs=None,
        verbose=False
):
    """Saves the lasts model
    Parameters
    ----------
    explainer : Lasts object
    file_path : str, optional (default = "")
        saved file path
        
    Returns
    -------
    None
    
    Notes
    -----
    Saving the model also deletes the sbgdt explainer
    """

    save_sbgdt(explainer.surrogate, file_path)
    explainer.surrogate = None
    if custom_autoencoder is not None:
        save_model(
            custom_autoencoder,
            custom_autoencoder_kwargs.get("input_shape"),
            custom_autoencoder_kwargs.get("latent_dim"),
            custom_autoencoder_kwargs.get("autencoder_kwargs"),
            path=file_path + "cust_aut_lasts"
        )
        explainer.encoder = None
        explainer.decoder = None
    if skip_encoder:
        explainer.encoder = None
    dump(explainer, file_path + "_lasts.pkl")


def load_lasts(file_path="", custom_autoencoder=False):
    """Loads the lasts model
    Parameters
    ----------
    file_path : str, optional (default = "")
        loaded file path
        
    Returns
    -------
    Lasts object
    """

    explainer = load(file_path + "_lasts.pkl")
    explainer.surrogate = load_sbgdt(file_path)
    if custom_autoencoder:
        explainer.encoder, explainer.decoder, _ = load_model(file_path + "cust_aut_lasts")
    return explainer


def save_reload_lasts(explainer, file_path=""):
    """Saves and reloads the lasts model
    Parameters
    ----------
    explainer : Lasts object
    file_path : str, optional (default = "")
        saved file path
        
    Returns
    -------
    Lasts object
    """
    save_lasts(explainer, file_path)
    explainer = load_lasts(file_path)
    return explainer


def save_multiple_lasts(lasts_list,  # type: list
                        file_path,
                        skip_encoder=False,
                        simple_dump=False,
                        verbose=False):
    """Saves a list of lasts models
    Parameters
    ----------
    lasts_list : list
        a list of Lasts objects
    file_path : str, optional (default = "")
        saved file path
    verbose : bool, optional (default = False)
        
    Returns
    -------
    None
    
    Notes
    -----
    Saving the models also deletes the sbgdt explainers
    """
    folder = file_path + "/"
    for i, lasts_ in enumerate(lasts_list):
        if verbose:
            print(i + 1, "/", len(lasts_list))
        if skip_encoder:
            lasts_.encoder = None
        if simple_dump:
            continue
        save_sbgdt(lasts_.surrogate, folder + "_" + str(i) + "_")
        lasts_.surrogate = None
    dump(lasts_list, file_path + "/" + "lasts_list.pkl")


def load_multiple_lasts(
        file_path,
        verbose=False,
        load_sbgdt_model=True,
        locator=True,
        model=True,
        transformer=True,
        simple_load=False,
        load_idxs=list()
):
    """Loads a list of lasts models
    Parameters
    ----------
    file_path : str, optional (default = "")
        saved file path
    verbose : bool, optional (default = False)
    locator : bool, optional (default = True)
        load the locator
    model : bool, optional (default = True)
        load the model
    transformer : bool, optional (default = True)
        load the transformer
    load_sbgdt_model : bool, optional (default = True)
        load the sbgdt model
        
    Returns
    -------
    List of Lasts objects
    
    Notes
    -----
    Loading all the models inside the sbgdt models takes a long time. Depending 
    on the specific task it could be desirable to load only some 
    """
    folder = file_path + "/"
    lasts_list = load(file_path + "/" + "lasts_list.pkl")
    if simple_load:
        return lasts_list
    if load_sbgdt_model:
        for i, lasts_ in enumerate(lasts_list):
            if len(load_idxs) > 0:
                if i not in load_idxs:
                    continue
            lasts_.surrogate = load_sbgdt(
                folder + "_" + str(i) + "_",
                locator=locator,
                model=model,
                transformer=transformer
            )
            if verbose:
                print(i + 1, "/", len(lasts_list))
    return lasts_list



def instability_lasts(lasts1, lasts2, ignore_mismatch=False, debug=False, divide_by_baseline=True):
    binarize_surrogate_labels = lasts1._binarize_surrogate_labels
    if binarize_surrogate_labels:
        pred_importances1, _ = lasts1.surrogate.predict_explanation(lasts1.z_tilde, 1)
    else:
        pred_importances1, _ = lasts1.surrogate.predict_explanation(lasts1.z_tilde, lasts1.x_label)

    if binarize_surrogate_labels:
        pred_importances2, _ = lasts2.surrogate.predict_explanation(lasts2.z_tilde, 1)
    else:
        pred_importances2, _ = lasts2.surrogate.predict_explanation(lasts2.z_tilde, lasts2.x_label)

    errors = list()
    if divide_by_baseline:
        baseline = baseline_error(pred_importances1[-1])
        if baseline == 0:
            if (pred_importances1[-1].sum() == 0) and (pred_importances2[-1].sum() == 0):
                errors.append(0)
            else:
                errors.append(1)
        else:
            errors.append(mean_absolute_error(pred_importances1[-1, :], pred_importances2[-1, :]) / baseline_error(
                pred_importances1[-1]))
    else:
        errors.append(mean_absolute_error(pred_importances1[-1, :], pred_importances2[-1, :]))

    if debug:
        print(mean_absolute_error(pred_importances1[-1, :], pred_importances2[-1, :]))

    pred_importances1 = pred_importances1[:-1]
    pred_importances2 = pred_importances2[:-1]
    if (len(pred_importances1) == 0) and (len(pred_importances2) == 0):
        errors.append(0)
    else:
        if not ignore_mismatch:
            #  doesn't ignore if there is a different number of not-contained subsequences i.e. the surplus vectors will
            #  be matched with all 0's vectors
            mismatch = len(pred_importances1) - len(pred_importances2)
            if mismatch != 0:
                if mismatch > 0:
                    if len(pred_importances2) == 0:
                        pred_importances2 = np.zeros(shape=(abs(mismatch), len(lasts1.x.ravel())))
                    else:
                        pred_importances2 = np.append(pred_importances2,
                                                      np.zeros(shape=(abs(mismatch), len(lasts1.x.ravel()))), axis=0)
                else:
                    if len(pred_importances1) == 0:
                        pred_importances1 = np.zeros(shape=(abs(mismatch), len(lasts1.x.ravel())))
                    else:
                        pred_importances1 = np.append(pred_importances1,
                                                      np.zeros(shape=(abs(mismatch), len(lasts1.x.ravel()))), axis=0)
        else:
            if ((len(pred_importances1) == 0) or (len(pred_importances2) == 0)):
                return np.mean(errors)
        distance_matrix = cdist(pred_importances1, pred_importances2, metric=mean_absolute_error)
        if debug:
            print(distance_matrix)
        while distance_matrix.shape[1] > 0:
            if debug:
                print(distance_matrix)
            to_delete = np.unravel_index(np.argmin(distance_matrix), shape=distance_matrix.shape)
            if divide_by_baseline:
                baseline = baseline_error(pred_importances1[to_delete[0]])
                if baseline == 0:
                    errors.append(1)  # FIXME: limit case
                else:
                    errors.append(distance_matrix.min() / baseline)
            else:
                errors.append(distance_matrix.min())
            distance_matrix = np.delete(distance_matrix, to_delete[0], axis=0)
            distance_matrix = np.delete(distance_matrix, to_delete[1], axis=1)
    return np.mean(errors)


def dump_metrics_multiple(lasts_list, surrogate=True):
    metrics_df = pd.DataFrame()
    for i, lasts_ in enumerate(lasts_list):
        metrics = dump_metrics(lasts_, surrogate=surrogate)
        metrics_df = metrics_df.append(pd.DataFrame(metrics, index=[i]))
    return metrics_df


def dump_metrics(lasts_, surrogate=True, explanation_error=False):
    metrics = dict()
    metrics = {**metrics, **lasts_.silhouette_scores()}
    metrics = {**metrics, **lasts_.lof_scores()}
    if surrogate:
        metrics["fidelity_neighborhood"] = lasts_.surrogate_fidelity_score()
        metrics["fidelity_x"] = lasts_.surrogate_fidelity_score_x()
        metrics = {**metrics, **lasts_.surrogate_coverage_scores()}
        metrics = {**metrics, **lasts_.surrogate_precision_scores()}
        metrics = {**metrics, **lasts_.get_rules_lengths()}
        if explanation_error:
            metrics["explanation_error"] = lasts_.explanation_error()
    return metrics


def plot_latent_space(
        Z,
        y,
        z,
        z_label,
        K=None,
        closest_counterfactual=None,
        **kwargs
):
    """Plots a 2d scatter representation of the latent space
    Parameters
    ----------
    Z : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    z : array of shape (n_features,)
        latent instance to explain
    z_label : int
        latent instance to explain label
    
    Returns
    -------
    None
    """

    z = z.ravel()
    if len(z) > 2:
        pca_2d = PCA(n_components=2)
        pca_2d.fit(Z)
        Z = pca_2d.transform(Z)
        z = pca_2d.transform(z.reshape(1, -1))
        z = z.ravel()
        if K is not None:
            K = pca_2d.transform(K)
        if closest_counterfactual is not None:
            closest_counterfactual = pca_2d.transform(closest_counterfactual)

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)),
                           dpi=kwargs.get("dpi", 72))
    ax.set_title(r"$Z$", fontsize=kwargs.get("fontsize", 12))

    # plots generated neighborhood points
    exemplars = np.argwhere(y == z_label)
    counterexemplars = np.argwhere(y != z_label)

    if K is not None:
        ax.scatter(
            K[:, 0],
            K[:, 1],
            c="gray",
            alpha=0  #alpha=0.2,
            #label=r"$\mathcal{N}$"
        )

    ax.scatter(
        Z[:, 0][exemplars],
        Z[:, 1][exemplars],
        c="#2ca02c",
        alpha=0.5,
        label=r"$Z_=$"
    )
    ax.scatter(
        Z[:, 0][counterexemplars],
        Z[:, 1][counterexemplars],
        c="#d62728",
        alpha=0.5,
        label=r"$Z_\neq$"
    )


    if closest_counterfactual is not None:
        ax.scatter(
            closest_counterfactual[:, 0],
            closest_counterfactual[:, 1],
            c="gray",
            alpha=0.9,
            label="closest counterfactual",
            marker="+",
            s=200
        )

    # marks the instance to explain with an X
    ax.scatter(
        z[0],
        z[1],
        label=r"z",
        c="royalblue",
        marker="X",
        edgecolors="white",
        s=200)

    if kwargs.get("legend"):
        ax.legend(fontsize=kwargs.get("fontsize", 12) - 2, frameon=True, loc="lower left")
    loc = matplotlib.ticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    plt.show()


def plot_latent_space_matrix_old(
        Z,
        y,
        z,
        z_label,
        K=None,
        **kwargs
):
    """Plots a scatter matrix of the latent space
    Parameters
    ----------
    Z : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    z : array of shape (n_features,)
        latent instance to explain
    z_label : int
        latent instance to explain label
    
    Returns
    -------
    None
    """

    cmap = matplotlib.colors.ListedColormap(["#d62728", '#2ca02c'])
    Z = list(Z)
    Z = pd.DataFrame(Z)
    colors = list(y == z_label)
    pd.plotting.scatter_matrix(Z,
                               c=colors,
                               cmap=cmap,
                               diagonal="kde",
                               alpha=0.5,
                               figsize=kwargs.get("figsize", (8, 8)))


def plot_latent_space_matrix(
        Z,
        y,
        z,
        z_label,
        K=None,
        **kwargs
):
    """Plots a scatter matrix of the latent space
    Parameters
    ----------
    Z : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    z : array of shape (n_features,)
        latent instance to explain
    z_label : int
        latent instance to explain label

    Returns
    -------
    None
    """

    y = 1 * (y == z_label)

    Z = np.concatenate([Z, z])
    y = np.concatenate([y, np.repeat(2, z.shape[0])])

    if K is not None:
        Z = np.concatenate([Z, K])
        y = np.concatenate([y, np.repeat(3, K.shape[0])])
        markers = [".", ".", "X", "."]
    else:
        markers = [".", ".", "X"]

    Z = list(Z)
    Z = pd.DataFrame(Z)
    Z["y"] = y

    g = sns.pairplot(
        Z,
        hue="y",
        markers=markers,
        palette={0: "#d62728", 1: "#2ca02c", 2: "royalblue", 3: "gray"},
        aspect=1,
        corner=True,
        height=4,
        plot_kws=dict(s=200)
    )
    g._legend.set_title("")
    if K is not None:
        new_labels = [r"$Z_\neq$", r"$Z_=$", r"$z$", r"$K$"]
    else:
        new_labels = [r"$Z_\neq$", r"$Z_=$", r"$z$"]
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)


def morphing_matrix(
        blackbox,
        decoder,
        x_label,
        labels=None,
        n=7,
        **kwargs
):
    """Plots a 2d matrix of instances sampled from a normal distribution
    only meaningful with a 2d normal latent space (es. with VAE, AAE)
    Parameters
    ----------
    blackbox : BlackboxWrapper object
        a wrapped blackbox
    decoder : object
        a trained decoder
    x_label : int
        instance to explain label
    labels : list of shape (n_classes,), optional (default = None)
        list of classes labels
    n : int, optional (default = 7)
        number of instances per latent dimension
    Returns
    -------
    self
    """

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))[::-1]
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    fig, axs = plt.subplots(
        n,
        n,
        figsize=kwargs.get("figsize", (10, 5)),
        dpi=kwargs.get("dpi", 72)
    )
    fig.suptitle("Classes Morphing", fontsize=kwargs.get("fontsize", 12))
    fig.patch.set_visible(False)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sampled = np.array([[xi, yi]])
            z_sampled_tilde = decoder.predict(z_sampled).ravel()
            z_sampled_label = blackbox.predict(z_sampled_tilde.reshape(1, -1, 1))[0]
            color = "#2ca02c" if z_sampled_label == x_label else "#d62728"
            if z_sampled_label == x_label:
                label = (r"$b(\tilde{z}) = $" + labels[z_sampled_label] if labels
                         else r"$b(\tilde{z}) = $" + str(z_sampled_label))
            else:
                label = (r"$b(\tilde{z}) \neq $" + labels[x_label] if labels
                         else r"$b(\tilde{z}) \neq $" + str(x_label))
            axs[i, j].plot(z_sampled_tilde, color=color, label=label)
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            axs[i, j].axis('off')

    d = dict()
    for a in fig.get_axes():
        if a.get_legend_handles_labels()[1][0] not in d:
            d[a.get_legend_handles_labels()[1][0]] = a.get_legend_handles_labels()[0][0]

    labels, handles = zip(*sorted(zip(d.keys(), d.values()), key=lambda t: t[0]))
    plt.legend(handles, labels, fontsize=kwargs.get("fontsize", 12))
    plt.show()


def plot_interpolation(z, z_prime, x_label, decoder, blackbox, kind="linear", n=100, **kwargs):
    interpolation_matrix = interpolate(z, z_prime, kind, n)
    decoded_interpolation_matrix = decoder.predict(interpolation_matrix)
    z_tilde = decoder.predict(z)
    z_prime_tilde = decoder.predict(z_prime)
    y = blackbox.predict(decoded_interpolation_matrix)

    exemplars_idxs = np.argwhere(y == x_label).ravel()
    counterexemplars_idxs = np.argwhere(y != x_label).ravel()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    plt.title("Interpolation", fontsize=kwargs.get("fontsize", 12))
    plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
    plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    plt.plot(decoded_interpolation_matrix[:, :, 0][exemplars_idxs].T, c="#2ca02c", alpha=kwargs.get("alpha", 0.1))
    if counterexemplars_idxs.shape[0] != 0:
        plt.plot(decoded_interpolation_matrix[:, :, 0][counterexemplars_idxs].T, c="#d62728",
                 alpha=kwargs.get("alpha", 0.1))
    plt.plot(z_tilde.ravel(), c="green", linestyle='-', lw=3, alpha=0.9, label=r"$\tilde{z}$")
    plt.plot(z_prime_tilde.ravel(), c="red", linestyle='-', lw=3, alpha=0.9, label=r"$\tilde{z}'$")
    plt.legend()
    plt.show()


def plot_exemplars_and_counterexemplars(
        Z_tilde,
        y,
        x,
        z_tilde,
        x_label,
        labels=None,
        plot_x=True,
        plot_z_tilde=True,
        legend=False,
        no_axes_labels=False,
        **kwargs
):
    """Plots x, z_tilde; exemplars; counterexemplars
    Parameters
    ----------
    Z_tilde : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    x : array of shape (n_features,)
        instance to explain
    z_tilde : array of shape (n_features,)
        autoencoded instance to explain
    x_label : int
        instance to explain label
    labels : list of shape (n_classes,), optional (default = None)
        list of classes labels
    
    Returns
    -------
    self
    """
    exemplars_idxs = np.argwhere(y == x_label).ravel()
    counterexemplars_idxs = np.argwhere(y != x_label).ravel()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Instance to explain: " + r"$b(x)$" + " = " + labels[x_label] if labels
    #           else "Instance to explain: " + r"$b(x)$" + " = " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))
    plt.title(r"$b(x)$" + " = " + labels[x_label] if labels else r"$b(x)$" + " = " + str(x_label),
              fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    if plot_x:
        plt.plot(x.ravel(), c="royalblue", linestyle='-', lw=3, alpha=0.9, label=r"$x$")
    if plot_z_tilde:
        plt.plot(z_tilde.ravel(), c="orange", linestyle='-', lw=3, alpha=0.9, label=r"$\tilde{z}$")
    if legend:
        plt.legend()
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Exemplars: " + r"$b(\tilde{Z}_{=})$" + " = " + labels[x_label] if labels
    #           else "Exemplars: " + r"$b(\tilde{Z}_{=})$" + " = " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))

    plt.title("LASTS - " + r"$b(\tilde{Z}_{=})$" + " = " + labels[x_label] if labels
              else r"$b(\tilde{Z}_{=})$" + " = " + str(x_label),
              fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    plt.plot(Z_tilde[:, :, 0][exemplars_idxs].T, c="#2ca02c", alpha=kwargs.get("alpha", 0.1))
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Counterexemplars: " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label] if labels
    #           else "Counterexemplars: " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))
    plt.title("LASTS - " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label] if labels
              else r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
              fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    plt.plot(Z_tilde[:, :, 0][counterexemplars_idxs].T, c="#d62728", alpha=kwargs.get("alpha", 0.1))
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    plt.title("Neighborhood: " + r"$\tilde{Z}$", fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    plt.plot(Z_tilde[:, :, 0][exemplars_idxs].T, c="#2ca02c", alpha=kwargs.get("alpha", 0.1))
    plt.plot(Z_tilde[:, :, 0][counterexemplars_idxs].T, c="#d62728", alpha=kwargs.get("alpha", 0.1))
    plt.plot(z_tilde.ravel(), c="royalblue", linestyle='-', lw=3, alpha=0.9)
    plt.show()


class Lasts(object):
    def __init__(self,
                 blackbox,
                 encoder,
                 decoder,
                 x,
                 neighborhood_generator,
                 labels=None,
                 z_fixed=None
                 ):
        """Local Agnostic Shapelet-based Time Series Explainer
        Parameters
        ----------
        blackbox : a wrapped classifier in BlackboxWrapper
        encoder : a trained encoder with input of shape [n_samples, n_features, 1]
        decoder : a trained decoder with input of shape [n_samples, n_features]
        x : array of shape [1, n_features, 1]
            instance to explain 
        neighborhood_generator : a fitted neighborhood generator
        labels : array of shape [n_classes,] (default = None)
            classes string labels
        
        Attributes
        ----------
        z : array of shape [1, n_latent_features]
            encoded instance to explain
        z_tilde : array of shape [1, n_features, 1]
            autoencoded instance to explain
        z_tilde_label : int
            autoencoded instance to explain label
        x_tilde_label : int
            instance to explain label
            
        surrogate : Sbgdt object
            scikit-style decision tree classifier based on shapelets
        _binarize_surrogate_labels : bool
            sbgdt binarized labels or not 
        """

        self.blackbox = blackbox
        self.encoder = encoder
        self.decoder = decoder
        self.x = x
        self.x_label = self.blackbox.predict(x)[0]
        if z_fixed is None:
            self.z = self.encoder.predict(x)
        else:
            self.z = z_fixed
        self.z_tilde = self.decoder.predict(self.z)
        self.z_tilde_label = self.blackbox.predict(self.z_tilde)[0]
        if self.x_label != self.z_tilde_label:
            warnings.warn("The x label before the autoencoding is " +
                          str(self.x_label) +
                          " but the label after the autoencoding is " +
                          str(self.z_tilde_label))
        self.neighborhood_generator = neighborhood_generator
        self.labels = labels

        self.surrogate = None
        self._binarize_surrogate_labels = None
        self.Z = None
        self.Z_tilde = None
        self.y = None

        self._neighborhood_backup = None

    def _decode_predict(self, X):
        """Decode and predict the class labels for the provided latent data.
        Parameters
        ----------
        X : array-like
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
        X : array-like
            Latent data. Shape [n_samples, n_features].
        Returns
        -------
        y : array of shape [n_samples, n_classes]
        """
        X_decoded = self.decoder.predict(X)
        y = self.blackbox.predict_proba(X_decoded)
        return y

    def generate_neighborhood(self, **generator_params):
        """Generates a neighborhood around the instance to explain
        Parameters
        ----------
        verbose
        generator_params : dict, optional
            parameters to pass to the neighborhood generator
        Returns
        -------
        dict with keys "Z", "Z_tilde" and "y"
        """
        Z = self.neighborhood_generator.generate_neighborhood(
            self.z,
            **generator_params
        )
        self.Z = Z
        self.Z_tilde = self.decoder.predict(Z)
        self.y = self.blackbox.predict(self.Z_tilde)

        if generator_params.get("verbose") == True:
            self.print_Z_balance()

        return {"Z": self.Z, "Z_tilde": self.Z_tilde, "y": self.y}

    def filter_neighborhood(
            self,
            ratio=0.5,
            balance=False,
            ignore_instance_after_first_match=True,
            inverse=False,
            verbose=True):
        if self._neighborhood_backup is None:
            self._neighborhood_backup = {"Z": self.Z.copy(), "Z_tilde": self.Z_tilde.copy(), "y": self.y.copy()}
        else:
            raise Exception("For now you can filter the neighborhood one time only.")
        self.Z = filter_neighborhood(
            self.Z,
            1 * (self.y == self.x_label),
            ratio=ratio,
            ignore_instance_after_first_match=ignore_instance_after_first_match,
            inverse=inverse
        )
        self.Z_tilde = self.decoder.predict(self.Z)
        self.y = self.blackbox.predict(self.Z_tilde)

        if balance:
            rus = RandomUnderSampler(random_state=0)
            self.Z, self.y = rus.fit_resample(self.Z, self.y)
            self.Z_tilde = self.decoder.predict(self.Z)

        if verbose:
            self.print_Z_balance()

        return {"Z": self.Z, "Z_tilde": self.Z_tilde, "y": self.y}

    def restore_neighborhood_backup(self, verbose=True):
        if self._neighborhood_backup is None:
            raise Exception("No backup available.")
        self.Z = self._neighborhood_backup["Z"]
        self.Z_tilde = self._neighborhood_backup["Z_tilde"]
        self.y = self._neighborhood_backup["y"]
        self._neighborhood_backup = None
        if verbose:
            self.print_Z_balance()
        return {"Z": self.Z, "Z_tilde": self.Z_tilde, "y": self.y}

    def print_Z_balance(self):
        unique, counts = np.unique(self.y, return_counts=True)
        print("\nLABELS BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

        unique, counts = np.unique(1 * (self.y == self.x_label), return_counts=True)
        binary_labels = ["not-" + str(self.x_label), str(self.x_label)]
        print("\nBINARY BALANCE")
        for i, label in enumerate(unique):
            print(binary_labels[label], ": ", round(counts[i] / sum(counts), 2))

    def plot_latent_space(self, scatter_matrix=False, K=None, **kwargs):
        """Plots a 2d scatter representation of the latent space
        Parameters
        ----------
        scatter_matrix : bool, optional (default = False)
            plot the scatter matrix also
        K: array, optional (default = None)
            plot the given dataset K
        Returns
        -------
        self
        """
        plot_latent_space(
            self.Z,
            self.y,
            self.z,
            self.x_label,
            K=K,
            **kwargs
        )
        if scatter_matrix:
            plot_latent_space_matrix(
                self.Z,
                self.y,
                self.z,
                self.x_label,
                K=K,
                **kwargs
            )
        return self

    def morphing_matrix(self, n=7, **kwargs):
        """Plots a 2d matrix of instances sampled from a normal distribution
        only meaningful with a 2d normal latent space (es. with VAE, AAE)
        Parameters
        ----------
        n : int, optional (default = 7)
            number of instances per latent dimension
        Returns
        -------
        self
        """
        morphing_matrix(
            self.blackbox,
            self.decoder,
            self.x_label,
            self.labels,
            n=n,
            **kwargs
        )
        return self

    def plot_exemplars_and_counterexemplars(self, **kwargs):
        """Plots x, z_tilde; exemplars; counterexemplars
        Returns
        -------
        self
        """
        plot_exemplars_and_counterexemplars(
            self.Z_tilde,
            self.y,
            self.x,
            self.z_tilde,
            self.x_label,
            self.labels,
            **kwargs
        )
        return self

    def fit_surrogate(
            self,
            surrogate,
            binarize_labels=False,
            **kwargs):
        """Builds the shapelet-based decision tree classifier
        Parameters
        ----------
        binarize_labels : bool, optional (default = False)
            train the sbgdt on exemplars (label = 1) and counterexemplars (label = 0)
        
        Returns
        -------
        sbgdt : a shapelet-based decision tree classifier
        """
        self._binarize_surrogate_labels = binarize_labels
        self.surrogate = surrogate
        if not binarize_labels:
            self.surrogate.labels = self.labels
            # self.surrogate.fit(self.Z_tilde[:, :, 0], self.y)
            self.surrogate.fit(self.Z_tilde, self.y)
        else:
            self.surrogate.labels = [
                "not " + str(self.x_label) if self.labels is None
                else "not " + self.labels[self.x_label],
                str(self.x_label) if self.labels is None
                else self.labels[self.x_label]
            ]
            # self.surrogate.fit(self.Z_tilde[:, :, 0], 1 * (self.y == self.x_label))
            self.surrogate.fit(self.Z_tilde, 1 * (self.y == self.x_label))
        return self.surrogate

    def plot_factual_and_counterfactual(self, **kwargs):
        """Plots the verbose and graphical factual and counterfactual rules
        Returns
        -------
        self
        """
        if not self._binarize_surrogate_labels:
            self.surrogate.plot_factual_and_counterfactual(
                # self.z_tilde[:, :, 0],
                self.z_tilde,
                self.x_label,
                draw_on=self.x[:, :, 0],
                **kwargs
            )
        else:
            self.surrogate.plot_factual_and_counterfactual(
                # self.z_tilde[:, :, 0],
                self.z_tilde,
                1,
                draw_on=self.x[:, :, 0],
                **kwargs
            )

    def plot_regression_coefs(self, **kwargs):
        """Plots the verbose and graphical factual and counterfactual rules
        Returns
        -------
        self
        """
        if not self._binarize_surrogate_labels:
            self.surrogate.plot_factual_and_counterfactual(
                # self.z_tilde[:, :, 0],
                self.z_tilde,
                self.x_label,
                draw_on=self.x[:, :, 0],
                **kwargs
            )
        else:
            self.surrogate.plot_factual_and_counterfactual(
                # self.z_tilde[:, :, 0],
                self.z_tilde,
                1,
                draw_on=self.x[:, :, 0],
                **kwargs
            )


        return self

    def plot_binary_heatmap(self, **kwargs):
        """Plots a heatmap of the contained and not contained shapelet
        Returns
        -------
        self
        """
        if not self._binarize_surrogate_labels:
            self.surrogate.plot_binary_heatmap(self.x_label, **kwargs)
        else:
            self.surrogate.plot_binary_heatmap(1, **kwargs)
        return self

    def plot_shapelet_space(self, binarize_labels=False, **kwargs):
        """Plots a 2d representation of the shapelet space
        Returns
        -------
        self
        """
        if not self._binarize_surrogate_labels:
            plot_shapelet_space(self.surrogate, self.z_tilde, self.x_label, **kwargs)
        else:
            plot_shapelet_space(self.surrogate, self.z_tilde, 1, **kwargs)
        return self

    def plot_closest_counterfactual_interpolation(self, kind="linear", n=100, **kwargs):
        plot_interpolation(
            self.z,
            self.neighborhood_generator.closest_counterfactual,
            self.x_label,
            self.decoder,
            self.blackbox,
            kind,
            n,
            **kwargs
        )
        return self

    def usefulness_scores(self, n=[1, 2, 4, 8, 16]):
        if self._binarize_surrogate_labels:
            y_by_n = usefulness_scores(
                self.Z_tilde[:, :, 0],
                1 * (self.y == self.x_label),
                self.x,  # FIXME: or z_tilde?
                1,
                n=n
            )
        else:
            y_by_n = usefulness_scores(
                self.Z_tilde[:, :, 0],
                self.y,
                self.x,  # FIXME: or z_tilde?
                self.x_label,
                n=n
            )
        return y_by_n

    def surrogate_fidelity_score(self):
        if self._binarize_surrogate_labels:
            # return self.surrogate.score(self.Z_tilde[:, :, 0], 1 * (self.y == self.x_label))
            return self.surrogate.score(self.Z_tilde, 1 * (self.y == self.x_label))
        else:
            # return self.surrogate.score(self.Z_tilde[:, :, 0], self.y)
            return self.surrogate.score(self.Z_tilde, self.y)

    def surrogate_fidelity_score_x(self):
        if self._binarize_surrogate_labels:
            # return self.surrogate.score(self.z_tilde[:, :, 0], np.array([1]))
            return self.surrogate.score(self.z_tilde, np.array([1]))
        else:
            # return self.surrogate.score(self.z_tilde[:, :, 0], np.array([self.x_label]))
            return self.surrogate.score(self.z_tilde, np.array([self.x_label]))

    def surrogate_coverage_scores(self):
        factual_leaf = self.surrogate.find_leaf_id(self.z_tilde)
        counterfactual_leaf = minimumDistance(
            self.surrogate.decision_tree_explorable.nodes[0],
            self.surrogate.decision_tree_explorable.nodes[factual_leaf]
        )[1]
        factual_coverage = self.surrogate.coverage_score(factual_leaf)
        counterfactual_coverage = self.surrogate.coverage_score(counterfactual_leaf)
        return {"factual_coverage": factual_coverage, "counterfactual_coverage": counterfactual_coverage}

    def surrogate_precision_scores(self):
        factual_leaf = self.surrogate.find_leaf_id(self.z_tilde)
        counterfactual_leaf = minimumDistance(
            self.surrogate.decision_tree_explorable.nodes[0],
            self.surrogate.decision_tree_explorable.nodes[factual_leaf]
        )[1]
        if self._binarize_surrogate_labels:
            y = 1 * (self.y == self.x_label)
        else:
            y = self.y
        factual_precision = self.surrogate.precision_score(factual_leaf, y, self.Z_tilde)
        counterfactual_precision = self.surrogate.precision_score(counterfactual_leaf, y, self.Z_tilde)
        return {"factual_precision": factual_precision, "counterfactual_precision": counterfactual_precision}

    def get_rules_lengths(self):
        factual_leaf = self.surrogate.find_leaf_id(self.z_tilde)
        counterfactual_leaf = minimumDistance(
            self.surrogate.decision_tree_explorable.nodes[0],
            self.surrogate.decision_tree_explorable.nodes[factual_leaf]
        )[1]
        factual_node = self.surrogate.decision_tree_explorable.nodes[factual_leaf]
        factual_length = get_branch_length(factual_node)

        if counterfactual_leaf is None:
            warnings.warn("counterfactual_leaf is None")
            counterfactual_length = 0
        else:
            counterfactual_node = self.surrogate.decision_tree_explorable.nodes[counterfactual_leaf]
            counterfactual_length = get_branch_length(counterfactual_node)
        return {"factual_length": factual_length, "counterfactual_length": counterfactual_length}

    def silhouette_scores(self, binarize_labels=True, manifest_metric="euclidean"):
        if binarize_labels:
            y = 1 * (self.y == self.x_label)

        silhouette_latent = silhouette_score(self.Z, y)

        if manifest_metric == "dtw":
            metric = dtw
        else:
            metric = manifest_metric
        silhouette_manifest = silhouette_score(self.Z_tilde[:, :, 0], y, metric=metric)
        return {"silhouette_latent": silhouette_latent, "silhouette_manifest_" + manifest_metric: silhouette_manifest}

    def lof_scores(self, manifest_metric="euclidean", aggregation="average"):
        if manifest_metric == "dtw":
            metric = dtw
        else:
            metric = manifest_metric

        lof_clf_latent = LocalOutlierFactor(metric="euclidean", novelty=True)
        lof_clf_latent.fit(self.Z)
        lof_scores_latent = lof_clf_latent.predict(self.Z)
        lof_score_latent_x = lof_clf_latent.predict(self.z)[0]

        lof_clf_manifest = LocalOutlierFactor(metric=metric, novelty=True)
        lof_clf_manifest.fit(self.Z_tilde[:, :, 0])
        lof_scores_manifest = lof_clf_manifest.predict(self.Z_tilde[:, :, 0])
        lof_score_manifest_x = lof_clf_manifest.predict(self.z_tilde[:, :, 0])[0]

        if aggregation == "average":
            lof_score_latent = lof_scores_latent.mean()
            lof_score_manifest = lof_scores_manifest.mean()
        else:
            raise Exception("Aggregation method not valid.")

        return {"lof_latent_" + aggregation: lof_score_latent,
                "lof_latent_x": lof_score_latent_x,
                "lof_manifest_" + manifest_metric + "_" + aggregation: lof_score_manifest,
                "lof_manifest_x_" + manifest_metric: lof_score_manifest_x}

    def explanation_error(self, skip_missing_contained=True, debug=False, divide_by_baseline=True, **kwargs):
        if self._binarize_surrogate_labels:
            pred_importances, tss = self.surrogate.predict_explanation(self.z_tilde, 1)
        else:
            pred_importances, tss = self.surrogate.predict_explanation(self.z_tilde, self.x_label)
        true_importances = list()
        for i in range(pred_importances.shape[0]):
            true_importances.append(self.blackbox.predict_explanation(tss[i, :].reshape(1, -1, 1)))
        true_importances = np.array(true_importances)

        contained_error = list()
        if (pred_importances[-1, :].sum() != 0) or (not skip_missing_contained):
            if divide_by_baseline:
                baseline = baseline_error(true_importances[-1, :])
                error = mean_absolute_error(true_importances[-1, :], pred_importances[-1, :]) / baseline
                contained_error.append(error)
            else:
                error = mean_absolute_error(true_importances[-1, :], pred_importances[-1, :])
                contained_error.append(error)
        contained_error = np.array(contained_error)

        pred_importances = pred_importances[:-1]
        true_importances = true_importances[:-1]
        tss = tss[:-1]
        if len(pred_importances) == 0:
            return contained_error.mean()

        notcontained_errors = list()
        if debug:
            for i in range(len(tss)):
                plt.plot(tss[i, :].T)
                plt.show()
            print("divider", np.prod(true_importances.shape))
            if pred_importances[-1, :].sum() == 0:
                print("contained-importance",
                      np.abs(true_importances[-1, :] - pred_importances[-1, :]).sum() / true_importances.shape[1])
            print("relative-importance",
                  np.abs(true_importances - pred_importances).sum(axis=1) / np.prod(true_importances.shape))
            print("importance", np.abs(true_importances - pred_importances).sum() / np.prod(true_importances.shape))
            return pred_importances, tss, true_importances

        for pred_importance, true_importance in zip(pred_importances, true_importances):
            if divide_by_baseline:
                baseline = baseline_error(true_importance)
                error = mean_absolute_error(true_importance, pred_importance) / baseline
                notcontained_errors.append(error)
            else:
                error = mean_absolute_error(true_importance, pred_importance)
                notcontained_errors.append(error)
        notcontained_errors = np.array(notcontained_errors)
        if len(contained_error) == 0:
            return notcontained_errors.mean()
        else:
            return (contained_error.mean() + notcontained_errors.mean()) / 2

if __name__ == '__main__':
    from blackbox_wrapper import BlackboxWrapper
    from datasets import build_cbf
    from variational_autoencoder import load_model
    from neighborhood_generators import NeighborhoodGenerator
    from utils import choose_z
    from saxdt import Saxdt

    random_state = 0
    np.random.seed(random_state)
    dataset_name = "cbf"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600,
                                                               random_state=random_state)
    
    knn = load("./trained_models/cbf/cbf_knn.joblib")
    blackbox = BlackboxWrapper(knn, 2, 1)

    _, _, autoencoder = load_model("./trained_models/cbf/cbf_vae")

    encoder = autoencoder.layers[2]
    decoder = autoencoder.layers[3]
    i = 2
    x = X_exp_test[i].ravel().reshape(1, -1, 1)
    z = choose_z(x, encoder, decoder, n=1000, x_label=blackbox.predict(x)[0],
                 blackbox=blackbox, check_label=True, mse=False)
    z_label = blackbox.predict(decoder.predict(z))[0]
    K = encoder.predict(X_exp_train)

    neighborhood_generator = NeighborhoodGenerator(blackbox, decoder)
    neigh_kwargs = {
        "balance": False,
        "n": 500,
        "n_search": 10000,
        "threshold": 2,
        "sampling_kind": "uniform_sphere",
        "kind": "gaussian_matched",
        "vicinity_sampler_kwargs": {"distribution": np.random.normal, "distribution_kwargs": dict()},
        "verbose": True,
        "stopping_ratio": 0.01,
        "downward_only": True,
        "redo_search": True,
        "forced_balance_ratio": 0.5,
        "cut_radius": True
    }

    lasts_ = Lasts(blackbox,
                   encoder,
                   decoder,
                   x,
                   neighborhood_generator,
                   z_fixed=z,
                   labels=["cylinder", "bell", "funnel"]
                   )

    out = lasts_.generate_neighborhood(**neigh_kwargs)

    # VARIOUS PLOTS
    lasts_.plot_exemplars_and_counterexemplars()
    lasts_.plot_latent_space(scatter_matrix=True)
    lasts_.plot_latent_space(scatter_matrix=False, K=K)

    # surrogate = Sbgdt(shapelet_model_params={"max_iter": 50}, random_state=random_state)
    surrogate = Saxdt(random_state=np.random.seed(0))

    # SHAPELET EXPLAINER
    lasts_.fit_surrogate(surrogate, binarize_labels=True)
    # VARIOUS PLOTS
    lasts_.plot_shapelet_space()
    lasts_.plot_binary_heatmap(step=100)
    lasts_.plot_factual_and_counterfactual()