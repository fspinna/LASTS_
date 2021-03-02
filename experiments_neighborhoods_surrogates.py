from utils import choose_z
from neighborhood_generators import NeighborhoodGenerator
from lasts import Lasts, save_multiple_lasts, dump_metrics
import numpy as np
import pandas as pd
import os
from sbgdt import save_sbgdt, Sbgdt
from saxdt import Saxdt
from joblib import dump
from utils import reconstruction_accuracy_vae, boxplots_from_df_list
from lore_wrapper import LoreWrapper
import itertools


def multiple_tests(
        X,
        blackbox,
        encoder,
        decoder,
        neighborhood_list,
        neighborhood_kwargs_list,
        neighborhood_names,
        surrogate_list,
        surrogate_kwargs_list,
        surrogate_names,
        file_path,
        random_state=0,
        n_instances_to_explain=None,
        verbose=True,
        simple_dump=None,
        labels=None,
        boxplots=True,
        Z_fixed=None,
        custom_lasts_lists=None,
        save=True
):
    if verbose:
        print("--- MULTIPLE LASTS TESTS ---\n")

    np.random.seed(random_state)

    if n_instances_to_explain is None:
        n_instances_to_explain = X.shape[0]

    os.makedirs(file_path + "/")
    if save:
        K = encoder.predict(X)
        dump(K, file_path + "/K.joblib")
    if verbose:
        print(" --> RECONSTRUCTION ACCURACY:",
              reconstruction_accuracy_vae(X, encoder, decoder, blackbox, check_label=True, verbose=False))
    y = blackbox.predict(X)


    if Z_fixed is None:
        Z_fixed = list()
        for i in range(n_instances_to_explain):
            x_label = y[i]
            z = choose_z(X[i][np.newaxis, :, :], encoder, decoder, 1000, x_label, blackbox, check_label=True)
            Z_fixed.append(z)
        Z_fixed = np.array(Z_fixed)
    if save:
        dump(Z_fixed, file_path + "/Z_fixed.joblib")

    if custom_lasts_lists is None:
        lasts_lists = list()
        if verbose:
            print("\n\n--- NEIGHBORHOOD GENERATIONS ---")
        for j in range(len(neighborhood_kwargs_list)):
            if verbose:
                print("\n -->", neighborhood_names[j].upper())
            lasts_list = list()
            for i in range(n_instances_to_explain):
                if verbose:
                    print("  NEIGHGEN:", j + 1, "/", len(neighborhood_kwargs_list),
                          "|", "INSTANCE:", i + 1, "/", n_instances_to_explain)
                neighborhood_generator = neighborhood_list[j](blackbox, decoder)  # NeighborhoodGenerator(blackbox, decoder)


                lasts_ = Lasts(
                    blackbox,
                    encoder,
                    decoder,
                    X[i][np.newaxis, :, :],
                    neighborhood_generator,
                    z_fixed=Z_fixed[i],
                    labels=labels
                )
                if isinstance(neighborhood_generator, LoreWrapper):
                    neighborhood_kwargs_list[j]["K"] = np.delete(X, i, axis=0)
                    neighborhood_kwargs_list[j]["K_encoded"] = np.delete(
                        Z_fixed.reshape((Z_fixed.shape[0], Z_fixed.shape[-1])), i, axis=0)
                lasts_.generate_neighborhood(**neighborhood_kwargs_list[j])


                lasts_list.append(lasts_)
            lasts_lists.append(lasts_list)
        if save:
            dump(lasts_lists, file_path + "/complete_neigh_backup.joblib")
    else:
        lasts_lists = custom_lasts_lists

    metrics_dict = dict()
    if verbose:
        print("\n\n--- SURROGATE FITTING ---")
    for i, lasts_list in enumerate(lasts_lists):
        new_metrics_dict = multiple_surrogate_fitting(
            lasts_list=lasts_list,
            neighborhood_name=neighborhood_names[i],
            surrogate_list=surrogate_list,
            surrogate_kwargs_list=surrogate_kwargs_list,
            surrogate_names=surrogate_names,
            file_path=file_path,
            verbose=verbose,
            simple_dump=simple_dump,
            save=save
        )
        metrics_dict = {**metrics_dict, **new_metrics_dict}

    if boxplots:
        if verbose:
            print("\n--- METRICS ---")
        boxplots_from_df_list(list(metrics_dict.values()), labels=list(metrics_dict.keys()))

    return metrics_dict


def multiple_surrogate_fitting(
        lasts_list,
        neighborhood_name,
        surrogate_list,
        surrogate_kwargs_list,
        surrogate_names,
        file_path,
        verbose=True,
        simple_dump=None,
        save=True,
        **kwargs
):
    combination_names = [neighborhood_name + "_" + surrogate_name for surrogate_name in surrogate_names]
    for combination_name in combination_names:
        os.makedirs(file_path + "/" + combination_name + "/")
    metrics_dict = dict()
    for combination_name in combination_names:
        metrics_dict[combination_name] = pd.DataFrame()
    for j, surrogate_ in enumerate(surrogate_list):
        if verbose:
            print("\n -->", combination_names[j].upper())
        for i, lasts_ in enumerate(lasts_list):
            if verbose:
                print("  INSTANCE:", i + 1, "/", len(lasts_list))
            surrogate = surrogate_(**surrogate_kwargs_list[j])

            lasts_.fit_surrogate(surrogate, binarize_labels=True)

            metrics = dump_metrics(lasts_)
            metrics_dict[combination_names[j]] = pd.concat([metrics_dict[combination_names[j]],
                                                            pd.DataFrame(metrics, index=[i])])

        if save:
            save_multiple_lasts(
                lasts_list,
                file_path + "/" + combination_names[j] + "/",
                skip_encoder=True,
                simple_dump=simple_dump[j] if simple_dump is not None else True
            )
    for metric_df in metrics_dict.keys():
        metrics_dict[metric_df].to_csv(file_path + "/" + metric_df + ".csv", sep=";", index=False)
    return metrics_dict
