import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from utils import reconstruction_accuracy_vae, usefulness_scores_real, usefulness_scores_lasts
from sklearn.metrics import f1_score, precision_score
from saxdt import Saxdt
from lasts import load_multiple_lasts, instability_lasts
from sbgdt import Sbgdt, generate_n_shapelets_per_size
from collections import defaultdict
from shap_explainer import ShapTimeSeries
from sklearn.neighbors import NearestNeighbors
from lasts import instability_lasts
from shap_explainer import instability_shap
from datasets import build_cbf, build_ecg200, build_gunpoint, build_coffee, build_synth
from joblib import load, dump
import keras
from variational_autoencoder import load_model
from blackbox_wrapper import BlackboxWrapper
from ts_generator import PatternClassifier, TimeSeriesGenerator, LocalPattern, PolynomialClassifier, GlobalFeature


def test_accuracy(clf_list, clf_names, X, y):
    scores = {"accuracy": list()}
    for clf in clf_list:
        scores["accuracy"].append(clf.score(X, y))
    scores = pd.DataFrame(scores, index=clf_names)
    return scores


def test_reconstruction_accuracy(clf_list, clf_names, encoder, decoder, X):
    scores = {"reconstruction_accuracy": list()}
    for clf in clf_list:
        scores["reconstruction_accuracy"].append(reconstruction_accuracy_vae(
            X, encoder, decoder, clf, verbose=False, check_label=True
        ))
    scores = pd.DataFrame(scores, index=clf_names)
    return scores


def test_fidelity(clf_list, clf_names, X_train, X_test, shapelet_list):
    scores = {"fidelity_sax": list(), "fidelity_stdt": list()}
    for i, clf in enumerate(clf_list):
        saxdt_ = Saxdt(**{"random_state": np.random.seed(0), "create_plotting_dictionaries": False})
        saxdt_.fit(X_train, clf.predict(X_train))
        scores["fidelity_sax"].append(saxdt_.score(X_test, clf.predict(X_test)))

        stdt_ = Sbgdt(**{"shapelet_model_params": {"max_iter": 50, "n_shapelets_per_size": shapelet_list[i]},
                       "random_state": 0})
        stdt_.fit(X_train, clf.predict(X_train))
        scores["fidelity_stdt"].append(stdt_.score(X_test, clf.predict(X_test)))
    scores = pd.DataFrame(scores, index=clf_names)
    return scores


def instability_multiple(lasts_list, combination_name="", latent=False, divide_by_baseline=True,
                         shap_kwargs=dict(), lasts_kwargs=dict(), verbose=False, compute_lasts=True, compute_shap=True,
                         **kwargs):
    X = list()
    Z = list()
    for lasts_ in lasts_list:
        X.append(lasts_.x.ravel())
        Z.append(lasts_.z.ravel())
    X = np.array(X)[:, :, np.newaxis]
    Z = np.array(Z)
    if latent:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Z)
        distances, idxs = nbrs.kneighbors(Z)
    else:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X[:, :, 0])
        distances, idxs = nbrs.kneighbors(X[:, :, 0])

    instabilities = defaultdict(list)
    if compute_lasts:
        if verbose:
            print("Computing LASTS instability... ", sep="")
        for i, idx in enumerate(idxs):
            instability_lasts_ = instability_lasts(
                lasts_list[i],
                lasts_list[idx[-1]],  # the first value is the idx of the instance itself
                ignore_mismatch=False,
                divide_by_baseline=divide_by_baseline,
                **lasts_kwargs
            )
            instabilities[combination_name + "_lasts_count_mismatch"].append(instability_lasts_)
        if verbose:
            print("Done!")


    if compute_shap:
        if verbose:
            print("Computing SHAP instability... ", sep="")
        for i, idx in enumerate(idxs):
            instability_shap_ = instability_shap(
                lasts_list[i].x,
                lasts_list[idx[-1]].x,
                lasts_list[0].blackbox,
                divide_by_baseline=divide_by_baseline,
                **shap_kwargs
            )
            instabilities[combination_name + "_shap_abs"].append(instability_shap_)

            instability_shap_ = instability_shap(
                lasts_list[i].x,
                lasts_list[idx[-1]].x,
                lasts_list[0].blackbox,
                divide_by_baseline=divide_by_baseline,
                kind="discard_negative",
                **shap_kwargs
            )
            instabilities[combination_name + "_shap_noneg"].append(instability_shap_)

            instability_shap_ = instability_shap(
                lasts_list[i].x,
                lasts_list[idx[-1]].x,
                lasts_list[0].blackbox,
                divide_by_baseline=divide_by_baseline,
                kind="no_change",
                **shap_kwargs
            )
            instabilities[combination_name + "_shap_nochange"].append(instability_shap_)
        if verbose:
            print("Done!")
    return pd.DataFrame(instabilities)


def get_predictions():

    lasts_cbf_knn = load_multiple_lasts("./saved/knn/cbf/matched_uniform_saxdt", simple_load=True)
    lasts_cbf_resnet = load_multiple_lasts("./saved/resnet/cbf/matched_uniform_saxdt", simple_load=True)

    lasts_coffee_knn = load_multiple_lasts("./saved/knn/coffee/matched_uniform_saxdt", simple_load=True)
    lasts_coffee_resnet = load_multiple_lasts("./saved/resnet/coffee/matched_uniform_saxdt", simple_load=True)

    lasts_ecg200_knn = load_multiple_lasts("./saved/knn/ecg200/matched_uniform_saxdt", simple_load=True)
    lasts_ecg200_resnet = load_multiple_lasts("./saved/resnet/ecg200/matched_uniform_saxdt", simple_load=True)

    lasts_gunpoint_knn = load_multiple_lasts("./saved/knn/gunpoint/matched_uniform_saxdt", simple_load=True)
    lasts_gunpoint_resnet = load_multiple_lasts("./saved/resnet/gunpoint/matched_uniform_saxdt", simple_load=True)

    lasts_lists = [lasts_cbf_knn, lasts_cbf_resnet, lasts_coffee_knn, lasts_coffee_resnet, lasts_ecg200_knn,
                   lasts_ecg200_knn, lasts_ecg200_resnet, lasts_gunpoint_knn, lasts_gunpoint_resnet]
    names = ["cbf_knn", "cbf_resnet", "coffee_knn", "coffee_resnet", "ecg200_knn", "ecg200_knn", "ecg200_resnet",
             "gunpoint_knn", "gunpoint_resnet"]

    dict_exp = defaultdict(list)
    dict_tss = defaultdict(list)
    i = 0
    for lasts_list, name in zip(lasts_lists, names):
        print(i + 1)
        for lasts_ in lasts_list:
            pred_importances, tss = lasts_.surrogate.predict_explanation(lasts_.z_tilde, 1)
            dict_tss[name].append(tss)
            dict_exp[name].append(pred_importances)
        dump(dict_exp, "./stab_temp/dict_exp_" + str(i) + ".joblib")
        dump(dict_tss, "./stab_temp/dict_tss_" + str(i) + ".joblib")
        i += 1
    return dict_exp, dict_tss


def get_predictions_synth():
    lasts_synth_pat = load_multiple_lasts("./saved/pat/pat/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat0 = load_multiple_lasts("./saved/pat/pat0/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat1 = load_multiple_lasts("./saved/pat/pat1/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat2 = load_multiple_lasts("./saved/pat/pat2/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat3 = load_multiple_lasts("./saved/pat/pat3/synth/matched_uniform_saxdt", simple_load=True)

    lasts_lists = [lasts_synth_pat, lasts_synth_pat0, lasts_synth_pat1, lasts_synth_pat2, lasts_synth_pat3]
    names = ["lasts_synth_pat", "lasts_synth_pat0", "lasts_synth_pat1", "lasts_synth_pat2", "lasts_synth_pat3"]

    dict_exp = defaultdict(list)
    dict_true = defaultdict(list)
    dict_tss = defaultdict(list)
    i = 0
    for lasts_list, name in zip(lasts_lists, names):
        print(i + 1)
        for lasts_ in lasts_list:
            pred_importances, tss = lasts_.surrogate.predict_explanation(lasts_.z_tilde, 1)
            true_importances = lasts_.blackbox.predict_explanation(lasts_.z_tilde)
            dict_tss[name].append(tss)
            dict_true[name].append(true_importances)
            dict_exp[name].append(pred_importances)
        dump(dict_exp, "./stab_temp/dict_exp_" + str(i) + ".joblib")
        dump(dict_true, "./stab_temp/dict_true_" + str(i) + ".joblib")
        dump(dict_tss, "./stab_temp/dict_tss_" + str(i) + ".joblib")
        i += 1
    return dict_exp, dict_true, dict_tss


def get_instability_df(latent=False, compute_lasts=True, compute_shap=True):
    shap_kwargs = {"nsamples": 1000, "background": "linear_consecutive", "pen": 1, "model": "rbf", "jump": 5, "plot": False,
              "figsize": (20, 3), "segments_size": None}

    lasts_cbf_knn = load_multiple_lasts("./saved/knn/cbf/matched_uniform_saxdt", simple_load=True)
    lasts_cbf_resnet = load_multiple_lasts("./saved/resnet/cbf/matched_uniform_saxdt", simple_load=True)

    lasts_coffee_knn = load_multiple_lasts("./saved/knn/coffee/matched_uniform_saxdt", simple_load=True)
    lasts_coffee_resnet = load_multiple_lasts("./saved/resnet/coffee/matched_uniform_saxdt", simple_load=True)

    lasts_ecg200_knn = load_multiple_lasts("./saved/knn/ecg200/matched_uniform_saxdt", simple_load=True)
    lasts_ecg200_resnet = load_multiple_lasts("./saved/resnet/ecg200/matched_uniform_saxdt", simple_load=True)

    lasts_gunpoint_knn = load_multiple_lasts("./saved/knn/gunpoint/matched_uniform_saxdt", simple_load=True)
    lasts_gunpoint_resnet = load_multiple_lasts("./saved/resnet/gunpoint/matched_uniform_saxdt", simple_load=True)

    lasts_lists = [lasts_cbf_knn, lasts_cbf_resnet, lasts_coffee_knn, lasts_coffee_resnet, lasts_ecg200_knn,
                   lasts_ecg200_knn, lasts_ecg200_resnet, lasts_gunpoint_knn, lasts_gunpoint_resnet]
    names = ["cbf_knn", "cbf_resnet", "coffee_knn", "coffee_resnet", "ecg200_knn", "ecg200_knn", "ecg200_resnet",
             "gunpoint_knn", "gunpoint_resnet"]
    df_dict = dict()
    i = 0
    for lasts_list, name in zip(lasts_lists, names):
        print(i + 1, "/", len(lasts_lists))
        i += 1
        df1 = instability_multiple(lasts_list=lasts_list, combination_name=name, shap_kwargs=shap_kwargs, latent=True,
                                  compute_lasts=compute_lasts, compute_shap=compute_shap
                                  )
        df2 = instability_multiple(lasts_list=lasts_list, combination_name=name, shap_kwargs=shap_kwargs, latent=False,
                                  compute_lasts=compute_lasts, compute_shap=compute_shap
                                  )
        df_dict[name + "_lasts_latent"] = df1
        df_dict[name + "_lasts_manifest"] = df2
        dump(df_dict, "./stability_saved/df_dict_" + str(i-1) + ".joblib")
    return df_dict


def get_explanation_error_df():
    kwargs = {"nsamples": 1000, "background": "linear_consecutive", "pen": 1, "model": "rbf", "jump": 5, "plot": False,
              "figsize": (20, 3), "segments_size": None}

    lasts_synth_pat = load_multiple_lasts("./saved/pat/pat/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat0 = load_multiple_lasts("./saved/pat/pat0/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat1 = load_multiple_lasts("./saved/pat/pat1/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat2 = load_multiple_lasts("./saved/pat/pat2/synth/matched_uniform_saxdt", simple_load=True)
    lasts_synth_pat3 = load_multiple_lasts("./saved/pat/pat3/synth/matched_uniform_saxdt", simple_load=True)

    lasts_lists = [lasts_synth_pat, lasts_synth_pat0, lasts_synth_pat1, lasts_synth_pat2, lasts_synth_pat3]
    names = ["lasts_synth_pat", "lasts_synth_pat0", "lasts_synth_pat1", "lasts_synth_pat2", "lasts_synth_pat3"]

    df = defaultdict(list)
    for j, lasts_list in enumerate(lasts_lists):
        for i, lasts_ in enumerate(lasts_list):
            print(i + 1, "/", len(lasts_list))
            # df["lasts_" + names[j]].append(lasts_.explanation_error(divide_by_baseline=False))
            shapts = ShapTimeSeries()
            shapts.shap_values(lasts_.x, lasts_.blackbox, **kwargs)
            df["shap_abs_" + names[j]].append(shapts.explanation_error(divide_by_baseline=False))
        print(pd.DataFrame(df))
    return pd.DataFrame(df)


def get_usefulness_df():
    cbf_X_exp = build_cbf(n_samples=600, random_state=0, verbose=False)[10:]
    ecg200_X_exp = build_ecg200(verbose=False, random_state=0)[10:]
    coffee_X_exp = build_coffee(verbose=False, random_state=0)[10:]
    gunpoint_X_exp = build_gunpoint(verbose=False, random_state=0)[10:]

    lasts_cbf_knn = load_multiple_lasts("./saved/knn/cbf/matched_uniform_saxdt", simple_load=True)
    lasts_cbf_resnet = load_multiple_lasts("./saved/resnet/cbf/matched_uniform_saxdt", simple_load=True)

    lasts_coffee_knn = load_multiple_lasts("./saved/knn/coffee/matched_uniform_saxdt", simple_load=True)
    lasts_coffee_resnet = load_multiple_lasts("./saved/resnet/coffee/matched_uniform_saxdt", simple_load=True)

    lasts_ecg200_knn = load_multiple_lasts("./saved/knn/ecg200/matched_uniform_saxdt", simple_load=True)
    lasts_ecg200_resnet = load_multiple_lasts("./saved/resnet/ecg200/matched_uniform_saxdt", simple_load=True)

    lasts_gunpoint_knn = load_multiple_lasts("./saved/knn/gunpoint/matched_uniform_saxdt", simple_load=True)
    lasts_gunpoint_resnet = load_multiple_lasts("./saved/resnet/gunpoint/matched_uniform_saxdt", simple_load=True)

    dataset_list = [cbf_X_exp, ecg200_X_exp, coffee_X_exp, gunpoint_X_exp]
    lasts_lists = [lasts_cbf_knn, lasts_cbf_resnet, lasts_coffee_knn, lasts_coffee_resnet, lasts_ecg200_knn,
                   lasts_ecg200_resnet, lasts_gunpoint_knn, lasts_gunpoint_resnet]
    lasts_lists_names = ["lasts_cbf_knn", "lasts_cbf_resnet", "lasts_coffee_knn", "lasts_coffee_resnet",
                        "lasts_ecg200_knn", "lasts_ecg200_resnet", "lasts_gunpoint_knn",
                        "lasts_gunpoint_resnet"]
    global_names = ["global_cbf", "global_coffee", "global_ecg200", "global_gunpoint"]
    df_dict = dict()
    for dataset, name in zip(dataset_list, global_names):
        df_dict[name] = usefulness_scores_real(dataset[0], dataset[1])
    for lasts_list, name in zip(lasts_lists, lasts_lists_names):
        df_dict[name] = usefulness_scores_lasts(lasts_list)
    return df_dict




def get_autoencoders_mse():
    # the indexes 4 and 5 are X_test, y_test
    cbf_X_exp = build_cbf(n_samples=600, random_state=0, verbose=False)[6:]
    ecg200_X_exp = build_ecg200(verbose=False, random_state=0)[6:]
    coffee_X_exp = build_coffee(verbose=False, random_state=0)[6:]
    gunpoint_X_exp = build_gunpoint(verbose=False, random_state=0)[6:]

    cbf_vae = load_model("./trained_models/cbf/cbf_vae")[2]
    ecg200_vae = load_model("./trained_models/ecg200/ecg200_vae")[2]
    coffee_vae = load_model("./trained_models/coffee/coffee_vae")[2]
    gunpoint_vae = load_model("./trained_models/gunpoint/gunpoint_vae")[2]

    dataset_list = [cbf_X_exp, ecg200_X_exp, coffee_X_exp, gunpoint_X_exp]
    dataset_names = ["cbf", "ecg200", "coffee", "gunpoint", "synth"]
    vae_list = [cbf_vae, ecg200_vae, coffee_vae, gunpoint_vae]

    df = dict()
    for dataset, dataset_name, vae in zip(dataset_list, dataset_names, vae_list):
        # print(dataset, dataset_name, clf_list, clf_name_list)
        df[dataset_name] = vae.evaluate(dataset[4], dataset[4])[1]
    return pd.DataFrame(df, index=[0])


def get_fidelities_df():
    # the indexes 4 and 5 are X_test, y_test
    cbf_X_exp = build_cbf(n_samples=600, random_state=0, verbose=False)[6:]
    ecg200_X_exp = build_ecg200(verbose=False, random_state=0)[6:]
    coffee_X_exp = build_coffee(verbose=False, random_state=0)[6:]
    gunpoint_X_exp = build_gunpoint(verbose=False, random_state=0)[6:]

    cbf_knn = load("./trained_models/cbf/cbf_knn.joblib")
    cbf_knn = BlackboxWrapper(cbf_knn, 2, 1)
    cbf_resnet = keras.models.load_model("./trained_models/cbf/cbf_resnet.h5")
    cbf_resnet = BlackboxWrapper(cbf_resnet, 3, 2)

    ecg200_knn = load("./trained_models/ecg200/ecg200_knn.joblib")
    ecg200_knn = BlackboxWrapper(ecg200_knn, 2, 1)
    ecg200_resnet = keras.models.load_model("./trained_models/ecg200/ecg200_resnet.h5")
    ecg200_resnet = BlackboxWrapper(ecg200_resnet, 3, 2)

    coffee_knn = load("./trained_models/coffee/coffee_knn.joblib")
    coffee_knn = BlackboxWrapper(coffee_knn, 2, 1)
    coffee_resnet = keras.models.load_model("./trained_models/coffee/coffee_resnet.h5")
    coffee_resnet = BlackboxWrapper(coffee_resnet, 3, 2)

    gunpoint_knn = load("./trained_models/gunpoint/gunpoint_knn.joblib")
    gunpoint_knn = BlackboxWrapper(gunpoint_knn, 2, 1)
    gunpoint_resnet = keras.models.load_model("./trained_models/gunpoint/gunpoint_resnet.h5")
    gunpoint_resnet = BlackboxWrapper(gunpoint_resnet, 3, 2)


    dataset_list = [cbf_X_exp, ecg200_X_exp, coffee_X_exp, gunpoint_X_exp]
    dataset_names = ["cbf", "ecg200", "coffee", "gunpoint"]
    clf_lists = [
        [cbf_knn, cbf_resnet],
        [ecg200_knn, ecg200_resnet],
        [coffee_knn, coffee_resnet],
        [gunpoint_knn, gunpoint_resnet]
    ]
    clf_names_lists = [
        ["cbf_knn", "cbf_resnet"],
        ["ecg200_knn", "ecg200_resnet"],
        ["coffee_knn", "coffee_resnet"],
        ["gunpoint_knn", "gunpoint_resnet"]
    ]

    shapelet_list = [
        generate_n_shapelets_per_size(cbf_X_exp[0].shape[1], n_shapelets_per_length=4, min_length=8, start_divider=2,
                                      divider_multiplier=1.2),
        generate_n_shapelets_per_size(ecg200_X_exp[0].shape[1], n_shapelets_per_length=4, min_length=8, start_divider=2,
                                      divider_multiplier=1.2),
        generate_n_shapelets_per_size(coffee_X_exp[0].shape[1], n_shapelets_per_length=2, min_length=8, start_divider=2,
                                      divider_multiplier=1.35),
        generate_n_shapelets_per_size(gunpoint_X_exp[0].shape[1], n_shapelets_per_length=2, min_length=8, start_divider=2,
                                      divider_multiplier=1.26),
    ]

    df = pd.DataFrame()
    for dataset, dataset_name, clf_list, clf_name_list in \
            zip(dataset_list, dataset_names, clf_lists, clf_names_lists):
        df_fid = test_fidelity(clf_list, clf_name_list, dataset[0], dataset[4], shapelet_list)
        df = df.append(df_fid)
    return df


def get_accuracies_df():
    # the indexes 4 and 5 are X_test, y_test
    cbf_test = build_cbf(n_samples=600, random_state=0, verbose=False)[4:6]
    ecg200_test = build_ecg200(verbose=False, random_state=0)[4:6]
    coffee_test = build_coffee(verbose=False, random_state=0)[4:6]
    gunpoint_test = build_gunpoint(verbose=False, random_state=0)[4:6]
    synth_test = build_synth(path="./datasets/synth04/", verbose=False, random_state=0)[4:6]

    cbf_knn = load("./trained_models/cbf/cbf_knn.joblib")
    cbf_knn = BlackboxWrapper(cbf_knn, 2, 1)
    cbf_resnet = keras.models.load_model("./trained_models/cbf/cbf_resnet.h5")
    cbf_resnet = BlackboxWrapper(cbf_resnet, 3, 2)

    ecg200_knn = load("./trained_models/ecg200/ecg200_knn.joblib")
    ecg200_knn = BlackboxWrapper(ecg200_knn, 2, 1)
    ecg200_resnet = keras.models.load_model("./trained_models/ecg200/ecg200_resnet.h5")
    ecg200_resnet = BlackboxWrapper(ecg200_resnet, 3, 2)

    coffee_knn = load("./trained_models/coffee/coffee_knn.joblib")
    coffee_knn = BlackboxWrapper(coffee_knn, 2, 1)
    coffee_resnet = keras.models.load_model("./trained_models/coffee/coffee_resnet.h5")
    coffee_resnet = BlackboxWrapper(coffee_resnet, 3, 2)

    gunpoint_knn = load("./trained_models/gunpoint/gunpoint_knn.joblib")
    gunpoint_knn = BlackboxWrapper(gunpoint_knn, 2, 1)
    gunpoint_resnet = keras.models.load_model("./trained_models/gunpoint/gunpoint_resnet.h5")
    gunpoint_resnet = BlackboxWrapper(gunpoint_resnet, 3, 2)

    cbf_vae = load_model("./trained_models/cbf/cbf_vae")[2]
    ecg200_vae = load_model("./trained_models/ecg200/ecg200_vae")[2]
    coffee_vae = load_model("./trained_models/coffee/coffee_vae")[2]
    gunpoint_vae = load_model("./trained_models/gunpoint/gunpoint_vae")[2]

    dataset_list = [cbf_test, ecg200_test, coffee_test, gunpoint_test]
    dataset_names = ["cbf", "ecg200", "coffee", "gunpoint"]
    clf_lists = [
        [cbf_knn, cbf_resnet],
        [ecg200_knn, ecg200_resnet],
        [coffee_knn, coffee_resnet],
        [gunpoint_knn, gunpoint_resnet]
    ]
    clf_names_lists = [
        ["cbf_knn", "cbf_resnet"],
        ["ecg200_knn", "ecg200_resnet"],
        ["coffee_knn", "coffee_resnet"],
        ["gunpoint_knn", "gunpoint_resnet"]
    ]
    vae_list = [cbf_vae, ecg200_vae, coffee_vae, gunpoint_vae]

    df = pd.DataFrame()
    for dataset, dataset_name, clf_list, clf_name_list, vae in \
            zip(dataset_list, dataset_names, clf_lists, clf_names_lists, vae_list):
        #print(dataset, dataset_name, clf_list, clf_name_list)
        df_acc = test_accuracy(clf_list, clf_name_list, dataset[0], dataset[1])
        df_rec_acc = test_reconstruction_accuracy(clf_list, clf_name_list, vae.layers[2], vae.layers[3], dataset[0])
        df_agg = pd.concat([df_acc, df_rec_acc], axis=1)
        df = df.append(df_agg)
    return df
