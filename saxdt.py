from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.shapelet_based.mrseql.mrseql import PySAX  # custom fork of sktime
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import graphviz
import scipy
from utils import convert_numpy_to_sktime, convert_sktime_to_numpy, sliding_window_distance, compute_medoid, sliding_window_euclidean
from tree_utils import (NewTree,
                        get_root_leaf_path,
                        get_thresholds_signs,
                        minimumDistance,
                        coverage_score_tree,
                        precision_score_tree,
                        prune_duplicate_leaves)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import math
import numpy as np
import warnings
import pandas as pd
from sbgdt import plot_binary_heatmap, plot_subsequences_grid


def plot_factual_and_counterfactual(
        saxdt_model,
        x,
        x_label,
        draw_on=None,
        verbose_explanation=True,
        graphical_explanation=True,
        figsize=(10, 3),
        fontsize=18,
        text_height=0.5,
        c_index=0,
        labelfontsize=12,
        dpi=60,
        loc=None,
        frameon=False,
        plot_dictionary=False,
        print_word=True,
        fixed_contained_subsequences=True,
        enhance_not_contained=False,
        no_axes_labels=False
):  # FIXME: check this monstruosity again

    if saxdt_model.subsequence_dictionary is None:
        saxdt_model.create_dictionaries()
    dtree = saxdt_model.decision_tree_explorable
    leaf_id = saxdt_model.find_leaf_id(x)

    factual = get_root_leaf_path(dtree.nodes[leaf_id])
    factual = get_thresholds_signs(dtree, factual)

    nearest_leaf = minimumDistance(dtree.nodes[0], dtree.nodes[leaf_id])[1]

    counterfactual = get_root_leaf_path(dtree.nodes[nearest_leaf])
    counterfactual = get_thresholds_signs(dtree, counterfactual)

    rules_list = [factual, counterfactual]

    if verbose_explanation:
        print("VERBOSE EXPLANATION")
        for i, rule in enumerate(rules_list):
            print()
            print("RULE" if i == 0 else "COUNTERFACTUAL")
            if i == 0:
                print("real class ==", saxdt_model.labels[x_label] if saxdt_model.labels else x_label)
            print("If", end=" ")
            for j, idx_word in enumerate(rule["features"][:-1]):
                word, _, _ = find_feature(idx_word, saxdt_model.seql_model.sequences)
                print("the word", "'" + word.decode("utf-8") + "'", "(" + str(idx_word) + ")",
                      "is", rule["thresholds_signs"][j], end="")
                if j != len(rule["features"][:-1]) - 1:
                    print(", and", end=" ")
                else:
                    print(",", end=" ")
            print("then the class is", rule["labels"][-1] if not saxdt_model.labels \
                else saxdt_model.labels[rule["labels"][-1]])

    if plot_dictionary:
        idx_word_list = list()
        for i, rule in enumerate(rules_list):
            for j, idx_word in enumerate(rule["features"][:-1]):
                if idx_word not in idx_word_list:
                    plot_subsequence_mapping(saxdt_model.subsequence_dictionary, saxdt_model.name_dictionary, idx_word)
                idx_word_list.append(idx_word)

    if graphical_explanation:
        contained_subsequences = dict()
        for i, idx_word in enumerate(rules_list[0]["features"][:-1]):
            threshold_sign = rules_list[0]["thresholds_signs"][i]
            if threshold_sign == "contained":
                start_idx, end_idx, feature = map_word_idx_to_ts(x, idx_word, saxdt_model.seql_model)
                if end_idx == len(x.ravel()):
                    end_idx -= 1
                subsequence = x.ravel()[start_idx:end_idx + 1]
                contained_subsequences[idx_word] = [subsequence]

        # counterfactual rule applied to a counterfactual z_tilde
        # get all the leave ids
        leave_ids = saxdt_model.decision_tree.apply(saxdt_model.X_transformed)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.argwhere(leave_ids == nearest_leaf)
        # choose one counterfactual
        counterfactual_idx = counterfactuals_idxs[c_index][0]
        counterfactual_ts = saxdt_model.X[counterfactual_idx:counterfactual_idx + 1]
        counterfactual_ts_sktime = convert_numpy_to_sktime(counterfactual_ts)
        counterfactual_y = saxdt_model.y[counterfactual_idx]
        for i, idx_word in enumerate(rules_list[1]["features"][:-1]):
            threshold_sign = rules_list[1]["thresholds_signs"][i]
            if threshold_sign == "contained":

                start_idx, end_idx, feature = map_word_idx_to_ts(counterfactual_ts, idx_word, saxdt_model.seql_model)
                if end_idx == len(counterfactual_ts.ravel()):
                    end_idx -= 1
                subsequence = counterfactual_ts.ravel()[start_idx:end_idx + 1]
                if idx_word not in contained_subsequences:
                    contained_subsequences[idx_word] = [subsequence]
                else:
                    contained_subsequences[idx_word].append(subsequence)

        title = ("LASTS - Factual Rule " + r"$p_s\rightarrow$" + " " +
                 saxdt_model.labels[rules_list[0]["labels"][-1]] if saxdt_model.labels
                 else "LASTS - Factual Rule " + r"$p_s\rightarrow$" + " " +
                      str(rules_list[0]["labels"][-1]))
        legend_label = r"$x$"
        y_lim = plot_graphical_explanation(
            saxdt_model=saxdt_model,
            x=x,
            rule=rules_list[0],
            title=title,
            legend_label=legend_label,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            text_height=text_height,
            labelfontsize=labelfontsize,
            loc=loc,
            frameon=frameon,
            forced_y_lim=None,
            return_y_lim=True,
            draw_on=draw_on,
            contained_subsequences=contained_subsequences,
            print_word=print_word,
            fixed_contained_subsequences=fixed_contained_subsequences,
            is_factual_for_counterexemplar=False,
            enhance_not_contained=enhance_not_contained,
            no_axes_labels=no_axes_labels
        )

        title = ("LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                 saxdt_model.labels[rules_list[1]["labels"][-1]] if saxdt_model.labels
                 else "LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                      str(rules_list[1]["labels"][-1]))
        legend_label = r"$x$"

        plot_graphical_explanation(
            saxdt_model=saxdt_model,
            x=x,
            rule=rules_list[1],
            title=title,
            legend_label=legend_label,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            text_height=text_height,
            labelfontsize=labelfontsize,
            loc=loc,
            frameon=frameon,
            forced_y_lim=None,
            return_y_lim=False,
            draw_on=draw_on,
            contained_subsequences=contained_subsequences,
            print_word=print_word,
            fixed_contained_subsequences=fixed_contained_subsequences,
            is_factual_for_counterexemplar=False,
            enhance_not_contained=enhance_not_contained,
            no_axes_labels=no_axes_labels
        )

        print("real class ==", saxdt_model.labels[counterfactual_y] if saxdt_model.labels else counterfactual_y)
        title = ("LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                 saxdt_model.labels[rules_list[1]["labels"][-1]] if saxdt_model.labels
                 else "LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                      str(rules_list[1]["labels"][-1]))
        legend_label = r"$\tilde{z}'$"

        plot_graphical_explanation(
            saxdt_model=saxdt_model,
            x=counterfactual_ts,
            rule=rules_list[1],
            title=title,
            legend_label=legend_label,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            text_height=text_height,
            labelfontsize=labelfontsize,
            loc=loc,
            frameon=frameon,
            forced_y_lim=y_lim,
            return_y_lim=False,
            contained_subsequences=contained_subsequences,
            print_word=print_word,
            fixed_contained_subsequences=fixed_contained_subsequences,
            is_factual_for_counterexemplar=True,
            enhance_not_contained=enhance_not_contained,
            no_axes_labels=no_axes_labels
        )


def plot_graphical_explanation(
        saxdt_model,
        x,
        rule,
        title,
        legend_label,
        figsize,
        dpi,
        fontsize,
        text_height,
        labelfontsize,
        loc,
        frameon,
        is_factual_for_counterexemplar,
        contained_subsequences,
        fixed_contained_subsequences=True,
        forced_y_lim=None,
        return_y_lim=False,
        draw_on=None,
        print_word=True,
        enhance_not_contained=False,
        no_axes_labels=False
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title, fontsize=fontsize)
    ax.plot(x.ravel().T if draw_on is None else draw_on.ravel().T, c="royalblue", alpha=0.2, lw=3, label=legend_label)
    #ax.plot(x.ravel().T if draw_on is None else draw_on.T, c="royalblue", alpha=0.2, lw=3, label=legend_label)

    for i, idx_word in enumerate(rule["features"][:-1]):
        feature = saxdt_model.name_dictionary[idx_word]
        threshold_sign = rule["thresholds_signs"][i]
        dummy_ts = np.full_like(x.ravel(), np.nan)
        if idx_word in contained_subsequences:
            if fixed_contained_subsequences:
                subsequence = contained_subsequences[idx_word][0]
            else:
                if is_factual_for_counterexemplar and (len(contained_subsequences[idx_word]) == 2):
                    subsequence = contained_subsequences[idx_word][1]
                else:
                    subsequence = contained_subsequences[idx_word][0]
        else:
            if enhance_not_contained:
                maximum = 0
                subseq = None
                for subsequence in saxdt_model.subsequence_dictionary[idx_word][:, :, 0]:
                    dist = sliding_window_euclidean(x.ravel(), subsequence)
                    if dist > maximum:
                        maximum = dist
                        subseq = subsequence
                subsequence = subseq
            else:
                subsequence = compute_medoid(saxdt_model.subsequence_dictionary[idx_word][:, :, 0])
        best_alignment_start_idx = sliding_window_distance(x.ravel(), subsequence)
        best_alignment_end_idx = best_alignment_start_idx + len(subsequence)
        start_idx = best_alignment_start_idx
        end_idx = best_alignment_end_idx
        if end_idx == len(x.ravel()):
            end_idx -= 1
            subsequence = subsequence[:-1]
        dummy_ts[start_idx:end_idx] = subsequence
        if threshold_sign == "contained":
            ax.plot(dummy_ts, c="#2ca02c", alpha=0.5, lw=5, label="contained")
            plt.text(
                (start_idx + end_idx) / 2,
                # np.nanmin(dummy_ts) + text_height + ((np.nanmin(dummy_ts) + np.nanmax(dummy_ts))/2),
                text_height + np.mean(subsequence),
                str(idx_word) if not print_word else str(idx_word) + " (" + feature.decode("utf-8") + ")",
                fontsize=fontsize - 2,
                c="#2ca02c",
                horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[patheffects.Stroke(linewidth=3, foreground='white', alpha=0.6),
                              patheffects.Normal()]
            )
        else:
            ax.plot(dummy_ts, c="#d62728", alpha=0.5, lw=5, linestyle="--", label="not-contained")
            plt.text(
                (best_alignment_start_idx + best_alignment_end_idx) / 2,
                # np.nanmin(dummy_ts) + text_height + ((np.nanmin(dummy_ts) + np.nanmax(dummy_ts))/2),
                text_height + np.mean(subsequence),
                str(idx_word) if not print_word else str(idx_word) + " (" + feature.decode("utf-8") + ")",
                fontsize=fontsize - 2,
                c="#d62728",
                horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[patheffects.Stroke(linewidth=3, foreground='white', alpha=0.6),
                              patheffects.Normal()]
            )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    if not no_axes_labels:
        plt.xlabel("time-steps", fontsize=fontsize)
        plt.ylabel("value", fontsize=fontsize)
    plt.legend(
        by_label.values(),
        by_label.keys(),
        frameon=frameon,
        fontsize=labelfontsize,
        loc=loc
    )
    if forced_y_lim is not None:
        plt.gca().set_ylim(forced_y_lim)
    if return_y_lim:
        y_lim = plt.gca().get_ylim()
    plt.show()
    if return_y_lim:
        return y_lim


def predict_explanation(
        saxdt_model,
        x,
        x_label
):
    if saxdt_model.subsequence_dictionary is None:
        saxdt_model.create_dictionaries()
    dtree = saxdt_model.decision_tree_explorable
    leaf_id = saxdt_model.find_leaf_id(x)

    factual = get_root_leaf_path(dtree.nodes[leaf_id])
    factual = get_thresholds_signs(dtree, factual)

    dummy_ts_x = np.repeat(np.nan, len(x.ravel()))
    dummy_list = list()
    tss = list()
    for i, idx_word in enumerate(factual["features"][:-1]):
        threshold_sign = factual["thresholds_signs"][i]
        if threshold_sign == "contained":
            start_idx, end_idx, _ = map_word_idx_to_ts(x, idx_word, saxdt_model.seql_model)
            if end_idx == len(x.ravel()):
                end_idx -= 1
            dummy_ts_x[start_idx:end_idx + 1] = 1
        else:
            dummy_ts = np.repeat(np.nan, len(x.ravel()))
            #  find other instances with different label wrt x and containing the subsequence
            idxs = np.argwhere((saxdt_model.y != x_label) & (saxdt_model.X_transformed[:, idx_word] == 1)).ravel()
            if len(idxs) == 0:
                dummy_list.append(dummy_ts)  # FIXME: if there are no ts that satisfy the condition above
            else:
                idx = idxs[0]
                tss.append(saxdt_model.X[idx].ravel())
                start_idx, end_idx, _ = map_word_idx_to_ts(saxdt_model.X[idx: idx + 1], idx_word, saxdt_model.seql_model)
                if end_idx == len(x.ravel()):
                    end_idx -= 1
                dummy_ts[start_idx:end_idx + 1] = 1
                dummy_list.append(dummy_ts)
    dummy_list.append(dummy_ts_x)
    tss.append(x.ravel())
    dummy_list = np.nan_to_num(np.array(dummy_list))
    tss = np.array(tss)
    return dummy_list, tss


def map_word_idx_to_ts(x, word_idx, seql_model):
    word, cfg_idx, _ = find_feature(word_idx, seql_model.sequences)
    start_idx, end_idx = map_word_to_ts(x, word, seql_model.config[cfg_idx])
    return start_idx, end_idx, word


def map_word_to_ts(x, word, cfg):
    word = [word]
    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
    idx_set = ps.map_patterns(x.ravel(), word)[0]
    # print(idx_set)
    if len(idx_set) == 0:
        return None, None
    idx_set_sorted = sorted(list(idx_set))
    start_idx = idx_set_sorted[0]
    end_idx = math.floor(start_idx + (len(word[0]) * cfg['window'] / cfg['word']) - 1)
    if end_idx < len(x) - 1:
        end_idx += 1
    return start_idx, end_idx


def create_subsequences_dictionary(X, X_transformed, n_features, seql_model):
    subsequence_dictionary = dict()
    subsequence_norm_dictionary = dict()
    name_dictionary = dict()
    for feature in range(n_features):
        subsequences = list()
        subsequences_norm = list()
        lengths = list()
        for i, x in enumerate(X):
            if X_transformed[i][feature] == 1:
                start_idx, end_idx, feature_string = map_word_idx_to_ts(x, feature, seql_model)
                x_norm = scipy.stats.zscore(x)
                if start_idx is not None:
                    subsequence = x[start_idx:end_idx]
                    subsequences.append(subsequence)

                    subsequence_norm = x_norm[start_idx:end_idx]
                    subsequences_norm.append(subsequence_norm)

                    lengths.append(end_idx - start_idx)
        mode = scipy.stats.mode(np.array(lengths))[0][0]
        subsequences_same_length = list()
        subsequences_norm_same_length = list()
        for i, subsequence in enumerate(
                subsequences):  # to avoid problems with sequences having slightly different lengths
            if len(subsequence) == mode:
                subsequences_same_length.append(subsequence)
                subsequences_norm_same_length.append(subsequences_norm[i])
        subsequence_dictionary[feature] = np.array(subsequences_same_length)
        subsequence_norm_dictionary[feature] = np.array(subsequences_norm_same_length)
        name_dictionary[feature] = feature_string
    return subsequence_dictionary, name_dictionary, subsequence_norm_dictionary


def plot_subsequence_mapping(subsequence_dictionary, name_dictionary, feature_idx):
    plt.title(str(feature_idx) + " : " + name_dictionary[feature_idx].decode("utf-8"))
    plt.plot(subsequence_dictionary[feature_idx][:, :, 0].T, c="gray", alpha=0.1)
    # plt.plot(subsequence_dictionary[feature_idx][:,:,0].mean(axis=0).ravel(), c="red")
    plt.plot(compute_medoid(subsequence_dictionary[feature_idx][:, :, 0]), c="red")
    plt.show()


def extract_mapped_subsequences(X, feature_idx, seql_model):
    subsequences = list()
    lengths = []
    for i, x in enumerate(X):
        start_idx, end_idx, feature = map_word_idx_to_ts(x.ravel(), feature_idx, seql_model)
        if start_idx is not None:
            lengths.append(end_idx - start_idx)
            subsequences.append(x.ravel()[start_idx:end_idx])
    same_length_subsequences = list()
    for i, s in enumerate(subsequences):
        if lengths[i] == scipy.stats.mode(np.array(lengths))[0][0]:
            same_length_subsequences.append(s)
    return np.array(same_length_subsequences)


def find_feature(feature_idx, sequences):  # FIXME: check if the count is correct
    idxs = 0
    for i, config in enumerate(sequences):
        if idxs + len(config) - 1 < feature_idx:
            idxs += len(config)
            continue
        elif idxs + len(config) - 1 >= feature_idx:
            j = feature_idx - idxs
            feature = config[j]
            break
    return feature, i, j


def map_word_to_window(word_length, window_size):
    mapping = list()
    paa_size = window_size / word_length
    for i in range(word_length):
        windowStartIdx = paa_size * i
        windowEndIdx = (paa_size * (i + 1)) - 1
        fullWindowStartIdx = math.ceil(windowStartIdx)
        fullWindowEndIdx = math.floor(windowEndIdx)
        startFraction = fullWindowStartIdx - windowStartIdx
        endFraction = windowEndIdx - fullWindowEndIdx
        if (startFraction > 0):
            fullWindowStartIdx = fullWindowStartIdx - 1
        if (endFraction > 0 and fullWindowEndIdx < window_size - 1):
            fullWindowEndIdx = fullWindowEndIdx + 1
        mapping.append([fullWindowStartIdx, fullWindowEndIdx + 1])
    return np.array(mapping)


class Saxdt(object):
    def __init__(
            self,
            labels=None,
            random_state=None,
            custom_config=None,
            decision_tree_grid_search_params={
                'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]
            },
            create_plotting_dictionaries=True
    ):
        self.labels = labels
        self.random_state = random_state
        self.decision_tree_grid_search_params = decision_tree_grid_search_params
        self.custom_config = custom_config

        self.X = None
        self.X_transformed = None
        self.y = None

        self.create_plotting_dictionaries = create_plotting_dictionaries
        self.subsequence_dictionary = None
        self.name_dictionary = None
        self.subsequences_norm_same_length = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        X = convert_numpy_to_sktime(X)
        seql_model = MrSEQLClassifier(seql_mode='fs', symrep='sax', custom_config=self.custom_config)
        seql_model.fit(X, y)
        mr_seqs = seql_model._transform_time_series(X)
        X_transformed = seql_model._to_feature_space(mr_seqs)

        clf = DecisionTreeClassifier()
        param_grid = self.decision_tree_grid_search_params
        grid = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=0)
        grid.fit(X_transformed, y)

        clf = DecisionTreeClassifier(**grid.best_params_, random_state=self.random_state)
        clf.fit(X_transformed, y)
        prune_duplicate_leaves(clf)

        self.X_transformed = X_transformed
        self.decision_tree = clf
        self.decision_tree_explorable = NewTree(clf)
        self.decision_tree_explorable.build_tree()
        self.seql_model = seql_model
        self._build_tree_graph()
        if self.create_plotting_dictionaries:
            self.create_dictionaries()
        return self

    def predict(self, X):
        X = convert_numpy_to_sktime(X)
        mr_seqs = self.seql_model._transform_time_series(X)
        X_transformed = self.seql_model._to_feature_space(mr_seqs)
        y = self.decision_tree.predict(X_transformed)
        return y

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _build_tree_graph(self, out_file=None):
        dot_data = tree.export_graphviz(self.decision_tree, out_file=out_file,
                                        class_names=self.labels,
                                        filled=True, rounded=True,
                                        special_characters=True)
        self.graph = graphviz.Source(dot_data)
        return self

    def plot_factual_and_counterfactual(
            self,
            x,
            x_label,
            **kwargs
    ):
        plot_factual_and_counterfactual(self, x, x_label, **kwargs)
        return self

    def plot_binary_heatmap(self, x_label, **kwargs):
        plot_binary_heatmap(x_label, self.y, self.X_transformed, **kwargs)
        return self

    def find_leaf_id(self, ts):
        ts = convert_numpy_to_sktime(ts)
        mr_seqs = self.seql_model._transform_time_series(ts)
        ts_transformed = self.seql_model._to_feature_space(mr_seqs)
        leaf_id = self.decision_tree.apply(ts_transformed)[0]
        return leaf_id

    def coverage_score(self, leaf_id):
        return coverage_score_tree(self.decision_tree, leaf_id)

    def precision_score(self, leaf_id, y, X=None):
        if X is None:
            X = self.X_transformed.copy()
        else:
            X = convert_numpy_to_sktime(X)
            mr_seqs = self.seql_model._transform_time_series(X)
            X_transformed = self.seql_model._to_feature_space(mr_seqs)
            X = X_transformed
        return precision_score_tree(self.decision_tree, X, y, leaf_id)

    def create_dictionaries(self):
        (self.subsequence_dictionary,
         self.name_dictionary,
         self.subsequence_norm_dictionary) = create_subsequences_dictionary(
            self.X, self.X_transformed, self.X_transformed.shape[1], self.seql_model
        )

    def predict_explanation(self, x, x_label, **kwargs):
        return predict_explanation(self, x, x_label)

    def plot_subsequences_grid(self, n, m, starting_idx=0, random=False, color="mediumblue", **kwargs):
        if self.subsequence_dictionary is None:
            self.create_dictionaries()
        subsequence_list = list()
        for key in self.subsequence_dictionary:
            subsequence_list.append(self.subsequence_dictionary[key].mean(axis=0).ravel())
        plot_subsequences_grid(subsequence_list, n=n, m=m, starting_idx=starting_idx, random=random, color=color,
                               **kwargs)
        return self


if __name__ == "__main__":
    from datasets import build_cbf

    random_state = 0
    np.random.seed(0)
    dataset_name = "cbf"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(random_state=random_state)

    clf = Saxdt(labels=["cylinder", "bell", "funnel"], random_state=np.random.seed(0))
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    i = 0
    clf.plot_factual_and_counterfactual(X_train[i:i + 1], y_train[i], plot_dictionary=True, print_word=False)
