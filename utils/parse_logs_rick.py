from glob import glob
from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

from sklearn.metrics import roc_auc_score

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.style.use('dark_background')

display(HTML("<style>"
             + "#notebook { padding-top:0px; } " ""
             + ".container { width:100%; } "
             + ".end_space { min-height:0px; } "
             + "</style>"))


def read_logs_til_exp(path_logs: str, id_exp: str):
    with open(path_logs, 'r') as f:
        line = 'start'
        while line:
            line = f.readline()
            if id_exp in line:
                break
    return literal_eval(line)


def print_exp_info(exp: dict):
    print({k: exp[k] for k in exp if k not in ['logs', 'results', 'knowledge']})


# usage example:  show_exp_images(experiments[0], True)
def show_exp_images(exp: dict, show_origins=False):
    for task in exp['results']:
        cur_images = exp['results'][task]['images']
        fig, axs = plt.subplots(2, 5, figsize=(15, 8))
        anomalies = [x['label'] for x in exp['results'][task]['images']
                     if x['label'] not in [str(z) for z in exp["knowledge"][task]]]
        fig.suptitle(f'TASK {task} - {str(exp["knowledge"][task])}\nanomalies - {anomalies} ', fontsize=30)
        for r, row in enumerate(axs):
            for c, cell in enumerate(row):
                idx = r * 5 + c
                image = np.zeros((28, 28, 3), dtype=float)
                cell.set_title(cur_images[idx]['label'])
                orig = np.array(cur_images[idx]['original'][0])
                recon = np.array(cur_images[idx]['reconstruction'][0]).clip(0, 1)
                if show_origins:
                    image[:, :, 1] = orig
                image[:, :, 0] = recon
                image[:, :, 2] = recon
                cell.imshow(image)
        plt.show()


def compute_weighted_auc(anomalies: np.array, scores: np.array):
    n_anomalies = anomalies.sum().item()
    n_normals = len(anomalies) - n_anomalies
    weights = np.zeros_like(scores)
    weights[anomalies == 0] = n_anomalies / len(anomalies)
    weights[anomalies == 1] = n_normals / len(anomalies)
    return roc_auc_score(anomalies, scores, sample_weight=weights)


def compute_all_aucs(anomalies, scores):
    total_auc = roc_auc_score(anomalies, scores)

    n_anomalies = anomalies.sum().item()
    n_normals = len(anomalies) - n_anomalies
    weighted_auc = compute_weighted_auc(anomalies, scores)

    min_label = 1 if n_anomalies < n_normals else 0
    max_label = 1 - min_label
    n_per_class = n_anomalies if n_anomalies < n_normals else n_normals
    idxs_norm = np.where(anomalies == min_label)[0]
    idxs_anom = np.random.choice(np.where(anomalies == max_label)[0], size=n_per_class, replace=False)
    idxs = np.concatenate((idxs_norm, idxs_anom))
    balanced_auc = roc_auc_score(anomalies[idxs], scores[idxs])

    return total_auc, weighted_auc, balanced_auc


def compute_exp_metrics(exp: dict, per_task=True):
    knowledge = []
    metrics = pd.DataFrame(index=exp['results'],
                           columns=[str(labels) for labels in exp['knowledge'].values()] + ['total'], dtype='float')
    for t, task in enumerate(exp['results']):
        knowledge.extend(exp['knowledge'][task])
        targets = np.array(exp['results'][task]['targets'])
        scores = np.array(exp['results'][task]['rec_errs'])
        anomalies = (~np.isin(targets, knowledge)).astype(int)
        auc = compute_weighted_auc(anomalies, scores)
        metrics.loc[task, 'total'] = auc
        # print(f'task {task}: {auc}')

        if t > 0 and per_task:
            for in_t, in_task in zip(range(t + 1), exp['results']):
                np_knowledge = np.array(knowledge)
                excluded_labels = np_knowledge[~np.isin(np_knowledge, exp['knowledge'][in_task])].tolist()
                mask = ~np.isin(targets, excluded_labels)
                in_targets = targets[mask]
                in_scores = scores[mask]
                in_anomalies = (~np.isin(in_targets, knowledge)).astype(int)
                in_auc = compute_weighted_auc(in_anomalies, in_scores)
                metrics.loc[task, str(exp['knowledge'][in_task])] = in_auc
                # print(f'  t{in_task} vs all: {in_auc}')
        else:
            metrics.loc[task, str(exp['knowledge'][task])] = auc

    final_auc = metrics.loc[task, 'total']
    average_auc = metrics.loc[:, "total"].mean()
    # print(f'final {final_auc} average {average_auc}')
    return final_auc, average_auc, metrics


def show_aucs_per_task(task_aucs: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(task_aucs, annot=True, ax=ax, cmap='Reds', cbar=False)
    plt.ylabel('Task')
    plt.xlabel('Class')
    plt.title('Auc per task and class')


def reconstruction_confusion_matrix(exp: dict):
    labels = np.unique(np.array(next(iter(exp['results'].values()))['targets'])).tolist()
    indexes = [key + str(exp['knowledge'][key]) for key in exp['knowledge']]
    matrix = pd.DataFrame(index=indexes,
                          columns=labels, dtype='float')

    for idx, task in zip(indexes, exp['results']):
        scores = np.array(exp['results'][task]['rec_errs'])
        targets = np.array(exp['results'][task]['targets'])
        for label in labels:
            matrix.loc[idx, label] = scores[targets == label].mean()

    return matrix


def show_conf_matrix(cmatrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data=cmatrix, ax=ax, annot=True, cbar=False, cmap='Reds')
    plt.ylabel('Task')
    plt.xlabel('Class')
    plt.title('Reconstruction error per task and class')


def analyze_experiment(exp: dict):
    ## print metrics of experiments

    print_exp_info(exp)

    # show_exp_images(exp, False)

    final_auc, average_auc, task_aucs = compute_exp_metrics(exp)
    # print(f'final {final_auc} average {average_auc}')
    show_aucs_per_task(task_aucs)

    cmatrix = reconstruction_confusion_matrix(exp)
    show_conf_matrix(cmatrix)

    # task_aucs.to_dict(orient='index')
    # pd.DataFrame.from_dict(task_aucs.to_dict(orient='index'), orient='index')

    show_exp_images(exp, False)
