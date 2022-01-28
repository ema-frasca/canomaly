import torch
from torch.functional import F

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def reconstruction_error(input_image: torch.Tensor, reconstruction: torch.Tensor):
    """
    Compute the reconstruction error on a batch of images
    """
    return F.mse_loss(input_image, reconstruction, reduction='none').sum((-3, -2, -1))


def reconstruction_confusion_matrix(log: dict):
    labels = np.unique(np.array(next(iter(log['results'].values()))['targets'])).tolist()
    indexes = [key + str(log['knowledge'][key]) for key in log['knowledge']]
    matrix = pd.DataFrame(index=indexes,
                          columns=labels, dtype='float')

    for idx, task in zip(indexes, log['results']):
        scores = np.array(log['results'][task]['rec_errs'])
        targets = np.array(log['results'][task]['targets'])
        for label in labels:
            matrix.loc[idx, label] = scores[targets == label].mean()

    return matrix.to_dict(orient='index')


def compute_weighted_auc(anomalies: np.array, scores: np.array):
    n_anomalies = anomalies.sum().item()
    n_normals = len(anomalies) - n_anomalies
    weights = np.zeros_like(scores)
    weights[anomalies == 0] = n_anomalies/len(anomalies)
    weights[anomalies == 1] = n_normals/len(anomalies)
    return roc_auc_score(anomalies, scores, sample_weight=weights)


def compute_task_auc(targets: list[int], scores: list[float], knowledge: list[int]):
    targets = np.array(targets)
    scores = np.array(scores)
    anomalies = np.logical_not(np.isin(targets, knowledge)).astype(int)
    return compute_weighted_auc(anomalies, scores)


def compute_exp_metrics(log: dict, per_task=True):
    knowledge = []
    metrics = pd.DataFrame(index=log['results'],
                           columns=[str(labels) for labels in log['knowledge'].values()] + ['total'], dtype='float')
    for t, task in enumerate(log['results']):
        knowledge.extend(log['knowledge'][task])
        targets = np.array(log['results'][task]['targets'])
        scores = np.array(log['results'][task]['rec_errs'])
        anomalies = (~np.isin(targets, knowledge)).astype(int)
        auc = compute_weighted_auc(anomalies, scores)
        metrics.loc[task, 'total'] = auc
        # print(f'task {task}: {auc}')

        if t > 0 and per_task:
            for in_t, in_task in zip(range(t+1), log['results']):
                np_knowledge = np.array(knowledge)
                excluded_labels = np_knowledge[~np.isin(np_knowledge, log['knowledge'][in_task])].tolist()
                mask = ~np.isin(targets, excluded_labels)
                in_targets = targets[mask]
                in_scores = scores[mask]
                in_anomalies = (~np.isin(in_targets, knowledge)).astype(int)
                in_auc = compute_weighted_auc(in_anomalies, in_scores)
                metrics.loc[task, str(log['knowledge'][in_task])] = in_auc
                # print(f'  t{in_task} vs all: {in_auc}')
        else:
            metrics.loc[task, str(log['knowledge'][task])] = auc

    final_auc = metrics.loc[task, 'total']
    average_auc = metrics.loc[:, "total"].mean()
    # print(f'final {final_auc} average {average_auc}')
    return final_auc, average_auc, metrics.to_dict(orient='index')
