import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from ddpmMed.utils.helpers import read_json_metrics


# directory = r"F:\diffusion\128x128 model\results"
# results = {}


def time_experiment(directory: str, metrics: list, labels: list):
    """
    """
    # define results dictionary
    results = {}

    # go over all directories
    for root, dir_name, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1].lower() == 'json':

                # key for current results entry
                key = os.path.basename(os.path.dirname(os.path.join(root)))
                key = key.split('-')[-1].strip()

                # results
                values = read_json_metrics(path=os.path.join(root, file), metrics=metrics, labels=labels)

                if key in results.keys():
                    results[key].append(values)
                else:
                    results[key] = [values]

    return results


metrics = ['dice', 'hd95', 'jaccard']
labels = ['TC', 'IT', 'ET']

results = time_experiment(directory=r"F:\diffusion\128x128 model\results", metrics=metrics, labels=labels)
sorted_keys = list(results.keys())
sorted_keys.sort(key=lambda x: int(x.split('t')[-1]))
mean_results = {k: {} for k in sorted_keys}

for key, entries in results.items():
    for entry in entries:
        for metric, values in entry.items():
            for label, scores in values.items():
                if len(list(mean_results[key].keys())) == 0:
                    mean_results[key] = {l: {m: [] for m in metrics} for l in labels}

                mean_results[key][label][metric].append(scores)

all_scores = {
    'TC': {
        'dice': [],
        'hd95': [],
        'jaccard': []
    },
    'IT': {
        'dice': [],
        'hd95': [],
        'jaccard': []
    },
    'ET': {
        'dice': [],
        'hd95': [],
        'jaccard': []
    }
}

for exp, entries in mean_results.items():
    for label, values in entries.items():
        for metric, value in values.items():
            mean_results[exp][label][metric] = np.mean(mean_results[exp][label][metric])
            print(exp, label, metric)
            all_scores[label][metric].append(mean_results[exp][label][metric])


fig, ax = plt.subplots(1, 3, figsize=(35, 8))
titles = ["Dice score", "Hausdorff 95 distance", "Jaccard score"]
legend_labels = {'TC': "Tumor Core", 'IT': "Invaded Tissue", 'ET': "Enhancing Tumor"}
markers = ["x", "^", "d"]

for i, metric in enumerate(metrics):
    for j, label in enumerate(labels):

        if metric == "hd95":
            ax[i].set_ylim([2, 10])
        elif metric == "jaccard":
            ax[i].set_ylim([0.25, 0.75])
        ax[i].plot(sorted_keys, all_scores[label][metric])
        ax[i].scatter(sorted_keys, all_scores[label][metric], label=legend_labels[label], marker=markers[j])
        ax[i].set_xlabel("Time step", fontsize=16)
        ax[i].set_ylabel("Score", fontsize=16)
        ax[i].legend(fontsize=12)
    ax[i].set_title(f"{titles[i]} vs. time", fontsize=16)

plt.show()
