import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Plot metrics from input")
parser.add_argument('--metric',choices=['Precision', 'Recall', 'F1-Score', 'all'], default='all',help="Metric to plot (default: all)")
parser.add_argument('--input', type=argparse.FileType('r'), nargs='+', required=True, help="Paths to input files (tab-separated format)")
args = parser.parse_args()


def process_file(input_file):
    scores = defaultdict(lambda: {'Recall': [], 'Precision': [], 'F1-Score': []})
    for line in input_file:
        line = line.strip()
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        key = parts[0]
        try:
            scores[key]['Recall'].append(float(parts[2]))
            scores[key]['Precision'].append(float(parts[1]))
            scores[key]['F1-Score'].append(float(parts[3]))
        except ValueError:
            print(f"Skipping line due to invalid float: {line}")
    return scores


all_scores = []
for file in args.input:
    scores = process_file(file)
    baseline = scores.pop('avg_segments')
    all_scores.append((scores, baseline))


metric_titles = {
    'Recall': 'Recall',
    'Precision': 'Precision',
    'F1-Score': 'F-score'
}
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray']
linestyles = ['-', '--', ':', '-.']

metrics_to_plot = (
    ['Recall', 'Precision', 'F1-Score']
    if args.metric == 'all' else [args.metric]
)

fig, axes = plt.subplots(
    3, 3, 
    sharex=True, #figsize=(15,5)
    figsize=(18, 15)
)  
axes = axes.flatten()  
lang_map = {'ces': 'cs', 'deu': 'de', 'eng': 'en', 'nld': 'nl', 'fin':'fn', 'hbs':'hb', 'hye':'hy' ,'kan':'kn', 'slk':'sk'}
for file_idx, (scores, baseline) in enumerate(all_scores):
    file_label = args.input[file_idx].name.split('/')[-1].split('.')[0].split('-')[0]

    for metric_idx, metric in enumerate(metrics_to_plot):
        ax_idx = file_idx * len(metrics_to_plot) + metric_idx
        if ax_idx >= len(axes):
            continue
        ax = axes[ax_idx]

        scores_by_type = {
            'Recall': defaultdict(lambda: defaultdict(dict)),
            'Precision': defaultdict(lambda: defaultdict(dict)),
            'F1-Score': defaultdict(lambda: defaultdict(dict))
        }

        for key, values in scores.items():
            threshold = float(key.split('-')[-1])
            mode = key.split('-')[-2]
            type_ = key.split('-')[-3]
            for m in metrics_to_plot:
                scores_by_type[m][type_][mode][threshold] = values[m][0]

        for ti, (type_, modes_dict) in enumerate(scores_by_type[metric].items()):
            for mi, (mode, threshold_dict) in enumerate(modes_dict.items()):
                sorted_items = sorted(threshold_dict.items())
                thresholds = [t for t, _ in sorted_items]
                values = [v for _, v in sorted_items]
                label = f'{type_}-{mode}'
                line, = ax.plot(
                    thresholds, values,
                    marker='o',
                    label=label,
                    color=colors[ti % len(colors)],
                    linestyle=linestyles[mi % len(linestyles)]
                )
            

        if baseline:
            ax.axhline(baseline[metric][0], linewidth=2.5, linestyle='--', color='red', label='Baseline')

        ax.set_title(f'{lang_map[file_label]}')
      
        ax.grid(True)

plt.legend(
    bbox_to_anchor=(1.05, 1.05),
    loc='best',
    fontsize=12,
    frameon=False
)

fig.supxlabel('Threshold', fontsize=12)  
fig.supylabel(args.metric, fontsize=12)  
plt.tight_layout()

plt.show()