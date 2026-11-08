import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image
from dataset import (
    CLASSES,
    IntelDataset,
    get_file_sha256,
    get_transforms,
    resolve_data_paths
)
from model import build_model_and_optimizer

def plot_training_curves(runs_data, out_path):
    epochs = list(range(1, 9))
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    colors = {
        'scratch': ['gray'],
        'frozen': ['darkblue', 'royalblue', 'deepskyblue'],
        'unfreeze': ['darkgreen', 'forestgreen', 'limegreen']
    }

    for item in runs_data:
        r = item['regime']
        s = item['seed']
        c = colors[r][s - 1]

        axes[0].plot(epochs, item['train_losses'], label = f'{r} (seed = {s})', color = c, marker = 'o', lw = 1.5)
        axes[1].plot(epochs, item['val_macro_f1s'], label = f'{r} (seed = {s})', color = c, marker = 's', lw = 1.5)

    axes[0].set_xlabel('Епоха', fontsize = 12)
    axes[0].set_ylabel('Втрата на train', fontsize = 12)
    axes[0].set_title('Втрата на навчальній вибірці за епохами', fontsize = 13)
    axes[0].grid(True, alpha = 0.5)
    axes[0].legend(fontsize = 9)

    axes[1].set_xlabel('Епоха', fontsize = 12)
    axes[1].set_ylabel('Macro-F1 на validation', fontsize = 12)
    axes[1].set_title('Macro-F1 на валідаційній вибірці за епохами', fontsize = 13)
    axes[1].grid(True, alpha = 0.5)
    axes[1].legend(fontsize = 9)

    plt.tight_layout()
    plt.savefig(out_path, dpi = 200)
    plt.close(fig)

def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize = (8, 7))
    im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    ax.figure.colorbar(im, ax = ax)

    ax.set(
        xticks = np.arange(cm.shape[1]), 
        yticks = np.arange(cm.shape[0]), 
        xticklabels = classes, 
        yticklabels = classes, 
        ylabel = 'Справжній клас', 
        xlabel = 'Передбачений клас'
    )

    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, format(val, 'd'), ha = 'center', va = 'center', color = color, fontsize = 11)

    ax.set_title('Матриця плутанини на тестовій вибірці', fontsize = 14)
    plt.tight_layout()
    plt.savefig(out_path, dpi = 200)
    plt.close(fig)

def plot_error_cases(error_samples, out_path):
    fig, axes = plt.subplots(1, len(error_samples), figsize = (16, 5))
    if len(error_samples) == 1:
        axes = [axes]

    for idx, sample in enumerate(error_samples):
        img = Image.open(sample['path']).convert('RGB')
        axes[idx].imshow(img)

        f_name = sample['file']
        t_cls = sample['true_class']
        p_cls = sample['pred_class']
        p_val = sample['prob']

        title_text = f'{f_name}\nTrue: {t_cls}\nPred: {p_cls}\nProb: {p_val:.3f}'
        axes[idx].set_title(title_text, fontsize = 9)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi = 200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type = str, required = True)
    parser.add_argument('--out-dir', type = str, required = True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {args.data_dir}')

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f'Output directory not found: {args.out_dir}')

    split_csv = base_dir / 'split.csv'
    if not split_csv.exists():
        raise FileNotFoundError(f'split.csv not found: {split_csv}')

    split_sha256 = get_file_sha256(split_csv)
    print('split check:')
    print(f'split_sha256: {split_sha256}')
    print()

    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents = True, exist_ok = True)

    model_path = out_dir / 'lab3_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    runs_order = [
        ('scratch', 1),
        ('frozen', 1),
        ('frozen', 2),
        ('frozen', 3),
        ('unfreeze', 1),
        ('unfreeze', 2),
        ('unfreeze', 3)
    ]

    runs_data = []
    for r, s in runs_order:
        run_file = out_dir / f'{r}_seed{s}' / 'run.json'
        if not run_file.exists():
            raise FileNotFoundError(f'Run file not found: {run_file}')

        with open(run_file, 'r', encoding = 'utf-8') as f:
            data = json.load(f)

        if data['split_sha256'] != split_sha256:
            raise ValueError(f'Mismatch split_sha256 for {r}_seed{s}')

        runs_data.append(data)

    print('training summary:')
    print('no | regime     | seed  | train_loss   | val_f1       | time_s  ')
    for idx, r_data in enumerate(runs_data, 1):
        print(f'{idx:<2d} | {r_data['regime']:10s} | {r_data['seed']:<5d} | {r_data['train_losses'][-1]:<12.4f} | {r_data['val_macro_f1s'][-1]:<12.4f} | {r_data['training_time_seconds']:<8.2f}')

    print()
    training_curves_path = figures_dir / 'training_curves.png'
    plot_training_curves(runs_data, training_curves_path)

    f_scratch = runs_data[0]['val_macro_f1s'][-1]
    f_frozen_vals = [d['val_macro_f1s'][-1] for d in runs_data if d['regime'] == 'frozen']
    f_unfreeze_vals = [d['val_macro_f1s'][-1] for d in runs_data if d['regime'] == 'unfreeze']

    f_mean_frozen = sum(f_frozen_vals) / len(f_frozen_vals)
    f_mean_unfreeze = sum(f_unfreeze_vals) / len(f_unfreeze_vals)

    r_frozen = max(f_frozen_vals) - min(f_frozen_vals)
    r_unfreeze = max(f_unfreeze_vals) - min(f_unfreeze_vals)

    noise_threshold = max(r_frozen, r_unfreeze)
    delta_pre = f_mean_frozen - f_scratch
    delta_unf = f_mean_unfreeze - f_mean_frozen

    print('effect decomposition:')
    print(f'F_bar_frozen: {f_mean_frozen:.4f}')
    print(f'R_frozen: {r_frozen:.4f}')
    print(f'F_bar_unfreeze: {f_mean_unfreeze:.4f}')
    print(f'R_unfreeze: {r_unfreeze:.4f}')
    print(f'R: {noise_threshold:.4f}')
    print(f'delta_pre: {delta_pre:.4f}')
    print(f'delta_unf: {delta_unf:.4f}')
    print(f'delta_total: {delta_pre + delta_unf:.4f}')

    verdict_pre = 'worth_checking' if abs(delta_pre) > noise_threshold else 'no_effect'
    verdict_unf = 'worth_checking' if abs(delta_unf) > noise_threshold else 'no_effect'
    print(f'verdict_pre: {verdict_pre}')
    print(f'verdict_unf: {verdict_unf}')
    print()

    regime_means = {
        'scratch': f_scratch,
        'frozen': f_mean_frozen,
        'unfreeze': f_mean_unfreeze
    }

    expected_best_regime = max(regime_means.keys(), key = lambda k: regime_means[k])
    best_candidates = [d for d in runs_data if d['regime'] == expected_best_regime]
    best_candidate_run = max(best_candidates, key = lambda d: d['val_macro_f1s'][-1])
    expected_best_seed = best_candidate_run['seed']

    ckpt = torch.load(model_path, map_location = 'cpu', weights_only = False)
    if ckpt['regime'] != expected_best_regime or ckpt['seed'] != expected_best_seed:
        raise ValueError('Model checkpoint does not match best validation run')

    print('selected model:')
    print(f'selected_model: {expected_best_regime}_seed{expected_best_seed} (val_f1={best_candidate_run['val_macro_f1s'][-1]:.4f})')
    print()

    device_cpu = torch.device('cpu')
    model, _ = build_model_and_optimizer(ckpt['regime'], seed = ckpt['seed'], device = device_cpu)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    _, eval_tf = get_transforms()
    test_ds = IntelDataset(split_csv, 'test', data_dir, eval_tf)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size = 64, shuffle = False, drop_last = False, num_workers = 0)

    test_preds = []
    test_targets = []
    all_logits = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device_cpu)
            logits = model(inputs)
            preds = torch.argmax(logits, dim = 1)

            all_logits.append(logits)
            test_preds.extend(preds.numpy())
            test_targets.extend(targets.numpy())

    test_logits = torch.cat(all_logits, dim = 0)
    test_macro_f1 = float(f1_score(test_targets, test_preds, average = 'macro'))
    test_acc = float(np.mean(np.array(test_targets) == np.array(test_preds)))

    n_test = len(test_targets)
    se_acc = math.sqrt(test_acc * (1.0 - test_acc) / n_test)

    print('test evaluation:')
    print(f'test_macro_f1: {test_macro_f1:.4f}')
    print(f'test_acc: {test_acc:.4f}')
    print(f'test_acc_se: {se_acc:.4f}')
    print()

    cm = confusion_matrix(test_targets, test_preds)
    cm_path = figures_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, CLASSES, cm_path)

    print('confusion matrix:')
    cols_header = ' | '.join(f'{c[:6]:>6s}' for c in CLASSES)
    print(f'true \\ pred  | {cols_header}')

    for i, cls in enumerate(CLASSES):
        row_str = ' | '.join(f'{cm[i, j]:>6d}' for j in range(6))
        print(f'{cls:12s} | {row_str}')

    print()
    probs = torch.softmax(test_logits, dim = 1).numpy()

    off_diag_pairs = []
    for i in range(6):
        for j in range(6):
            if i != j and cm[i, j] > 0:
                off_diag_pairs.append((cm[i, j], i, j))

    off_diag_pairs.sort(key = lambda item: (-item[0], item[1], item[2]))
    top_pairs = off_diag_pairs[:5]

    error_cases = []
    for count, i, j in top_pairs:
        matching_indices = [idx for idx, (t, p) in enumerate(zip(test_targets, test_preds)) if t == i and p == j]
        matching_samples = []

        for idx in matching_indices:
            path, _, fname = test_ds.samples[idx]
            matching_samples.append((fname, path, probs[idx, j]))

        matching_samples.sort(key = lambda s: s[0])
        first_sample = matching_samples[0]

        error_cases.append({
            'file': first_sample[0],
            'path': first_sample[1],
            'true_class': CLASSES[i],
            'pred_class': CLASSES[j],
            'prob': float(first_sample[2]),
            'error_count': count
        })

    print('top error cases:')
    print('file       | true_class   | pred_class   | prob     | count ')
    for err in error_cases:
        print(f'{err['file']:10s} | {err['true_class']:12s} | {err['pred_class']:12s} | {err['prob']:<8.4f} | {err['error_count']:<6d}')

    error_cases_path = figures_dir / 'error_cases.png'
    plot_error_cases(error_cases, error_cases_path)

if __name__ == '__main__':
    main()
