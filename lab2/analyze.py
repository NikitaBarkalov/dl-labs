import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from pathlib import Path
from thresholds import (
    compute_metrics,
    pick_best_threshold,
    run_unit_tests,
    search_threshold_fast,
)

def load_saved_data(results_dir):
    archive_path = results_dir / 'predictions.npz'
    
    with np.load(archive_path, allow_pickle = False) as archive:
        return {name: archive[name] for name in archive.files}

def plot_validation_curves(y_val, s_val, out_dir):
    fpr, tpr, _ = roc_curve(y_val, s_val)
    roc_auc_val = float(auc(fpr, tpr))

    prec, rec, _ = precision_recall_curve(y_val, s_val)
    pr_auc_val = float(auc(rec, prec))

    fig, axes = plt.subplots(1, 2, figsize = (14, 6))

    axes[0].plot(fpr, tpr, color = 'blue', lw = 2)
    axes[0].plot([0, 1], [0, 1], color = 'gray', linestyle = '--', lw = 1)
    axes[0].set_xlabel('FPR', fontsize = 14)
    axes[0].set_ylabel('TPR', fontsize = 14)
    axes[0].set_title('ROC-curve', fontsize = 16)
    axes[0].grid(True, alpha = 0.6)

    axes[1].plot(rec, prec, color = 'green', lw = 2)
    axes[1].set_xlabel('Recall', fontsize = 14)
    axes[1].set_ylabel('Precision', fontsize = 14)
    axes[1].set_title('PR-curve', fontsize = 16)
    axes[1].grid(True, alpha = 0.6)

    fig.tight_layout()
    fig.savefig(out_dir / 'validation_curves.png', dpi = 300)

    return roc_auc_val, pr_auc_val

def evaluate_specific_threshold(scores, y, t):
    preds = scores >= t
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))

    metrics = compute_metrics(tp, fp, fn, tn)
    metrics['threshold'] = t

    return metrics

def extract_error_cases(row_ids, y, scores, amounts, t_star, n_cases):
    preds = scores >= t_star
    records = []

    fp_mask = (preds == 1) & (y == 0)
    fp_indices = np.where(fp_mask)[0]
    fp_order = fp_indices[np.argsort(row_ids[fp_indices])][:n_cases]

    for idx in fp_order:
        records.append(
            {
                'csv_row': int(row_ids[idx]),
                'error_type': 'FP',
                'label': int(y[idx]),
                'score': float(scores[idx]),
                'distance_to_threshold': float(abs(scores[idx] - t_star)),
                'amount': float(amounts[idx]),
            }
        )

    fn_mask = (preds == 0) & (y == 1)
    fn_indices = np.where(fn_mask)[0]
    fn_order = fn_indices[np.argsort(row_ids[fn_indices])][:n_cases]

    for idx in fn_order:
        records.append(
            {
                'csv_row': int(row_ids[idx]),
                'error_type': 'FN',
                'label': int(y[idx]),
                'score': float(scores[idx]),
                'distance_to_threshold': float(abs(scores[idx] - t_star)),
                'amount': float(amounts[idx]),
            }
        )

    return pd.DataFrame(records)

def run_full_analysis(results_dir):
    results_dir = Path(results_dir)
    data = load_saved_data(results_dir)

    val_y = data['validation_y']
    val_scores = data['validation_scores']
    test_y = data['test_y']
    test_scores = data['test_scores']
    test_amounts = data['test_amount']
    test_row_ids = data['test_row_ids']

    candidates_df = search_threshold_fast(val_scores, val_y)
    candidates_csv_path = results_dir / 'candidates.csv'
    candidates_df.to_csv(candidates_csv_path, index = False)

    t_star, best_val_row = pick_best_threshold(candidates_df)

    std_metrics = evaluate_specific_threshold(val_scores, val_y, 0.5)
    all_neg_metrics = evaluate_specific_threshold(val_scores, val_y, np.inf)

    print('\nvalidation:')
    print(f't = 0.5: TP = {std_metrics['TP']}, FP = {std_metrics['FP']}, FN = {std_metrics['FN']}, TN = {std_metrics['TN']}, C = {std_metrics['C']}')
    print(f't* = {t_star}: TP = {int(best_val_row['TP'])}, FP = {int(best_val_row['FP'])}, FN = {int(best_val_row['FN'])}, TN = {int(best_val_row['TN'])}, C = {int(best_val_row['C'])}')
    print(f't = inf: TP = {all_neg_metrics['TP']}, FP = {all_neg_metrics['FP']}, FN = {all_neg_metrics['FN']}, TN = {all_neg_metrics['TN']}, C = {all_neg_metrics['C']}')

    roc_auc_val, pr_auc_val = plot_validation_curves(val_y, val_scores, results_dir)
    print(f'ROC-AUC = {roc_auc_val}')
    print(f'PR-AUC = {pr_auc_val}\n')

    unit_results = run_unit_tests()
    for name, res in unit_results.items():
        print(f'[{name}]:')
        for _, row in res['fast_df'].iterrows():
            t_val = '+inf' if np.isinf(row['threshold']) else f'{row['threshold']}'
            print(f't = {t_val}: TP = {int(row['TP'])}, FP = {int(row['FP'])}, FN = {int(row['FN'])}, TN = {int(row['TN'])}, C = {int(row['C'])}')
        t_star_val = '+inf' if np.isinf(res['t_star']) else f'{res['t_star']}'
        print(f't* = {t_star_val}, C = {res['min_cost']}\n')

    test_metrics_50 = evaluate_specific_threshold(test_scores, test_y, 0.5)
    test_metrics_star = evaluate_specific_threshold(test_scores, test_y, t_star)

    print('test:')
    print(f't = 0.5: TP = {test_metrics_50['TP']}, FP = {test_metrics_50['FP']}, FN = {test_metrics_50['FN']}, TN = {test_metrics_50['TN']}, C = {test_metrics_50['C']}, precision = {test_metrics_50['precision']:.4f}, recall = {test_metrics_50['recall']:.4f}')
    print(f't* = {t_star:.6f}: TP = {test_metrics_star['TP']}, FP = {test_metrics_star['FP']}, FN = {test_metrics_star['FN']}, TN = {test_metrics_star['TN']}, C = {test_metrics_star['C']}, precision = {test_metrics_star['precision']:.4f}, recall = {test_metrics_star['recall']:.4f}\n')

    errors_df = extract_error_cases(test_row_ids, test_y, test_scores, test_amounts, t_star, 5)
    errors_csv_path = results_dir / 'error_cases.csv'
    errors_df.to_csv(errors_csv_path, index = False)

    print('FP:')
    for _, row in errors_df[errors_df['error_type'] == 'FP'].iterrows():
        print(f'csv_row = {int(row['csv_row'])}, y = {int(row['label'])}, s = {row['score']:.6f}, |s - t*| = {row['distance_to_threshold']:.6f}, amount = {row['amount']:.2f}')

    print('\nFN:')
    for _, row in errors_df[errors_df['error_type'] == 'FN'].iterrows():
        print(f'csv_row = {int(row['csv_row'])}, y = {int(row['label'])}, s = {row['score']:.6f}, |s - t*| = {row['distance_to_threshold']:.6f}, amount = {row['amount']:.2f}')