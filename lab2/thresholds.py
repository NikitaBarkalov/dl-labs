import numpy as np
import pandas as pd

def compute_metrics(tp, fp, fn, tn):
    cost = 10 * fn + fp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'C': cost,
        'precision': precision,
        'recall': recall,
    }

def search_threshold_fast(scores, y):
    total_p = int(np.sum(y == 1))
    total_n = int(len(y) - total_p)

    order = np.argsort(scores, descending = True)
    s_sorted = scores[order]
    y_sorted = y[order]

    distinct_mask = np.append(s_sorted[:-1] != s_sorted[1:], True)
    distinct_idx = np.where(distinct_mask)[0]

    cum_tp = np.cumsum(y_sorted)[distinct_idx]
    cum_fp = np.cumsum(1 - y_sorted)[distinct_idx]
    unique_thresholds = s_sorted[distinct_idx]

    thresholds = np.concatenate(([np.inf], unique_thresholds))
    tp = np.concatenate(([0], cum_tp))
    fp = np.concatenate(([0], cum_fp))

    fn = total_p - tp
    tn = total_n - fp
    cost = 10 * fn + fp

    return pd.DataFrame(
        {
            'threshold': thresholds,
            'TP': tp.astype(int),
            'FP': fp.astype(int),
            'FN': fn.astype(int),
            'TN': tn.astype(int),
            'C': cost.astype(int)
        }
    )

def search_threshold_slow(scores, y):
    unique_scores = sorted(np.unique(scores), reverse = True)
    candidates = [np.inf] + unique_scores

    records = []
    for t in candidates:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        tn = int(np.sum((preds == 0) & (y == 0)))
        c = 10 * fn + fp
        records.append({'threshold': t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'C': c})

    return pd.DataFrame(records)

def pick_best_threshold(candidates_df):
    min_cost = candidates_df['C'].min()
    min_rows = candidates_df[candidates_df['C'] == min_cost]
    best_threshold = float(min_rows['threshold'].max())
    best_row = min_rows[min_rows['threshold'] == best_threshold].iloc[0]

    return best_threshold, best_row

def get_small_sets():
    s_a = np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], dtype = np.float64)
    y_a = np.array([0, 1, 0, 1, 0, 0], dtype = np.int64)

    s_b = np.array([0.9, 0.6, 0.3], dtype = np.float64)
    y_b = np.array([0, 0, 0], dtype = np.int64)

    s_c = np.array([0.8] + [0.5] * 11 + [0.2], dtype = np.float64)
    y_c = np.array([1, 1] + [0] * 10 + [0], dtype = np.int64)

    return {'A': (s_a, y_a), 'B': (s_b, y_b), 'C': (s_c, y_c)}

def run_unit_tests():
    sets = get_small_sets()
    results = {}

    for name, (scores, y) in sets.items():
        fast_df = search_threshold_fast(scores, y)
        slow_df = search_threshold_slow(scores, y)
        t_star, row = pick_best_threshold(fast_df)

        results[name] = {
            't_star': t_star,
            'min_cost': int(row['C']),
            'fast_df': fast_df,
            'slow_df': slow_df,
        }

    return results