import argparse
import json
import platform
import subprocess
import time
import numpy as np
import torch
import torchvision

from pathlib import Path
from sklearn.metrics import f1_score
from checks import run_all_checks
from dataset import (
    CLASSES,
    IntelDataset,
    get_file_sha256,
    get_transforms
)
from model import (
    build_model_and_optimizer,
    set_train_mode
)

EPOCHS = 8

def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr = subprocess.DEVNULL).decode('utf-8').strip()
        status_output = subprocess.check_output(['git', 'status', '--porcelain', '-u'], stderr = subprocess.DEVNULL).decode('utf-8')
        dirty = False
        for line in status_output.splitlines():
            if len(line) < 4:
                continue
            path_str = line[3:].strip().strip('"\'')
            if path_str.endswith('.py'):
                dirty = True
                break
        return commit, dirty
    except Exception:
        return 'unknown', False

def train_single_run(regime, seed, split_csv, data_dir, device, runs_dir, epochs = EPOCHS):
    run_dir = Path(runs_dir) / f'{regime}_seed{seed}'
    run_dir.mkdir(parents = True, exist_ok = True)

    train_tf, eval_tf = get_transforms()
    train_ds = IntelDataset(split_csv, 'train', data_dir, train_tf)
    val_ds = IntelDataset(split_csv, 'validation', data_dir, eval_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = 32, shuffle = True, drop_last = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size = 64, shuffle = False, drop_last = False, num_workers = 0)

    model, optimizer = build_model_and_optimizer(regime, seed, device)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_f1s = []
    final_val_logits = None

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        set_train_mode(model, regime)

        total_epoch_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            b_size = inputs.size(0)
            total_epoch_loss += loss.item() * b_size
            total_samples += b_size

        epoch_loss = total_epoch_loss / total_samples
        train_losses.append(epoch_loss)

        model.eval()
        epoch_val_preds = []
        epoch_val_targets = []
        epoch_val_logits = [] if epoch == epochs else None

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                logits = model(val_inputs)
                preds = torch.argmax(logits, dim = 1)

                if epoch == epochs:
                    epoch_val_logits.append(logits.cpu().numpy())

                epoch_val_preds.extend(preds.cpu().numpy())
                epoch_val_targets.extend(val_targets.numpy())

        epoch_f1 = float(f1_score(epoch_val_targets, epoch_val_preds, average = 'macro'))
        val_f1s.append(epoch_f1)
        print(f'  epoch {epoch}/{epochs} | train_loss: {epoch_loss:.4f} | val_f1: {epoch_f1:.4f}')

        if epoch == epochs:
            final_val_logits = np.concatenate(epoch_val_logits, axis = 0)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    duration = time.time() - t_start
    np.save(run_dir / 'val_logits.npy', final_val_logits.astype(np.float32))

    commit_hash, dirty = get_git_info()
    split_hash = get_file_sha256(split_csv)

    t_crop = train_tf.transforms[0]
    t_flip = train_tf.transforms[1]
    t_norm = train_tf.transforms[3]
    v_resize = eval_tf.transforms[0]

    opt_pg0 = optimizer.param_groups[0]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    run_info = {
        'regime': regime,
        'seed': seed,
        'git_commit': commit_hash,
        'git_dirty': dirty,
        'split_sha256': split_hash,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else ('Apple MPS' if device.type == 'mps' else 'CPU'),
        'img_size': t_crop.size[0] if isinstance(t_crop.size, (list, tuple)) else t_crop.size,
        'train_crop_scale': list(t_crop.scale),
        'train_crop_ratio': list(t_crop.ratio),
        'train_crop_interpolation': str(t_crop.interpolation).split('.')[-1].lower(),
        'train_horizontal_flip_p': t_flip.p,
        'normalize_mean': list(t_norm.mean),
        'normalize_std': list(t_norm.std),
        'val_resize_size': list(v_resize.size) if isinstance(v_resize.size, (list, tuple)) else [v_resize.size, v_resize.size],
        'val_resize_interpolation': str(v_resize.interpolation).split('.')[-1].lower(),
        'architecture': model.__class__.__name__.lower(),
        'weights': 'None' if regime == 'scratch' else 'ResNet18_Weights.IMAGENET1K_V1',
        'num_classes': model.fc.out_features,
        'fc_in_features': model.fc.in_features,
        'fc_out_features': model.fc.out_features,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'optimizer': optimizer.__class__.__name__,
        'weight_decay': opt_pg0['weight_decay'],
        'learning_rate_head': opt_pg0['lr'],
        'learning_rate_backbone': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None,
        'adamw_betas': list(opt_pg0['betas']),
        'adamw_eps': opt_pg0['eps'],
        'adamw_amsgrad': opt_pg0['amsgrad'],
        'bn_momentum': model.bn1.momentum,
        'bn_eps': model.bn1.eps,
        'batch_size_train': train_loader.batch_size,
        'batch_size_val': val_loader.batch_size,
        'shuffle_train': isinstance(train_loader.sampler, torch.utils.data.RandomSampler),
        'shuffle_val': isinstance(val_loader.sampler, torch.utils.data.RandomSampler),
        'num_workers': train_loader.num_workers,
        'drop_last': train_loader.drop_last,
        'pin_memory': train_loader.pin_memory,
        'loss_fn': criterion.__class__.__name__,
        'loss_reduction': criterion.reduction,
        'label_smoothing': criterion.label_smoothing,
        'epochs': epochs,
        'steps_per_epoch': len(train_loader),
        'total_optimizer_steps': len(train_loader) * epochs,
        'train_losses': train_losses,
        'val_macro_f1s': val_f1s,
        'training_time_seconds': duration
    }

    with open(run_dir / 'run.json', 'w', encoding = 'utf-8') as f:
        json.dump(run_info, f, indent = 2)

    return {
        'regime': regime,
        'seed': seed,
        'train_loss': train_losses[-1],
        'val_f1': val_f1s[-1],
        'training_time_seconds': duration,
        'model_state': model.state_dict()
    }

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
    if out_dir.exists():
        raise FileExistsError(f'Output directory already exists: {args.out_dir}')

    out_dir.mkdir(parents = True)

    split_csv = base_dir / 'split.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f'device: {device}')

    checks_passed = run_all_checks(data_dir, split_csv, device, out_dir)
    if not checks_passed:
        raise RuntimeError('Pre-training checks failed')

    runs_configs = [
        ('scratch', 1),
        ('frozen', 1),
        ('frozen', 2),
        ('frozen', 3),
        ('unfreeze', 1),
        ('unfreeze', 2),
        ('unfreeze', 3)
    ]

    results_list = []
    saved_states = {}

    print('training runs:')
    for r, s in runs_configs:
        print(f'{r}_seed{s}:')
        res = train_single_run(r, s, split_csv, data_dir, device, out_dir, epochs = EPOCHS)
        results_list.append(res)
        saved_states[(r, s)] = res['model_state']
        print(f'{r}_seed{s} done | time: {res['training_time_seconds']:.2f}s')
        print()

    print('training summary:')
    print('no | regime     | seed  | train_loss   | val_f1       | time_s  ')
    for idx, res in enumerate(results_list, 1):
        print(f'{idx:<2d} | {res['regime']:10s} | {res['seed']:<5d} | {res['train_loss']:<12.4f} | {res['val_f1']:<12.4f} | {res['training_time_seconds']:<8.2f}')

    regime_scores = {}
    for r in ['scratch', 'frozen', 'unfreeze']:
        vals = [res['val_f1'] for res in results_list if res['regime'] == r]
        regime_scores[r] = sum(vals) / len(vals)

    best_regime = max(regime_scores.keys(), key = lambda k: regime_scores[k])
    best_candidates = [res for res in results_list if res['regime'] == best_regime]
    best_run = max(best_candidates, key = lambda x: x['val_f1'])
    best_seed = best_run['seed']
    print()
    print('best model:')
    print(f'best_model: {best_regime}_seed{best_seed} (val_f1={best_run['val_f1']:.4f})')

    model_checkpoint = {
        'state_dict': saved_states[(best_regime, best_seed)],
        'regime': best_regime,
        'seed': best_seed,
        'classes': CLASSES,
        'img_size': 160
    }

    model_path = out_dir / 'lab3_model.pt'
    torch.save(model_checkpoint, model_path)

if __name__ == '__main__':
    main()
