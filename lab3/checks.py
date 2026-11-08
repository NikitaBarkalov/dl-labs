import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from dataset import (
    CLASSES,
    IntelDataset,
    get_transforms,
    make_split,
    unnormalize
)
from model import (
    build_model_and_optimizer,
    set_train_mode
)

def save_batch_visualization(loader, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)

    for images, labels in loader:
        fig, axes = plt.subplots(4, 4, figsize = (10, 10))

        for i in range(16):
            r = i // 4
            c = i % 4
            ax = axes[r, c]

            img_np = unnormalize(images[i])
            ax.imshow(img_np)
            ax.set_title(CLASSES[labels[i].item()], fontsize = 10)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(out_path, dpi = 200)
        plt.close(fig)
        break

def check_parameters_info(regime):
    device = torch.device('cpu')
    model, opt = build_model_and_optimizer(regime, seed = 1, device = device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    groups_info = []
    opt_params_total = 0
    for g in opt.param_groups:
        count = sum(p.numel() for p in g['params'])
        opt_params_total += count
        groups_info.append({'count': count, 'lr': g['lr']})

    has_frozen_in_opt = any(not p.requires_grad for g in opt.param_groups for p in g['params'])

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'groups': groups_info,
        'has_frozen_in_opt': has_frozen_in_opt
    }

def check_modules_changes(regime, train_loader, device):
    model, opt = build_model_and_optimizer(regime, seed = 1, device = device)
    init_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    criterion = torch.nn.CrossEntropyLoss()
    set_train_mode(model, regime)

    step = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()

        step += 1
        if step == 5:
            break

    curr_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    module_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    results = {}

    for mod_name in module_names:
        submod = getattr(model, mod_name)

        param_changed = False
        for p_name, _ in submod.named_parameters():
            full_key = f'{mod_name}.{p_name}'
            if not torch.equal(init_state[full_key], curr_state[full_key]):
                param_changed = True
                break

        has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in submod.modules())
        if has_bn:
            bn_changed = False
            for b_name, _ in submod.named_buffers():
                if any(stat in b_name for stat in ['running_mean', 'running_var', 'num_batches_tracked']):
                    full_key = f'{mod_name}.{b_name}'
                    if not torch.equal(init_state[full_key], curr_state[full_key]):
                        bn_changed = True
                        break
            bn_str = 'yes' if bn_changed else 'no'
        else:
            bn_str = 'no_bn'

        param_str = 'yes' if param_changed else 'no'
        expected = False

        if regime == 'frozen':
            if mod_name == 'fc':
                expected = (param_str == 'yes' and bn_str == 'no_bn')
            else:
                expected = (param_str == 'no' and bn_str in ['no', 'no_bn'])
        elif regime == 'unfreeze':
            if mod_name in ['layer4', 'fc']:
                expected = (param_str == 'yes' and bn_str in ['yes', 'no_bn'])
            else:
                expected = (param_str == 'no' and bn_str in ['no', 'no_bn'])

        results[mod_name] = {
            'param_changed': param_str,
            'bn_changed': bn_str,
            'matches_expected': 'yes' if expected else 'no'
        }

    return results

def run_buggy_frozen_experiment(train_loader, device):
    model, opt = build_model_and_optimizer('frozen', seed = 1, device = device)
    criterion = torch.nn.CrossEntropyLoss()

    target_bn = model.layer1[0].bn1
    init_mean = target_bn.running_mean.clone().cpu()
    init_var = target_bn.running_var.clone().cpu()

    model.train()

    step = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()

        step += 1
        if step == 5:
            break

    curr_mean = target_bn.running_mean.clone().cpu()
    curr_var = target_bn.running_var.clone().cpu()

    max_mean_diff = (curr_mean - init_mean).abs().max().item()
    max_var_diff = (curr_var - init_var).abs().max().item()

    return max_mean_diff, max_var_diff

def run_all_checks(data_dir, split_csv, device, out_dir):
    control_data = make_split(data_dir, split_csv)
    expected_control = {
        'train': {
            'buildings': (200, 1996761),
            'forest': (200, 2032625),
            'glacier': (200, 2183923),
            'mountain': (200, 1950345),
            'sea': (200, 1987613),
            'street': (200, 1841817)
        },
        'validation': {
            'buildings': (218, 4837323),
            'forest': (237, 5276962),
            'glacier': (276, 6135652),
            'mountain': (262, 5789535),
            'sea': (255, 5655068),
            'street': (250, 5541845)
        },
        'test': {
            'buildings': (219, 4837559),
            'forest': (237, 5291326),
            'glacier': (277, 6157250),
            'mountain': (263, 5849765),
            'sea': (255, 5668011),
            'street': (251, 5547533)
        }
    }

    all_matched = True
    print('split check:')
    print('class      | train            | val              | test            ')

    for cls in CLASSES:
        tr_act = control_data['train'][cls]
        tr_exp = expected_control['train'][cls]
        val_act = control_data['validation'][cls]
        val_exp = expected_control['validation'][cls]
        ts_act = control_data['test'][cls]
        ts_exp = expected_control['test'][cls]

        if tr_act != tr_exp or val_act != val_exp or ts_act != ts_exp:
            all_matched = False

        print(f'{cls:10s} | {tr_act[0]} / {tr_act[1]:<9d} | {val_act[0]} / {val_act[1]:<9d} | {ts_act[0]} / {ts_act[1]:<9d}')

    tr_tot = (sum(control_data['train'][c][0] for c in CLASSES), sum(control_data['train'][c][1] for c in CLASSES))
    val_tot = (sum(control_data['validation'][c][0] for c in CLASSES), sum(control_data['validation'][c][1] for c in CLASSES))
    ts_tot = (sum(control_data['test'][c][0] for c in CLASSES), sum(control_data['test'][c][1] for c in CLASSES))

    with open(split_csv, 'r', encoding = 'utf-8') as f:
        csv_rows = [line.strip().split(',') for line in f.readlines()[1:] if line.strip()]

    tr_files = {r[0] for r in csv_rows if r[2] == 'train'}
    val_files = {r[0] for r in csv_rows if r[2] == 'validation'}
    ts_files = {r[0] for r in csv_rows if r[2] == 'test'}

    intersections_empty = (len(tr_files & val_files) == 0) and (len(tr_files & ts_files) == 0) and (len(val_files & ts_files) == 0)
    total_unique_ok = (len(tr_files | val_files | ts_files) == 4200)

    if not (intersections_empty and total_unique_ok):
        all_matched = False

    print(f'total      | {tr_tot[0]} / {tr_tot[1]:<9d} | {val_tot[0]} / {val_tot[1]:<9d} | {ts_tot[0]} / {ts_tot[1]:<9d}')
    print(f'split_match: {all_matched}')
    print(f'intersections_empty: {intersections_empty}')
    print(f'total_unique_files: {len(tr_files | val_files | ts_files)}')
    print()

    train_tf, _ = get_transforms()
    train_ds = IntelDataset(split_csv, 'train', data_dir, train_tf)
    check_loader = torch.utils.data.DataLoader(train_ds, batch_size = 32, shuffle = True, drop_last = False, num_workers = 0)

    print('initial loss check:')
    criterion = torch.nn.CrossEntropyLoss()
    for regime in ['scratch', 'frozen']:
        torch.manual_seed(1)
        np.random.seed(1)

        m_init, _ = build_model_and_optimizer(regime, seed = 1, device = device)
        set_train_mode(m_init, regime)

        for inputs, targets in check_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                logits = m_init(inputs)
                l0 = criterion(logits, targets).item()
            break

        print(f'L0_{regime}: {l0:.5f}')

    print()
    print('parameters check:')
    expected_params = {
        'scratch': {'total': 11179590, 'trainable': 11179590, 'groups': [(11179590, 3e-3)]},
        'frozen': {'total': 11179590, 'trainable': 3078, 'groups': [(3078, 3e-3)]},
        'unfreeze': {'total': 11179590, 'trainable': 8396806, 'groups': [(3078, 3e-3), (8393728, 3e-4)]}
    }

    params_ok = True
    for r in ['scratch', 'frozen', 'unfreeze']:
        info = check_parameters_info(r)
        print(f'{r}: total={info['total_params']}, trainable={info['trainable_params']}, frozen_in_opt={info['has_frozen_in_opt']}')

        exp_p = expected_params[r]
        if info['total_params'] != exp_p['total'] or info['trainable_params'] != exp_p['trainable'] or info['has_frozen_in_opt']:
            params_ok = False

        if len(info['groups']) != len(exp_p['groups']):
            params_ok = False
        else:
            for idx, g in enumerate(info['groups']):
                exp_cnt, exp_lr = exp_p['groups'][idx]
                if g['count'] != exp_cnt or g['lr'] != exp_lr:
                    params_ok = False
                print(f'  group_{idx}: count={g['count']}, lr={g['lr']}')

    print()
    print('module changes check:')
    modules_ok = True
    for r in ['frozen', 'unfreeze']:
        print(f'regime: {r}')
        mod_results = check_modules_changes(r, check_loader, device)
        print('module     | param_ch   | bn_ch      | match     ')

        for m_name, res in mod_results.items():
            if res['matches_expected'] != 'yes':
                modules_ok = False
            print(f'{m_name:10s} | {res['param_changed']:10s} | {res['bn_changed']:10s} | {res['matches_expected']:10s}')

    print()
    print('batchnorm bug check:')
    max_mean_diff, max_var_diff = run_buggy_frozen_experiment(check_loader, device)
    print(f'bn_bug_max_mean_diff: {max_mean_diff:.8f}')
    print(f'bn_bug_max_var_diff: {max_var_diff:.8f}')
    print()
    bn_bug_ok = (max_mean_diff > 0.0) and (max_var_diff > 0.0)

    checks_passed = (all_matched and params_ok and modules_ok and bn_bug_ok)
    if checks_passed:
        figures_dir = Path(out_dir) / 'figures'
        figures_dir.mkdir(parents = True, exist_ok = True)
        save_batch_visualization(check_loader, figures_dir / 'batch_samples.png')

    print('checks status:')
    print(f'checks_passed: {checks_passed}')
    print()
    return checks_passed
