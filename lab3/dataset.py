import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

MEAN = (0.485, 0.456, 0.406)

STD = (0.229, 0.224, 0.225)

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(160, scale = (0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_tf, eval_tf

def unnormalize(tensor):
    mean_tensor = torch.tensor(MEAN).view(3, 1, 1)
    std_tensor = torch.tensor(STD).view(3, 1, 1)

    img = tensor.clone().detach().cpu() * std_tensor + mean_tensor
    img = img.clamp(0.0, 1.0)

    return img.permute(1, 2, 0).numpy()

def resolve_data_paths(data_dir):
    p = Path(data_dir)

    train_root = p / 'seg_train' / 'seg_train'
    test_root = p / 'seg_test' / 'seg_test'

    return train_root, test_root

def make_split(data_dir, output_csv = None):
    if output_csv is None:
        output_csv = Path(__file__).resolve().parent / 'split.csv'

    train_root, test_root = resolve_data_paths(data_dir)

    rows = []
    control_data = {
        'train': {},
        'validation': {},
        'test': {}
    }

    rng_train = np.random.default_rng(0)
    for cls in CLASSES:
        cls_dir = train_root / cls
        files = sorted([f.name for f in cls_dir.glob('*.jpg')])
        n_k = len(files)
        idx = np.arange(n_k)
        rng_train.shuffle(idx)

        selected_idx = idx[:200]
        selected_files = [files[i] for i in selected_idx]

        control_data['train'][cls] = (len(selected_files), sum(int(f.split('.')[0]) for f in selected_files))

        for f in selected_files:
            rows.append({'file': f, 'label': cls, 'part': 'train'})

    rng_val_test = np.random.default_rng(0)
    for cls in CLASSES:
        cls_dir = test_root / cls
        files = sorted([f.name for f in cls_dir.glob('*.jpg')])
        n_k = len(files)
        idx = np.arange(n_k)
        rng_val_test.shuffle(idx)

        cut = n_k // 2
        val_idx = idx[:cut]
        test_idx = idx[cut:]

        val_files = [files[i] for i in val_idx]
        test_files = [files[i] for i in test_idx]

        control_data['validation'][cls] = (len(val_files), sum(int(f.split('.')[0]) for f in val_files))
        control_data['test'][cls] = (len(test_files), sum(int(f.split('.')[0]) for f in test_files))

        for f in val_files:
            rows.append({'file': f, 'label': cls, 'part': 'validation'})

        for f in test_files:
            rows.append({'file': f, 'label': cls, 'part': 'test'})

    csv_path = Path(output_csv)

    with open(csv_path, 'w', encoding = 'utf-8') as f:
        f.write('file,label,part\n')
        for r in rows:
            f.write(f'{r['file']},{r['label']},{r['part']}\n')

    return control_data

def get_file_sha256(file_path):
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

class IntelDataset(Dataset):
    def __init__(self, split_csv, part, data_dir, transform):
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.samples = []

        train_root, test_root = resolve_data_paths(data_dir)

        with open(split_csv, 'r', encoding = 'utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        header = lines[0].split(',')
        file_idx = header.index('file')
        label_idx = header.index('label')
        part_idx = header.index('part')

        for line in lines[1:]:
            parts = line.split(',')
            f_name = parts[file_idx]
            label = parts[label_idx]
            p = parts[part_idx]

            if p == part:
                root = train_root if p == 'train' else test_root
                img_path = root / label / f_name
                self.samples.append((img_path, self.class_to_idx[label], f_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img)

        return tensor, label
