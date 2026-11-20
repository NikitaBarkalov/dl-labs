import csv
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from onnxruntime.quantization import CalibrationDataReader

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXPECTED_TRAIN_COUNT = 1200
EXPECTED_TRAIN_SUM = 11993084
EXPECTED_TEST_COUNT = 1502
EXPECTED_TEST_SUM = 33351444
EXPECTED_CALIB_COUNT = 200
EXPECTED_CALIB_SUM = 1921470
EXPECTED_CALIB_CLASSES = {
    'buildings': 40,
    'forest': 35,
    'glacier': 36,
    'mountain': 33,
    'sea': 29,
    'street': 27
}

def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

def parse_file_number(filename):
    return int(Path(filename).stem)

def resolve_image_path(data_dir, part, label, filename):
    root = Path(data_dir)
    if part == 'train':
        return root / 'seg_train' / 'seg_train' / label / filename
    return root / 'seg_test' / 'seg_test' / label / filename

def load_and_verify_split(split_csv):
    train_rows = []
    test_rows = []

    with open(split_csv, 'r', encoding = 'utf-8', newline = '') as f:
        reader = csv.DictReader(f)

        for row in reader:
            part = row['part'].strip()
            entry = {
                'file': row['file'].strip(),
                'label': row['label'].strip(),
                'part': part
            }
            if part == 'train':
                train_rows.append(entry)
            elif part == 'test':
                test_rows.append(entry)

    train_count = len(train_rows)
    train_sum = sum(parse_file_number(r['file']) for r in train_rows)
    test_count = len(test_rows)
    test_sum = sum(parse_file_number(r['file']) for r in test_rows)

    if train_count != EXPECTED_TRAIN_COUNT or train_sum != EXPECTED_TRAIN_SUM:
        raise ValueError(f'Train split mismatch: count = {train_count}, sum = {train_sum}')

    if test_count != EXPECTED_TEST_COUNT or test_sum != EXPECTED_TEST_SUM:
        raise ValueError(f'Test split mismatch: count={test_count}, sum={test_sum}')

    split_control = {
        'train': {
            'count': train_count,
            'name_sum': train_sum
        },
        'test': {
            'count': test_count,
            'name_sum': test_sum
        }
    }

    return train_rows, test_rows, split_control

def select_calibration_rows(train_rows):
    sorted_train = sorted(train_rows, key = lambda r: r['file'])
    idx = np.random.default_rng(0).permutation(len(sorted_train))[:200]
    calib_rows = [sorted_train[int(i)] for i in idx]

    calib_count = len(calib_rows)
    calib_sum = sum(parse_file_number(r['file']) for r in calib_rows)
    class_counts = {c: 0 for c in CLASSES}
    for row in calib_rows:
        class_counts[row['label']] += 1

    if calib_count != EXPECTED_CALIB_COUNT or calib_sum != EXPECTED_CALIB_SUM:
        raise ValueError(f'Calibration set mismatch: count = {calib_count}, sum = {calib_sum}')

    if class_counts != EXPECTED_CALIB_CLASSES:
        raise ValueError(f'Calibration class counts mismatch: {class_counts}')

    calib_control = {
        'count': calib_count,
        'name_sum': calib_sum,
        'class_counts': class_counts
    }

    return calib_rows, calib_control

def load_image_tensor(img_path, transform):
    with Image.open(img_path) as img:
        rgb_img = img.convert('RGB')
        return transform(rgb_img)

class SplitImageDataset(Dataset):
    def __init__(self, rows, data_dir, transform):
        self.rows = rows
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = resolve_image_path(self.data_dir, row['part'], row['label'], row['file'])
        tensor = load_image_tensor(img_path, self.transform)
        label_idx = self.class_to_idx[row['label']]
        return tensor, label_idx

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calib_rows, data_dir, transform, input_name = 'input'):
        self.calib_rows = calib_rows
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.calib_rows):
            return None
        
        row = self.calib_rows[self.index]
        self.index += 1
        img_path = resolve_image_path(self.data_dir, row['part'], row['label'], row['file'])
        tensor = load_image_tensor(img_path, self.transform)
        return {self.input_name: tensor.unsqueeze(0).numpy()}

    def rewind(self):
        self.index = 0
