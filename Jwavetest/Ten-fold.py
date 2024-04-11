import os
from sklearn.model_selection import StratifiedKFold
import shutil


data_dir = 'data/'


classes = [class_name for class_name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, class_name))]


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


all_files = []
all_labels = []

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    files = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]
    labels = [class_name] * len(files)
    all_files.extend(files)
    all_labels.extend(labels)


for fold, (train_idx, test_idx) in enumerate(kfold.split(all_files, all_labels), 1):

    train_dir = os.path.join(data_dir, f'fold_{fold}', 'train_data')
    test_dir = os.path.join(data_dir, f'fold_{fold}', 'test_data')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


    train_files = [all_files[idx] for idx in train_idx]
    test_files = [all_files[idx] for idx in test_idx]


    for file in train_files:
        dest_file = os.path.join(train_dir, os.path.basename(file))
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy(file, dest_file)

    for file in test_files:
        dest_file = os.path.join(test_dir, os.path.basename(file))
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy(file, dest_file)
