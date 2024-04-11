
import os
import shutil
import random

data_dir = 'data/'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for type_folder in os.listdir(data_dir):
    type_path = os.path.join(data_dir, type_folder)
    if os.path.isdir(type_path):
        files = os.listdir(type_path)
        random.shuffle(files)
        total_samples = len(files)
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)
        test_samples = total_samples - train_samples - val_samples

        train_set = files[:train_samples]
        val_set = files[train_samples:train_samples+val_samples]
        test_set = files[train_samples+val_samples:]

        os.makedirs(f'train_data/{type_folder}', exist_ok=True)
        os.makedirs(f'val_data/{type_folder}', exist_ok=True)
        os.makedirs(f'test_data/{type_folder}', exist_ok=True)

        for file in train_set:
            src = os.path.join(type_path, file)
            dest = os.path.join(f'train_data/{type_folder}', file)
            shutil.copy(src, dest)

        for file in val_set:
            src = os.path.join(type_path, file)
            dest = os.path.join(f'val_data/{type_folder}', file)
            shutil.copy(src, dest)

        for file in test_set:
            src = os.path.join(type_path, file)
            dest = os.path.join(f'test_data/{type_folder}', file)
            shutil.copy(src, dest)