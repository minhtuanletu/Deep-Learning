import os
import random
import shutil

def split_dataset(input_folder, output_folder, train_ratio, val_ratio, test_ratio):
    list_dirs = os.listdir(input_folder)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    for dir in list_dirs:
        train_dir = os.path.join(train_folder, dir)
        val_dir = os.path.join(val_folder, dir)
        test_dir = os.path.join(test_folder, dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        folder = os.path.join(input_folder, dir)
        images_name = os.listdir(folder)
        random.shuffle(images_name)
        num_train = int(train_ratio * len(images_name))
        num_val = int(val_ratio * len(images_name))
        train_images = images_name[:num_train]
        val_images = images_name[num_train:num_train+num_val]
        test_images = images_name[num_train + num_val:]
        for image in train_images:
            src_path = os.path.join(folder, image)
            dst_path = os.path.join(train_dir, image)
            shutil.copy(src_path, dst_path)
        for image in val_images:
            src_path = os.path.join(folder, image)
            dst_path = os.path.join(val_dir, image)
            shutil.copy(src_path, dst_path)
        for image in test_images:
            src_path = os.path.join(folder, image)
            dst_path = os.path.join(test_dir, image)
            shutil.copy(src_path, dst_path)

input_folder = '/home/minhtuan/Desktop/Learn/Flower_Recognition/data/flowers'
output_folder = '/home/minhtuan/Desktop/Learn/Flower_Recognition/total_data'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
split_dataset(input_folder, output_folder, train_ratio, val_ratio, test_ratio)