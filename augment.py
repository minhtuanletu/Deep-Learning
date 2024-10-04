import os
from common.data_augmentation import AugmentDataset

input_dirs = 'dataset/Flower_Recognition'
output_dirs = 'dataset/Flower_Recognition_augment'
for task in ['train', 'val', 'test']:
    input_dir = os.path.join(input_dirs, task)
    output_dir = os.path.join(output_dirs, task)
    os.makedirs(output_dir, exist_ok=True)
    augment = AugmentDataset(input_dir, output_dir)
    augment.handle()
    print(f'Create {len(os.listdir(output_dir))} in {output_dir}')