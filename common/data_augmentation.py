import os
import random
from PIL import Image
from torchvision.transforms import v2

class AugmentDataset:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.list_transform = []
        self.build_random_augment()

    def build_random_augment(self):
        self.list_transform = [
            v2.Compose([
                v2.PILToTensor(),
                v2.RandomHorizontalFlip(),
                v2.ToPILImage()
            ]),
            v2.Compose([
                v2.PILToTensor(),
                v2.RandomCrop(size=(224, 224)),
                v2.ToPILImage()
            ]),
            v2.Compose([
                v2.PILToTensor(),
                v2.RandomAffine(degrees=(30, 60), translate=(0.1, 0.2)),
                v2.ToPILImage()
            ]),
            v2.Compose([
                v2.PILToTensor(),
                v2.GaussianBlur(kernel_size=(3, 7)),
                v2.ToPILImage()
            ]),
            v2.Compose([
                v2.PILToTensor(),
                v2.AugMix(),
                v2.ToPILImage(),
            ])
        ]

    def handle(self):
        for label in os.listdir(self.input_dir):
            label_folder = os.path.join(self.input_dir, label)
            output_folder = os.path.join(self.output_dir, label)
            os.makedirs(output_folder, exist_ok=True)
            idx = 0
            num_of_augment = random.randint(3, 6)
            for _ in range(num_of_augment):
                for dir in os.listdir(label_folder):
                    while True:
                        try:
                            src_image_path = os.path.join(label_folder, dir)
                            image = Image.open(src_image_path)
                            transform = random.choice(self.list_transform)
                            new_image = transform(image)
                            dst_image_path = os.path.join(output_folder, f'img{idx}.png')
                            new_image.save(dst_image_path)
                            idx += 1
                            break
                        except: 
                            continue