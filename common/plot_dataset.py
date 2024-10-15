import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def random_choice_image(folder_path, num_rows=3, num_cols=3):
    num_images = num_cols * num_rows
    images_path = []
    for dir in os.listdir(folder_path):
        folder = os.path.join(folder_path, dir)
        for name in os.listdir(folder):
            images_path.append(os.path.join(folder, name))
    random_images = random.choices(images_path, k=num_images)
    return random_images

def plot_image(folder_path, num_rows=3, num_cols=3):
    random_images = random_choice_image(folder_path, num_rows, num_cols)
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    axs = axs.flatten()
    for i in range(len(random_images)):
        image_path = random_images[i]
        label = image_path.split('\\')[-2]
        image = Image.open(image_path).convert('L')
        image = image.resize((144, 144))
        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(label)
        axs[i].axis('off')
    plt.show()

folder_path = 'dataset/Flower_Recognition_augment/train'
plot_image(folder_path)