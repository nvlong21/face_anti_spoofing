import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
import PIL.Image as Image

class DataLoader(data.Dataset):
    """Data Loader class"""

    def __init__(self, transform):
        self.transform = transform
        self.data, self.labels = get_files()

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.data[index]
        # random flip with ratio of 0.5
        flip = np.random.choice(2) * 2 - 1
        img = img[:, ::flip, :]
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        return img, label
    def __len__(self):
        return len(self.data)

def get_files():
    all_image_list = []
    all_label_list = []
    real_dir = './data/ClientFace'
    fake_dir = './data/ImposterFace'
    # load the real image
    count_real = 0
    count_fake = 0
    for sub_dir in os.listdir(real_dir):
        if os.path.isdir(real_dir + '/' + sub_dir):
            for file_name in os.listdir(real_dir + '/' + sub_dir):
                if not file_name.endswith('.jpg') or file_name.startswith('.'):
                    continue  # Skip!
                
                image = cv2.imread(real_dir + '/' + sub_dir + '/' + file_name, cv2.IMREAD_COLOR)
                all_image_list.append(image)
                all_label_list.append(1)
                count_real += 1

    for sub_dir_fake in os.listdir(fake_dir):
        if os.path.isdir(fake_dir + '/' + sub_dir_fake):
            for fake_file_name in os.listdir(fake_dir + '/' + sub_dir_fake):
                if not fake_file_name.endswith('.jpg') or fake_file_name.startswith('.'):
                    continue  # Skip!

                image = cv2.imread(fake_dir + '/' + sub_dir_fake + '/' + fake_file_name, cv2.IMREAD_COLOR)
                all_image_list.append(image)
                all_label_list.append(0)
                count_fake += 1

    print('There are %d real images\nThere are %d fake images' % (count_real, count_fake))
    all_image_list = np.array(all_image_list)

    return all_image_list, np.array([label for label in all_label_list])

