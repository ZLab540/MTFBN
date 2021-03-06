# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import random
import torchvision
from torchvision import transforms

#data augmentation for image rotate
def augment(hazy, clean):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        return hazy, clean
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)
        return hazy, clean
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)
        return hazy, clean
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return hazy, clean

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        train_list = train_data_dir + 'trainlist.txt'
        with open(train_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_img = Image.open(self.train_data_dir + 'hazy/' + haze_name)

        try:
            gt_img = Image.open(self.train_data_dir + 'clear/' + gt_name + '.jpg')
        except:
            gt_img = Image.open(self.train_data_dir + 'clear/' + gt_name + '.png').convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        haze_crop_img, gt_crop_img = augment(haze_crop_img, gt_crop_img)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

