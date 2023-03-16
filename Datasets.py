from os import listdir
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dir):
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.dir = dir
        self.image_urls = listdir(dir)
        self.targets = [self.get_target(x) for x in self.image_urls]
        self.data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, index):
        image_url = self.image_urls[index]
        image = Image.open(self.dir + "/" + image_url)
        image_arr = np.array(image)
        image_arr = np.expand_dims(image_arr, -1)
        image_arr = image_arr.repeat(3, axis=-1)
        rgb_image = Image.fromarray(image_arr)
        image = self.data_transform(rgb_image)
        return image, self.targets[index]

    def get_target(self, label):
        return 1 if label[0] == "A" else 2
    
class TriangleDataset(Dataset):
    def __init__(self, file_path: str, repeat = 1, rescale = True):
        raw_trial = self.open_data(file_path)
        trials = np.repeat(self.rescale_trials(raw_trial) if rescale else raw_trial, repeat, axis=0)
        rng = np.random.default_rng()
        rng.shuffle(trials)
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]
        return trial[1:], int(trial[0])

    def open_data(self, file_path: str):
        with open(file_path, "r") as file:
            data = []
            for entry in file:
                data.append(np.array(list(map(float, entry.split()))))
            return np.array(data)
        
    def rescale_trials(self, trials):
        orientation = trials[:, 1]  # "x"
        spatial = trials[:, 2]  # "y"
        min_orientation = abs(min(orientation))
        min_spatial = abs(min(spatial))
        padding = 5
        orientation = [x + min_orientation for x in orientation]
        spatial = [y + min_spatial for y in spatial]
        plane = np.column_stack((orientation, spatial))
        plane *= (100 - 2 * padding) / plane.max()
        orientation = plane[:, 0]
        spatial = plane[:, 1]
        orientation = [x + padding for x in orientation]
        spatial = [y + padding for y in spatial]
        return np.column_stack((trials[:, 0], orientation, spatial))
