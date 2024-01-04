import warnings
from pathlib import Path

import torch
import torch.utils.data
import torchvision.transforms as T
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from datasets.utils import *

warnings.simplefilter('ignore', category=VerifyWarning)

def get_transforms(img_size=128):
    return T.Compose([
            RemoveNaNs(),
            ZScale(),
            SigmaClip(),
            ToTensor(),
            torch.nn.Tanh(),
            MinMaxNormalize(),
            Unsqueeze(),
            T.Resize((img_size, img_size), antialias=True)
        ])

class RGDataset(Dataset):
    def __init__(self, data_dir, img_paths, img_size=128):
        super().__init__()
        data_dir = Path(data_dir)
        with open(img_paths) as f:
            self.img_paths = f.read().splitlines()
        self.img_paths = [data_dir / p for p in self.img_paths]

        self.transforms = get_transforms(img_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        img = fits.getdata(image_path)
        img = self.transforms(img)

        return img


if __name__ == '__main__':
    import sys
    sys.path.append('/home/apilzer/Documents/inaf/radiff')
    rgtrain = RGDataset('data/rg-dataset/data',
                        'data/rg-dataset/train_all.txt')
    batch = next(iter(rgtrain))
    image = batch
    to_pil_image(image).save('image.png')

    loader = torch.utils.data.DataLoader(
        rgtrain, batch_size=4, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        image = batch
        print(i, image.shape)
