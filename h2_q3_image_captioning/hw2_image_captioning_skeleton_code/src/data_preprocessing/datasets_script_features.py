from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import time


class CaptionDataset(Dataset):

    def __init__(
        self,
        data_file,
        images_folder
    ):

        with open(data_file) as json_file:
            dataset = json.load(json_file)

        self.images_names = list(set(dataset["images_names"]))

        self.dataset_size = len(self.images_names)

        self.images_folder = images_folder

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                    std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, i):
        image_name = self.images_folder + self.images_names[i]
        
        image_id= int(self.images_names[i].split(".")[0]) -1 #image id is the same number as the name -1 (ex_tif1.jpg, id=0)
        image = Image.open(image_name)
        image = self.transform(image)

        return image, image_id

    def __len__(self):
        return self.dataset_size


