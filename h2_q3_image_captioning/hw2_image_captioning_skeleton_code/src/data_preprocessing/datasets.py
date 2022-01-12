import torch
from torch.utils.data import Dataset
from PIL import Image
from data_preprocessing.preprocess_tokens import convert_captions_to_Y, START_TOKEN, END_TOKEN
from torchvision import transforms
import json
import h5py
import numpy as np


class CaptionDataset(Dataset):

    def __init__(
        self,
        data_file,
        images_folder,
        max_len,
        token_to_id
    ):
        self.images_folder = images_folder
        with open(data_file) as json_file:
            dataset = json.load(json_file)
        self._init_caption(dataset, max_len, token_to_id)
        self._init_images(dataset, images_folder)

    def _init_caption(self, dataset, max_len, token_to_id):
        captions_of_tokens = dataset["captions_tokens"]

        self.input_captions, self.captions_lengths = convert_captions_to_Y(
            captions_of_tokens, max_len, token_to_id)

    def _init_images(self, dataset, images_folder):
        self.images_folder = images_folder
        self.images_names = dataset["images_names"]
        self.dataset_size = len(self.images_names)
        self.image_features = h5py.File("data/raw_dataset/images/all_features.hdf5", "r")["features"]


    def __getitem__(self, i):
        image_name = self.images_folder + self.images_names[i]
        
        image_id= int(self.images_names[i].split(".")[0]) -1 #image id is the same number as the name -1 (ex_tif1.jpg, id=0)
 
        input_caption = self.input_captions[i]
        caption_lenght = self.captions_lengths[i]

        enc_output = self.image_features[image_id][()]
        enc_output = torch.FloatTensor(enc_output)

        return enc_output, torch.LongTensor(input_caption), torch.LongTensor([caption_lenght])

    def __len__(self):
        return self.dataset_size


class CaptionValDataset(CaptionDataset):

    def __init__(
        self,
        data_file,
        images_folder,
        max_len,
        token_to_id
    ):
        self.images_folder = images_folder
        with open(data_file) as json_file:
            dataset = json.load(json_file)
        self._init_images(dataset, images_folder)
        self.all_refs= dataset["all_refs"]
        self.max_len= max_len
        self.token_to_id = token_to_id

    def __getitem__(self, i):

        image_name = self.images_folder + self.images_names[i]
        
        image_id= int(self.images_names[i].split(".")[0]) -1 #image id is the same number as the name -1 (ex_tif1.jpg, id=0)

        enc_output = self.image_features[image_id][()]
        enc_output = torch.FloatTensor(enc_output)
        
        return enc_output, self.images_names[i]

