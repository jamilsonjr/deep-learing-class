
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from args_parser import get_args
from data_preprocessing.datasets_script_features import CaptionDataset
from models.encoder import Encoder
import time


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def add_features(imgs, imgs_ids, features_h5):
    encoder_out = encoder(imgs)
    encoder_out = encoder_out.view(encoder_out.size(0), -1, encoder_out.size(-1))  # flatten

    for i in range(len(imgs_ids)):
        features_h5[imgs_ids[i]] = encoder_out[i].cpu().numpy()

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    args = get_args()
    logging.info(args.__dict__)

    dataset_jsons = "data/datasets/"
    dataset_images_folder= "data/raw_dataset/images/"

    train_dataloader = DataLoader(
        CaptionDataset(dataset_jsons + "train.json", dataset_images_folder),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = DataLoader(
        CaptionDataset(dataset_jsons + "val.json", dataset_images_folder),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_dataloader = DataLoader(
        CaptionDataset(dataset_jsons + "test.json", dataset_images_folder),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    start = time.time()

    encoder = Encoder()

    with h5py.File(dataset_images_folder+'all_features.hdf5', 'w') as h:
        features_h5 = h.create_dataset('features', (613, 256, 512))

        logging.info("starting train features")
        for batch_i, (imgs, imgs_ids) in enumerate(train_dataloader):
            add_features(imgs, imgs_ids, features_h5)


        logging.info("starting val features")
        for batch_i, (imgs, imgs_ids) in enumerate(val_dataloader):
            add_features(imgs, imgs_ids, features_h5)

        
        logging.info("starting test features")
        for batch_i, (imgs, imgs_ids) in enumerate(test_dataloader):
            add_features(imgs, imgs_ids, features_h5)

    logging.info('Time taken for 1 epoch {:.4f} sec'.format(
            time.time() - start))