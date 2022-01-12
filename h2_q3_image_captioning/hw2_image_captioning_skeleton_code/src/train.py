
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from args_parser import get_args
import json
from data_preprocessing.datasets import CaptionDataset, CaptionValDataset
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN

from models.encoder import Encoder
from models.decoder import Decoder
from models.decoder_with_attention import DecoderWithAttention
from optimizer import get_optimizer,clip_gradient
import time
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import random


# Run this with: PYTHONHASHSEED=0 python3 src/train.py
def prepare_inputs(encoder_out, caps, caption_lengths):
    encoder_out = encoder_out.to(device)
    caps = caps.to(device)
    caption_lengths = caption_lengths.to(device)

    encoder_out = encoder_out.view(
        encoder_out.size(0), -1, encoder_out.size(-1))  # flatten

    # sorted captions
    caption_lengths, sort_ind = caption_lengths.squeeze(
        1).sort(dim=0, descending=True)
    encoder_out = encoder_out[sort_ind]
    caps_sorted = caps[sort_ind]

    # input captions must not have "end_token"
    caption_lengths = (caption_lengths - 1).tolist()

    return encoder_out, caps_sorted, caption_lengths


def get_loss(criterion, predict_output, targets, caption_lengths):
    targets = targets[:, 1:]  # targets captions dont have start token 

    # pack scores and target
    predictions = pack_padded_sequence(
        predict_output, caption_lengths, batch_first=True)
    targets = pack_padded_sequence(
        targets, caption_lengths, batch_first=True)

    loss = criterion(predictions.data, targets.data)

    return loss

def predict(encoder_out, decoder, caps, caption_lengths, vocab_size):
    batch_size = encoder_out.size(0)

    all_predictions = torch.zeros(batch_size, max(
        caption_lengths), vocab_size).to(device)

    h, c = decoder.init_hidden_state(encoder_out)

    for t in range(max(
            caption_lengths)):
        # batchsizes of current time_step are the ones with lenght bigger than time-step 
        # (i.e have not fineshed yet)
        batch_size_t = sum([l > t for l in caption_lengths])

        predictions, h, c = decoder(
            caps[:batch_size_t, t], h[:batch_size_t], c[:batch_size_t], encoder_out[:batch_size_t])

        all_predictions[:batch_size_t, t, :] = predictions

    return all_predictions
    
def train_batch(decoder, decoder_optimizer, encoder_out, caps_input, cap_len, vocab_size, criterion):
    encoder_out, caps_sorted, caption_lengths = prepare_inputs(encoder_out, 
        caps_input, cap_len)

    predict_output = predict(encoder_out, decoder, caps_sorted, caption_lengths, vocab_size)

    loss= get_loss(criterion, predict_output, caps_sorted, caption_lengths)

    decoder_optimizer.zero_grad()
    
    loss.backward()

    # Clip gradients
    clip_gradient(decoder_optimizer, args.grad_clip)

    # Update weights
    decoder_optimizer.step()

    return loss

def val_step(decoder, encoder_out, all_captions, vocab_size, token_to_id, id_to_token, max_len):    
    #greedy decodying mechanism for inference
    generated_tokens = []

    input_word = torch.tensor([token_to_id[START_TOKEN]])

    h, c = decoder.init_hidden_state(encoder_out)

    for _ in range(max_len):

        decoder_output, h, c = decoder(input_word, h, c, encoder_out)
        
        top_index = decoder_output.argmax().item()

        if top_index == token_to_id[END_TOKEN]:
            break
        else:
            generated_tokens.append(id_to_token[top_index])

        #next word will be the one with highest prob
        input_word = torch.tensor([top_index]) 
    
    bleu_4=sentence_bleu(all_captions, generated_tokens)
    return bleu_4,generated_tokens

def plot(plottable, ylabel='', name=''):
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(plottable)
    plt.savefig('results/{}.png'.format(name))
    plt.show()


def configure_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    args = get_args()
    logging.info(args.__dict__)

    configure_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    dataset_jsons = "data/datasets/"
    dataset_images_folder= "data/raw_dataset/images/"

    with open(dataset_jsons + "vocab_info.json") as json_file:
        vocab_info = json.load(json_file)

    # given it was loaded from a json, the dict id_to_token has keys as strings instead of int, as supposed. To fix:
    vocab_info["id_to_token"] = {
        int(k): v for k, v in vocab_info["id_to_token"].items()}

    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    logging.info("vocab size %s", vocab_size)

    train_dataloader = DataLoader(
        CaptionDataset(dataset_jsons + "train.json", dataset_images_folder,max_len,token_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = DataLoader(
        CaptionValDataset(dataset_jsons + "val.json", dataset_images_folder,max_len,token_to_id),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_dataloader = DataLoader(
        CaptionValDataset(dataset_jsons + "test.json", dataset_images_folder,max_len,token_to_id),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.use_attention:
        decoder = DecoderWithAttention(
            attention_dim = args.attention_dim,
            decoder_dim=args.decoder_dim,
            embed_dim=args.embed_dim,
            vocab_size=vocab_size,
            dropout_rate=args.dropout
        )
    else:
        decoder = Decoder(
            decoder_dim=args.decoder_dim,
            embed_dim=args.embed_dim,
            vocab_size=vocab_size,
            dropout_rate=args.dropout
        )

    #in this case encoder doesnot need optimizar since we are using we are using pretrained encoder without fine-tuning it
    #just use optimizer for decoder:
    decoder_optimizer = get_optimizer(args.optimizer_type, decoder.parameters(), args.decoder_lr)
    criterion= nn.CrossEntropyLoss().to(device)

    train_epoch_losses = []
    val_epoch_bleus = []

    # Iterate by epoch
    start_training = time.time()

    for epoch in range(args.epochs):
        start = time.time()
        train_total_loss = 0.0
        val_total_bleu = 0.0

        len_train=len(train_dataloader)
        len_val=len(train_dataloader)

        decoder.train()

        for batch_i, (enc_output, caps, caplens) in enumerate(train_dataloader):
            
            train_loss = train_batch(
                decoder,decoder_optimizer, enc_output, caps, caplens, vocab_size, criterion
            ).data.item()

            train_total_loss += train_loss

            if batch_i % args.print_freq == 0:
                logging.info(
                    "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                        "TRAIN", epoch, args.epochs, batch_i, len_train, train_loss
                    )
                )

            # (only for debug: interrupt val after 1 step)
            if args.disable_steps:
                break


        # End training
        epoch_loss = train_total_loss / (batch_i + 1)
        train_epoch_losses.append(epoch_loss)
        
        logging.info('Time taken for 1 epoch {:.4f} sec'.format(
            time.time() - start))
        logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                train_total_loss / (batch_i + 1)))

        # Start validation
        decoder.eval()  # eval mode (no dropout or batchnorm)

        with torch.no_grad():
            for batch_i, (enc_output, image_name) in enumerate(val_dataloader):
                all_captions = val_dataloader.dataset.all_refs[image_name[0]]
                val_bleu,_ = val_step(decoder, enc_output, all_captions, vocab_size, token_to_id, id_to_token, max_len)
                
                if batch_i % args.print_freq == 0:
                    logging.info(
                        "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t BLEU: {:.4f}\t".format(
                            "VAL", epoch, args.epochs, batch_i, len_val, val_bleu
                        )
                    )

                val_total_bleu += val_bleu

                # (only for debug: interrupt val after 1 step)
                if args.disable_steps:
                    break

        # End validation
        epoch_val_bleu = val_total_bleu / (batch_i + 1)
        val_epoch_bleus.append(epoch_val_bleu)

        logging.info('\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Bleu:{:.4f}; -------------\n'.format(
            epoch, args.epochs, epoch_loss, epoch_val_bleu))

    #testing
    decoder.eval()
    results=[]
    total_bleu=0.0
    with torch.no_grad():
        for batch_i, (enc_output, image_name) in enumerate(test_dataloader):
            image_name = image_name[0] #batch size is 1 for testing and val
            all_captions = test_dataloader.dataset.all_refs[image_name]
            bleu,caption_generated = val_step(decoder, enc_output, all_captions, vocab_size, token_to_id, id_to_token, max_len)
            total_bleu += bleu

            results.append({
                "image_name": image_name,
                "caption": caption_generated,
            })
    
    logging.info("time for training and testing %s", (time.time() - start_training))

    if args.use_attention:
        model_name= 'enc_dec_w_attention'
    else:
        model_name= 'enc_dec'

    logging.info("Saving generated captions to a file")
    
    with open("results/caps_generated_"+ model_name+".json", 'w+') as f:
        json.dump(results, f, indent=2)

    logging.info("Generated captions %s", results)

    #Final score:
    logging.info("Test Bleu-4: %s", (total_bleu / (batch_i + 1)))
      
    plot(train_epoch_losses, ylabel='Loss', name='training_loss_'+model_name)
    plot(val_epoch_bleus, ylabel='BLEU_4', name='validation_bleu4_'+model_name)
