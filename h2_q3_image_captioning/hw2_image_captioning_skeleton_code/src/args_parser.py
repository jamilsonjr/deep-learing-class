import argparse

from optimizer import (
    OptimizerType,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=8,
                        help='define batch size to train the model')

    parser.add_argument('--epochs', type=int, default=6,
                        help='define epochs to train the model')

    parser.add_argument('--optimizer_type', type=str, default=OptimizerType.ADAM.value,
                        choices=[optimizer.value for optimizer in OptimizerType])

    parser.add_argument('--decoder_lr', type=float, default=4e-4)

    parser.add_argument('--embed_dim', type=int, default=100,
                        help='define dims of embeddings for words')

    parser.add_argument('--decoder_dim', type=int,
                        default=256, help='define units of decoder')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='define units of decoder')

    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='define value to clip gradients')

    parser.add_argument(
        '--use_attention', action='store_true', default=False,
        help='Use the decoder with attention mechanism')


    parser.add_argument('--attention_dim', type=int,
                        default=100, help='define units of attention')

    parser.add_argument('--print_freq', type=int,
                        default=5, help='define print freq of loss')

    parser.add_argument(
        '--disable_steps', action='store_true', default=False,
        help='Conf just for debug: make the model run only 1 steps instead of the steps that was supposed')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='define num_works to dataloader')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='define seed to reproduce')

    args = parser.parse_args()

    return args
