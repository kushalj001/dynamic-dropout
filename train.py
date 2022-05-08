import argparse
import datetime
import os
from datetime import datetime as dt
import matplotlib
matplotlib.use('Agg')
from lib.SentenceVAE import SentenceVAE
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lib.utils import run_cuda_diagnostics, make_reproducible
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def main(arguments=None):
    """ Main function used for training various models.
    :param arguments: list of arguments passed (i) through python script; or (ii) command line;
    """
    parser = get_parser()

    # parse arguments
    args = parser.parse_args(arguments.split()) if arguments else parser.parse_args()

    # for reproducibility
    make_reproducible(args.seed)

    # cuda diagnostics
    run_cuda_diagnostics(args.gpus)

    # select model based on the dataset
    dataset_to_model_map = dict({'PTB': SentenceVAE,
                                 'Yahoo': SentenceVAE,
                                 'Yelp': SentenceVAE,
                                 'SNLI': SentenceVAE})
    model_name = dataset_to_model_map[args.dataset]

    # instantiate and initialize model
    model = model_name(**vars(args))
    model.init_weights()
    print("model signature", model.signature())

    # monitor learning rate
    callbacks = [LearningRateMonitor(logging_interval='step')]

    # construct full experiment name which contains hyper parameters
    exp_name = args.exp_name + "_" + dt.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%f') + model.signature()

    # checkpoint directory
    checkpoint_dir = os.getcwd() + '/checkpoints/' + args.exp_name
    os.makedirs(checkpoint_dir, exist_ok=True)

    # load existing, or create a new checkpoint
    detected_checkpoint = None
    if args.use_checkpoint:
        checkpoint_list = os.listdir(checkpoint_dir)
        checkpoint_list.sort(reverse=True)
        print("Checkpoint list: ", checkpoint_list)
        for checkpoint in checkpoint_list:
            if checkpoint.startswith(model.signature()):
                print("Checkpoint found!")
                detected_checkpoint = checkpoint_dir + "/" + checkpoint
                # exp_name = "CHK_" + exp_name
                break

    # setup a checkpoint callback
    checkpoint_flag = False
    if args.make_checkpoint:
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                              monitor='neg_elbo',
                                              mode='min',
                                              filename=model.signature()+'-{epoch:02d}-{neg_elbo:.2f}',
                                              period=args.check_val_every_n_epoch)
        callbacks.append(checkpoint_callback)
        checkpoint_flag = True

    # fit model
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=WandbLogger(save_dir="logs/", name=exp_name,
                                                               project='dynamic-dropout-'+args.dataset),
                                            progress_bar_refresh_rate=1000,
                                            flush_logs_every_n_steps=1000,
                                            log_every_n_steps=1000,
                                            distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
                                            terminate_on_nan=True,
                                            max_epochs=10000,
                                            callbacks=callbacks,
                                            resume_from_checkpoint=detected_checkpoint,
                                            default_root_dir=checkpoint_dir,
                                            num_sanity_val_steps=0,
                                            checkpoint_callback=checkpoint_flag,
                                            automatic_optimization=False
                                            )
    trainer.fit(model)


def get_parser():

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--verbose_frequency', type=int, default=0, help='The frequency of logs. 0 means None.')
    parser.add_argument('--root', type=str, default="./data/", help='Path to data folder.')
    parser.add_argument('--exp_name', type=str, default="NoName", help='The name of the experiment.')
    parser.add_argument('--seed', type=int, default=13, help='Random seed.')
    parser.add_argument('--lrate', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--lrate_decay', type=float, default=0.96, help='Learning rate decay.')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--dataset', type=str, default="", help='Dataset.')
    parser.add_argument('--task', type=str, default="", help='Task.')
    parser.add_argument('--iters', type=int, default=2000000, help='Number of training iterations.')
    parser.add_argument('--hid_dim', type=int, default=128, help='Hidden states of recurrent units dimensionality.')
    parser.add_argument('--z_dim', type=int, default=32, help='Latent representation z dimensionality.')
    parser.add_argument('--eval_iterations', type=int, default=10000, help='Frequency of evaluation logs.')
    parser.add_argument('--kl_rate', type=float, default=0, help='Linear KL annealing rate: 1 means no annealing; '
                                                                 '0 means there is no KL term i.e. vanilla AE.')
    parser.add_argument('--kl_start', type=float, default=0, help='Initial KL(z) term value.')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length.')
    parser.add_argument('--interrupt_step', type=int, default=1000000, help='In which step to interrupt training.')
    parser.add_argument('--use_testset', action='store_true', help='Whether to validate on testset during training.')
    parser.add_argument('--hid_dropout', type=float, default=0.0, help='Dropout on hidden states, not teacher forcing.')
    parser.add_argument('--ema_rate', type=float, default=0.0, help='Exponential moving average.')
    parser.add_argument('--free_bits', type=float, default=0, help='Free bits threshold.')
    parser.add_argument('--make_checkpoint', action='store_true', help='Flag to indicate whether to make a checkpoint.')
    parser.add_argument('--use_checkpoint', action='store_true', help='Flag to indicate whether to resume training.')
    # for baselines
    parser.add_argument('--tf_level', type=float, default=1.0, help='The level of teacher forcing.')
    parser.add_argument('--decoder_dropout', type=float, default=0.0, help='Normal dropout level, in decoder.')
    parser.add_argument('--wdp', type=float, default=0.0, help='Word dropout level, in decoder.')
    # for adversarial dropout
    parser.add_argument('--ddr', type=float, default=0.0, help='Dynamic dropout rate.')
    parser.add_argument('--use_soft_mask', action='store_true', help='Use soft mask?')
    parser.add_argument('--use_adam', action='store_true', help='Learning rate of dynamic dropout.')
    parser.add_argument('--dd_lrate', type=float, default=0.1, help='Learning rate of dynamic dropout.')
    parser.add_argument('--lambd', type=float, default=0.001, help='Lambda coefficient for regularization.')
    parser.add_argument('--random_scores', action='store_true', help='Debug mode: generate random scores.')
    parser.add_argument('--do_not_use_double_lstm', action='store_true', help='Ablation study on double-lstm effects.')

    return parser


if __name__ == '__main__':
    main()