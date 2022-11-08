""" The main script for running SNLI experiments. """

from lib import computation

# BestSoFar:
# [Adam]-2021-05-23_22-36-296435_[rst-False]-s-13-lr-1.0_lrd-0.96_b-32_h-1024_z-32_klr-3e-05-tf-1.0_dp-0.0_wdp-0.0_hdp-0.5_ema-0.9995_fb-0.0_ddr-0.2_lmbd-0.01_sm-False_ddlr-0.1

# Name the experiment somehow
exp_name = "[Ablation][Double-LSTM]"

# select computation nodes
nodes = [computation.Local()]

# parameters
parameters = dict()

# used for debugging only:
# parameters['random_scores'] = ['']
parameters['verbose_frequency'] = [100]
parameters['do_not_use_double_lstm'] = [""]  # uncomment to use normal LSTM
# parameters['limit_val_batches'] = [1]
# parameters['limit_train_batches'] = [1]

# checkpointing:
# parameters['use_checkpoint'] = ['']
parameters['make_checkpoint'] = ['']

# baseline-dropout parameters
parameters['tf_level'] = [1]  # from previous work
parameters['decoder_dropout'] = [0.5]  # default is 0.5
parameters['wdp'] = [0.3]  # default is 0.3

# dynamic dropout parameters:
parameters['use_adam'] = ['']
parameters['ddr'] = [0.0] # [0.2]
parameters['lambd'] = [0.0]  # 0.1 is best so far
parameters['dd_lrate'] = [0.01]

# Other parameters
parameters['kl_rate'] = [0.00003]  # from previous work
parameters['kl_start'] = [0.1]  # from previous work
parameters['lrate'] = [1.0]  # from previous work
parameters['lrate_decay'] = [0.96]  # -----> TODO: maybe change this ???
parameters['batch_size'] = [32]  # default is 32 | previous work is 32
parameters['hid_dim'] = [1024]  # default is 1024  TODO: maybe change this ???
parameters['z_dim'] = [32]  # from previous work
parameters['interrupt_step'] = [300000]
parameters['use_testset'] = ['']  # using testset in validation
parameters['hid_dropout'] = [0.5]  # from previous work
parameters['free_bits'] = [0]  # default is 10
parameters['ema_rate'] = [0.9995]
# parameters['stochastic_weight_avg'] = ['']  # not working yet....

# Data and computation
gpus = 1
gpu_memory = 10000
execution_time = 240  # = 4 hours
backend = 'ddp'
parameters['dataset'] = ["SNLI"]
parameters['distributed_backend'] = [backend]
parameters['num_workers'] = [0]
parameters['seed'] = [1234]
parameters['check_val_every_n_epoch'] = [1]

# grid search
computation.grid_search(nodes, exp_name, parameters, gpus, gpu_memory, execution_time)