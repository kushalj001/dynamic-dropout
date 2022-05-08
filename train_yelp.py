""" The main script for running Yelp experiments. """

from lib import computation

# Name the experiment somehow
exp_name = "[Leo][YLP][Adam]"
# exp_name = "[Leo][Baseline]"
# exp_name = "[Leo][WDP-Baseline]"

nodes = [computation.Leonhard()]

# parameters
parameters = dict()

# for debugging:
# parameters['random_scores'] = ['']
parameters['verbose_frequency'] = [200]

# checkpointing:
parameters['use_checkpoint'] = ['']
parameters['make_checkpoint'] = ['']

# baseline dropout parameters
parameters['tf_level'] = [1]  # from previous work
parameters['decoder_dropout'] = [0]  # default is 0.5
parameters['wdp'] = [0]  # default is 0.3

# adversarial dropout parameters
parameters['use_adam'] = ['']  # do not change
parameters['ddr'] = [0.2]  # vary for ablations!
parameters['lambd'] = [3]  # not sure what is the best, so far lambda=2.0 | definitely lambda>1
parameters['dd_lrate'] = [0.1, 0.01, 0.001]  # do not change!

# other parameters
parameters['kl_rate'] = [0.00003]  # from previous work
parameters['kl_start'] = [0.1]  # from previous work
parameters['lrate'] = [1.0]  # from previous work
parameters['lrate_decay'] = [0.96]  # -----> TODO: not exactly the same -- original paper reduces lr on plateau
parameters['batch_size'] = [32]  # default is 32 | previous work is 32
parameters['hid_dim'] = [2048]  # default is 2048 | previous work is 1024  TODO: might be causing overfitting!!!!!!
parameters['z_dim'] = [32]  # default is 32 | previous work is 32
parameters['interrupt_step'] = [300000]
parameters['use_testset'] = ['']  # using testset in validation
parameters['hid_dropout'] = [0.5]  # from previous work
parameters['free_bits'] = [0]
parameters['ema_rate'] = [0.9995]

# data and computation
gpus = 1
gpu_memory = 10000
execution_time = 240
backend = 'ddp'
parameters['dataset'] = ["Yelp"]
parameters['distributed_backend'] = [backend]
parameters['num_workers'] = [0]
parameters['seed'] = [1234]
parameters['check_val_every_n_epoch'] = [1]

# grid search
computation.grid_search(nodes, exp_name, parameters, gpus, gpu_memory, execution_time)
