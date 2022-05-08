""" The main script for running Yahoo experiments. """

from lib import computation

# Name the experiment somehow
exp_name = "[Leo][PLS][CKPT][Adam]"
# exp_name = "[Leo][Adam]Adversary-2048H"

# BestSoFar:
# [SGD][rst-False]-s-1234-lr-1.0_lrd-0.96_b-32_h-2048_z-32_klr-3e-05-tf-1.0_dp-0.0_wdp-0.0_hdp-0.5_ema-0.9995_fb-0.0_ddr-0.2_lmbd-0.01_sm-False_ddlr-1.0_adam-False
# BestOfAdam:
# [Adam][rst-False]-s-1234-lr-1.0_lrd-0.96_b-32_h-2048_z-32_klr-3e-05-tf-1.0_dp-0.0_wdp-0.0_hdp-0.5_ema-0.9995_fb-0.0_ddr-0.2_lmbd-0.001_sm-False_ddlr-0.003_adam-True
# [Adam][rst-False]-s-1234-lr-1.0_lrd-0.96_b-32_h-2048_z-32_klr-3e-05-tf-1.0_dp-0.0_wdp-0.0_hdp-0.5_ema-0.9995_fb-0.0_ddr-0.2_lmbd-0.01_sm-False_ddlr-0.003_adam-True
# High Lambda:
# [Adam]s-1234-lr-1.0_lrd-0.96_b-32_h-2048_z-32_klr-3e-05-tf-1.0_dp-0.0_wdp-0.0_hdp-0.5_ema-0.9995_fb-0.0_ddr-0.2_lmbd-2.0_sm-False_ddlr-0.002_adam-True
# [SGD]s-1234-lr-1.0_lrd-0.96_b-32_h-2048_z-32_klr-3e-05-tf-1.0_dp-0.0_wdp-0.0_hdp-0.5_ema-0.9995_fb-0.0_ddr-0.2_lmbd-3.0_sm-False_ddlr-0.1_adam-False
# select computation nodes
nodes = [computation.Local()]

# parameters
parameters = dict()

# for debugging:
# parameters['limit_val_batches'] = [2]
# parameters['limit_train_batches'] = [2]
# parameters['random_scores'] = ['']
parameters['verbose_frequency'] = [200]

# checkpointing:
parameters['use_checkpoint'] = ['']
parameters['make_checkpoint'] = ['']

# baseline dropout parameters
parameters['tf_level'] = [1]  # from previous work
parameters['decoder_dropout'] = [0]  # default is 0.5
parameters['wdp'] = [0]  # default is 0.3

# adversarial dropout parameters -- the trick is to use small learning rates and large lambdas (as opposed to SNLI!)
parameters['use_adam'] = ['']  # do not change
parameters['ddr'] = [0.2]  # vary for ablations!
parameters['lambd'] = [3, 4, 5]  # not sure what is the best, so far lambda=2.0 | definitely lambda>1
parameters['dd_lrate'] = [0.001]  # do not change!

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
gpus = 0
gpu_memory = 10000
execution_time = 240
backend = 'ddp'
parameters['dataset'] = ["Yahoo"]
parameters['distributed_backend'] = [backend]
parameters['num_workers'] = [0]
parameters['seed'] = [1234]
parameters['check_val_every_n_epoch'] = [1]

# grid search
computation.grid_search(nodes, exp_name, parameters, gpus, gpu_memory, execution_time)

