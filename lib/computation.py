"""
For ETHZ users:
To use this script, you need to change for each computing node,
the project folder on each computing node and data folder.
"""

from train import main as local_main
import subprocess
from abc import abstractmethod
import itertools
import random
import time


class Node(object):
    """
    Base class for describing computing nodes of snn_experiments.
    For every new computing node, one should make a realization of this class.
    * :attr: 'address': str the address of the computing node
    * :attr: 'root': str the home folder of all data on the computing node
    """
    def __init__(self):
        self.address = None
        self.root = None
        self.project_root = None

    def get_address(self):
        return self.address

    def get_root(self):
        return self.root

    def get_project_root(self):
        return self.project_root

    @abstractmethod
    def run_experiment(self, experiment_arguments, exp_name, gpus, gpu_memory, execution_time):
        """ Do the computation.
        :param experiment_arguments: string of arguments
        :param exp_name: custom name of the experiment
        :param gpus: number of gpus to run on
        :param gpu_memory: minimum gpu memory
        :param execution_time: expected execution time in minutes
        :return: [False/True] depending if job submission was successful
        """


class Local(Node):
    """ For executing computations locally. """
    def __init__(self):
        super(Local, self).__init__()
        self.project_root = "/mnt/d/dynamic-dropout-master/"
        self.root = self.project_root + "data/"
        self.address = "local"

    def run_experiment(self, experiment_arguments, exp_name, gpus, gpu_memory, execution_time):
        """ this is a blocking run """
        experiment_arguments = experiment_arguments + " --root {}".format(self.root)
        local_main(experiment_arguments)
        return True


class ISE(Node):
    """ For executing computations on ETHZ computing cluster. """
    def __init__(self):
        super(ISE, self).__init__()
        self.address = "isegpu2"
        self.root = "~/projects/exposure-bias/data/"
        self.project_name = "~/projects/exposure-bias"
        self.last_used = -1

    def run_experiment(self, experiment_arguments, exp_name, gpus, gpu_memory, execution_time):
        """ this is a non-blocking run """
        # augment experiment arguments
        experiment_arguments = experiment_arguments + " --root {}".format(self.root)
        # read out gpu usage
        ssh = subprocess.Popen(["ssh", "%s" % "djordjem@isegpu2.ethz.ch",
                                " nvidia-smi --query-gpu=memory.used  --format=csv"],
                               shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        gpu_usage = ssh.stdout.readlines()
        del gpu_usage[0]

        # find the first free GPU card and run the experiment on it (skip first one - is not working..)
        for i in range(8):
            if int(gpu_usage[i].split()[0]) < 50 and i > self.last_used:

                print("GPU number", i, "at ISEGPU2 seems to be available!")

                # construct the command
                command = "cd " + self.project_name + "; CUDA_VISIBLE_DEVICES=" + str(i) + " " \
                          "nohup python3.7 train.py " + experiment_arguments + " &" \
                          # " >> ./logdir/" + exp_name + str(random.uniform(0, 1)) + ".log &"

                print(command)
                print("Sleep for 3 seconds..")
                time.sleep(3)

                # submit the job
                subprocess.Popen(["ssh", "%s" % "djordjem@isegpu2.ethz.ch", command],
                                 shell=False,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
                self.last_used = i
                return True
        return False


class Leonhard(Node):
    """ For executing computations on ETHZ Leonhard. """
    def __init__(self):
        super(Leonhard, self).__init__()
        self.address = "login.leonhard.ethz.ch"
        self.root = "./data/"
        # self.root = "/nfs/nas12.ethz.ch/fs1201/infk_jbuhmann_project_leonhard/cardioml/projects/snn"
        self.client = None
        self.project_path = "/nfs/nas12.ethz.ch/fs1201/infk_jbuhmann_project_leonhard/cardioml/projects/dynamic-dropout/SequenceVAE"

    def run_experiment(self, experiment_arguments, exp_name, gpus, gpu_memory, execution_time):
        """ this is a non-blocking run """
        print("Sending a job to Leonhard.")
        # augment experiment arguments
        experiment_arguments = experiment_arguments + " --root {}".format(self.root)
        # construct the command                  #"module load cudnn/7.5; " \
        command = "source environment_setup.sh; cd " + self.project_path + "; " \
                  "bsub -n "+str(gpus*2) + " " \
                  "-R 'rusage[ngpus_excl_p="+str(gpus)+",mem=10000]' -R 'select[gpu_mtotal0>="+str(gpu_memory)+"]' " \
                  "-W " + str(execution_time) + " " \
                  "'python train.py"+experiment_arguments+"' "

        print(command)
        print("Sleep for 3 seconds..")

        time.sleep(3)

        # submit the job
        subprocess.Popen(["ssh", "-o StrictHostKeyChecking=no", "%s" % "djordjem@login.leonhard.ethz.ch", command],
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        print("Job sent!")

        return True


class Euler(Node):
    """ For executing computations on ETHZ Euler. """
    def __init__(self):
        super().__init__()
        self.address = "euler.ethz.ch"
        self.root = "./data/"
        self.client = None
        self.project_path = "~/projects/dynamic-dropout"

    def run_experiment(self, experiment_arguments, exp_name, gpus, gpu_memory, execution_time):
        """ this is a non-blocking run """
        # print("Sending a job to Euler.")
        # augment experiment arguments
        experiment_arguments = experiment_arguments + " --root {}".format(self.root)
        # construct the command                  #"module load cudnn/7.5; " \
        command_suffix = \
                  "bsub -n "+str(gpus*2) + " " \
                  "-R 'rusage[ngpus_excl_p="+str(gpus)+",mem=10000]' -R 'select[gpu_mtotal0>="+str(gpu_memory)+"]' " \
                  "-W " + str(execution_time) + " " \
                  "'python train.py"+experiment_arguments+"' "
        command = "source .bash_profile; cd " + self.project_path + "; " + command_suffix

        print(command_suffix)
        # print("Sleep for 3 seconds..")

        time.sleep(3)

        # submit the job
        subprocess.Popen(["ssh", "-o StrictHostKeyChecking=no", "%s" % "djordjem@euler.ethz.ch", command],
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        # print("Job sent!")

        return True


class MPI(Node):
    """ For executing computations on MPI cluster. """
    def __init__(self):
        super().__init__()
        self.address = "login.cluster.is.localnet"
        self.root = "./data/"
        self.client = None
        self.project_path = "/home/sbauer/dj/snn/"
        self.memory = 60 * 1024

    def run_experiment(self, experiment_arguments, exp_name, gpus, gpu_memory, execution_time):
        """ this is a non-blocking run """
        print("Sending a job to MPI Cluster.")
        memory = self.memory + gpus * 3 * 1024
        cpus = gpus * 2
        print("Requested memory", memory)
        # augment experiment arguments
        experiment_arguments += " --root {}".format(self.root)
        output_name = exp_name + str(random.uniform(0, 1))
        # construct the command
        command = "source .bash_profile; cd " + self.project_path + "; " \
                  "module load cuda/10.1; module load cudnn/7.6.5-cu10.1; " \
                  "printf 'executable = /usr/bin/python3" + "\n" +\
                  "arguments = train.py " + experiment_arguments + "\n" + \
                  "error = logsub/"+output_name+".err" + "\n" + \
                  "output = logsub/"+output_name+".out" + "\n" + \
                  "log = logsub/"+output_name+".log" + "\n" + \
                  "request_memory = " + str(memory) + "\n" + \
                  "requirements = TARGET.CUDAGlobalMemoryMb > "+str(gpu_memory)+" " + "\n" + \
                  "request_cpus = " + str(cpus) + "\n" + \
                  "request_gpus = " + str(gpus) + "\n" + \
                  "queue' > submission_file; " + \
                  "condor_submit_bid 1000 submission_file"

        print(command)
        print("Sleep for 5 seconds..")
        time.sleep(5)

        # submit the job
        subprocess.Popen(["ssh", "%s" % "sbauer@login.cluster.is.localnet", command],
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        print("Job sent!")

        return True


def execute_experiment(nodes, configuration, exp_name, gpus, gpu_memory, execution_time):
    # integrate all experiment pars into a string
    exp_name += ("-" + configuration['dataset'])
    exp_str = " --exp_name {} ".format(exp_name + "_")
    combination_str = " "
    # iterate over parameter configurations
    for key, val in configuration.items():
        if type(val) != list:
            combination_str += ' --{} {}'.format(key, val)
        else:
            combination_str += ' --{}'.format(key)
            for v in val:
                combination_str += ' {}'.format(v)
    # combination_str = ' '.join('--{} {}'.format(key, val) for key, val in combination.items())
    experiment_arguments = ' --gpus ' + str(gpus) + ' '
    experiment_arguments += exp_str + combination_str
    # find a computing node and run the experiment on it
    for node in nodes:
        # print("Trying node: ", node.address)
        success = node.run_experiment(experiment_arguments, exp_name, gpus, gpu_memory, execution_time)
        if success:
            # print("Connection established!")
            break
        else:
            print("This node is busy.")


def grid_search(nodes, exp_name, parameters, gpus, gpu_memory, execution_time=1440):

    # assemble parameter combinations
    keys, values = zip(*parameters.items())
    parameter_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # iterate over configurations
    for configuration in parameter_configurations:
        execute_experiment(nodes, configuration, exp_name, gpus, gpu_memory, execution_time)


def random_search(nodes, exp_name, parameters, number_of_experiments, gpus, gpu_memory, execution_time=1440):

    # assemble parameter combinations
    keys, values = zip(*parameters.items())
    parameter_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_number_of_configurations = len(parameter_configurations)
    probability_of_experiment = number_of_experiments * 1.0 / total_number_of_configurations

    # iterate over configurations
    for configuration in parameter_configurations:
        if random.uniform(0, 1) > probability_of_experiment:
            continue
        execute_experiment(nodes, configuration, exp_name, gpus, gpu_memory, execution_time)
