import logging
import sys, os
import time
import yaml


# Class to redirect all print() outputs and errors to both terminal and log file
class Logger(object):
    def __init__(self, log_filename, stream):
        self.terminal = stream
        self.log = open(log_filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):    # needed for python 3 compatibility.
        # self.terminal.flush()
        # self.log.flush()
        pass


def get_params_filename_from_cmd_args():
    num_args = len(sys.argv)
    if num_args != 2:
        print('Usage:')
        print('python train_dnn_mnist my_config.yaml')
        sys.exit(1)

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print(f'Config File {filename} does not exist')
        sys.exit()

    return filename


def init_logging(log_filename):
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)


def get_params_from_file(params_filename):
    file = open(params_filename, "r")
    params = yaml.load(file, Loader=yaml.FullLoader)
    return params


# ---------------------------------------------------------
# Global item management (figures, dataframes to save etc)

# A global list to keep figures and dataframes (to be saved at the end of the program)
figures_list = []
dataframes_list = []

def add_figure_to_save(fig, name=None):
    figures_list.append((fig, name))

def add_dataframe_to_save(df, name=None):
    dataframes_list.append((df, name))

def clear_all_figures():
    for (fig, name) in figures_list:
        fig.clear()
    figures_list.clear()

def clear_all_dataframes():
    dataframes_list.clear()

def save_all_figures(dir):
    for i, (fig, name) in enumerate(figures_list):
        figname = ''
        if name: figname = '_' + name

        filename = dir + '/fig_' + str(i+1) + figname + '.png'
        fig.savefig(filename)
    print('All figures saved to {}'.format(dir))

def save_all_dataframes(dir):
    for i, (df, name) in enumerate(dataframes_list):
        dfname = ''
        if name: dfname = '_' + name

        filename = dir + '/df_' + str(i+1) + dfname + '.csv'
        df.to_csv(filename)
    print('All dataframes saved to {}'.format(dir))