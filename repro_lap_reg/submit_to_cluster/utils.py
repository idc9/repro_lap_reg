import os
from os.path import join


def which_computer():
    """
    Detect if we are working on Iain's laptop or on the cluster
    """
    cwd = os.getcwd()
    if 'iaincarmichael' in cwd:
        return 'iain_laptop'  # running on Iain's laptop

    elif 'idc9' in cwd:
        return 'bayes'  # running on the Bayes cluster

    else:
        return None


class Paths(object):

    def __init__(self, computer=None):

        if computer is None:
            computer = which_computer()
        assert computer in ['iain_laptop', 'bayes']

        if computer == 'iain_laptop':
            self.script_dir = '/Users/iaincarmichael/Dropbox/Research/local_packages/python/repro_lap_reg/scripts'

            self.top_dir = '/Users/iaincarmichael/Dropbox/Research/laplacian_reg/sim'

        elif computer == 'bayes':

            self.script_dir = '/home/guests/idc9/local_packages/repro_lap_reg/scripts'

            self.top_dir = '/home/guests/idc9/projects/repro_lap_reg'

        self.cluster_out_dir = '/home/guests/idc9/cluster_out'

        self.out_data_dir = join(self.top_dir, 'out_data')
        self.results_dir = join(self.top_dir, 'results')
