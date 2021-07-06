import os

from repro_lap_reg.submit_to_cluster.utils import Paths


def iain_to_bayes(iain, bayes, password, print_only=False):
    """
    Sends data from Iain's laptop to the Bayes cluster.

    Parameters
    ----------
    iain: str
        Path to location on Iain's laptop.

    bayes: str
        Path to location on the Bayes cluster.

    password: str
        Password for Bayes cluster. Don't worry, this is not also the password for Iain's email!

    print_only: bool
        Only print the command, don't run it (for debugging)
    """

    command = "sshpass -p '{}' rsync -aP --delete {} " \
              "idc9@bayes.biostat.washington.edu:{}".format(password,
                                                            iain, bayes)

    if print_only:
        print(command)
    else:
        os.system(command)


def bayes_to_iain(bayes, password, iain='./', print_only=False):
    """
    Sends data from the Bayes cluster to Iain's laptop.

    Parameters
    ----------
    bayes: str
        Path to location on the Bayes cluster.

    iain: str
        Path to location on Iain's laptop.

    password: str
        Password for Bayes cluster. Don't worry-- this is not also the password for my email!

    print_only: bool
        Only print the command, don't run it (for debugging)
    """
    command = "sshpass -p '{}' rsync -aP idc9@bayes.biostat.washington.edu:"\
              "{} {}".format(password, bayes, iain)

    if print_only:
        print(command)
    else:
        os.system(command)


def get_sim_results(model, name, password, print_only=False):
    """
    Gets simulation results from the Bayes cluster.

    Parameters
    ----------
    name: str
        Name of simulation

    password: str
        Password for Bayes cluster. Don't worry-- this is not also the password for my email!

    print_only: bool
        Only print the command, don't run it (for debugging)
    """
    bayes = os.path.join(Paths('bayes').results_dir, model, name)
    iain = os.path.join(Paths('iain_laptop').results_dir, model)
    bayes_to_iain(bayes=bayes, iain=iain, password=password,
                  print_only=print_only)


def get_sim_outdata(model, name, password, print_only=False):
    """
    Gets simulation out data from the Bayes cluster.

    Parameters
    ----------
    name: str
        Name of simulation

    password: str
        Password for Bayes cluster. Don't worry-- this is not also the password for my email!

    print_only: bool
        Only print the command, don't run it (for debugging)
    """
    bayes = os.path.join(Paths('bayes').out_data_dir, model, name)
    iain = os.path.join(Paths('iain_laptop').out_data_dir, model)
    bayes_to_iain(bayes=bayes, iain=iain, password=password,
                  print_only=print_only)
