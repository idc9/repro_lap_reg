import os
import sys
import __main__
import numpy as np

from warnings import warn
from repro_lap_reg.submit_to_cluster.utils import Paths, which_computer


def add_args_for_bayes(parser, force_add=False):
    """
    Adds arguments to argparse for the Bayes cluster at UW's biostat department.

    Parameters
    ----------
    parser: argparse.ArgumentParser()

    force_add: bool

    Output
    ------
    parser:argparse.ArgumentParser()

    Example
    -------
    parser = argparse.ArgumentParser(description='A parser.')
    parser = add_args_for_bayes(parser)
    args = parser.parse_args()

    """
    if which_computer() != 'bayes' and not force_add:
        return parser

    parser.add_argument('--mem', default='8G',
                        help='Memory to allocate for each job.')

    parser.add_argument('--queue', default='w-bigmem.q',
                        choices=['w-bigmem.q', 'w-normal.q', 'normal.q'],
                        help='Which queue to submit to.')

    parser.add_argument('--time', default=None,
                        help='Time to allocate for the job.')

    parser.add_argument('--node', default=None,
                        help='Which node to use e.g. b34 or b34|b35.')

    parser.add_argument('--n_slots', default=None,
                        help='Number of job slots to request.')

    parser.add_argument('--job_name', default=None,
                        help='Name of the job.')

    parser.add_argument('--submit', action='store_true',
                        help='Whether to submit the script or to just run it.')

    parser.add_argument('--override_submit', action='store_true',
                        help='Do not submit, even if --submit is called.')

    parser.add_argument('--print', action='store_true', dest='print_only',
                        help='Only print the command.')

    return parser


def maybe_submit_on_bayes(args):
    """
    Submits job on bayes then exits. If run from a computer other than the Bayes cluster this function will not do anything.

    Parameters
    ----------
    parser: argparse.ArgumentParser()

    Example
    -------
    parser = argparse.ArgumentParser(description='A parser.')
    parser = add_args_for_bayes(parser)
    args = parser.parse_args()
    maybe_submit_on_bayes(args)

    """

    if which_computer() == 'bayes' and args.submit \
            and not args.override_submit:

        py_fpath = os.path.join(os.getcwd(), __main__.__file__)
        # new_args = ' '.join(sys.argv[1:]) + ' --override_submit'
        # new_args = ' '.join(argv) + ' --override_submit'

        # reformat node argument
        argv = sys.argv[1:]
        if '--node' in argv:
            idx = np.where(np.array(argv) == '--node')[0].item()
            argv[idx + 1] = "'{}'".format(argv[idx + 1])
        new_args = " ".join(argv) + " --override_submit"

        py_command = '{} {}'.format(py_fpath, new_args)

        bayes_command = get_bayes_command(py_command, args)

        print('submission command:', bayes_command)

        if not args.print_only:
            os.system(bayes_command)

        sys.exit(0)


def get_bayes_command(py_command, args):
    """
    Constructs the submission command for the Bayes cluster.

    Parameters
    ----------
    py_command: str
        The python command to be run.

    args: argparse.ArgumentParser()
        The parser after having been passed through add_args()

    Output
    ------
    command: str
        The command that will run the python script on the cluster.
    """

    command = 'qsub -q {} -V'.format(args.queue)

    if args.time is not None:
        command += ' -l h_rt={}'.format(args.time)

    if args.mem is not None:
        command += ' -l h_vmem={}'.format(args.mem)

    if args.node is not None:
        command += ' -l h="{}"'.format(args.node)

    if args.n_slots is not None:

        n_slots = args.n_slots

        if n_slots == 'all':
            n_slots = 12

        if int(n_slots) > 12:
            warn('n_slots={} > 12 which is the maximum number of '
                 'available slots (Im pretty sure)'.format(n_slots))

        command += ' -pe local {}'.format(n_slots)

    if args.job_name is not None:
        command += ' -N {}'.format(args.job_name)

    # output files
    # out_fpath = os.path.join(Paths().cluster_out_dir,
    #                          '\$JOB_NAME__\$JOB_ID.out')
    # err_fpath = os.path.join(Paths().cluster_out_dir,
    #                          '\$JOB_NAME__\$JOB_ID.err')
    out_fpath = os.path.join(Paths().cluster_out_dir,
                             '$JOB_NAME__$JOB_ID.out')
    err_fpath = os.path.join(Paths().cluster_out_dir,
                             '$JOB_NAME__$JOB_ID.err')

    command += ' -o {} -e {}'.format(out_fpath, err_fpath)

    command += ' /home/guests/idc9/bayes_scripts/run_python.sh' \
               ' {}'.format(py_command)

    # command += ' ' + py_command
    # command = 'qsub -q {queue} -V -l h_rt={time} -l h_vmem={mem} '\
    #     '/home/guests/idc9/bayes_scripts/run_python.sh
    # {py_command}'.format(**bayes_args)

    # command = 'qsub -V /home/guests/idc9/bayes_scripts/run_python.sh
    # {py_command}'.format(**bayes_args)
    return command
