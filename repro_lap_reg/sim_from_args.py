from copy import deepcopy


def add_one_sim_config(parser):
    parser = add_simulation_config(parser)
    parser = add_block_config(parser)
    parser = add_data_config(parser)
    parser = add_penalty_config(parser)
    parser = add_cv_config(parser)
    return parser


def add_penalty_config(parser):
    parser.add_argument('--n_pen_vals', default=100, type=int,
                        help='Number of penalty values to tune over.')

    parser.add_argument('--pen_min_mult_fcp', default=1e-3, type=float,
                        help="Multiplier to determine the min pen value for FCP algorithms. If no argument is provided, will use the model's default.")

    parser.add_argument('--pen_min_mult_fclsp', default=1e-3, type=float,
                        help="Multiplier to determine the min pen value for FCLSP algorithms. If no argument is provided, will use the model's default.")

    parser.add_argument('--pen_spacing', default='log', type=str,
                        choices=['lin', 'log'],
                        help='Penalty sequence spacing.')

    # which penalty function to use
    parser.add_argument('--pen_func', default='scad', type=str,
                        choices=['scad'],
                        help='Which concave penalty function to use.')

    # TODO: what should we use for the default here?
    # for SCAD it is normaly 3.7
    parser.add_argument('--pen_second_param', default=2.1, type=float,
                        help='Secondary parameter for penalty functions e.g. this is the a parameter for SCAD and MCP.')

    parser.add_argument('--n_lla_steps_max', default=1000,
                        type=int,
                        help='Maximum number of LLA steps when we run the LLA algorithm to convergence.')

    parser.add_argument('--lla_tracking', default=1,
                        type=int,
                        help='How much data to track during LLA optimization.')

    return parser


def add_cv_config(parser):
    parser.add_argument('--cv', default=5, type=int,
                        help='Number of cross-validation folds.')

    parser.add_argument('--cv_n_jobs', default=None, type=int,
                        help='Cross-validation parallelization.')

    parser.add_argument('--cv_select_rule', default='best', type=str,
                        choices=['best', '1se'],
                        help='Cross-validation selection rule.')

    parser.add_argument('--cv_verbose', default=0, type=int,
                        help='Amount of printout during cross-validation.')

    return parser


def get_pen_kws_from_config(args):

    pen_kws = {'pen_func': args.pen_func,
               'n_pen_vals': args.n_pen_vals,
               'pen_spacing': args.pen_spacing,

               'cv_select_rule': args.cv_select_rule,
               'cv': args.cv,
               'cv_n_jobs': args.cv_n_jobs,
               'cv_verbose': args.cv_verbose,

               'lla_kws': {'tracking_level': args.lla_tracking}
               }

    fcp_kws = deepcopy(pen_kws)
    fclsp_kws = deepcopy(pen_kws)

    if args.pen_func == 'scad':
        fclsp_kws['pen_func_kws'] = {'a': args.pen_second_param}
        fcp_kws['pen_func_kws'] = {'a': 3.7}  # standard default
    else:
        raise NotImplementedError

    fclsp_kws['pen_min_mult'] = args.pen_min_mult_fclsp
    fcp_kws['pen_min_mult'] = args.pen_min_mult_fcp
    # # only add if not None
    # if args.pen_min_mult_fcp is not None:
    #     fcp_kws['pen_min_mult'] = args.pen_min_mult_fcp

    # # only add if not None
    # if args.pen_min_mult_fclsp is not None:
    #     fclsp_kws['pen_min_mult'] = args.pen_min_mult_fclsp

    return fclsp_kws, fcp_kws


def add_data_config(parser):
    parser.add_argument('--n_samples', default=100, type=int,
                        help='Size of dataset.')

    parser.add_argument('--data_seed', default=2342, type=int,
                        help='Seed for sampling dataset.')

    return parser


def add_block_config(parser):
    parser.add_argument('--block_sizes', nargs='+', default=[5, 5, 5],
                        help='Sizes of the blocks.')

    parser.add_argument('--block_size_str', default=None,
                        help='String input to describe block sizes. This string should either be formatted like 1) S_N  or 2) S_N_I where S = size of blocks, N = number of blocks and I = number of isolated vertices')

    return parser


def get_block_sizes_from_config(args):
    # format block sizes
    if args.block_size_str is not None:
        split_str = args.block_size_str.split('_')

        if len(split_str) == 2:
            S, N = split_str
            block_sizes = [int(S)] * int(N)
        elif len(split_str) == 3:
            S, N, n_iso = split_str
            block_sizes = [int(S)] * int(N)
            block_sizes += [0] * int(n_iso)
    else:
        block_sizes = [int(b) for b in args.block_sizes]
        assert len(block_sizes) >= 1

    return block_sizes


def add_simulation_config(parser):
    # simulation setup
    parser.add_argument('--out_data_dir',
                        default='out_data',
                        help='Directory for output data.')

    parser.add_argument('--script_dir',
                        default='scripts',
                        help='Directory where scripts are stored.')

    parser.add_argument('--results_dir',
                        default='results',
                        help='Directory where results are stored.')

    parser.add_argument('--name', default='meow',
                        help='Name of the experiment.')

    parser.add_argument('--mini', action='store_true', default=False,
                        help='Run a mini simulation for debugging.')

    parser.add_argument('--save_models', action='store_true', default=False,
                        help='Save the models to disk.')

    parser.add_argument('--make_viz', action='store_true', default=False,
                        help='Call visualization script.')

    parser.add_argument('--results_zero_tol', default=1e-9, type=float,
                        help='Zero tolerance for results about parameter support.')

    return parser


def add_multiple_submit_config(parser):
    parser.add_argument('--out_data_dir',
                        default='out_data',
                        help='Directory for output data.')

    parser.add_argument('--results_dir',
                        default='results',
                        help='Directory for results.')

    parser.add_argument('--script_dir',
                        default='scripts',
                        help='Directory where scripts are stored.')

    parser.add_argument('--mini', action='store_true', default=False,
                        help='Run a mini simulation for debugging.')

    parser.add_argument('--metaseed', default=234234, type=int,
                        help='Seed to setup all seeds for the experiment.')

    parser.add_argument('--n_mc_reps', default=3, type=int,
                        help='Number of Monte-Carlo reptitions.')

    parser.add_argument('--mc_start_idx', default=0, type=int,
                        help='Which monte carlo index to start at (this allows'
                        'you to run additional monte-carlo simulations).')

    parser.add_argument('--make_viz', action='store_true', default=False,
                        help='Run the visualization script,'
                             'not the simulation.')

    return parser
