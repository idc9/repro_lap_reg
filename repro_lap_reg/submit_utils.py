from repro_lap_reg.utils import merge_dicts


def get_command_func(script_fpath,
                     base_kwargs={}, base_options=[]):
    """
    Returns a function that gets the command string to run the simulation.

    Parameters
    -----------
    script_fpath: str
        Path to the python script that runs the simulation.

    base_kwargs: dict
        Key word arguments that will be supplied to all simulations. These can be overwritten.

    base_options: list
        Options that that will be supplied to all simulations. These can be overwritten.

    Output
    ------
    get_command: callable(add_kwargs, add_options) -> str
        A function that returns the command to run the simulation.
        add_kwargs: dict
            Additional keyword arguments

        add_options: list
            Additional options to add.
    """

    def get_command(add_kwargs={}, add_options=[]):
        """
        Gets the command string to run a simulation.

        Parameters
        ----------
        add_kwargs: dict
            Additional keyword arguments

        add_options: list
            Additional options to add.

        Output
        ------
        command: str
            The command string to run a simulation. command will look like
            'python my_awesome_simulation.py --param_a 2 --option'
            and can be run by calling os.system(command).
        """

        # add in keywork arguments
        if add_kwargs is not None:
            kwargs = merge_dicts(base_kwargs, add_kwargs)
        else:
            kwargs = base_kwargs

        # add in options
        if add_options is not None:
            options = list(set(base_options).union(add_options))
        else:
            options = base_options

        command = 'python {}'.format(script_fpath)

        # key work arguments are parsed as --key value
        for k, v in kwargs.items():
            command += ' --{} {}'.format(k, v)

        # options are parsed as --option
        for o in options:
            command += ' --{}'.format(o)

        return command

    return get_command
