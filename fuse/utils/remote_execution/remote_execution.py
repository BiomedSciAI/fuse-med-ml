import os
from typing import Optional, List, Tuple
import getpass
from .shell_handler import ShellHandler
from os.path import abspath, join, dirname
from glob import glob
from collections import namedtuple

SCRIPT_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script_runner")


def get_script_runner_path(postfix="_DETACHED"):
    return SCRIPT_RUNNER_PATH + postfix + ".sh"


RemoteCommand = namedtuple("RemoteCommand", "machine command")


class RemoteExecution:
    def __init__(
        self,
        conda_env: str,
        password: str,
        user: Optional[str] = None,
        allow_manual_password_typing=True,
        verbose=0,
    ):
        self._conda_env = conda_env

        if password is None:
            if allow_manual_password_typing:
                password = getpass.getpass(prompt="password:")
            else:
                raise Exception("No password provided (and allow_manual_password_typing is disabled)")
        self._password = password

        if user is None:
            user = getpass.getuser()
        self._user = user
        self._verbose = verbose

    def run_remote_machine_command(self, machine: str, cmd: str) -> Tuple[str, str]:
        """
        connects to [machine] with ssh and runs [cmd].
        For example:
        run_remote_machine_command('server12', 'nvidia-smi')
        """
        shell = ShellHandler(machine, self._user, self._password)
        shin, shout, sherr = shell.execute(cmd)
        if self._verbose > 0:
            print("ssh returned:")
            print("shout:")
            _ = [print(x) for x in shout]
            print("sherr:")
            _ = [print(x) for x in sherr]
            print("\n\n\n")
        return shout, sherr

    def run_remote_python_command(
        self,
        *python_args,
        machine: str = None,
        gpu: str = None,
        log_output: str = None,
        auto_add_repo_root_dirs_to_PYTHONPATH=False,
        verbose=0,
    ) -> Tuple[str, str]:
        """
        connects to [machine] with ssh and runs [cmd].
        :param gpu: str that will be used directly as CUDA_VISIBLE_DEVICES
        For example:
        run_remote_python_command(
            '/path/to/some/script.py',
            '12',
            4,
            'banana',
            #

            machine='server12',
            gpu='1,4',
            log_output='./my_log1.txt',
        )

        this will run the python 'script.py', passing args '12 4 banana' to it.
        It will run it on machine 'server12', enabling gpus 1 and 4,
        and logging stdout+stderr to 'my_log1.txt'
        """

        print(f"total of {len(python_args)} python args")
        for a in python_args:
            print(a)

        assert len(python_args) > 0
        python_args = " ".join(python_args)

        assert isinstance(machine, str)
        assert isinstance(gpu, str)
        assert isinstance(log_output, str)

        if auto_add_repo_root_dirs_to_PYTHONPATH:
            all_repos_base_dir = abspath(join(dirname(abspath(__file__)), "../../"))
            repo_dirs = glob(all_repos_base_dir + "/*/")
            PYTHONPATH = ":".join(repo_dirs)
        else:
            PYTHONPATH = "/dummy"  # it is a positional arg, so we gotta give it something

        script_runner = get_script_runner_path()

        cmd = f"bash {script_runner} "
        cmd += f"{gpu} {self._conda_env} {PYTHONPATH} {log_output} "
        cmd += python_args

        print(cmd)
        # assert False #

        shell = ShellHandler(machine, self._user, self._password)
        shin, shout, sherr = shell.execute(cmd)
        if verbose > 0:
            print("ssh returned:")
            print("shout:")
            _ = [print(x) for x in shout]
            print("sherr:")
            _ = [print(x) for x in sherr]
            print("\n\n\n")
        return shout, sherr

    def run_multi_machines_commands(self, commands: List[RemoteCommand], verbose=0) -> None:
        """
        Execute a list of RemoteCommand-s on remote machines

        For example:
        cmds = [
            RemoteCommand(machine='server11', cmd='echo test123'),
            RemoteCommand(machine='server23', cmd='echo test456'),
        ]
        run_multi_machines_commands(cmds)
        """

        _ERR_TEXT = '"commands" is expected to be a list of RemoteCommand instances'

        if not isinstance(commands, list):
            raise Exception(_ERR_TEXT)
        if 0 == len(commands):
            raise Exception("commands is empty")
        if not isinstance(commands[0], RemoteCommand):
            raise Exception(_ERR_TEXT)

        for rem_cmd in commands:
            self.run_remote_machine_command(rem_cmd.machine, rem_cmd.command, verbose=verbose)
