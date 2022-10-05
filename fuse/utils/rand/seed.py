import torch
import numpy as np
import random
import os


class Seed:
    """
    Random seed functionality - static methods
    """

    @staticmethod
    def set_seed(seed: int, deterministic_mode: bool = True) -> torch.Generator:
        """
        Fix a seed (numpy, torch and random) and set PyTorch mode to deterministic to reproduce results
        :param seed: seed to use
        :param deterministic_mode: set PyTorch deterministic mode
        :return: Random generator to be used when creating Dataloader
        """
        if deterministic_mode:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # random
        random.seed(seed)

        # torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(deterministic_mode)
        dataloader_rand_gen = torch.Generator()
        dataloader_rand_gen.manual_seed(seed)

        # numpy
        np.random.seed(seed)

        return dataloader_rand_gen

    @staticmethod
    def seed_worker_init(worker_id: int):
        """
        Function to provide to torch dataloader to set a seed per worker
        """
        worker_seed = torch.initial_seed() % 2**32
        Seed.set_seed(worker_seed)
