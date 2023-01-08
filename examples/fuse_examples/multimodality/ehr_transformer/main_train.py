from dataset import PhysioNetCinC
from typing import Any, Optional
import hydra
from omegaconf import DictConfig
from ehrtransformers.model.utils import WordVocab


@hydra.main(config_path=".", config_name="config_PhysioNetCinC")
def main(cfg: DictConfig):
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)

    corpus, ds_train, ds_valid, _ = PhysioNetCinC.dataset(**cfg.dataset)
    token2idx = WordVocab(corpus, max_size=None, min_freq=1).get_stoi()


if __name__ == "__main__":
    main()
