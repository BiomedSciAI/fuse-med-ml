from dataset import PhysioNetCinC
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="config_PhysioNetCinC")
def main(cfg: DictConfig):
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    token2idx, ds_train, ds_valid, _ = PhysioNetCinC.dataset(**cfg.dataset)


if __name__ == "__main__":
    main()
