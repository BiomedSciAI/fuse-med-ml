import hydra
from omegaconf import DictConfig, OmegaConf
from fuse.data.tokenizers.modular_tokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict, Any


@hydra.main(
    config_path="../pretrained_tokenizers/configs",
    config_name="tokenizer_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    cfg_tokenizer: Dict[str, Any] = cfg_raw["data"]["tokenizer"]
    t_mult = ModularTokenizer(
        **cfg_tokenizer,
    )

    t_mult.save(path=cfg_raw["data"]["tokenizer"]["out_path"])


if __name__ == "__main__":
    main()
