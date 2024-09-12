import hydra
from omegaconf import DictConfig, OmegaConf
from fuse.data.tokenizers.modular_tokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict, Any


@hydra.main(
    config_path="../pretrained_tokenizers/configs",
    config_name="config_add_tokenizer_to_multi_tokenizer",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """script to add a tokenizer (and all special tokens from it and special_tokens.py) to an existing tokenizer.
    The old tokenizer is read from the in_path, tokenizer to add is taken from the tokenizer_to_add variable.
    max_possible_token_id will be updated if the new max is larger then the old one.
    Add the tokenizer_info of the new tokenizer, as usual.


    Args:
        cfg (DictConfig): the config file.
    """

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    new_max_token_id = cfg_raw["data"]["tokenizer"]["max_possible_token_id"]

    t_mult = ModularTokenizer.load(path=cfg_raw["data"]["tokenizer"]["in_path"])

    # Update tokenizer with special tokens:
    new_tokenizer_name = cfg_raw["data"]["tokenizer"]["tokenizer_to_add"]
    cfg_tokenizer_info = {
        info["name"]: info for info in cfg_raw["data"]["tokenizer"]["tokenizers_info"]
    }
    new_tokenizer_info = cfg_tokenizer_info[new_tokenizer_name]
    if new_max_token_id > t_mult._max_possible_token_id:
        print(
            f"updating the max possible token ID from {t_mult._max_possible_token_id} to {new_max_token_id}"
        )
        t_mult._max_possible_token_id = new_max_token_id

    t_mult.add_single_tokenizer(new_tokenizer_info)
    t_mult.save(path=cfg_raw["data"]["tokenizer"]["out_path"])
    print("Fin")


if __name__ == "__main__":
    main()
