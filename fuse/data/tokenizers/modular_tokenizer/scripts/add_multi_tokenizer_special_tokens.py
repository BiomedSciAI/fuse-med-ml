import click
from typing import Union, List
from fuse.data.tokenizers.modular_tokenizer.modular_tokenizer import ModularTokenizer


@click.command()
@click.argument(
    "tokenizer-path",
    # "-p",
    default="../pretrained_tokenizers/bmfm_modular_tokenizer",
    # help="the directory containing the modular tokenizer",
)
@click.option(
    "--added-tokens",
    default=None,
    help="list of tokens to add",
)
@click.option(
    "--output-path",
    "-o",
    default=None,
    help="path to write tokenizer in",
)
# # this needs to be run on all the related modular tokenizers
def main(
    tokenizer_path: str, output_path: Union[str, None], added_tokens: List[str]
) -> None:
    print(f"adding special tokens to {tokenizer_path}")
    if output_path is None:
        output_path = tokenizer_path
    else:
        print(f"output into  {output_path}")

    tokenizer = ModularTokenizer.load(path=tokenizer_path)

    # Update tokenizer with special tokens:
    added_tokens = added_tokens.split(",")
    tokenizer.update_special_tokens(
        added_tokens=added_tokens,
        save_tokenizer_path=output_path,
    )

    print("Fin")


if __name__ == "__main__":
    main()
