from typing import Optional, List, Tuple, Dict, Union
from tokenizers import Encoding
import torch
import re
from fuse.utils import NDict
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import (
    TypedInput,
    list_to_tokenizer_string,
)


class InjectorToModularTokenizerLib:
    """
    InjectorTokenizer builds on top of ModularTokenizer.
    !!!!
    Note - this file contains only few utility (static) functions for InjectorTokenizerOp
    as a user, you are not expected to InjectorTokenizer directly, instead you should use fusedrug.data.tokenizer.ops.injector_tokenizer_ops.InjectorTokenizerOp
    !!!!

    applies a injector tokenizer

    injector tokenizer builds on top of modular tokenizer.
    its purpose is to build inputs_emb for the model (instead of input_ids)
        this allows to support more advanced inputs beyond token ids, like:
        * scalars inputs
        * embeddings vector within a single input

    supported syntax/format:

    for text following <@TOKENIZER-TYPE=SCALARS_LITERALS> supports the following format:
    ',' separated float values and/or <MASK> tokens -
        for example: "2.7,3.99,-12.9" or "<MASK><MASK>" or "2.19,<MASK>,3.19,<MASK>"

    for text following <@TOKENIZER-TYPE=SCALARS_FROM_DICT> is expected to be a key to the sample NDict
        for example: "blah.boo.banana"  or "data.input.encoder_input"
        note: in SCALARS_FROM_DICT you can't describe masked scalars (outputs) you can only describe inputs

    example usage:

    encoder_input:
    <@TOKENIZER-TYPE=AA><MOLECULAR_WEIGHT_IN_SOME_UNIT><@TOKENIZER-TYPE=SCALARS_LITERALS>0.3<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_NANOMOLAR><@TOKENIZER-TYPE=SCALARS_LITERALS><MASK><@TOKENIZER-TYPE=AA><SEQUENCE_NATURAL_START>ISGGDAIYSSTGRCSLGFNVRSGSTYYFLTAGICTDGATTWWANSARTTVLGTTSGSSFPNNDYGIVRYTNTTIPKDGTVGGQDITSAANATVGMAVTRRGSTTGTISGSVTALNATVNYGGGDVVYGMIRTNVCAEPGDSGGPLYSGTRAIGLTSGGSGNCSSGGTTFFQPVTEALVAYGVSVY<SEQUENCE_NATURAL_END>
    labels:
    <@TOKENIZER-TYPE=AA><MOLECULAR_WEIGHT_IN_SOME_UNIT><@TOKENIZER-TYPE=SCALARS_LITERALS>0.3<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_NANOMOLAR><@TOKENIZER-TYPE=SCALARS_LITERALS>12.4<@TOKENIZER-TYPE=AA><SEQUENCE_NATURAL_START>ISGGDAIYSSTGRCSLGFNVRSGSTYYFLTAGICTDGATTWWANSARTTVLGTTSGSSFPNNDYGIVRYTNTTIPKDGTVGGQDITSAANATVGMAVTRRGSTTGTISGSVTALNATVNYGGGDVVYGMIRTNVCAEPGDSGGPLYSGTRAIGLTSGGSGNCSSGGTTFFQPVTEALVAYGVSVY<SEQUENCE_NATURAL_END>

    """

    @staticmethod
    def build_placeholder_meta_tokenization(
        *,
        sequence: Union[str, list, tuple],
        sample_dict: Optional[NDict] = None,
    ) -> Tuple[str, List[str]]:
        """
        In order to avoid modifying and rewriting the logic in modular tokenizer, especially regarding padding, limitation of max length of certain sub-parts,
         we put placeholders to make sure that the total size is known/fixed and respects the meta instructions to the modular tokenizer

         Returns: a tuple with 2 elements
         (
            a single string with the full query containing placeholder tokens for FLOAT and VECTOR meta tokenizer parts,
            a list of [meta-tokenizer name, data, meta-tokenizer name, data, meta-tokenizer name, data,  ...]
         )
        """
        if not isinstance(sequence, str):
            if not isinstance(sequence, (list, tuple)):
                raise Exception(
                    f"Expected sequence to be either string or a list of TypedInput elements. Instead got {type(sequence)}"
                )
            if len(sequence) > 0:
                if isinstance(sequence[0], TypedInput):
                    sequence_str = list_to_tokenizer_string(
                        sequence
                    )  # currently supporting it in this simple way. Consider optimizing if it causes a bottleneck.
                else:
                    raise Exception(
                        f"Expected sequence to be either string or a list of TypedInput elements. Got a list, but the first element is of type {type(sequence[0])}"
                    )
        else:
            sequence_str = sequence
        hints_and_subseq = re.split("<@TOKENIZER-TYPE=([^>]*)>", sequence_str)[
            1:
        ]  # the first element is blank - removing it
        assert (
            len(hints_and_subseq) > 0 and len(hints_and_subseq) % 2 == 0
        ), f"Error: expecting leading modular tokenizer hints followed by a sequence to tokenize, got {sequence}"

        with_placeholders = []

        for tokenizer_type, subseq in zip(
            hints_and_subseq[::2], hints_and_subseq[1::2]
        ):
            if tokenizer_type.startswith("SCALARS_"):
                with_placeholders.append(
                    "<@TOKENIZER-TYPE=AA>"
                )  # won't use AA tokens, just an arbitrary one to be able to use a token like <SCALAR>

                if (
                    tokenizer_type == "SCALARS_LITERALS"
                ):  # note: masking is only supported in literals (not in "from dict")
                    values = subseq.split(",")
                    # seq = "<SCALAR>" * len(values)
                    seq = "".join(
                        [
                            "<MASKED_SCALAR>" if x == "<MASK>" else "<SCALAR>"
                            for x in values
                        ]
                    )
                elif tokenizer_type == "SCALARS_FROM_DICT":
                    if sample_dict is None:
                        raise Exception(
                            "SCALARS_FROM_DICT used but the provided sample_dict is None"
                        )
                    values = sample_dict[subseq]
                    assert len(values.shape) == 1
                    seq = "<SCALAR>" * len(values)
                else:
                    raise Exception(f"tokenizer_type={tokenizer_type} is not supported")

                with_placeholders.append(seq)

            elif tokenizer_type.startswith("VECTORS_"):
                raise Exception("VECTOR_* are not supported yet")
            else:
                with_placeholders.append("<@TOKENIZER-TYPE=" + tokenizer_type + ">")
                with_placeholders.append(subseq)

        return "".join(with_placeholders), hints_and_subseq

    @staticmethod
    def prepare_info_for_model_step(
        *,
        per_meta_tokenizer_data: List[str],
        per_meta_encoding_including_placeholders: List[Encoding],
        sample_dict: Optional[NDict] = None,
    ) -> Dict:
        """
        since we:
        1. Need to use the model embedding layer (allowing gradients flow if needed)
        2. We prefer not to use the model during the data pipeline

        In this function we prepare everything so that during the train/val/test_step we'll be able to do what's needed before doing the forward pass

        Args:
            per_meta_tokenizer_data: a list of [meta-tokenizer name, data, meta-tokenizer name, data, meta-tokenizer name, data,  ...]
            per_meta_encoding_including_placeholders: a list of Encoding elements. This is used to extract per tokenizer final tokens num (after all of the padding and cropping logic was already done)
            sample_dict: a fuse sample_dict - optional.
                needed only if the meta tokenizer instruction uses a syntax of lookup from the dictionary


        """
        scalars_indices = []
        scalars_values = []
        scalars_masked_indices = []
        prev_index_end = -1

        for tokenizer_name, curr_str_data, curr_placeholder_encoding in zip(
            per_meta_tokenizer_data[::2],
            per_meta_tokenizer_data[1::2],
            per_meta_encoding_including_placeholders,
        ):
            if tokenizer_name.startswith("SCALARS_"):
                if "SCALARS_LITERALS" == tokenizer_name:
                    curr_str_data = curr_str_data.strip().split(",")
                    if len(curr_str_data) != len(curr_placeholder_encoding.ids):
                        raise Exception(
                            f"should match expected length. Found length {len(curr_str_data)} but placeholders length was {len(curr_placeholder_encoding.ids)}"
                        )

                    curr_indices = []
                    curr_data = []

                    for i, val in enumerate(curr_str_data):
                        if val != "<MASK>":
                            curr_indices.append(i + prev_index_end + 1)
                            curr_data.append(float(val))
                        else:
                            scalars_masked_indices.append(i + prev_index_end + 1)

                    if len(curr_indices) > 0:
                        curr_indices = torch.tensor(curr_indices, dtype=torch.int64)
                        curr_data = torch.tensor(curr_data, dtype=torch.float32)

                        scalars_indices.append(curr_indices)
                        scalars_values.append(curr_data)

                        assert len(curr_data.shape) == 1

                    prev_index_end += len(curr_str_data)
                elif "SCALARS_FROM_DICT" == tokenizer_name:
                    if sample_dict is None:
                        raise Exception(
                            "SCALARS_FROM_DICT used but the provided sample_dict is None"
                        )
                    curr_data = sample_dict[curr_str_data]
                    assert len(curr_data.shape) == 1
                    curr_indices = torch.arange(
                        prev_index_end + 1, prev_index_end + 1 + curr_data.shape[0]
                    )

                    scalars_indices.append(curr_indices)
                    scalars_values.append(curr_data)

                    prev_index_end += curr_data.shape[0]

                else:
                    raise Exception(
                        "Only supported SCALARS_* tokenizers are SCALARS_LITERALS and SCALARS_FROM_DICT"
                    )

            elif tokenizer_name.startswith("VECTORS_"):
                raise NotImplementedError
            else:
                prev_index_end += len(curr_placeholder_encoding.ids)

        if len(scalars_indices) > 0:
            scalars_indices = torch.concat(scalars_indices)
            scalars_values = torch.concat(scalars_values)
        else:
            scalars_indices = None
            scalars_values = None

        if len(scalars_masked_indices) > 0:
            scalars_masked_indices = torch.tensor(
                scalars_masked_indices, dtype=torch.int64
            )
        else:
            scalars_masked_indices = None

        return {
            "scalars_indices": scalars_indices,  # 1d - its length is the number of actual scalars (provided) found
            "scalars_values": scalars_values,  # 1d - values of provided scalars
            "scalars_masked_indices": scalars_masked_indices,  # 1d - indices of masked scalars
        }
