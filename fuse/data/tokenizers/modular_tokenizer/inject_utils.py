from typing import Optional, List, Tuple, Dict, Union
from tokenizers import Encoding
import torch
import re
from fuse.utils import NDict
from fuse.data.tokenizers.modular_tokenizer.modular_tokenizer import (
    TypedInput,
    list_to_tokenizer_string,
)
from warnings import warn


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
    ',' separated float values. For example: "2.7,3.99,-12.9"

    for text following <@TOKENIZER-TYPE=SCALARS_FROM_DICT> is expected to be a key to the sample NDict
        for example: "blah.boo.banana"  or "data.input.encoder_input"

    example usage:

    encoder_input:
    <@TOKENIZER-TYPE=AA><MOLECULAR_WEIGHT_IN_SOME_UNIT><@TOKENIZER-TYPE=SCALARS_LITERALS>0.3<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_NANOMOLAR><MASK><@TOKENIZER-TYPE=AA><SEQUENCE_NATURAL_START>ISGGDAIYSSTGRCSLGFNVRSGSTYYFLTAGICTDGATTWWANSARTTVLGTTSGSSFPNNDYGIVRYTNTTIPKDGTVGGQDITSAANATVGMAVTRRGSTTGTISGSVTALNATVNYGGGDVVYGMIRTNVCAEPGDSGGPLYSGTRAIGLTSGGSGNCSSGGTTFFQPVTEALVAYGVSVY<SEQUENCE_NATURAL_END>
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
                    sequence = list_to_tokenizer_string(
                        sequence
                    )  # currently supporting it in this simple way. Consider optimizing if it causes a bottleneck.
                else:
                    raise Exception(
                        f"Expected sequence to be either string or a list of TypedInput elements. Got a list, but the first element is of type {type(sequence[0])}"
                    )

        hints_and_subseq = re.split("<@TOKENIZER-TYPE=([^>]*)>", sequence)[
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
                )  # AA tokenizer selection is arbitrary, we only take the special token <SCALAR> from it

                if tokenizer_type == "SCALARS_LITERALS":
                    values = subseq.split(",")
                    # validate that all values can be converted to float
                    try:
                        [float(x) for x in values]
                    except:
                        raise ValueError(
                            f'expected a string with "," separated values that can each be converted to float. Got {subseq}'
                        )
                    seq = "<SCALAR>" * len(values)
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
    def build_scalars(
        *,
        per_meta_tokenizer_data: List[str],
        per_meta_encoding_including_placeholders: List[Encoding],
        token_ids: List[int],
        sample_dict: Optional[NDict] = None,
        crop_report: str = "warn",
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
            crop_report: one of None (no action), 'warn' - print a warning, 'raise' - raise an exception
                will be triggered if cropping happened

        """
        assert crop_report in ["warn", "raise", None]
        ## both `all_scalars_values` and `all_scalars_valid_mask` will contain torch tensors, which will be concatanated in the end of this function

        # one scalar for every element, `scalar_default_unfound_value` is used for elements that aren't scalars
        all_scalars_values = []
        # for each element, whether it's a scalar or not
        all_scalars_valid_mask = []
        scalar_default_unfound_value = -1000.0

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

                    curr_scalar_values = [float(val) for val in curr_str_data]
                    curr_scalar_values = torch.tensor(
                        curr_scalar_values, dtype=torch.float32
                    )
                    all_scalars_values.append(curr_scalar_values)
                    all_scalars_valid_mask.append(
                        torch.full_like(
                            curr_scalar_values, fill_value=True, dtype=torch.bool
                        )
                    )
                elif "SCALARS_FROM_DICT" == tokenizer_name:
                    if sample_dict is None:
                        raise Exception(
                            "SCALARS_FROM_DICT used but the provided sample_dict is None"
                        )
                    curr_scalar_values = sample_dict[curr_str_data]
                    assert len(curr_scalar_values.shape) == 1
                    all_scalars_values.append(curr_scalar_values)
                    all_scalars_valid_mask.append(
                        torch.full_like(
                            curr_scalar_values, fill_value=True, dtype=torch.bool
                        )
                    )

                else:
                    raise Exception(
                        "Only supported SCALARS_* tokenizers are SCALARS_LITERALS and SCALARS_FROM_DICT"
                    )

            elif tokenizer_name.startswith("VECTORS_"):
                raise NotImplementedError
            else:
                # prev_index_end += len(curr_placeholder_encoding.ids)
                curr_scalar_values = torch.full(
                    (len(curr_placeholder_encoding.ids),),
                    fill_value=scalar_default_unfound_value,
                )
                all_scalars_values.append(curr_scalar_values)
                all_scalars_valid_mask.append(
                    torch.full_like(
                        curr_scalar_values, fill_value=False, dtype=torch.bool
                    )
                )

        all_scalars_values = torch.concat(all_scalars_values)
        all_scalars_valid_mask = torch.concat(all_scalars_valid_mask)

        assert all_scalars_values.shape == all_scalars_valid_mask.shape

        # pad if needed
        full_query_len = len(token_ids)
        if full_query_len > all_scalars_values.shape[0]:
            pad_len = full_query_len - all_scalars_values.shape[0]
            all_scalars_values = torch.concat(
                [
                    all_scalars_values,
                    torch.full(
                        (pad_len,),
                        fill_value=scalar_default_unfound_value,
                        dtype=all_scalars_values.dtype,
                    ),
                ]
            )
            all_scalars_valid_mask = torch.concat(
                [
                    all_scalars_valid_mask,
                    torch.full(
                        (pad_len,), fill_value=False, dtype=all_scalars_valid_mask.dtype
                    ),
                ]
            )
        elif full_query_len < all_scalars_values.shape[0]:
            if crop_report in ["warn", "raise"]:
                _msg = f"warning: scalars sequence had to be cropped. The full (including all subtokenizers) length was {all_scalars_values.shape[0]} after cropping it is {full_query_len}"
                if crop_report == "warn":
                    warn(_msg)
                elif crop_report == "raise":
                    raise Exception(_msg)
                else:
                    assert False, "should not get here"
            all_scalars_values = all_scalars_values[:full_query_len]
            all_scalars_valid_mask = all_scalars_valid_mask[:full_query_len]

        return {
            "scalars_values": all_scalars_values,  # 1d - its length is the number of actual scalars (provided) found
            "scalars_valid_mask": all_scalars_valid_mask,  # 1d - values of provided scalars
        }
