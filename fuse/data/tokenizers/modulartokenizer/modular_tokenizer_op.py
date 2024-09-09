from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from fusedrug.data.tokenizer.modulartokenizer.inject_utils import (
    InjectorToModularTokenizerLib,
)

from warnings import warn
from collections import defaultdict
from typing import Tuple, Optional, Union, Any
import os
import re


class ModularTokenizerWithoutInjectOp(OpBase):
    """
    Modular tokenizer without an option to inject into the model scalars or embedding
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_size: Union[int, None] = None,
        pad_token: Union[str, None] = None,
        pad_type_id: Union[int, None] = None,
        validate_ends_with_eos: Optional[bool] = True,
        eos: Optional[str] = "<EOS>",
        verbose: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            tokenizer_path: full path to a directory that the tokenizer will be loaded from
            max_size: sequences below this size will be padded, and above this size will be truncated
            pad: a string of the pad token
            pad_type_id: see tokenizers.Tokenizer.enable_padding() docstring
            validate_ends_with_eos: during encoder request (a _call_ to the op) will make sure that it ends with the provided eos token, and raise exception otherwise.
                having an eos (end of sentence) token in the end is useful for multiple scenarios, for example in a generative transformer (like T5 encoder-decoder)
            verbose:
        """
        super().__init__(**kwargs)

        if verbose:
            print(
                f"DEBUG:{self.__class__.__name__} __init__ called for path {tokenizer_path}"
            )

        self._tokenizer_path = tokenizer_path
        self._tokenizer = ModularTokenizer.from_file(self._tokenizer_path)
        pad_id = self._tokenizer.token_to_id(pad_token)

        if pad_token is None:
            raise Exception(
                f"Could not find pad token = {pad_token} in {tokenizer_path}"
            )

        self._validate_ends_with_eos = validate_ends_with_eos
        self._eos = eos

        if self._validate_ends_with_eos:
            eos_id = self._tokenizer.token_to_id(self._eos)
            if eos_id is None:
                raise Exception(
                    f"Could not find eos token = {self._eos} in {tokenizer_path}. You can disable the validation by setting validate_ends_with_eos=False"
                )

        self._pad_id = pad_id
        self._verbose = verbose

        if max_size is not None:
            assert isinstance(max_size, int)
            assert max_size > 0
            assert isinstance(pad_id, int)

            padding_kwargs = dict(length=max_size, pad_id=pad_id)
            if pad_type_id is not None:
                assert isinstance(pad_type_id, int)
                padding_kwargs["pad_type_id"] = pad_type_id

            self._tokenizer.enable_padding(direction="right", **padding_kwargs)

            self._tokenizer.enable_truncation(
                max_length=max_size,
                direction="right",
            )

        self._max_size = max_size

        if self._verbose:
            self._debug_max_tokenized_len_encountered = defaultdict(int)

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_max_token_id(self) -> Tuple[str, int]:
        """
        scans the vocab for the max observed token id and returns a tuple for it
            [its token string (str), the token id (int)]
        """
        max_token_id = self._tokenizer.get_max_id()
        token_str_of_max_token_id = self._tokenizer.id_to_token(max_token_id)

        if max_token_id is None:
            raise Exception("Could not find max_token_id! possibly an empty vocab.")
        if token_str_of_max_token_id is None:
            warn(
                "max_token_id does not correspond to a token. It may be an upper bound placeholder."
            )
            return "placeholder upper-bound token", max_token_id
        return token_str_of_max_token_id, max_token_id

    def get_min_max_sentinels(
        self,
        sentinel_prefix: Optional[str] = "<SENTINEL_ID_",
        integer_find_regex: Optional[str] = "\d{1,}",
    ) -> Tuple[int, int]:
        """
        returns a Tuple [min encountered sentinel name, max encountered sentinel name]

        For example, if the vocab contains:

        SENTINEL_ID_101: 1000,
        SENTINEL_ID_102: 1001,
        SENTINEL_ID_103: 1002,

        will return [101,103]
        """
        min_token = None
        max_token = None

        for k, _ in self._tokenizer.get_added_vocab().items():
            if sentinel_prefix in k:
                val = re.findall(integer_find_regex, k)
                if len(val) != 1:
                    raise Exception(
                        f"expected exactly one integer number in {k} but found {val}"
                    )
                val = val[0]
                val = int(val)

                if (min_token is None) or (val < min_token):
                    min_token = val

                if (max_token is None) or (val > max_token):
                    max_token = val

        if (min_token is None) or (max_token is None):
            raise Exception(
                f'Could not find any sentinels with the prefix "{sentinel_prefix}"'
            )

        return (min_token, max_token)

    def get_token_id(self, token_str: str, t_type: Optional[str] = None) -> int:
        """
        Args:
            token_str (:obj:`str`):
                The token to convert
            t_type (:obj:`str`): The sub-tokenizer to use. If None, the first (in order defined in the config)
                sub-tokenizer is used. If the token is special, type should not be set.

        Returns:
            :obj:`int`: The token's id under the tokenizer type (if given)
        """
        ans = self._tokenizer.token_to_id(token_str, t_type)
        assert ans is not None, f"could not find token id for token:{token_str}!"
        return ans

    def get_max_len(
        self, override_max_len: Union[int, None] = None
    ) -> Union[int, None]:
        """Returns the expected max_len of any encoding. max_len is given by internal state (set during initialization of the tokenizer), or it can be overridden
        during call to encode_list (applicable only to that specific encoding), or enable_padding/enable_truncation (applicable to all encodings produced
        following the call).

        Args:
            override_max_len (Optional[int], optional): Returns the resulting max_len, if the internal max_len were to be overridden by override_max_len
            during call to encode_list, or enable_padding/enable_truncation. Defaults to None.

        Returns:
            Optional[int]: _description_
        """
        return self._tokenizer.get_expected_max_len(override_max_len=override_max_len)

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out_tokenized_object: Optional[str] = None,
        key_out_tokens_ids: Optional[str] = None,
        key_out_attention_mask: Optional[str] = None,
        convert_attention_mask_to_bool: Optional[bool] = True,
        max_seq_len: Optional[int] = None,
        on_unknown: Optional[str] = "warn",
        verbose: Optional[int] = 1,
        validate_ends_with_eos: Optional[bool] = None,
        additional_caller_info_text: Optional[str] = "",
        key_out_encoding_per_meta: Optional[str] = None,
    ) -> NDict:
        """_summary_

        Args:
            sample_dict (NDict): _description_
            key_in (str): key to either a:
                (1) string that contains, in addition to the text that is to be tokenized, special delimiters signifying the type
                of input within each span of text (e.g. <@TOKENIZER-TYPE=AA> sequence, <@TOKENIZER-TYPE=SMILES>, etc.).
                (2) list of modular_tokenizer.TypedInput specifying the tokenizer type and the subsequence to tokenize
            key_out_tokenized_object (Optional[str], optional): _description_. Defaults to None.
            key_out_tokens_ids (Optional[str], optional): _description_. Defaults to None.
            key_out_attention_mask (Optional[str], optional): _description_. Defaults to None.
            convert_attention_mask_to_bool (Optional[bool], optional): _description_. Defaults to True.
            max_seq_len (Optional[int], optional): set maximum sequence len dynamically, used for both padding and truncation.. Defaults to None.
            on_unknown (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'. Defaults to "warn".
            verbose (Optional[int], optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
            validate_ends_with_eos (Optional[bool], optional): if not None, overrides self._validate_ends_with_eos
            key_out_encoding_per_meta: optional key out. If set to a string will put in it the per-meta-instruction encoded parts as a list of Encoding elements

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            NDict: _description_
        """

        data = sample_dict[key_in]
        if not isinstance(data, (list, str)):
            # data is a list of named tuples of type collections.namedtuple("TypedInput", ["input_type", "input_string", "max_len"])
            raise Exception(
                f"Expected key_in={key_in} to point to a list of inputs or string with builtin tokenizer hints, and instead got a {type(data)}. value={data}"
            )
        if validate_ends_with_eos is None:
            validate_ends_with_eos = self._validate_ends_with_eos

        if validate_ends_with_eos:
            if isinstance(data, str):
                last_seq = data
            else:
                last_seq = data[-1].input_string
            if not last_seq.rstrip().endswith(self._eos):
                raise Exception(
                    f"validate_ends_with_eos was set to {validate_ends_with_eos}, but about to encode a string that does not end with {self._eos}. The str end was: {last_seq}"
                )

        if isinstance(data, str):
            _ans = self._tokenizer.encode(
                data,
                max_len=max_seq_len,
                return_overflow_info=True,
                on_unknown=on_unknown,
                verbose=verbose,
                also_return_split=key_out_encoding_per_meta is not None,
            )
        else:
            _ans = self._tokenizer.encode_list(
                data,
                max_len=max_seq_len,
                return_overflow_info=True,
                on_unknown=on_unknown,
                verbose=verbose,
                also_return_split=key_out_encoding_per_meta is not None,
            )

        if key_out_encoding_per_meta is not None:
            encoded, overflow_info, per_meta_encoded = _ans
            sample_dict[key_out_encoding_per_meta] = per_meta_encoded
        else:
            encoded, overflow_info = _ans

        expected_max_len = self.get_max_len(override_max_len=max_seq_len)
        if (
            expected_max_len is not None
        ):  # we tightly couple padding length and max size.
            assert expected_max_len == len(encoded.ids)

        if self._verbose:
            if self._pad_id in encoded.ids:
                _encoded_len_unpadded = encoded.ids.index(self._pad_id)
            else:
                # no padding, therefore it was fully used (either exactly the size, or most likely it was clipped)
                _encoded_len_unpadded = len(encoded.ids)

            if (
                _encoded_len_unpadded
                > self._debug_max_tokenized_len_encountered[self._tokenizer_path]
            ):
                print(
                    f"DEBUG: {self.__class__.__name__} : encountered new max encoded size:",
                    _encoded_len_unpadded,
                    " for tokenizer: ",
                    self._tokenizer_path,
                )
                self._debug_max_tokenized_len_encountered[
                    self._tokenizer_path
                ] = _encoded_len_unpadded

        # KEEP THIS AS DOC FOR NOW
        # encoded has attributes [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]
        # ids are the encoded tokens,
        # type_ids are for things like "which sentence is this from"
        # tokens are the actual tokens (for example - ['c1ccc(', '/C(', '=N/N', 'c2nc3ccccc3', 's2)', 'c2cccc', 'n2)cc1', '[PAD]', '[PAD]', '[PAD]'])
        # offsets describe the starting point and length of each original token
        # attention_mask - by default puts 1 for everything that isn't padding, and 0 for those that are padding
        # special_tokens_mask - 1 for anything that is a special token (e.g. padding, separator, etc.) 0 for the rest
        # overflowing - it's encoded original content that get clipped out due to max length definition

        if (
            len(encoded.overflowing) > 0 and verbose > 0
        ):  # note, encoded.overflowing may have multiple items, and each item can contain multiple items
            print(
                f"Warning: {self.__class__.__name__}  (pid={os.getpid()}, {additional_caller_info_text}) had to truncate sequence: [{overflow_info}]  \
                    for tokenizer: {self._tokenizer_path} for sample_id {get_sample_id(sample_dict)}"
            )

        if key_out_tokenized_object is not None:
            # if requested, store the entire tokenizer.Encoding object (which provides access to attributes such as  [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
            sample_dict[key_out_tokenized_object] = encoded

        if key_out_tokens_ids is not None:
            sample_dict[key_out_tokens_ids] = encoded.ids

        if key_out_attention_mask is not None:
            sample_dict[key_out_attention_mask] = encoded.attention_mask
            if convert_attention_mask_to_bool:
                sample_dict[key_out_attention_mask] = [
                    bool(x) for x in sample_dict[key_out_attention_mask]
                ]

        if (key_out_tokens_ids is None) and (key_out_tokenized_object is None):
            warn(
                f"{self.__class__.__name__}  Op got key_out_tokens_ids=None and key_out_tokenized_object=None, which means it will not modify anything in the sample. Is this intended?"
            )

        return sample_dict


# backward compatibility
FastModularTokenizer = ModularTokenizerWithoutInjectOp


class ModularTokenizerOp(ModularTokenizerWithoutInjectOp):
    """
    Extends ModularTokenizerWithoutInjectOp and adds the option to inject scalars or embedding as an input to the model.

    injector tokenizer builds on top of modular tokenizer op.
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

    def __init__(
        self,
        tokenizer_path: str,
        max_size: Union[int, None] = None,
        pad_token: Union[str, None] = None,
        pad_type_id: Union[int, None] = None,
        validate_ends_with_eos: Optional[bool] = True,
        eos: Optional[str] = "<EOS>",
        verbose: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            tokenizer_path: full path to a directory that the tokenizer will be loaded from
            max_size: sequences below this size will be padded, and above this size will be truncated
            pad: a string of the pad token
            pad_type_id: see tokenizers.Tokenizer.enable_padding() docstring
            validate_ends_with_eos: during encoder request (a _call_ to the op) will make sure that it ends with the provided eos token, and raise exception otherwise.
                having an eos (end of sentence) token in the end is useful for multiple scenarios, for example in a generative transformer (like T5 encoder-decoder)
            verbose:
        """
        if verbose:
            print(
                f"DEBUG:{self.__class__.__name__} __init__ called for path {tokenizer_path}"
            )

        super().__init__(
            tokenizer_path=tokenizer_path,
            max_size=max_size,
            pad_token=pad_token,
            pad_type_id=pad_type_id,
            validate_ends_with_eos=validate_ends_with_eos,
            eos=eos,
            verbose=verbose,
            **kwargs,
        )

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out_tokenized_object: Optional[str] = None,
        key_out_tokens_ids: Optional[str] = None,
        key_out_attention_mask: Optional[str] = None,
        convert_attention_mask_to_bool: Optional[bool] = True,
        max_seq_len: Optional[int] = None,
        on_unknown: Optional[str] = "warn",
        verbose: Optional[int] = 1,
        validate_ends_with_eos: Optional[bool] = None,
        key_out_scalars_indices: Optional[str] = None,
        key_out_scalars_values: Optional[str] = None,
        key_out_masked_scalars_indices: Optional[str] = None,
    ) -> NDict:
        """_summary_

        Args:
            sample_dict (NDict): _description_
            key_in (str): key to either a:
                (1) string that contains, in addition to the text that is to be tokenized, special delimiters signifying the type
                of input within each span of text (e.g. <@TOKENIZER-TYPE=AA> sequence, <@TOKENIZER-TYPE=SMILES>, etc.).
                (2) list of modular_tokenizer.TypedInput specifying the tokenizer type and the subsequence to tokenize
            key_out_tokenized_object (Optional[str], optional): _description_. Defaults to None.
            key_out_tokens_ids (Optional[str], optional): _description_. Defaults to None.
            key_out_attention_mask (Optional[str], optional): _description_. Defaults to None.
            convert_attention_mask_to_bool (Optional[bool], optional): _description_. Defaults to True.
            max_seq_len (Optional[int], optional): set maximum sequence len dynamically, used for both padding and truncation.. Defaults to None.
            on_unknown (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'. Defaults to "warn".
            verbose (Optional[int], optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
            validate_ends_with_eos (Optional[bool], optional): if not None, overrides self._validate_ends_with_eos
            key_out_scalars_inputs_indices:str optional
                if provided, will write to sample_dict in this key a 1D torch tensor with indices of all inputs scalar elements.
            key_out_scalars_inputs_values:str optional
                if provided, will write to sample_dict in this key a 1D torch tensor with indices of all inputs scalar values.

        Returns:
            NDict: _description_
        """

        (
            with_placeholders_str,
            per_meta_orig,
        ) = InjectorToModularTokenizerLib.build_placeholder_meta_tokenization(
            sequence=sample_dict[key_in], sample_dict=sample_dict
        )
        sample_dict[key_in + ".with_placeholders"] = with_placeholders_str

        super().__call__(
            sample_dict=sample_dict,
            key_in=key_in + ".with_placeholders",
            key_out_tokenized_object=key_out_tokenized_object,
            key_out_tokens_ids=key_out_tokens_ids,
            key_out_attention_mask=key_out_attention_mask,
            convert_attention_mask_to_bool=convert_attention_mask_to_bool,
            max_seq_len=max_seq_len,
            on_unknown=on_unknown,
            verbose=verbose,
            validate_ends_with_eos=validate_ends_with_eos,
            key_out_encoding_per_meta=key_in
            + ".per_meta_part_encoding",  # using the key_in as base for the name because key_out_* are optional
        )

        prepared_data = InjectorToModularTokenizerLib.prepare_info_for_model_step(
            per_meta_tokenizer_data=per_meta_orig,
            per_meta_encoding_including_placeholders=sample_dict[
                key_in + ".per_meta_part_encoding"
            ],
            sample_dict=sample_dict,
        )

        if key_out_scalars_indices is not None:
            sample_dict[key_out_scalars_indices] = prepared_data["scalars_indices"]
        else:
            if prepared_data["scalars_indices"] is not None:
                raise Exception(
                    "non None scalars_indices found but no key_out_scalars_indices found"
                )

        if key_out_scalars_values is not None:
            sample_dict[key_out_scalars_values] = prepared_data["scalars_values"]
        else:
            if prepared_data["scalars_values"] is not None:
                raise Exception(
                    "non None scalars_value found but no key_out_scalars_values found"
                )

        if key_out_masked_scalars_indices is not None:
            sample_dict[key_out_masked_scalars_indices] = prepared_data[
                "scalars_masked_indices"
            ]
        else:
            if prepared_data["scalars_masked_indices"] is not None:
                raise Exception(
                    "non None scalars_masked_indices found but no key_out_masked_scalars_indices found"
                )

        return sample_dict
