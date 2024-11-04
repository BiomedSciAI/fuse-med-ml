from torch import Tensor
from typing import Dict
from collections.abc import Iterable
from tokenizers import Tokenizer, Encoding
import tokenizers
from warnings import warn
from typing import Optional, List, Set, Union, Tuple, Any, Iterator
import json
import transformers
import os
from omegaconf import OmegaConf
import collections
import omegaconf
import copy
import traceback
import re
from fuse.data.tokenizers.modular_tokenizer.special_tokens import special_wrap_input


TypedInput = collections.namedtuple(
    "TypedInput", ["input_type", "input_string", "max_len"]
)


def list_to_tokenizer_string(lst: List[TypedInput]) -> str:
    out = ""
    # prev_tokenizer = None
    for in_named_tuple in lst:
        curr_tokenizer = in_named_tuple.input_type
        curr_len = in_named_tuple.max_len
        # NOTE: For now we don't combine consequent strings encoded by the same tokenizer,
        # since they may have different max lengths, so we create a new entry, even if curr_tokenizer == prev_tokenizer:
        if curr_len is None:
            out += f"<@TOKENIZER-TYPE={curr_tokenizer}>"
        else:
            out += f"<@TOKENIZER-TYPE={curr_tokenizer}@MAX-LEN={curr_len}>"
        out += in_named_tuple.input_string
        # prev_tokenizer = curr_tokenizer
    return out


class ModularTokenizer(transformers.PreTrainedTokenizerFast):
    def __init__(
        self,
        tokenizers_info: Union[List, omegaconf.listconfig.ListConfig],
        load_adjusted_jsons: Optional[bool] = False,
        special_tokens_list: Optional[list] = None,
        max_possible_token_id: Optional[int] = None,
        max_special_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Creates a modular tokenizer that combines multiple existing tokenizers, adjusting them so that:
        a. They all share the same special tokens (combined special tokens from all the source tokenizers),
        b. Each tokenizer retains its regular tokens, however their IDs are remapped to a single space, with no overlaps.

        ModularTokenizer inherits the interface of PreTrainedTokenizerBase, but not the underlying logic.

        Args:
            tokenizers_info (List): A list of dictionaries containing the following:
            [
                   {
                        "name": a name of a tokenizer
                        "modular_json_path":out_path for tokenizer_type
                        "json_path": Optional - path to a json of the original sub-tokenizer, which will be merged into the modular tokenizer
                    }
            ]
            load_adjusted_jsons (Optional[bool], optional): Whether to load json files created by ModularTokenizer (True),
                or to adjust the indices of given non-modular jsons (False). This should not ordinarily be set to False, as loading
                from modular jsons is best done through the load_from_jsons method. Defaults to False.
            special_tokens_list (Optional[list], optional): A list of special tokens that should be common among all tokenizers
            max_possible_token_id (Optional[int], optional): An upper limit to a token ID. When IDs of tokens added to modular tokenizer
                go above this, an exception is thrown. Defaults to None (i.e. no limit is set).
            max_special_token_id (Optional[int], optional): An upper limit to special token ID. Special tokens are shared between all sub-tokenizers.
                If max_special_token_id is set, when special tokens are added, they are mapped to IDs between 0 and max_special_token_id
                (after which come regular token IDs). Once max_special_token_id is reached, no more special tokens may be added.
                If it is not set, new special tokens may be mapped to IDs higher that regular token IDs. If Defaults to None (i.e. no limit is set).
        """
        # ModularTokenizer inherits the interface of PreTrainedTokenizerBase, but not the underlying logic, therefore super.__init__() is not called

        # If there is only one tokenizer, remapping it is not needed - if there's only one, we can just load its json using load_from_jsons.
        if isinstance(tokenizers_info, omegaconf.listconfig.ListConfig) or isinstance(
            tokenizers_info, omegaconf.dictconfig.DictConfig
        ):
            tokenizers_info_list: List = OmegaConf.to_object(tokenizers_info)
        elif isinstance(tokenizers_info, list):
            tokenizers_info_list = tokenizers_info
        else:
            raise Exception("unexpected tokenizers_info type")
        self.tokenizers_info_raw_cfg = copy.deepcopy(tokenizers_info_list)
        self.tokenizers_info = ModularTokenizer.cfg_list_2_dict(
            copy.deepcopy(tokenizers_info_list)
        )
        self._max_possible_token_id = max_possible_token_id
        self._max_special_token_id = max_special_token_id

        if not load_adjusted_jsons:
            # store special tokens in a list to preserve their order:
            all_special_tokens = (
                list(special_tokens_list) if special_tokens_list is not None else []
            )

            # collect all special tokens (without indices):
            for t_type in self.tokenizers_info:
                t_info = self.tokenizers_info[t_type]
                with open(t_info["json_path"]) as json_file:
                    t_json = json.load(json_file)
                self.tokenizers_info[t_type]["json_instance"] = t_json

                part_special_tokens = ModularTokenizer.get_subtokenizer_added_tokens(
                    t_json,
                    enforce_special=False,
                )
                part_special_tokens = [
                    t for t in part_special_tokens if t not in all_special_tokens
                ]
                all_special_tokens = all_special_tokens + part_special_tokens

            all_special_token_structs = ModularTokenizer.build_special_token_list(
                all_special_tokens
            )

            # Set the starting ID for regular token mapping:
            next_index = max([t["id"] for t in all_special_token_structs]) + 1
            if self._max_special_token_id is not None:
                if next_index > self._max_special_token_id:
                    raise Exception(
                        f"Max special token ID {self._max_special_token_id} is too small to contain all special tokens {next_index}. Either increase or do not set it."
                    )
                next_index = self._max_special_token_id + 1
        else:
            if special_tokens_list is not None:
                raise Exception(
                    "When loading a tokenizer special_tokens_list must be None. Use ModularTokenizer.add_special_tokens instead"
                )
            for t_type in self.tokenizers_info:
                t_info = self.tokenizers_info[t_type]
                with open(t_info["modular_json_path"]) as modular_file:
                    t_json = json.load(modular_file)
                self.tokenizers_info[t_type]["json_instance"] = t_json

        # rearrange regular token indices to map to IDs starting from next_index:
        for t_type in self.tokenizers_info:
            t_info = self.tokenizers_info[t_type]
            t_json = self.tokenizers_info[t_type]["json_instance"]
            # operations on the tokenizer json
            if not load_adjusted_jsons:
                min_id = t_info.get("minimal_token_id", 0)
                next_index = max(min_id, next_index)
                t_json["added_tokens"] = all_special_token_structs
                (t_json["model"]["vocab"], next_index,) = ModularTokenizer.remap_vocab(
                    vocab=t_json["model"]["vocab"],
                    special_token_structs=all_special_token_structs,
                    starting_index=next_index,
                )
            # end operations on json
            # operations on the tokenizer instance (if possible, operations should be done here, using built-in tokenizer methods)
            json_str = json.dumps(t_json)
            tokenizer_inst = Tokenizer.from_str(json_str)
            if special_tokens_list is not None:
                # At this point, tokens from special_tokens_list are in every tokenizer. This is only to test that all special tokens were added.
                num_add = tokenizer_inst.add_special_tokens(special_tokens_list)
                if num_add > 0:
                    raise Exception(
                        f"All special tokens should have been in the vocabulary at this point. {num_add} were added - need to check why."
                    )
            if "max_len" in t_info and t_info["max_len"] is not None:
                max_size = t_info["max_len"]
                tokenizer_inst.enable_truncation(
                    max_length=max_size,
                    direction="right",
                )
            json_str = tokenizer_inst.to_str()
            t_json = json.loads(json_str)
            self.tokenizers_info[t_type]["tokenizer_inst"] = tokenizer_inst
            self.tokenizers_info[t_type]["json_instance"] = t_json

        self.max_len: Union[
            int, None
        ] = None  # determines the final length of the overall encoding (and therefore padding/truncation length)
        self._pad_token_id: Union[int, None] = None

        self._pad_token_type_id = 0
        self._pad_token: Union[str, None] = None

        test_res, test_res_detail = self.diagnose()
        assert (
            False not in test_res.values()
        ), f"resulting tokenizer is not consistent: {test_res_detail}"
        self.build_inner_decoder()
        if self._max_possible_token_id is not None:
            if self._get_max_mapped_id() > self._max_possible_token_id:
                raise Exception(
                    f"tokenizer remapping resulted in IDs greater (max_id={self._get_max_mapped_id()}) than max_possible_id ({self._max_possible_token_id}). Reinitialize the modular tokenizer with larger max_possible_id"
                )

    @staticmethod
    def remap_vocab(
        vocab: Dict,
        special_token_structs: Optional[List] = None,
        starting_index: Optional[int] = None,
    ) -> Tuple[Dict, int]:
        """Receives a vocabulary, a list of special token structures and a starting index. Returns a new vocabulary that
        a. contains all the special tokens with their IDs, as were given in special_token_structs.
        b. contains all the tokens in vocab (except special ones), numbered consecutively starting with starting_index.
        c. the order of the regular tokens remains unchanged (they are usually ordered by appearance frequency - we do not want to change that)

        Args:
            vocab (Dict): vocabulary of tokens to be included in the ModularTokenizer. If there is an overlap between tokens in vocab and tokens
            in special_token_structs, the special tokens are removed from vocab, and will be present in the resulting ModularTokenizer as special tokens.
            special_token_structs (Optional[List]): a list of special token structures to be added to the tokenizer. If None or empty, no special tokens are added. Defaults to None
            starting_index (Optional[int], optional): Starting id of regular tokens. If None - inferred from special_tokens. Defaults to None.

        Returns:
            Tuple[Dict,int]: Returns the updated vocabulary and the next starting index (its max ID + 1)
        """
        if special_token_structs is not None and len(special_token_structs) > 0:
            init_vocab = {t["content"]: t["id"] for t in special_token_structs}
            special_tokens = set(init_vocab.keys())
            special_inds = list(init_vocab.values())
            if starting_index is None:
                starting_index = max(special_inds) + 1
        else:
            special_tokens = set()
            init_vocab = {}
            if starting_index is None:
                starting_index = 0

        # At this point the vocab we're building contains all the special tokens with their IDs, and we know from which ID to start the regular token mappings
        # First, remove any tokens that are special from the input vocabulary, so that if any are present there as regular tokens, they won't be duplicated
        regular_tokens = (
            set(vocab.keys()) - special_tokens
        )  # TODO: make this an ordered operation, to make sure we have a consistent mapping
        # (maybe order the regular tokens in an alphabetic order)
        regular_vocab = {
            r_t: vocab[r_t] for r_t in regular_tokens
        }  # These are only regular tokens with their original indices
        regular_sorted = sorted(
            regular_vocab.items(), key=lambda x: x[1], reverse=False
        )  # regular tokens sorted by their ID in ascending order.

        regular_vocab = {t[0]: i + starting_index for i, t in enumerate(regular_sorted)}
        init_vocab.update(regular_vocab)
        starting_index_new = max(regular_vocab.values()) + 1
        return init_vocab, starting_index_new

    @staticmethod
    def build_special_token_list(
        special_tokens: Union[List, Set],
        starting_index: Union[int, None] = None,
        token_ids: Union[List, Set, None] = None,
    ) -> List:
        """Creates a list of special token structures with consecutive indices, according to the following template
            special token template:
            {
                'id': int id value,
                'content': string token name,
                'single_word': False,
                'lstrip': False,
                'rstrip': False,
                'normalized': False,
                'special': True
            }


        Args:
            special_tokens (Union[List, Set]): a list of token names to be added
            starting_index (Optional[int], optional): The tokens are mapped to consecutive IDs starting from this. Defaults to 0.
            token_ids (Union[List, Set, None], optional): a list of corresponding token IDs to be set as is. Defaults to None
            (i.e. new consecutive IDs, starting with starting index are used). Cannot be set together with starting_index

        Returns:
            List: _description_
        """
        if starting_index is None:
            starting_index_to_use = 0
        else:
            starting_index_to_use = starting_index
        if token_ids is None:
            special_tokens = [
                {
                    "id": i + starting_index_to_use,
                    "content": v,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for i, v in enumerate(special_tokens)
            ]
        else:
            assert len(special_tokens) == len(
                token_ids
            ), f"Number of tokens {len(special_tokens)} and number of IDs {len(token_ids)} must be the same."
            assert (
                starting_index is None
            ), "Either starting index, or a list of IDs may be given, not both."
            special_tokens = [
                {
                    "id": i,
                    "content": v,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for i, v in zip(token_ids, special_tokens)
            ]
        return special_tokens

    @staticmethod
    def get_subtokenizer_added_tokens(
        tokenizer_json_inst: Dict,
        enforce_special: Optional[bool] = False,
    ) -> List:
        """returns the special tokens from a tokenizer defined by json_inst.
            Note: An alternative would be to call tokenizer_inst.get_vocab(with_added_tokens), using with_added_tokens False and True, which
            should've given us just regular and regular+special tokens, but for some reason both these options return the same output,
            so we must resort to json parsing.


        Args:
            json_inst (Dict): _description_
            enforce_special (Optional[bool], optional): If False, treats all added tokens as special. If True, considers only those that have "special": True as special. Defaults to False.

        Returns:
            Set: _description_
        """
        special_token_structs = tokenizer_json_inst["added_tokens"]
        if enforce_special:
            special_tokens = [
                t["content"] for t in special_token_structs if t["special"]
            ]
        else:
            special_tokens = [t["content"] for t in special_token_structs]

        return special_tokens

    @staticmethod
    def get_subtokenizer_regular_tokens(
        tokenizer_json_inst: Dict, enforce_special: Optional[bool] = False
    ) -> Set:
        """returns the regular tokens from tokenizer defined by json_inst.
            Note: An alternative would be to call tokenizer_inst.get_vocab(with_added_tokens), using with_added_tokens False and True, which
            should've given us just regular and regular+special tokens, but for some reason both these options return the same output,
            so we must resort to json parsing.


        Args:
            json_inst (Dict): _description_
            enforce_special (Optional[bool], optional): If False, treats all added tokens as special when deciding which token is regular and which is special.
                If True, considers only those that have "special": True as special. Defaults to False.

        Returns:
            Set: _description_
        """
        special_tokens = ModularTokenizer.get_subtokenizer_added_tokens(
            tokenizer_json_inst=tokenizer_json_inst, enforce_special=enforce_special
        )
        all_tokens = set(tokenizer_json_inst["model"]["vocab"].keys())
        return all_tokens - set(special_tokens)

    @staticmethod
    def get_subtokenizer_vocab(
        tokenizer_json_inst: Dict, token_list: Optional[List] = None
    ) -> Dict:
        """Returns a dictionary of {token:id} of tokenizer tokenizer_json_inst for all tokens in token_list

        Args:
            tokenizer_json_inst (Dict): json instance representation of a tokenizer
            token_list (Optional[List], optional): list of tokens. If None - all tokens are used. Defaults to None.

        Returns:
            Dict: _description_
        """
        all_vocab = tokenizer_json_inst["model"]["vocab"]
        if token_list is None:
            return all_vocab
        output = {t: all_vocab[t] for t in token_list if t in all_vocab}
        return output

    @staticmethod
    def load_from_jsons(tokenizers_info: List) -> "ModularTokenizer":
        """Reads a list of json paths (from tokenizer_info dictionary, as defined in the config), that were created by ModularTokenizer.save_jsons, and creates a modular tokenizer, keeping the ID mappings
        of the jsons.

        Args:
            tokenizer_info (List): A list of dictionaries containing the following:
            [
                   {
                        "name": a name of a tokenizer
                        "modular_json_path":out_path for tokenizer_type
                    }
            ]

        Returns:
            object: _description_
        """
        return ModularTokenizer(
            tokenizers_info=tokenizers_info, load_adjusted_jsons=True
        )

    @staticmethod
    def load(path: str) -> "ModularTokenizer":
        """Reads all information that was saved by ModularTokenizer.save(), and creates a modular tokenizer based on it.

        Args:
            path: directory that contains a file named config.yaml which defines list of dictionaries containing the following:
            [
                   {
                        "name": a name of a tokenizer
                        "modular_json_path":out_path for tokenizer_type
                    }
            ]

        Returns:
            object: Loaded ModularTokenizer
        """

        def fix_json_paths(
            loaded_conf: omegaconf.listconfig.ListConfig,
            path: str,
        ) -> omegaconf.listconfig.ListConfig:
            """Since the path passed to ModularTokenizer.load() must contain all of the modular jsons and the config that defines their relations, all json
            paths in the config must point to path. This function replaces any dirname of any json in the loaded config (found in path) with path.
            Args:
                loaded_conf (Union[List, omegaconf.listconfig.ListConfig]): _description_
                path (str): _description_

            Returns:
                Union[omegaconf.listconfig.ListConfig]: _description_
            """
            for ind, t_conf in enumerate(loaded_conf):
                if ("json_path" in t_conf) and (t_conf["json_path"] is not None):
                    loaded_conf[ind]["json_path"] = os.path.join(
                        path, os.path.basename(t_conf["json_path"])
                    )
                loaded_conf[ind]["modular_json_path"] = os.path.join(
                    path, os.path.basename(t_conf["modular_json_path"])
                )
            return loaded_conf

        try:
            loaded_conf: omegaconf.dictconfig.DictConfig = OmegaConf.load(
                os.path.join(path, "config.yaml")
            )
        except:
            traceback.print_exc()
            raise Exception(f"couldn't load config.yaml from {path}")
        tokenizers_info_fixed = fix_json_paths(loaded_conf["tokenizers_info"], path)

        if "max_possible_token_id" in loaded_conf:
            max_possible_token_id: Union[int, None] = loaded_conf[
                "max_possible_token_id"
            ]
        else:
            max_possible_token_id = None

        if "max_special_token_id" in loaded_conf:
            max_special_token_id: Union[int, None] = loaded_conf["max_special_token_id"]
        else:
            max_special_token_id = None

        return ModularTokenizer(
            tokenizers_info=tokenizers_info_fixed,
            load_adjusted_jsons=True,
            max_possible_token_id=max_possible_token_id,
            max_special_token_id=max_special_token_id,
        )

    @staticmethod
    def update_id2token_mapping(
        id2token: Dict[int, Dict], add_vocab: Dict, is_special: Optional[bool] = False
    ) -> Dict[int, Dict]:
        """Updates id2token mapping with tokens from add_vocab. Returns the updated id2token

        Args:
            id2token (Dict): A dictionary of int:{
                "token":int,
                "is_special":bool
                }
            add_vocab (Dict): vocabulary as returned
            is_special (Optional[bool], optional): whether or not add_vocab holds special tokens. Defaults to False.

        Returns:
            Dict: _description_
        """

        for token in add_vocab:
            if add_vocab[token] in id2token:
                print(
                    "Warning: ID collision during update_id2token_mapping for token {token}, id {add_vocab[token]}"
                )
            else:
                tmp_dict = {"token": token, "is_special": is_special}
                id2token[add_vocab[token]] = tmp_dict
        return id2token

    def build_inner_decoder(self) -> None:
        """Goes over all the inner tokenizers and builds an id-to-token mapping with the following structure:
        self.decoder_dict = {id: {
                                    token:token_id,     #token corresponding to the id
                                    is_special:bool,    #whether the token is special or not
                                    }
                                }
        There are two ways to implement this:
        - automatic understanding of relevant tokenizer for each subsequence (subsequences can be recognized from sequence_ids and mask), and using tokenizer.decode
            Pros:
            -   Decoding takes less time by using efficient implementation from tokenizers
            Cons:
            -   Inference may be difficult/inefficient (need to divide into sequences of regular tokens)
        - maintaining a single decoder dictionary, and using it. Currently implemented this option.
            Pros:
            -   straightforward implementation
            Cons:
            -   not as efficient as built-in tokenizer decode.

        """
        self.decoder_dict: Dict = {}
        for t_type in self.tokenizers_info:
            t_info = self.tokenizers_info[t_type]
            assert (
                "json_instance" in t_info
            ), f"tokenizer of type {t_type} hasn't been instantiated yet. Call init first."
            if len(self.decoder_dict) == 0:  # Add
                sp_tokens = ModularTokenizer.get_subtokenizer_added_tokens(
                    t_info["json_instance"]
                )
                sp_vocab = ModularTokenizer.get_subtokenizer_vocab(
                    tokenizer_json_inst=t_info["json_instance"], token_list=sp_tokens
                )
                self.decoder_dict = ModularTokenizer.update_id2token_mapping(
                    id2token=self.decoder_dict, add_vocab=sp_vocab, is_special=True
                )
            reg_tokens = ModularTokenizer.get_subtokenizer_regular_tokens(
                t_info["json_instance"]
            )
            reg_vocab = ModularTokenizer.get_subtokenizer_vocab(
                tokenizer_json_inst=t_info["json_instance"], token_list=list(reg_tokens)
            )
            self.decoder_dict = ModularTokenizer.update_id2token_mapping(
                id2token=self.decoder_dict, add_vocab=reg_vocab, is_special=False
            )

    def diagnose(self) -> Tuple[Dict, Dict]:
        """_summary_

        Returns:
            Tuple[Dict, Dict]: brief (pass/fail for each test) and detailed (which tokenizers failed) description of failed tests
        """
        tests = [
            "special token consistency",  # Special tokens are the same (and map to the same indices) across all the tokenizers
            "ID duplicates in vocab",  # Regular token ID mappings of any given tokenizer do not collide with special token mappings
            "ID collisions across vocabs",  # Regular token ID mappings of any given tokenizer do not collide with ID mappings of other tokenizers
        ]
        result = {t_name: True for t_name in tests}
        result_details: Dict[str, Any] = {t_name: [] for t_name in tests}
        tokenizer_types = list(self.tokenizers_info.keys())
        # TODO: If there are multiple tokenizer files that were derived from the same file - use only one for diagnosis
        all_inds_set: Set[int] = set()
        all_inds_len = 0
        if len(tokenizer_types) > 1:
            special_tokens = list(
                ModularTokenizer.get_subtokenizer_added_tokens(
                    self.tokenizers_info[tokenizer_types[0]]["json_instance"]
                )
            )
            special_tokens_vocab = ModularTokenizer.get_subtokenizer_vocab(
                tokenizer_json_inst=self.tokenizers_info[tokenizer_types[0]][
                    "json_instance"
                ],
                token_list=special_tokens,
            )

            # check if all special tokens are the same across all tokenizers
            for t_type in tokenizer_types:
                special_tokens_t = list(
                    ModularTokenizer.get_subtokenizer_added_tokens(
                        self.tokenizers_info[t_type]["json_instance"]
                    )
                )
                special_tokens_vocab_t = ModularTokenizer.get_subtokenizer_vocab(
                    tokenizer_json_inst=self.tokenizers_info[t_type]["json_instance"],
                    token_list=special_tokens_t,
                )

                if special_tokens_vocab != special_tokens_vocab_t:
                    result["special token consistency"] = False
                    result_details["special token consistency"].append(t_type)

            # check if there are no ID collisions within/between vocabs
            for t_type in tokenizer_types:
                regular_tokens = list(
                    ModularTokenizer.get_subtokenizer_regular_tokens(
                        self.tokenizers_info[t_type]["json_instance"]
                    )
                )
                regular_tokens_vocab = ModularTokenizer.get_subtokenizer_vocab(
                    tokenizer_json_inst=self.tokenizers_info[t_type]["json_instance"],
                    token_list=regular_tokens,
                )
                regular_tokens_IDs = regular_tokens_vocab.values()
                regular_tokens_ID_set = set(regular_tokens_IDs)
                if len(regular_tokens_IDs) != len(regular_tokens_ID_set):
                    result["ID duplicates in vocab"] = False
                    result_details["ID duplicates in vocab"].append(t_type)

                old_all_inds_set = all_inds_set
                all_inds_set = all_inds_set.union(regular_tokens_ID_set)
                if len(all_inds_set) != all_inds_len + len(regular_tokens_ID_set):
                    result["ID collisions across vocabs"] = False
                    result_details["ID collisions across vocabs"].append(
                        f"{t_type}:{old_all_inds_set.intersection(regular_tokens_ID_set)}"
                    )
                all_inds_len = len(all_inds_set)

            special_tokens_ID_set = set(special_tokens_vocab.values())
            if len(special_tokens_vocab.values()) != len(special_tokens_ID_set):
                result["ID duplicates in vocab"] = False
                result_details["ID duplicates in vocab"].append("special")

            all_inds_set = all_inds_set.union(special_tokens_ID_set)
            if len(all_inds_set) != all_inds_len + len(set(special_tokens_ID_set)):
                result["ID collisions across vocabs"] = False
                result_details["ID collisions across vocabs"].append("special")
            all_inds_len = len(all_inds_set)

        return result, result_details

    def is_consistent(self) -> bool:
        """Returns True if the modular tokenizer is consistent, i.e.:
        a. Special tokens are the same (and map to the same indices) across all the tokenizers
        b. Regular token ID mappings of any given tokenizer do not collide with special token mappings, nor with ID mappings of other tokenizers

        Returns:
            bool: True is the tokenizer is consistent, False otherwise
        """
        test_res, test_res_detail = self.diagnose()
        return False not in test_res.values()

    @staticmethod
    def cfg_list_2_dict(dict_list: List) -> Dict[str, Any]:
        """Receives a list of dicts, each containing a key "name" and changes it to

        Args:
            dict_list (List): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        return {d["name"]: d for d in dict_list}

    def save_jsons(self, tokenizers_info: Optional[List] = None) -> None:
        """_summary_

        Args:
            tokenizers_info_list (Optional[List], optional): A list of dictionaries containing the following:
            [
                   {
                        "name": a name of a tokenizer
                        "modular_json_path":out_path for tokenizer_type
                    }
            ]
            In case of None, paths stored in self.tokenizers_info (modular_json_path for each tokenizer) are used.
            In case of partial tokenizer_types, only those tokenizers will be saved
            Defaults to None.
            TODO: also save the config yaml there
        """
        if tokenizers_info is None:
            for t_type in self.tokenizers_info:
                tokenizer_inst = self.tokenizers_info[t_type]["tokenizer_inst"]
                out_path = self.tokenizers_info[t_type]["modular_json_path"]
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                tokenizer_inst.save(out_path)
        else:
            tokenizers_info_dict = ModularTokenizer.cfg_list_2_dict(tokenizers_info)
            for t_type in tokenizers_info_dict:
                tokenizer_inst = self.tokenizers_info[t_type]["tokenizer_inst"]
                out_path = tokenizers_info_dict[t_type]["modular_json_path"]
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                tokenizer_inst.save(out_path)

    def save(self, path: str) -> None:
        """Saves all information needed to reconstruct the modular tokenizer to path.
        After saving, path will contain the following:
        - json files: modular json files (i.e. that have common special tokens, and that all map to consistent ID space)
        - config.yaml: a config file that contains the tokenizer_info list with information from self.tokenizer_info, defining the relations between the different tokenizers.
        ** Since all modular tokenizers are found in the same path, and the path may change when it's passed between users, the path is not included in the config
        (i.e. all json paths have base dir of './'). The correct path is updated upon calling ModularTokenizer.load()

        Args:
            path (str): a directory where the modular tokenizer info will be saved. For compatibility with huggingface format, this can also be a path to a json
            file, in which case its base directory will be used.
        """

        def get_out_path(input_json_path: str, base_path: Optional[str] = None) -> str:
            """_summary_

            Args:
                input_json_path (str): _description_
                base_path (str, optional): _description_. Defaults to None.

            Returns:
                str: _description_
            """
            fname: str = os.path.basename(input_json_path)
            if base_path is None:
                return os.path.join("./", fname)
            return os.path.join(base_path, fname)

        def set_field(tokenizers_info_cfg: List, name: str, key: str, val: Any) -> List:
            for i, t in enumerate(tokenizers_info_cfg):
                if t["name"] == name:
                    t[key] = val
                    tokenizers_info_cfg[i] = t
                    return tokenizers_info_cfg
            raise Exception(f"name {name} not found")

        if path.endswith(".json") or path.endswith(".yaml"):
            path = os.path.dirname(path)

        tokenizers_info_cfg = self.tokenizers_info_raw_cfg

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        for t_type in self.tokenizers_info:
            tokenizer_inst = self.tokenizers_info[t_type]["tokenizer_inst"]
            if self.tokenizers_info[t_type]["json_path"] is not None:
                input_json_path = self.tokenizers_info[t_type]["json_path"]
            elif self.tokenizers_info[t_type]["modular_json_path"] is not None:
                input_json_path = self.tokenizers_info[t_type]["modular_json_path"]
            else:
                raise Exception(f"Couldn't find json path for subtokenizer {t_type}")
            write_out_path = get_out_path(
                input_json_path=input_json_path,
                base_path=path,
            )
            # json paths in the save config are meaningless, since the saves may pass between machines, and the base directory for the save may change,
            # therefore the json path in config is set to ./json_filename.json
            config_out_path = get_out_path(
                input_json_path=input_json_path,
                base_path=None,
            )
            tokenizers_info_cfg = set_field(
                tokenizers_info_cfg=tokenizers_info_cfg,
                name=t_type,
                key="modular_json_path",
                val=config_out_path,
            )
            # Original json path (for the json of the tokenizer used to create the original instance of this ModularTokenizer) is no longer relevant,
            # since it may be located on another machine. config_out_path is used instead.
            tokenizers_info_cfg = set_field(
                tokenizers_info_cfg=tokenizers_info_cfg,
                name=t_type,
                key="json_path",
                val=config_out_path,
            )
            tokenizer_dir = os.path.dirname(write_out_path)
            if not os.path.exists(tokenizer_dir):
                os.mkdir(tokenizer_dir)
            tokenizer_inst.save(write_out_path)
        tokenizer_config_overall = {
            "tokenizers_info": tokenizers_info_cfg,
            "max_possible_token_id": self._max_possible_token_id,
            "max_special_token_id": self._max_special_token_id,
        }
        # yaml_data: str = OmegaConf.to_yaml(tokenizers_info_cfg)
        with open(os.path.join(path, "config.yaml"), "w") as f:
            OmegaConf.save(tokenizer_config_overall, f)

    def update_special_tokens(
        self, added_tokens: List, save_tokenizer_path: str = None
    ) -> None:
        self.add_special_tokens(tokens=added_tokens)
        if save_tokenizer_path:
            self.save(path=save_tokenizer_path)

    def add_single_tokenizer(
        self,
        tokenizer_info: Dict,
    ) -> None:

        # first we load the new tokenizer

        # TODO: rename variable
        new_tokenizer_info = tokenizer_info

        self.tokenizers_info_raw_cfg.append(
            {
                k: v
                for k, v in new_tokenizer_info.items()
                if k
                not in [
                    "json_instance",
                    "tokenizer_inst",
                ]
            }
        )
        new_tokenizer_path = new_tokenizer_info["json_path"]
        with open(new_tokenizer_path) as new_tokenizer:
            t_json = json.load(new_tokenizer)
        new_tokenizer_info["json_instance"] = t_json

        # we can set a minimal starting id for each tokenizer.  Used for the extended tokenizer.
        min_id = new_tokenizer_info.get("minimal_token_id", 0)
        next_index = self._get_max_mapped_id() + 1
        next_index = max(min_id, next_index)

        # the new tokenizer may not have all the special tokens in the vocabulary
        # We keep the old vocabulary,

        old_vocab = t_json["model"]["vocab"]

        # and special tokens

        new_tokenize_special_tokens = self.get_subtokenizer_added_tokens(t_json)

        # update the ids
        updated_vocab, next_index = ModularTokenizer.remap_vocab(
            vocab=old_vocab,
            starting_index=next_index,
        )

        # Take the special tokens as they appear in the first tokenizer inside the modular tokenizer
        first_tok_key = list(self.tokenizers_info.keys())[0]
        all_special_token_structs = self.tokenizers_info[first_tok_key][
            "json_instance"
        ]["added_tokens"]

        # check if we are really adding any tokens from the new tokenizer
        new_special_tokens = set(new_tokenize_special_tokens) - set(
            [tok["content"] for tok in all_special_token_structs]
        )
        # in the begging of the list, with their id's

        t_json["model"]["vocab"] = {
            sp["content"]: sp["id"] for sp in all_special_token_structs
        }
        # and finally we add the updated vocabulary.

        t_json["model"]["vocab"].update(updated_vocab)

        # we also need to add the tokens to the added_tokens section as special tokens
        t_json["added_tokens"] = copy.deepcopy(all_special_token_structs)

        # create the tokenizer instance TODO: do we need this?
        json_str = json.dumps(t_json)
        tokenizer_inst = tokenizers.Tokenizer.from_str(json_str)
        new_tokenizer_info["tokenizer_inst"] = tokenizer_inst

        self.tokenizers_info[new_tokenizer_info["name"]] = new_tokenizer_info
        added_tokens = self.get_added_vocab()
        if new_special_tokens:
            new_special_tokens = set(new_special_tokens) - set(added_tokens)
        if new_special_tokens:
            warn(
                f"Added tokenizer {new_tokenizer_info['name']} came with {len(new_special_tokens)} new special tokens: {','.join(new_special_tokens)}"
            )

        self.build_inner_decoder()
        if self._max_possible_token_id is not None:
            if self._get_max_mapped_id() > self._max_possible_token_id:
                raise Exception(
                    f"tokenizer remapping resulted in IDs greater (max_id={self._get_max_mapped_id()}) than max_possible_id ({self._max_possible_token_id}). Reinitialize the modular tokenizer with larger max_possible_id"
                )
        # we update the special tokens but do not save here.  remember to save yourself.
        self.update_special_tokens(
            special_tokens=new_tokenize_special_tokens,
        )

    def add_tokenizers(
        self,
    ) -> None:
        raise Exception("Not implemented")

    def _encode_single_type(
        self,
        data_str: str,
        input_type: str,
        sequence_id: Optional[int] = None,
        verbose: int = 1,
    ) -> Encoding:
        """_summary_

        Args:
            data_str (str): input string to be tokenized
            input_type (str): type of a tokenizer to use
            sequence_id (Optional[int], optional): setting sequence_id does not always work.
                instead of changing the sequence IDS, it sometimes does nothing (probably due to nonunique seq. ids)
                In order for this to work, IDs must start with 0 and continue as a sequence of integers, without repetitions.
                If None, does nothing. Defaults to None.

        Raises:
            Exception: _description_

        Returns:
            Encoding: _description_
        """
        assert isinstance(data_str, str)
        assert isinstance(input_type, str)

        if input_type not in self.tokenizers_info:
            raise Exception(f"Input type {input_type} not found")
        try:
            encoded = self.tokenizers_info[input_type]["tokenizer_inst"].encode(
                data_str
            )
        except Exception as e:
            raise e

        if (len(encoded.overflowing) > 0) and (verbose > 0):
            print(
                f"Warning: {self.__class__.__name__} had to truncate sequence. Original Sequence Length = {len(data_str)}, max tokens supported = {self.tokenizers_info[input_type]['max_len']}, exceeded by {len(encoded.overflowing[0].ids)} tokens, for tokenizer: {input_type}"
            )

        if sequence_id is None:
            sequence_id = int(self.tokenizers_info[input_type]["tokenizer_id"])
        # set_sequence_id does not always work.
        # Instead of changing the sequence IDS, it sometimes does nothing (probably due to nonunique seq. ids, if we use the same tokenizer for several sequences)
        # In order for this to work, IDs must start with 0 and continue as a sequence of integers.
        for ind_id in range(1, sequence_id + 1):
            encoded.set_sequence_id(ind_id)

        return encoded

    def get_expected_max_len(
        self, override_max_len: Optional[int] = None
    ) -> Optional[int]:
        """Returns the expected max_len of any encoding. max_len is given by internal state (set during initialization of the tokenizer), or it can be overridden
        during call to encode_list (applicable only to that specific encoding), or enable_padding/enable_truncation (applicable to all encodings produced
        following the call).

        Args:
            override_max_len (Optional[int], optional): Returns the resulting max_len, if the internal max_len were to be overridden by override_max_len
            during call to encode_list, or enable_padding/enable_truncation. Defaults to None.

        Returns:
            Optional[int]: _description_
        """
        if override_max_len is not None:
            return override_max_len
        return self.max_len

    def count_unknowns(
        self, encoding: Union[List, Encoding], unk_token: Optional[str] = None
    ) -> int:
        """Counts the number of unknown tokens in the encoding

        Args:
            encoding (Union[List, Encoding]): Either a list of IDs or an Encoding. If it is an encoding, only encoding.ids is parsed. Overflowing
                information is ignored, i.e. there may be an unknown token in overflowing ids (i.e. those that were cut due to encoding length limit)
            unk_token (Optional[str], optional): Unknown token string (usually "<UNK>"). If None, locates the token in special_tokens.special_tokens.
                Defaults to None.

        Returns:
            int: count of unknown tokens in the encoding
        """
        if isinstance(encoding, list):
            ids = encoding
        elif isinstance(encoding, Encoding):
            ids = encoding.ids
        else:
            raise Exception(
                f"Unexpected type of encoding {type(encoding)}, should be list or Encoding"
            )
        if unk_token is None:
            unk_token = special_wrap_input("UNK")
        unk_token_id = self.token_to_id(unk_token)
        return ids.count(unk_token_id)

    def encode_list(
        self,
        typed_input_list: List,
        max_len: Optional[int] = None,
        padding_token_id: Optional[int] = None,
        padding_token: Optional[str] = "<PAD>",
        pad_type_id: Optional[int] = None,
        return_overflow_info: Optional[bool] = False,
        on_unknown: Optional[str] = "raise",
        verbose: int = 1,
        also_return_split: bool = False,
    ) -> Union[
        Encoding,
        Tuple[Encoding, str],
        Tuple[Encoding, List[Encoding]],
        Tuple[Encoding, str, List[Encoding]],
    ]:
        """_summary_

        Args:
            typed_input_list (List): list of collections.namedtuple("input_type", ["input_string", "max_len"]), with
                input type: the name of input type,
                input_string: the string to be encoded
                max_len: maximal length of the encoding (in tokens). Only relevant for truncation, as we do not need to
                pad individual sub-tokenizer encodings - we only pad the final encoding of the ModularTokenizer.
                The smallest value between config-defined and tuple-defined is used. If None, the max_len
                that was defined for the sub-tokenizer in the config is used.
            max_len (Optional[int], optional): _description_. Defaults to None.
            padding_token_id (Optional[str], optional): _description_. Defaults to 0. TODO: default to None and infer it
            padding_token (Optional[str], optional): _description_. Defaults to "<PAD>".
            pad_type_id (Optional[int], optional): _description_. Defaults to 0. (TODO: raise exception)
            return_overflow_info (Optional[bool], optional): If True return an additional string with overflow information. Defaults to False.
            on_unknown: (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'
            verbose (Optional[int], optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
            also_return_split: defaults to False. If set to True, the return value will also contain a list that contains per meta-tokenizer-instruction element of Encoding
        Returns:
            Encoding: _description_
        """
        encoded_list = []
        # sequence_ids and sequence_names are an initial implementation of a currently broken huggingface
        # tokenizer functionality (Encoding.merge() does not preserve all sequence IDs).
        sequence_ids = []  # sequence id for each token (starting with 1)
        sequence_types = []  # encoder name used for each token
        curr_sequence_id = 1
        overflow_info = ""
        for inpt in typed_input_list:
            input_type = inpt.input_type
            data_str = inpt.input_string
            sub_max_len = inpt.max_len
            sub_encoding = self._encode_single_type(
                data_str=data_str,
                input_type=input_type,
                sequence_id=curr_sequence_id,
                verbose=verbose,
            )
            if sub_max_len is not None:
                if len(sub_encoding) > sub_max_len:
                    overflow_info += f"{input_type}:{len(sub_encoding)}=>{sub_max_len}|"
                sub_encoding.truncate(max_length=sub_max_len)
            encoded_list.append(sub_encoding)
            sequence_ids.extend([curr_sequence_id] * len(sub_encoding))
            sequence_types.extend([input_type] * len(sub_encoding))
            curr_sequence_id += 1
            # KEEP THIS AS DOC FOR NOW
            # encoded has attributes [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]
            # ids are the encoded tokens,
            # type_ids are for things like "which sentence is this from". There seem to be several limitations on those:
            #   - they must be unique, i.e. several different sentences cannot receive the same type_id from outside (for some reason they can be mapped
            #     to the same value if wrong ids were supplied by the user
            #   - it seems like the ids must be consecutive, starting with 0, in order to work as expected. If they do not start with 0,
            #     it is forced on the last sequence.
            # tokens are the actual tokens (for example - ['c1ccc(', '/C(', '=N/N', 'c2nc3ccccc3', 's2)', 'c2cccc', 'n2)cc1', '[PAD]', '[PAD]', '[PAD]'])
            # offsets describe the starting point and length of each original token
            # attention_mask - by default puts 1 for everything that isn't padding, and 0 for those that are padding
            # special_tokens_mask - 1 for anything that is a special token (e.g. padding, separator, etc.) 0 for the rest
            # overflowing - It's a list of Encoding structures of original content that got clipped out due to max length definition.
            #               In my experience, only the zeroth index contains anything. Don't know when there's more then one member in the list.

        merged_encoding = Encoding.merge(encoded_list)

        max_len = self.get_expected_max_len(override_max_len=max_len)
        if max_len is not None:
            if len(merged_encoding) > max_len:
                overflow_info += f"OVERALL:{len(merged_encoding)}=>{max_len}|"
            merged_encoding.truncate(max_length=max_len)

        if padding_token_id is None and padding_token is None:
            # if either padding token or id were given, or enable_padding was called earlier
            padding_token_id = self._pad_token_id
            padding_token = self._pad_token
        if padding_token is not None:
            # find the actual padding token ID from padding token
            padding_token_id = self.token_to_id(padding_token)
        else:
            if padding_token_id is not None:
                padding_token = self.id_to_token(padding_token_id)
        if pad_type_id is None:
            pad_type_id = self._pad_token_type_id
        if (
            padding_token_id is not None
            and padding_token is not None
            and max_len is not None
        ):
            merged_encoding.pad(
                length=max_len,
                direction="right",
                pad_id=padding_token_id,
                pad_token=padding_token,
                pad_type_id=pad_type_id,
            )
        else:
            if max_len is not None:
                warn(
                    f"both padding token and padding id are None, but padding length is {max_len}. It's possible that it was set for truncation alone."
                )

        # Check if we have any unknown tokens in our input
        unk_count = self.count_unknowns(merged_encoding)
        if unk_count > 0:
            # At some point we may want to know which parts of text were mapped to "UNK"
            # This is not quite intuitive:
            # - Encoding does not contain original text information
            # - Encoding does contain a field called Encoding.offsets, which contains a tuple
            #   (start index, end index) that point to the beginning and end of each token in
            #   the input text. That would allow us to access the original strings, if we knew
            #   the mapping between our input strings and the actual tokenized strings. However,
            #   this is not the case since we merge multiple inputs, and we may truncate each
            #   of them by number of tokens (not actual string length).
            # - Encoding.overflowing may contain relevant information, but I don't think it works
            #   properly, or, at least, I don't understand its logic. E.g. one instance of a
            #   TypedInput list containing 3 members resulted in 17 Encodings in overflowing...
            # - Finally, it would seem that encoding and then decoding (or, at this point,
            #   combining the contents of Embedding.tokens), and carefully parsing and comparing
            #   the input string and its encoded-decoded variant, will allow us to identify parts
            #   of input mapped to unk.
            if on_unknown == "raise":
                raise RuntimeError(
                    f"Encountered {unk_count} unknown tokens out of {len(merged_encoding.ids)} in input starting with {typed_input_list[0].input_string}"
                )
            elif on_unknown == "warn":
                if verbose == 0:
                    warning_message = None
                if verbose == 1:
                    warning_message = "Encountered unknown tokens in input"
                elif verbose == 2:
                    warning_message = f"Encountered {unk_count} unknown tokens in input starting with {typed_input_list[0].input_string[:20]}"
                elif verbose >= 3:
                    warning_message = f"Encountered {unk_count} unknown tokens out of {len(merged_encoding.ids)} in input starting with {typed_input_list[0].input_string}"
                else:
                    ValueError("We shouldn't be here")
                if warning_message is not None:
                    warn(warning_message)
            else:
                raise ValueError(
                    f"Unexpected on_unknown value {on_unknown}. Should be 'warn' or 'raise'"
                )

        if (not return_overflow_info) and (not also_return_split):
            return merged_encoding
        ans = [merged_encoding]
        if return_overflow_info:
            ans += [overflow_info]
        if also_return_split:
            ans += [encoded_list]

        return tuple(ans)

    def decode(self, ids: Iterable, skip_special_tokens: Optional[bool] = False) -> str:
        """Receives a list of IDs and returns a string of tokens
            TODO: possibly output also the type of token (AA, SMILES, etc)
        Args:
            ids (Iterable): _description_
            skip_special_tokens (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            str: _description_
        """
        if isinstance(ids, Tensor):
            # Tokens in 'self.decoder_dict' are integers, and not tensors scalars
            ids = ids.tolist()

        if skip_special_tokens:
            ret_val = [
                self.decoder_dict[id]["token"]
                for id in ids
                if id in self.decoder_dict and not self.decoder_dict[id]["is_special"]
            ]
        else:
            ret_val = [
                self.decoder_dict[id]["token"]
                if id in self.decoder_dict
                else f"<@TOKEN_MISSING-{id}>"
                for id in ids
            ]
        return "".join(ret_val)

    def encode(
        self,
        sequence: str,
        max_len: Optional[int] = None,
        padding_token_id: Optional[int] = 0,
        padding_token: Optional[str] = "<PAD>",
        pad_type_id: Optional[int] = 0,
        return_overflow_info: Optional[bool] = False,
        on_unknown: Optional[str] = "warn",
        verbose: Optional[int] = 1,
        also_return_split: bool = False,
    ) -> Encoding:
        # (self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True)
        """Receives a user-supplied string that contains, in addition to the text that is to be tokenized, special delimiters signifying the type
        of input within each span of text (e.g. <@TOKENIZER-TYPE=AA> sequence, <@TOKENIZER-TYPE=SMILES>, etc.). These determine the type of tokenizer to use on each span,
        and are not encoded.
        Optionaly, you may also describe maximum length per section, for example:
            "<@TOKENIZER-TYPE=AA><BLAH><BLAH2>QKPGQAPRLLIYG<@TOKENIZER-TYPE=AA@MAX-LEN=122><BLAH3>SGSDFSDFSFD"
            would not have a local limitation of the first AA section, but will have a local maximum length of 122 on the second section.
            local in this context means that the maximum length will be imposed on the individual section prior to applying any global "entire sequence" maximum size limitations (if any).

        Args:
            input_string (str): _description_
            max_len (Optional[int], optional): _description_. Defaults to None.
            padding_token_id (Optional[str], optional): _description_. Defaults to 0.
            padding_token (Optional[str], optional): _description_. Defaults to "<PAD>".
            pad_type_id (Optional[int], optional): _description_. Defaults to 0.
            return_overflow_info (Optional[bool], optional): _description_. If True return an additional string with overflow information. Defaults to False.
            on_unknown: (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'
            verbose (int, optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
            also_return_split: also return the per-meta-instruction encoded parts as a list of Encoding elements
        Returns:
            Encoding: _description_
            str: _description_ information on overflow, if return_overflow_info=True
        """
        # split sequence to token hints and the following sequence

        hints_and_subseq = re.split("<@TOKENIZER-TYPE=([^>]*)>", sequence)[
            1:
        ]  # the first element is blank - removing it
        assert (
            len(hints_and_subseq) > 0 and len(hints_and_subseq) % 2 == 0
        ), f"Error: expecting leading modular tokenizer hints followed by a sequence to tokenize, got {sequence}"
        # arrange as a list of TypedInput - each one will include the type and the following sequence
        encode_list_format = []
        for tokenizer_type, subseq in zip(
            hints_and_subseq[::2], hints_and_subseq[1::2]
        ):
            max_len_str = "@MAX-LEN="
            curr_max_len_idx = tokenizer_type.find(max_len_str)
            if curr_max_len_idx > 0:
                curr_max_len = tokenizer_type[curr_max_len_idx + len(max_len_str) :]
                try:
                    curr_max_len = int(curr_max_len)
                except:
                    raise Exception(
                        f"Had a problem casting curr_max_len={curr_max_len} to int! it was found inside modular tokenizer meta TOKENIZER-TYPE={tokenizer_type}"
                    )
                tokenizer_type = tokenizer_type[:curr_max_len_idx]
            else:
                curr_max_len = None
            encode_list_format.append(TypedInput(tokenizer_type, subseq, curr_max_len))

        return self.encode_list(
            typed_input_list=encode_list_format,
            max_len=max_len,
            padding_token_id=padding_token_id,
            padding_token=padding_token,
            pad_type_id=pad_type_id,
            return_overflow_info=return_overflow_info,
            on_unknown=on_unknown,
            verbose=verbose,
            also_return_split=also_return_split,
        )

    def get_tokenizer_types(self) -> List:
        return list(self.tokenizers_info.keys())

    ########## Original Tokenizer functions: ##################
    def add_special_tokens(self, tokens: List[str]) -> int:
        """
        Add the given special tokens to the Tokenizer. If max_special_token_id was set, the new token are mapped to IDs below it,
        and if some tokens on the list are existing regular tokens,

        If these tokens are already part of the vocabulary, it just lets the Tokenizer know about
        them. If they don't exist, the Tokenizer creates them, giving them a new id.

        These special tokens will never be processed by the model (ie won't be split into
        multiple tokens), and they can be removed from the output when decoding.

        General token addition notes:
        There are three general cases when adding:
        1. There is no limitation on IDs. In this case, new IDs are added after the last taken ID, and the ID space is
            compact, with no holes.
        2. There's an upper limit on all IDs (self._max_possible_token_id). In this case, the new IDs (regular and special)
            are also added after the last taken ID, and ID space is compact, but limited in size. Any tokens added beyond
            the limit must raise an exception.
        3. There's an upper limit in special IDs (self._max_special_token_id). In this case, special IDs are added after
            the last taken special ID and before special ID limit, and regular IDs are added after last taken regular ID
            (and before all ID limit, if such is defined). Any tokens added beyond the limit must raise an exception. The
            ID space consists of two parts (with an empty buffer between them):
            - [0..upper special ID limit], containing special IDs only, compacted at the beginning of the range
            - (upper special id limit, infinity or upper all ID limit], containing regular IDs only, compacted at the
                beginning of the range.

        When we add special tokens, there are three options:
        - Special tokens that already exist as special tokens in the tokenizer:
            - These do not need to be added
        - Special tokens that already exist as regular tokens: For now, this causes an error.
            - If there is an upper limit on special token IDs, then the regular tokens that turn into special need to
                have their IDs remapped to values lower than the upper limit. This is not an acceptable option, since
                remapping IDs will break all models trained with this tokenizer. The only way around this is to choose
                other token names to add.
            - If there is no upper limit on special token IDs, we could add them to the special token struct, retaining
                their original IDs. However this may not be possible, since some tokens map to multiple IDs, depending
                on context. It could be possible to detect such cases and raise an error only when there is no 1:1 mapping
                between tokens and IDs, but for now, we raise an error in any case.
        - Special tokens that do not exist in the tokenizer
            - Depending on whether there is or there is not an upper limit on special IDs, these are added after the
                last taken special ID, or after the last taken general ID (special or regular).

        When we add regular tokens as part of a new sub-tokenizer:
        - Tokens that already exist as special tokens: These need not be added to the tokenizer - we just need to add
            the entire common special token vocab to the new subtokenizer.
        - Tokens that already exist in another subtokenizer as regular tokens/tokens that do not exist: Since we allow
            similar regular token names in different tokenizers (as long as they map to different IDs), these two are
            the same - we just add the tokens to the new subtokenizer, remapping their IDs after the last taken ID (and
            before all ID limit, if there is such).

        When we add regular tokens to a given sub-tokenizer):
        - Tokens that already exist as special tokens: These need not be added to the tokenizer.
        - Tokens that already exist in the same sub-tokenizer: need not be added
        - Tokens that already exist in another subtokenizer as regular tokens/tokens that do not exist: We just add the
            tokens to the new subtokenizer, remapping their IDs after the last taken ID (and before all ID limit, if
            there is such).

        Args:
            tokens (A :obj:`List` of  :obj:`str`):
                The list of special tokens we want to add to the vocabulary. Each token must
                be a string.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary

        TODO: If we try to add special tokens and reach max_special ID, allow the option to add part of the tokens to the
        remaining buffer space, and the rest after max taken regular ID
        """

        def update_vocab(
            vocab: Dict,
            special_token_structs: List,
        ) -> Dict:
            """Receives a vocabulary and a list of special token structures. Returns a new vocabulary that
            a. contains all the special tokens with their IDs, as were given in special_token_structs.
            b. contains all the tokens in vocab (except special ones), with their original IDs.

            Args:
                vocab (Dict): vocabulary of tokens to be included in the ModularTokenizer. If there is an overlap between tokens in vocab and tokens
                in special_token_structs, raises an exception.
                special_token_structs (Optional[List]): a list of special token structures to be added to the tokenizer.

            Returns:
                Dict: Returns the updated vocabulary, sorted by value
            """
            if special_token_structs is not None and len(special_token_structs) > 0:
                special_vocab = {t["content"]: t["id"] for t in special_token_structs}
            else:
                raise Exception("Got empty special tokens")
            vocab.update(special_vocab)
            return dict(sorted(vocab.items(), key=lambda x: x[1], reverse=False))

        # remove from tokens all currently existing special_tokens
        special_vocab = self.get_added_vocab()
        current_special_tokens = set(special_vocab.keys())
        tokens = list(set(tokens) - current_special_tokens)

        # go over all tokens that already exist as regular tokens, and if such exist and self._max_special_token_id is
        # not set, mark them as special (adding them to special token strictures for all sub-tokenizers), otherwise raise an exception.
        #   An alternative would be to either:
        #   - Make them special, while keeping their ID (works only if self._max_special_token_id is not set,
        #   because if not it'll break the condition that they must have IDs below self._max_special_token_id) or
        #   - Remove them from regular tokens (so that when they're added to special tokens they'll be assigned new IDs)
        #       This way we preserve the condition of special token IDs being below self._max_special_token_id. However
        #       this will change IDs of existing tokens, and break the logic of the model,
        #       which will require retraining - not something we want
        # tokens <- new tokens, without existing special tokens, and, possibly, without existing regular tokens (depending on the above choice)

        # At this point tokens contain to existing special tokens, but may contain regular tokens
        all_existing_tokens = set([x["token"] for x in self.decoder_dict.values()])
        tokens_regular = sorted(list(set(tokens).intersection(all_existing_tokens)))
        tokens = sorted(list(set(tokens) - set(tokens_regular)))
        # At this point tokens contain no tokens that currently exist in the modular tokenizer, and tokens_regular contain
        # special tokens to be added that are currently regular tokens in the tokenizer

        if len(tokens_regular) > 0:
            raise Exception(
                f"Trying to add to the tokenizer tokens that are currently regular tokens in the tokenizer. Choose other token names. Conflicting tokens are {tokens_regular}"
            )

        if len(tokens) == 0:
            return 0

        if self._max_special_token_id is not None:
            if len(tokens_regular) > 0:
                raise Exception(
                    f"Trying to add to the tokenizer tokens that are currently regular tokens in the tokenizer. Since _max_special_token_id is set, there is no way to uphold it without remapping existing IDs. Conflicting tokens are {tokens_regular}"
                )
            max_id = self._max_special_token_id
            next_id: int = max(special_vocab.values()) + 1
            if max_id - next_id < len(tokens):
                raise Exception("Not enough free special token space left")
        elif self._max_possible_token_id is not None:
            max_id = self.self._max_possible_token_id
            next_id = self._get_max_mapped_id() + 1
            if max_id - next_id < len(tokens):
                raise Exception("Not enough free token space left")
        else:
            next_id = self._get_max_mapped_id() + 1

        all_special_token_structs = ModularTokenizer.build_special_token_list(
            special_tokens=tokens, starting_index=next_id
        )

        for t_type in self.tokenizers_info:
            t_info = self.tokenizers_info[t_type]
            t_json = self.tokenizers_info[t_type]["json_instance"]
            # operations on the tokenizer json
            if "added_tokens" in t_json and t_json["added_tokens"] is not None:
                t_json["added_tokens"] += all_special_token_structs
            else:
                t_json["added_tokens"] = all_special_token_structs

            t_json["model"]["vocab"] = update_vocab(
                vocab=t_json["model"]["vocab"],
                special_token_structs=all_special_token_structs,
            )
            # end operations on json
            # operations on the tokenizer instance (if possible, operations should be done here, using built-in tokenizer methods)
            json_str = json.dumps(t_json)
            tokenizer_inst = Tokenizer.from_str(json_str)

            # restore truncation that was lost when we reset the tokenizer instance
            if "max_len" in t_info and t_info["max_len"] is not None:
                max_size = t_info["max_len"]
                tokenizer_inst.enable_truncation(
                    max_length=max_size,
                    direction="right",
                )
            json_str = tokenizer_inst.to_str()
            t_json = json.loads(json_str)
            self.tokenizers_info[t_type]["tokenizer_inst"] = tokenizer_inst
            self.tokenizers_info[t_type]["json_instance"] = t_json

        # Rebuild inner decoder information
        self.build_inner_decoder()
        return len(tokens)

    def add_tokens(self, tokens: Union[List, str]) -> int:
        """
        Add the given tokens to the vocabulary

        The given tokens are added only if they don't already exist in the vocabulary.
        Each token then gets a new attributed id.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of tokens we want to add to the vocabulary. Each token can be either a
                string or an instance of :class:`~tokenizers.AddedToken` for more customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        raise Exception("Not implemented")
        # self.build_inner_decoder()
        # if self._max_possible_token_id is not None:
        #     if self._get_max_mapped_id() > self._max_possible_token_id:
        #         raise Exception(
        #             f"tokenizer remapping resulted in IDs greater (max_id={self._get_max_mapped_id()}) than max_possible_id ({self._max_possible_token_id}). Reinitialize the modular tokenizer with larger max_possible_id"
        #         )

    def decode_batch(
        self, sequences: List, skip_special_tokens: Optional[bool] = True
    ) -> List:
        """
        Decode a batch of ids back to their corresponding string

        Args:
            sequences (:obj:`List` of :obj:`List[int]`):
                The batch of sequences we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded strings

        Returns:
            :obj:`List[str]`: A list of decoded strings
        """
        raise Exception("Not implemented")

    @property
    def decoder(self) -> None:
        """
        The `optional` :class:`~tokenizers.decoders.Decoder` in use by the Tokenizer
        """
        raise Exception("Not implemented")

    def enable_padding(
        self,
        direction: Optional[str] = "right",
        pad_id: Optional[int] = None,
        pad_type_id: Optional[int] = 0,
        pad_token: Optional[str] = "<PAD>",
        length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        """
        Enable the padding. Note: Also enables truncation to the same max_len

        Args:
            direction (:obj:`str`, `optional`, defaults to :obj:`right`):
                The direction in which to pad. Can be either ``right`` or ``left``

            pad_to_multiple_of (:obj:`int`, `optional`):
                If specified, the padding length should always snap to the next multiple of the
                given value. For example if we were going to pad witha length of 250 but
                ``pad_to_multiple_of=8`` then we will pad to 256.

            pad_id (:obj:`int`, defaults to 0):
                The id to be used when padding

            pad_type_id (:obj:`int`, defaults to 0):
                The type id to be used when padding

            pad_token (:obj:`str`, defaults to :obj:`[PAD]`):
                The pad token to be used when padding

            length (:obj:`int`, `optional`):
                If specified, the length at which to pad. If not specified we pad using the size of
                the longest sequence in a batch.
        """
        assert direction == "right", "pad direction other than right is not implemented"

        assert pad_to_multiple_of is None, "pad_multiple_of is not implemented"

        if pad_token is None and pad_id is None:
            raise Exception(
                "enable_padding was called, but neither padding token nor padding id were given"
            )
        if pad_token is not None and pad_id is not None:
            # if both are given, make sure they map to each other
            if self.token_to_id(pad_token) != pad_id:
                raise Exception(
                    f"pad_token {pad_token} does not correspond to pad_id {pad_id}"
                )
        # at this point either padding token or id (or both) must be not None:
        if pad_id is None:
            if pad_token is not None:
                pad_id = self.token_to_id(pad_token)
        if pad_token is None:
            if pad_id is not None:
                pad_token = self.id_to_token(pad_id)

        self._pad_token_type_id = pad_type_id
        self._pad_token = pad_token
        self._pad_token_id = pad_id
        self.max_len = length

    def enable_truncation(
        self,
        max_length: int,
        stride: Optional[int] = 0,
        strategy: Optional[str] = "longest_first",
        direction: Optional[str] = "right",
    ) -> None:
        """
        Enable truncation. Note: Also sets padding length to max_len, if padding is enabled.

        Args:
            max_length (:obj:`int`):
                The max length at which to truncate

            stride (:obj:`int`, `optional`):
                The length of the previous first sequence to be included in the overflowing
                sequence

            strategy (:obj:`str`, `optional`, defaults to :obj:`longest_first`):
                The strategy used to truncation. Can be one of ``longest_first``, ``only_first`` or
                ``only_second``.

            direction (:obj:`str`, defaults to :obj:`right`):
                Truncate direction
        """
        assert stride == 0, "stride not implemented"
        assert strategy == "longest_first", "strategy not implemented"
        assert direction == "right", "direction setting not implemented"
        self.max_len = max_length

    def encode_batch(
        self,
        input: List,
        is_pretokenized: Optional[bool] = False,
        add_special_tokens: Optional[bool] = True,
    ) -> List:
        """
        Encode the given batch of inputs. This method accept both raw text sequences
        as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode_batch([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch

        """
        raise Exception("Not implemented")

    @staticmethod
    def from_buffer(buffer: object) -> object:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.

        Args:
            buffer (:obj:`bytes`):
                A buffer containing a previously serialized :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        raise Exception("Not implemented")

    @staticmethod
    def from_file(path: str) -> "ModularTokenizer":
        """
        Accepts a file or directory, and loads a modular tokenizer stored in that directory

        Args:
            path (:obj:`str`):
                A path to a local config file representing a previously saved
                :class:`ModularTokenizer`

        Returns:
            :class:`ModularTokenizer`: The new tokenizer
        """
        if os.path.isfile(path):
            path = os.path.dirname(path)
        return ModularTokenizer.load(path)

    @staticmethod
    def from_pretrained(
        identifier: str,
        revision: Optional[str] = "main",
        auth_token: Optional[str] = None,
    ) -> Any:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from an existing file on the
        Hugging Face Hub.

        Args:
            identifier (:obj:`str`):
                The identifier of a Model on the Hugging Face Hub, that contains
                a tokenizer.json file
            revision (:obj:`str`, defaults to `main`):
                A branch or commit id
            auth_token (:obj:`str`, `optional`, defaults to `None`):
                An optional auth token used to access private repositories on the
                Hugging Face Hub

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        if not os.path.isdir(identifier):
            raise Exception(
                f"Error: expecting {identifier} to be a directory that contains config.yaml file and sub-tokenizers config"
            )
        path_to_config_file = os.path.join(identifier, "config.yaml")
        if not os.path.isfile(path_to_config_file):
            raise Exception(
                f"Error: expecting {identifier} to be a directory that contains config.yaml file"
            )
        path_to_config_file = os.path.dirname(path_to_config_file)

        return ModularTokenizer.load(path_to_config_file)

    @staticmethod
    def from_str(json: str) -> object:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.

        Args:
            json (:obj:`str`):
                A valid JSON string representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        raise Exception("Not implemented")

    def get_vocab(self, with_added_tokens: Optional[bool] = True) -> Dict:
        """
        Get the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[str, int]`: The vocabulary

        Note: Irrelevant to ModularTokenizer, since it may not be possible to express with a single vocabulary
        """
        raise Exception(
            "Not implemented, because the functionality is not defined for ModularTokenizer. Use either get_added_vocab() or get_typed_vocab()"
        )

    def get_added_vocab(self) -> Dict:
        """
        Get the underlying vocabulary including only the added tokens (i.e. the ones that are present in all sub-tokenizers, and map to the same ids)
        as token to id dictionary.

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[str, int]`: The vocabulary

        Note: Irrelevant to ModularTokenizer, since it may not be possible to express with a single vocabulary
        """
        t_types = self.get_tokenizer_types()
        assert len(t_types) >= 1
        t_type = t_types[
            0
        ]  # TODO: Here we consider the special tokens of the first subtokenizer alone,
        # Assuming all subtokenizers are consistent. This, however, is not necessarily the case. For example,
        # if a second multitokenizer (Z) was created from the first one (Y) by adding another subtokenizer (A).
        # The first multitokenizer (Y) then was updated with additional special tokens, then all its subtokenizers
        # were updated, but subtokenizer A was not (since it's only part of Z and not of Y). Next time multitokenizer
        # Z is loaded, it will no longer be consistent - its subtokenizer A will be missing special tokens.
        # If we try to add the missing tokens to Z, we'll fail because they're found in its first subtokenizer.
        # Possible solutions:
        # A. Test a ModularTokenizer for consistency each time it's loaded.
        #       - If it is not consistent, add a consolidation function that will identify missing tokens (and their IDs)
        #           from each subtokenizer and add them, if possible (throwing an exception if not)
        # B. Test ModularTokenizer for consistency each time before it is changed (e.g. by add_special_tokens), and consilidate it
        #       if needed/possible
        tokenizer_json_inst = self.tokenizers_info[t_type]["json_instance"]
        special_tokens_list = ModularTokenizer.get_subtokenizer_added_tokens(
            tokenizer_json_inst=tokenizer_json_inst
        )
        special_tokens_dict = ModularTokenizer.get_subtokenizer_vocab(
            tokenizer_json_inst=tokenizer_json_inst, token_list=special_tokens_list
        )
        return special_tokens_dict

    def get_typed_vocab(self, with_added_tokens: Optional[bool] = True) -> Dict:
        """
        Get the underlying vocabulary, as (token type, token) to id dictionary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[Tuple[str, str], int]`: The following dictionary:
            {
                (type of tokenizer, token):id
                }
            with type of tokenizer being the sub-tokenizer name, token - a token in the sub-tokenizer and id - the id of the token in the sub-tokenizer.
            If a token is present in several sub-tokenizers, with the same id (e.g. if it's an added/special token), only one occurrence is present. If a token
            is present in several sub-tokenizers, but with different ids - all occurrences are present.

        Note: Irrelevant to ModularTokenizer, since it may not be possible to express with a single vocabulary
        """
        raise Exception("Not implemented")

    def get_vocab_size(self, with_added_tokens: Optional[bool] = True) -> int:
        """
        Get the size of the underlying vocabulary, in terms of number of tokens. This does NOT return max token id.

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        # TODO: It may be possible to count all the unique IDs (i.e. sum of numbers of
        # regular tokens of each subtokenizer plus the number of special tokens, which
        # are the common to all subtokenizers)
        if not with_added_tokens:
            raise Exception("Not implemented")
        else:
            return len(list(self.decoder_dict.values()))

    def _get_max_mapped_id(self, with_added_tokens: Optional[bool] = True) -> int:
        """
        Get value of the highest used ID of the underlying vocabulary (i.e. the highest ID that has a token mapped to it)

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        if not with_added_tokens:
            raise Exception("Not implemented")
        else:
            return max(list(self.decoder_dict.keys()))

    def _get_max_mapped_special_id(
        self, with_added_tokens: Optional[bool] = True
    ) -> int:
        """
        Get value of the highest used ID of the underlying vocabulary (i.e. the highest ID that has a token mapped to it)

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        if not with_added_tokens:
            raise Exception("Not implemented")
        else:
            return max(list(self.decoder_dict.keys()))

    def get_max_id(self, with_added_tokens: Optional[bool] = True) -> int:
        """
        Get value of the highest ID of the underlying vocabulary. If there is an upper limit defined for the modular tokenizer, it is returned

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        if self._max_possible_token_id is not None:
            return self._max_possible_token_id
        return self._get_max_mapped_id(with_added_tokens=with_added_tokens)

    def id_to_token(self, id: int) -> Optional[str]:
        """
        Convert the given id to its corresponding token if it exists
        In general, id_to_token is undefined for MultiTokenizer bec
        Args:
            id (:obj:`int`):
                The id to convert

        Returns:
            :obj:`Optional[str]`: An optional token, :obj:`None` if out of vocabulary
        """
        if not isinstance(id, int):
            raise TypeError(
                f"Expected 'id' to be an integer, got {type(id).__name__} instead."
            )

        token_info = self.decoder_dict.get(id, None)
        if token_info is not None:
            return token_info["token"]
        return None

    @property
    def model(self) -> None:
        """
        The :class:`~tokenizers.models.Model` in use by the Tokenizer
        """
        raise Exception("Not implemented")

    def no_padding(self) -> None:
        """
        Disable padding
        """
        raise Exception("Not implemented")

    def no_truncation(self) -> None:
        """
        Disable truncation
        """
        raise Exception("Not implemented")

    @property
    def normalizer(self) -> None:
        """
        The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer
        """
        raise Exception("Not implemented")

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        raise Exception("Not implemented")

    @property
    def padding(self) -> Optional[Dict]:
        """
        Get the current padding parameters

        `Cannot be set, use` :meth:`~tokenizers.Tokenizer.enable_padding` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current padding parameters if padding is enabled
        """
        raise Exception("Not implemented")

    def post_process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: Optional[bool] = True,
    ) -> Encoding:
        """
        Apply all the post-processing steps to the given encodings.

        The various steps are:

            1. Truncate according to the set truncation params (provided with
               :meth:`~tokenizers.Tokenizer.enable_truncation`)
            2. Apply the :class:`~tokenizers.processors.PostProcessor`
            3. Pad according to the set padding params (provided with
               :meth:`~tokenizers.Tokenizer.enable_padding`)

        Args:
            encoding (:class:`~tokenizers.Encoding`):
                The :class:`~tokenizers.Encoding` corresponding to the main sequence.

            pair (:class:`~tokenizers.Encoding`, `optional`):
                An optional :class:`~tokenizers.Encoding` corresponding to the pair sequence.

            add_special_tokens (:obj:`bool`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The final post-processed encoding
        """
        raise Exception("Not implemented")

    @property
    def post_processor(self) -> None:
        """
        The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer
        """
        raise Exception("Not implemented")

    @property
    def pre_tokenizer(self) -> None:
        """
        The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer
        """
        raise Exception("Not implemented")

    def to_str(self, pretty: Optional[bool] = False) -> None:
        """
        Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.

        Args:
            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON string should be pretty formatted.

        Returns:
            :obj:`str`: A string representing the serialized Tokenizer
        """
        raise Exception("Not implemented")

    def token_to_id(self, token: str, t_type: Optional[str] = None) -> Union[int, None]:
        """
        Convert the given token to its corresponding id if it exists
        In general, token_to_id is undefined for MultiTokenizer because the same
        token may get mapped to different ids, depending on the subtokenizer type

        Args:
            token (:obj:`str`):
                The token to convert
            t_type (:obj:`str`): The subtokenizer to use. If None, the first (in order defined in the config)
                subtokenizer is used. If the token is special, type should not be set. TODO: raise a warning
                if type=None and the token is not special

        Returns:
            :obj:`Optional[int]`: An optional id, :obj:`None` if out of vocabulary
        """
        if t_type is None:
            t_type_val = list(self.tokenizers_info.keys())[0]
            possible_ids = []
            possible_id_types = []
            for t_type_val in self.tokenizers_info.keys():
                tok_id = self.tokenizers_info[t_type_val]["tokenizer_inst"].token_to_id(
                    token
                )
                if tok_id is not None:
                    possible_ids.append(tok_id)
                    possible_id_types.append(t_type_val)
            possible_ids_unique = list(set(possible_ids))
            if len(possible_ids_unique) == 0:
                return None
            elif len(possible_ids_unique) == 1:
                return possible_ids_unique[0]
            else:
                raise Exception(
                    f"Token {token} maps to several possible ids {possible_ids}, of types {possible_id_types}, and the t_type argument was not set"
                )
        else:
            t_type_val = str(t_type)
            return self.tokenizers_info[t_type_val]["tokenizer_inst"].token_to_id(token)

    def train(
        self, files: List, trainer: Optional[tokenizers.trainers.Trainer] = None
    ) -> None:
        """
        Train the Tokenizer using the given files.

        Reads the files line by line, while keeping all the whitespace, even new lines.
        If you want to train from data store in-memory, you can check
        :meth:`~tokenizers.Tokenizer.train_from_iterator`

        Args:
            files (:obj:`List[str]`):
                A list of path to the files that we should use for training

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model
        """
        raise Exception("Not implemented")

    def train_from_iterator(
        self,
        iterator: Iterator,
        trainer: Optional[tokenizers.trainers.Trainer] = None,
        length: Optional[int] = None,
    ) -> None:
        """
        Train the Tokenizer using the provided iterator.

        You can provide anything that is a Python Iterator

            * A list of sequences :obj:`List[str]`
            * A generator that yields :obj:`str` or :obj:`List[str]`
            * A Numpy array of strings
            * ...

        Args:
            iterator (:obj:`Iterator`):
                Any iterator over strings or list of strings

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model

            length (:obj:`int`, `optional`):
                The total number of sequences in the iterator. This is used to
                provide meaningful progress tracking
        """
        raise Exception("Not implemented")

    @property
    def truncation(self) -> Optional[Dict]:
        """
        Get the currently set truncation parameters

        `Cannot set, use` :meth:`~tokenizers.Tokenizer.enable_truncation` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current truncation parameters if truncation is enabled
        """
        raise Exception("Not implemented")
