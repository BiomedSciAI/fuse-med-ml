# Modular Tokenizer:
* A modular tokenizer combines multiple pre-trained (huggingface-based) tokenizers and maps their tokens to a single, consistent ID space. It's useful for sequence-to-sequence problems, where different tokenizers should be used for different parts of the input sequence, depending on the context, and straightforward merging of the tokenizers may not be possible due to token overlap (e.g. 'C' in an amino acid sequence, standing for Cysteine, and 'C' in a SMILES sequence, standing for Carbon, should be mapped to different IDs).
* The modular tokenizer retains most of huggingface tokenizer interface (but not the underlying logic), so it can be plugged into existing code with very few (if any at all) changes.
## Definitions:
* __sub-tokenizer__: One of several underlying tokenizers of the modular tokenizer. Maps to a consistent ID space.
* __initial-tokenizer__: One of several pre-trained tokenizers used to create a modular tokenizer. Maps to an ID space of its own, which may (and probably does) overlap with those of other initial tokenizers.
## Interface:
### __init__():
Creates a modular tokenizer that combines multiple initial tokenizers, adjusting them so that:
* They all share the same special tokens (combined special tokens from all the source tokenizers),
* Each tokenizer retains its regular tokens, however their IDs are remapped to a single space, with no overlaps.
Note: If a token has the same meaning across all input types (e.g. special tokens, like SEP, EOS, sentinel tokens), it should be defined as special token in at least one of the initial (input) tokenizers.

* __init__() has two optional parameters: upper special token ID limit, and upper ID limit for all tokens. Depending on their values, there are three options of ID mapping:

        1. There is no limitation on IDs. In this case, new IDs are added after the last taken ID, and the ID space is
            compact, with no holes.

        2. There's an upper limit on all IDs (self._max_possible_token_id). In this case, the new IDs (regular and special)
            are also mapped after the last taken ID, and ID space is compact, but limited in size. Any tokens added beyond
            the limit must raise an exception.

        3. There's an upper limit in special IDs (self._max_special_token_id). In this case, special IDs are mapped after
            the last taken special ID and before special ID limit, and regular IDs are mapped after last taken regular ID
            (and before all ID limit, if such is defined). Any tokens mapped beyond the limit must raise an exception. The
            ID space consists of two parts (with an empty buffer between them):
            - [0..upper special ID limit], containing special IDs only, compacted at the beginning of the range
            - (upper special id limit, infinity or upper all ID limit], containing regular IDs only, compacted at the
                beginning of the range.

### add_special_tokens():
Adds a list of special tokens to the modular tokenizer. This does not change the existing tokenizer IDs, just adds new ones. If the modular tokenizer was created
with max_special_token_id, the special tokens will be mapped to IDs between max current special token ID and max_special_token_id.
### decode():
Decodes a list of tokens
### diagnose():
Tests a modular tokenizer for underlying ID mapping consistency, checking that the following hold for all sub-tokenizers:
* Special tokens are the same (and map to the same indices) across all the tokenizers
* Regular token ID mappings of any given tokenizer do not collide with special token mappings
* Regular token ID mappings of any given tokenizer do not collide with ID mappings of other tokenizers
### enable_padding():
Enables padding to a given length, using a given padding token. Also enables truncation of sequences to the same length.
### enable_truncation():
Enables truncation of encoded sequences to a given length. If padding is enabled, padding length is also set to given length.
### encode():
(Not implemented yet) Receives a string, infers which tokenizers to use on it and returns its tokenization.
### encode_list():
Receives a list of named tuples, each containing the type of tokenizer to use, a string to be tokenized, and, optionally, maximum length (in tokens) of the result. Tokenizes each input string.
### from_file():
Receives a path to a file or a directory and loads a modular tokenizer from that directory.
### from_pretrained():
Receives a path to a directory and loads a modular tokenizer from that directory.
### get_added_vocab():
Returns a vocabulary of all special tokens (ones common between all subtokenizers)
### get_max_id():
Returns the highest mapped ID in the vocabulary, or the upper limit to ID, if it was set
### get_vocab_size():
Returns the size of the vocabulary of the modular tokenizer (i.e. the number of unique IDs, which may be greater than the number of unique tokens)
### id_to_token():
Returns the token that maps to the input ID.
### load():
Loads a modular tokenizer saved by save()
### load_from_jsons():
Loads a group of adjusted tokenizers (created by __init__, andsaved by save_jsons), and returns a modular tokenizer with the same ID mapping.
### save():
Saves all mudular tokenizer information to a given path.
### save_jsons():
Saves the underlying adjusted tokenizers as jsons.
### token_to_id():
Returns the input token's corresponding ID.
## Use example
### Creation:
A script for creating new modular tokenizers can be found in [ModularTokenizer creation](scripts/create_multi_tokenizer.py).
Example of configuration file for the script that adds Amino-Acid tokenizer, smiles tokenizer and cell attributes tokenizer can be found here(pretrained_tokenizers/configs/tokenizer_config.yaml).
Note: this line [path definition](pretrained_tokenizers/configs/tokenizer_config.yaml#L3) needs to be changed so that _your_path_ points to cloned fuse-drug parent directory.

#### General creation steps:
* Collect all sub-tokenizer jsons, and add them to a config, similarly to [tokenizer_config.yaml](pretrained_tokenizers/configs/tokenizer_config.yaml)
* Run [ModularTokenizer creation](create_multi_tokenizer.py). The script will a. create a new tokenizer that contains all required added tokens and all the sub-tokenizers; and b. Test the resulting tokenizer for consistency.

### Adding special tokens steps:
An example of adding special tokens to an existing tokenizer is found here: [ModularTokenizer update](scripts/add_multi_tokenizer_special_tokens.py). Note, that these are instructions to add new tokens, not to remove existing ones.
Removing tokens from special_tokens.py will accomplish nothing, since they are already in the saved version of modular tokenizer, and our current strategy only allows building upon an existing tokenizer by adding tokens. The only way to remove a token would be to directly edit tokenizer json files, which must be done with extreme caution, to avoid changing token-ID mapping. Also, removing tokens should be avoided, since we may break models that use them. The steps are as follows:

# Pretrained Modular tokenizers
We currently have a full modular tokenizer *bmfm_extended_modular_tokenizer* and a smaller tokenizer, the *bmfm_modular_tokenizer* which is a strict subset of the first.
This relationship must be maintained, as it will allow mixing models and tasks between different modalities.  To maintain this consistency, whenever adding tokens to one of the modular tokenizers, the same tokens must be added to each the modular tokenizers.
To add tokens to all those tokenizer at once use[update_special_tokens_for_all_modular_tokenizers.sh](pretrained/tokenizers/update_special_tokens_for_all_modular_tokenizers.sh)
A test in the Jenkins run will prevent merging of inconsistent tokenizers.
#### Backward compatibility and deprecation of old modular tokenizer:
The *modular_AA_SMILES_single_path* modular tokenizer was renamed to *bmfm_modular_tokenizer*, and is temporarily kept as an identical copy for backwards compatibility.  Remember to update this version as well to maintain compatibility, and update your code to new name.


## Config structure:
The init and load_from_jsons functions both receive a list of dictionaries, each defining a single type of tokenizer. The dictionaries have the following fields:
* name: Name of the tokenizer (usually depicting its use context - AA sequences, SMILES, etc)
* tokenizer_id:    unique int identifier of the tokenizer
* json_path:       a path to a json file containing the initial input tokenizer
* modular_json_path: a path to json that will contain the updated (remapped) sub-tokenizer that will be derived from the initial tokenizer (automatically generated by save())
* max_len: (Optional) maximum number of tokens encoded by each instance of this tokenizer. If not given or None - no limit is set. If max_len is defined both here and during a call to encode_list, the smallest one is used.
*
## Adding new tokens
There are two ways to add new tokens:
* Adding a list of special tokens (e.g. task-related), by running [add_multi_tokenizer_special_tokens](scripts/add_multi_tokenizer_special_tokens.py)
* Adding a whole new tokenizer, by running [add_tokenizer_to_multi_tokenizer.py](scripts/add_tokenizer_to_multi_tokenizer.py)
  - note - this will also update the special tokens as above.


## Adding new tokenizer
To add a new tokenizer:
* create a config file similar to the one in the general creation step.
* The new tokenizer should be added to the tokenizers section (you may keep the full list or just the new tokenizer)
* Remember to increase the tokenizer_id
* Add any new special tokens to the special_tokens.py as above.
* set `modular_tokenizers_out_path` to the new output directory
* set `in_path` to the base version of the tokenizer you wish to add to
* run the `add_tokenizer_to_multi_tokenizer.py` script

## Extended tokenizer
The default tokenizer has the highest token ID set to 5000. This leaves plenty of free space to add tokenizers, heavy memory usage.
The human genes taxonomy we use includes at least about 24000 genes, and will clearly not fit inside the 5k space.
For the time being, the gene taxonomy is only used by the BMFM-targets team.  For this use case, we extended to tokenizer to 35000 id's by adding the genes from id 5000 and on.
This allows for an "all but genes" tokenizer which is small, while the extended tokenizer is kept consistent.
To maintain this, one need to place all special-tokens from the extended tokenizers (for now, only GENE) in the [special-tokens.py] file, and update both the [bmfm_modular_tokenizer] and the [bmfm_extended_modular_tokenizer] when new spacial tokens are added.
