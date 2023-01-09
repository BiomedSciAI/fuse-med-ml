"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Jan 09, 2023

"""


special_tokens = {"padding": "PAD", "unknown": "UNK", "separator": "SEP", "cls": "CLS"}


def seq_translate(tokens, translate_dict):
    """
    returns a list of tokens translated using translate_dict
    :param tokens:
    :param translate_dict:
    :return:
    """
    return ([translate_dict.get(token, translate_dict[special_tokens["unknown"]]) for token in tokens],)


def position_idx(tokens, symbol=special_tokens["separator"]):
    """
    Given a sequence of codes divided into groups (visits) by symbol ('SEP') tokens, returns a sequence of the same
    size of visit indices.
    :param tokens:
    :param symbol:
    :return:
    """
    group_inds = []
    flag = 0
    for token in tokens:
        group_inds.append(flag)
        if token == symbol:
            flag += 1
    return group_inds


def seq_pad(tokens, max_len, symbol=special_tokens["padding"]):
    """
    Returns a list of tokens padded by symbol to length max_len.
    :param tokens:
    :param max_len:
    :param symbol:
    :return:
    """
    token_len = len(tokens)
    if token_len < max_len:
        return list(tokens) + [symbol] * (max_len - token_len)

    else:
        return tokens[:max_len]
