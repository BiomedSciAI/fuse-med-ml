"""
Taken from https://github.com/LuoweiZhou/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py,
which is, in turn, based on HuggingFace implementation of Bert
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_pretrained_bert as Bert


from typing import Dict
#from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get("vocab_size"),
            hidden_size=config.get("hidden_size"),
            num_hidden_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            intermediate_size=config.get("intermediate_size"),
            hidden_act=config.get("hidden_act"),
            hidden_dropout_prob=config.get("hidden_dropout_prob"),
            attention_probs_dropout_prob=config.get("attention_probs_dropout_prob"),
            max_position_embeddings=config.get("max_position_embedding"),
            initializer_range=config.get("initializer_range"),
        )

class BertEmbeddings(nn.Module):
    # Same as pytorch_pretrained_bert.BertEmbeddings, only without token_type_embeddings to save some memory
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            position_ids = torch.zeros_like(input_ids)

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids=input_ids,position_ids=position_ids) #embedding vectors of Bert
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoded_layers[-1] # this is the embedding of all tokens
        pooled_output = self.pooler(sequence_output) #"'pooling' is simply taking the embedding of the first token
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertBackbone(Bert.modeling.BertPreTrainedModel):
    # This is a class that wraps BertModel,outputting the embedding vector for Fuse.
    def __init__(self, config, feature_dict):
        super(BertBackbone, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(
            self,
            input_ids,
            position_ids=None,
            attention_mask=None,
    ):
        _, pooled_output = self.bert(
            input_ids,
            position_ids,
            attention_mask,
            output_all_encoded_layers=False,
        )
        return pooled_output

class BertForMultiLabelPredictionHead(nn.Module):
    def __init__(self, config, num_labels, feat_inputs, head_name='classifier'):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)  # This is the output 'head'
        self.feat_inputs = feat_inputs
        self.head_name=head_name


    def forward(self, batch_dict: Dict) -> Dict:
        feat_input = torch.cat(
            [batch_dict[feat_input[0]] for feat_input in self.feat_inputs])
        feat_input = self.dropout(feat_input)
        logits = self.classifier(feat_input)

        cls_preds = F.softmax(logits, dim=1) #Since we're dealing with multilabel prediction, no softmax is needed

        batch_dict['model.logits.' + self.head_name] = logits
        batch_dict['model.output.' + self.head_name] = cls_preds

        return batch_dict

class BertContrastiveHead(nn.Module):
    def __init__(self, config, feat_inputs, head_name='contrastive'):
        super().__init__()

        self.feat_inputs = feat_inputs
        self.head_name=head_name


    def forward(self, batch_dict: Dict) -> Dict:
        feat_input = torch.cat(
            [batch_dict[feat_input[0]] for feat_input in self.feat_inputs])

        batch_dict['model.logits.' + self.head_name] = feat_input
        batch_dict['model.output.' + self.head_name] = feat_input
        # batch_dict['model.output.' + self.head_name] = cls_preds

        return batch_dict
