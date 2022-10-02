from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from fuse.dl.losses.loss_default import LossDefault
from ehrtransformers.model.metrics_FUSE import FuseMetricAUC
import torch.nn as nn
import torch

from ehrtransformers.utils.common import load_pkl
from ehrtransformers.configs.config import get_vocab_path
from ehrtransformers.data_access.utils import (
    seq_translate,
    DataAdder,
)
from ehrtransformers.model.OutcomeFuse import (
    cleanup_vocab
)
from ehrtransformers.model.model_selector import BertForMultiLabelPredictionHead, BertContrastiveHead, BertConfig, model_type

from ehrtransformers.model.losses import SupConLoss
from ehrtransformers.model.utils import (
    cluster_vocab,
)
import pandas as pd

def derive_cluster_vocab(cluster_fnames, cluster_column_name='0'):
    """

    :param cluster_fnames: a list
    :param cluster_column_name:
    :return:
    """
    if not isinstance(cluster_fnames, list):
        cluster_fnames = [cluster_fnames]
    cluster_inds = set()

    for cluster_fname in cluster_fnames:
        if isinstance(cluster_fname, str):
            if cluster_fname.lower().endswith('.csv'):
                df = pd.read_csv(cluster_fname)
                cluster_inds = cluster_inds.union(set(df[cluster_column_name]))

    clust2idx, idx2clust = cluster_vocab(list(cluster_inds), symbol=None)
    return clust2idx

def get_event_head(head_name:str, vocab_path:dict, conf, source_key: str, loss_weight: float) -> dict:
    """
    returns a definition of an event-prediction-type network head
    :param head_name: name of the head, used in naming metrict, losses and output fields
    :param vocab_path: pathname of a saved vocabulary file (pickled dictionary with at least a 'token2idx' field)
    :param conf: Behrt config
    :param source_key: column name in the input DF which contains for every input a list of strings depicting events
    that the head is to predict.
    :param loss_weight: weight of the loss of this head
    :return: dictionary of head definitions (see HeadConfig for reference)
    """
    event_vocab = load_pkl(vocab_path)
    event_vocab = cleanup_vocab(event_vocab['token2idx'])

    mlb = MultiLabelBinarizer(classes=list(event_vocab.values()))
    mlb.fit([[each] for each in list(event_vocab.values())])

    head_dict = {'name': head_name,
                 'output_type': 'vector',  # 'boolean',
                 'binarizer': mlb,
                 'loss': LossDefault(pred='model.logits.' + head_name, target='data.' + head_name,
                                         callable=nn.MultiLabelSoftMarginLoss(), weight=loss_weight),  # Weights of the
                 # different heads need not sum to 1. If they don't, it will increase
                 # (or decrease) overall learning rate.
                 'metrics': {
                     'AUC.' + head_name: FuseMetricAUC(pred_name='model.logits.' + head_name,
                                                       target_name='data.' + head_name)
                 },
                 'outputs_to_save': 'data.' + head_name,
                 'source_key': source_key,
                 'vocab': event_vocab,
                 'head': BertForMultiLabelPredictionHead(config=conf,
                                                         head_name=head_name,
                                                         feat_inputs=[('model.backbone_features', conf.hidden_size)],
                                                         num_labels=len(set(event_vocab.values())),
                                                         ),
                 }
    return head_dict

class HeadConfig():
    output_format = {}
    """
    output_format is a dictionary defining the task for each head:
        'name' : head name ('event', 'next_vis', 'gender')
        'output_type' : kind of output the head produces (and its corresponding GT) - 'vector', 'scalar' or 'boolean'
        'loss' : Fuse loss class (e.g. FuseDefaultLoss) for the head
        'metrics': a dictionary of metric_name:fuse_metric_class for every metric we wish to calculate for this head
        'outputs_to_save': list of names (strings) of outputs we wish to save for this head,
        'source_key': a column in the input DB (for expected column names see OutcomeFuse.rename_columns_for_behrt)
                    that contains the GT values for this head,
        'vocab': a dictionary to transform general-type GT (e.g. strings) to integers, prior to binarization
        'binarizer' : a binarizer class (e.g. MultiLabelBinarizer) to translate integer GT values to a binary vector        
        """
    def __init__(self, head_names = None, naming_conventions = None, model_config=None, file_config=None,  visit_vocab = None, gender_vocab = None, disease_vocab = None):
        """

        :param head_names:
        :param naming_conventions:
        :param model_config:
        :param event_vocab:
        :param visit_vocab:
        """
        self.output_format = {}
        conf = BertConfig(model_config)
        if 'event' in head_names:
            head_name = 'event'
            head_weight = model_config['head_weights'][model_config['heads'].index(head_name)]
            head_dict = get_event_head(head_name=head_name,
                                       vocab_path=get_vocab_path(root_vocab_path=file_config['gt_vocab'],
                                                            event_type_identifier=naming_conventions['event_key']),
                                       conf=conf,
                                       source_key='label',
                                       loss_weight=head_weight)
            self.output_format[head_name] = head_dict

        if 'treatment_event' in head_names:
            head_name = 'treatment_event'
            head_weight = model_config['head_weights'][model_config['heads'].index(head_name)]
            head_dict = get_event_head(head_name=head_name,
                                       vocab_path=get_vocab_path(root_vocab_path=file_config['gt_vocab'],
                                                                 event_type_identifier=naming_conventions['treatment_event_key']),
                                       conf=conf,
                                       source_key='treatment_event',
                                       loss_weight=head_weight)
            self.output_format[head_name] = head_dict

        if 'next_vis' in head_names:
            assert visit_vocab is not None
            mlb = MultiLabelBinarizer(classes=list(set(visit_vocab.values())))
            mlb.fit([[each] for each in set(visit_vocab.values())])
            head_name='next_vis'
            head_weight = model_config['head_weights'][model_config['heads'].index(head_name)]
            head_dict = {'name': head_name,
                         'output_type': 'vector',  # 'boolean',
                         'binarizer': mlb,
                         'loss': LossDefault(pred='model.logits.'+head_name, target='data.'+head_name, callable=nn.MultiLabelSoftMarginLoss(), weight=head_weight), #(50 for marketscan)
                                        # different heads need not sum to 1. If they don't, it will increase
                                        # (or decrease) overall learning rate.
                         'metrics': {
                                    'AUC.'+head_name: FuseMetricAUC(pred='model.logits.'+head_name, target='data.'+head_name)
                         },
                         'outputs_to_save': 'data.'+head_name,
                         'source_key': 'next_visit',
                         'vocab': visit_vocab,
                         'head': BertForMultiLabelPredictionHead(config=conf,
                                                    head_name=head_name,
                                                    feat_inputs=[('model.backbone_features', conf.hidden_size)],
                                                    num_labels=len(set(visit_vocab.values())),
                                                    ),
                         }
            self.output_format[head_name] = head_dict       

        if 'disease_prediction' in head_names:
            lb_disease_prediction = LabelBinarizer(neg_label=0, pos_label=1)
            lb_disease_prediction.fit([0, 1])
            head_name = 'disease_prediction'
            head_weight = model_config['head_weights'][model_config['heads'].index(head_name)]
            head_dict = {'name': head_name,
                         'output_type': 'boolean',  # 'boolean',
                         'binarizer': lb_disease_prediction,
                         'loss': LossDefault(pred='model.logits.' + head_name, target='data.' + head_name,
                                                 callable=nn.MultiLabelSoftMarginLoss(), weight=head_weight),  # Weights of the
                         # different heads need not sum to 1. If they don't, it will increase
                         # (or decrease) overall learning rate.
                         'metrics': {
                             'AUC.' + head_name: FuseMetricAUC(pred='model.logits.' + head_name,
                                                               target='data.' + head_name)
                         },
                         'outputs_to_save': 'data.' + head_name,
                         'source_key': naming_conventions['outcome_key'],
                         'vocab': None,
                         'head': BertForMultiLabelPredictionHead(config=conf,
                                                                 head_name=head_name,
                                                                 feat_inputs=[
                                                                     ('model.backbone_features', conf.hidden_size)],
                                                                 num_labels=1,
                                                                 ),
                         }
            self.output_format[head_dict['name']] = head_dict

        if 'gender' in head_names:
            lb_gender = LabelBinarizer(neg_label=0, pos_label=1)
            if gender_vocab == None:
                lb_gender.fit([naming_conventions['female_val'], naming_conventions['male_val']])
            else:
                lb_gender.fit(gender_vocab.values())
            head_name = 'gender'
            head_weight = model_config['head_weights'][model_config['heads'].index(head_name)]
            head_dict = {'name': head_name,
                         'output_type': 'boolean',  # 'boolean',
                         'binarizer': lb_gender,
                         'loss': LossDefault(pred='model.logits.'+head_name, target='data.'+head_name,
                                           callable=nn.MultiLabelSoftMarginLoss(), weight=head_weight),  # Weights of the
                                         # different heads need not sum to 1. If they don't, it will increase
                                         # (or decrease) overall learning rate.
                         'metrics': {
                            'AUC.'+head_name: FuseMetricAUC(pred='model.logits.'+head_name, target='data.'+head_name),
                         },
                         'outputs_to_save': 'data.'+head_name,
                         'source_key': 'gender',
                         'vocab': None,
                         'head':BertForMultiLabelPredictionHead(config=conf,
                                                    head_name=head_name,
                                                    feat_inputs=[('model.backbone_features', conf.hidden_size)],
                                                    num_labels=1,
                                                    ),
                         }
            self.output_format[head_dict['name']] = head_dict
        
    def add_external_gt_to_df(self, main_df, set_names):
        """
        reads external GT for all heads (where needed), and adds it to main_df
        :param main_df: df to which we want to add the GT column
        :param set_names: a vector containing any of 'train', 'val', or 'test'. name of the set the df describes. If None - we add merged train/val/test external datasets
        :return:
        """
        for head_name in self.output_format:
            if 'external_data_combiner' in self.output_format[head_name].keys():
                if self.output_format[head_name]['external_data_combiner'] is not None:
                    adder_inst = self.output_format[head_name]['external_data_combiner']
                    main_df = adder_inst.combine(main_df=main_df, set_names=set_names)
        return main_df

    def get_head_names(self, only_with_outputs=True):
        """
        Outputs a list of head names. If only_with_outputs is True, only heads that require GT from the dataset are returned (i.e. self-supervised heads are ignored)
        :param only_with_outputs:
        :return:
        """
        if only_with_outputs:
            head_names = [head_name for head_name in self.output_format if self.output_format[head_name]['source_key'] != None]
        else:
            head_names = [head_name for head_name in self.output_format]
        return head_names

    def get_add_save_outputs(self):
        """
        returns names (fuse) of additional network outputs that need to be saved
        :return:
        """
        outs = []
        for head_name in self.output_format:
            if 'outputs_to_save' in self.output_format[head_name]:
                if self.output_format[head_name]['outputs_to_save'] != None:
                    outs.append(self.output_format[head_name]['outputs_to_save'])
        return outs
        # return [self.output_format[head_name]['outputs_to_save'] for head_name in self.output_format]

    def get_heads(self):
        """
        returns head NN modules
        :return:
        """
        return [self.output_format[head_name]['head'] for head_name in self.output_format]

    def translate_input_value(self, head_name, inp_val):
        trans_input = inp_val
        if trans_input is None:
            trans_input = []
        head_dict = self.output_format[head_name]
        if head_dict['output_type'] == 'vector':
            if ('vocab' in head_dict) and (head_dict['vocab'] is not None):
                    _, trans_input = seq_translate(trans_input, head_dict['vocab'])

            if ('dtype' in head_dict) and (head_dict['dtype'] == 'float'):
                trans_input = torch.FloatTensor(trans_input)
            else:
                trans_input = torch.LongTensor(trans_input)

            if ('binarizer' in head_dict) and (head_dict['binarizer'] is not None):
                trans_input = trans_input.unsqueeze(0)
                trans_input = torch.tensor(head_dict['binarizer'].transform(trans_input.numpy()), dtype=torch.float32)
                trans_input = trans_input.squeeze(0)
        elif head_dict['output_type'] == 'boolean':
            if head_dict['binarizer'] is not None:
                trans_input = head_dict['binarizer'].transform([trans_input])
                trans_input = torch.LongTensor(trans_input).squeeze(0)
        elif head_dict['output_type'] == 'scalar':
            if head_dict['vocab'] is not None:
                    _, trans_input = seq_translate([trans_input], head_dict['vocab'])

            trans_input = torch.LongTensor(trans_input)

            if head_dict['binarizer'] is not None:
                trans_input = trans_input.unsqueeze(0)
                trans_input = torch.tensor(head_dict['binarizer'].transform(trans_input.numpy()), dtype=torch.float32)
                trans_input = trans_input.squeeze(0)
        return trans_input

    def get_head_inputs(self, df):
        """
        Returns dataframe column names for GT values of the heads, if such are needed (e.g. when the heads are not self-supervised)
        :param df:
        :return:
        """
        return {head_name: df[self.output_format[head_name]['source_key']] for head_name in self.output_format if (('source_key' in self.output_format[head_name]) and (self.output_format[head_name]['source_key'] != None))}

    def get_metrics(self):
        metrics = {}
        for head_name in self.output_format:
            if 'metrics' in self.output_format[head_name]:
                if self.output_format[head_name]['metrics'] != None:
                    metrics.update(self.output_format[head_name]['metrics'])
        return metrics

    def get_losses(self):
        losses = {}
        for head_name in self.output_format:
            if 'loss' in self.output_format[head_name]:
                if self.output_format[head_name]['loss'] != None:
                    loss_name = head_name+'_loss'
                    losses[loss_name] = self.output_format[head_name]['loss']
        return losses

