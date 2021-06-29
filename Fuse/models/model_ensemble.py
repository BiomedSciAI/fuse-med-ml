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

Created on June 30, 2021

"""

import copy
import os

import torch
from typing import Sequence, Dict, List


class FuseModelEnsemble(torch.nn.Module):
    """
    Ensemble Module - runs several sub-modules sequentially.
    In addition to producing a dictionary with predictions of each model in the ensemble,
    the class produces average over the predictions, and majority vote.

    """

    def __init__(self, input_model_dirs: Sequence[str]) -> None:
        super().__init__()
        self.nets = []
        self.input_model_dirs = input_model_dirs

        # Load multiple modules
        for input_model_dir in input_model_dirs:
            net_path = os.path.join(input_model_dir, 'net.pth')
            net = torch.load(net_path, map_location='cpu')
            self.nets.append(net)

        # Register modules
        self.nets = torch.nn.ModuleList(self.nets)

    def eval(self) -> None:
        for net in self.nets:
            net.eval()

    def load_state_dict(self, *net_state_dicts: List[Dict], strict: bool) -> None:

        for idx, net_state_dict in enumerate(net_state_dicts):
            self.nets[idx].load_state_dict(net_state_dict, strict=strict)

    def calculate_ensemble_avg(self, ensemble_pred: List[Dict]) -> Dict:
        ensemble_avg_dict = {}
        for key in ensemble_pred[0].keys():
            if 'loss' in key:
                continue
            accumulate_prediction = []
            for pred in ensemble_pred:
                accumulate_prediction.append(pred[key])
            accumulate_prediction = torch.stack(accumulate_prediction)
            ensemble_avg_dict[key + '_ensemble_average'] = torch.mean(accumulate_prediction, dim=0)
        return ensemble_avg_dict

    def calculate_ensemble_majority_vote(self, ensemble_pred: List[Dict]) -> Dict:
        ensemble_majority_dict = {}
        for key in ensemble_pred[0].keys():
            if 'loss' in key:
                continue
            accumulate_prediction = []
            for pred in ensemble_pred:
                accumulate_prediction.append(torch.argmax(pred[key], dim=-1))
            accumulate_prediction = torch.stack(accumulate_prediction)
            unique_val, unique_count = torch.unique(accumulate_prediction, dim=0, return_counts=True)
            ensemble_majority_dict[key + '_ensemble_majority_vote'] = unique_val[torch.argmax(unique_count)]
        return ensemble_majority_dict

    def forward(self, batch_dict: Dict) -> Dict:
        ensemble_model_dict = {'output': {}}
        ensemble_pred = []
        for ensemble_idx, ensemble_net in enumerate(self.nets):
            ensemble_model_dict['output']['ensemble_output_' + str(ensemble_idx)] = copy.deepcopy(ensemble_net(batch_dict)['output'])
            ensemble_pred.append(ensemble_model_dict['output']['ensemble_output_' + str(ensemble_idx)])
        ensemble_model_dict['output'].update(self.calculate_ensemble_avg(ensemble_pred))
        ensemble_model_dict['output'].update(self.calculate_ensemble_majority_vote(ensemble_pred))

        return ensemble_model_dict
