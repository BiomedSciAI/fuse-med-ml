from sys import argv
import os
import copy
import pandas as pd
import socket
import torch
import torch.optim as optim
import pytorch_lightning as pl
from yaml import safe_load

from ehrtransformers.utils.tb_parser import parse_tb_output, plot_tb_summary
from ehrtransformers.utils.common import load_pkl
from ehrtransformers.configs.config import get_config #, get_config_path
from ehrtransformers.utils.log_utils import Logger, get_output_dir, get_run_index_from_model_location
from ehrtransformers.model.utils import age_vocab
from ehrtransformers.model.OutcomeFuse import (
    load_outcome_data,
    cleanup_vocab
)

from ehrtransformers.model.model_selector import BertBackbone, BertConfig, model_type
from ehrtransformers.data_access.utils import reduce_vocab
from ehrtransformers.data_access.icd_to_ccs import read_icd_ccs_dict
from ehrtransformers.utils.save_results import save_translated_outputs
from ehrtransformers.configs.head_config import HeadConfig

from fuse.utils.gpu import choose_and_enable_multiple_gpus
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from fuse.dl.models.model_multihead import ModelMultiHead
from fuse.utils.file_io.file_io import create_dir, save_dataframe

import pytorch_pretrained_bert as Bert


def get_paths_and_train_params_fuse(global_params, model_config, optim_config, head_config, device='cuda'):

    ROOT = global_params['output_dir']
    PATHS = {'model_dir': os.path.join(ROOT, 'model_dir'),
             'force_reset_model_dir': False,
             # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
             'cache_dir': os.path.join(ROOT, 'cache_dir'),
             'inference_dir': os.path.join(ROOT, 'infer_dir'),
             'analyze_dir': os.path.join(ROOT, 'analyze_dir')}

    TRAIN_COMMON_PARAMS = {}

    ### Data ###
    TRAIN_COMMON_PARAMS['data.batch_size'] = global_params['batch_size']
    TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
    TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8

    ### PL Trainer ###
    TRAIN_COMMON_PARAMS['trainer.num_epochs'] = model_config['train_epochs']
    TRAIN_COMMON_PARAMS['trainer.num_devices'] = 1
    TRAIN_COMMON_PARAMS['trainer.accelerator'] = 'gpu' if device.startswith('cuda') else 'cpu'
    TRAIN_COMMON_PARAMS['trainer.ckpt_path'] = None  # if not None, will try to load the checkpoint

    heads = head_config.get_head_names()
    if 'event' in heads:
        best_source = 'validation.metrics.AUC.event'
    elif 'next_vis' in heads:
        best_source = 'validation.metrics.AUC.next_vis'
    elif 'disease_prediction' in heads:
        best_source = 'validation.metrics.AUC.disease_prediction'
    else:
        raise Exception('Need to define the metric for choosing best epoch, as neither event nor netx_vis are in heads')
    TRAIN_COMMON_PARAMS['best_epoch_source'] = {
        'source': best_source,  # can be any key from 'epoch_results'
    }

    ### Optimizer ###
    TRAIN_COMMON_PARAMS["opt.lr"] = optim_config['lr']
    TRAIN_COMMON_PARAMS["opt.weight_decay"] = optim_config['weight_decay']
    return PATHS, TRAIN_COMMON_PARAMS

def get_train_manager_multihead(paths, model, train_params, optim_config, head_config):

    losses = head_config.get_losses()

    train_metrics = head_config.get_metrics()
    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # create optimizer

    # optimizer = optimiser.adam(params=list(model.named_parameters()), config=optim_config)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    model_params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]

    optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                       lr=optim_config['lr'],
                                       warmup=optim_config['warmup_proportion'])

    # create learning scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    best_epoch_source = dict(monitor=train_params['best_epoch_source']['source'], mode="max")

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=paths["model_dir"],
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        devices=train_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    return pl_trainer, pl_module

def load_and_filter_df(input_file, target_path):
    a=1


def train_fuse(config_file: str):
    argv = ['train_fuse', config_file]
    if len(argv) > 1:
        config_file_path = argv[1]  
    else:
        config_file_path = None

    (
        global_params,
        file_config,
        model_config,
        optim_config,
        data_config,
        naming_conventions,
    ) = get_config(config_file_path)

    do_train = True #True #False
    do_infer = True #True #False
    do_translate = True #True #False
    
    run_index = 0 #104 #Run index to use for inference and translation. Only relevant if do_train = False. Otherwise the current index is used
    model_init_weights_location = model_config['pretrain_model'] #None #'/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_5_days/EVENT/models/PD/outcome_admdate_visit/run_102/Logs/model_dir/checkpoint_last_epoch.pth'
    if (model_init_weights_location==None) or ('None' in model_init_weights_location):
        model_init_weights_location = None
    # '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_with_procedures/EVENT/models/PD/outcome_admdate_visit/run_113/Logs/model_dir/checkpoint_last_epoch.pth' #if None: train from scratch. If integer: get the path for that run index and continue from it. If string: this is the path to the model checkpoint
    load_checkpoint_mode = 'train' #'infer' #'train' loads weights, epoch and learning rate. 'infer' only loads weights.
    if (not do_train) and (model_init_weights_location != None):
        run_index = get_run_index_from_model_location(model_init_weights_location)

    # create a logger instance:
    # This is important for two reasons:
    # 1. It creates and sets up a log directory where all run information (logs, config files and code) is saved
    # 2. It sets up global_config['output_dir'] which is where model checkpoints and inference files are saved
    if do_train:
        logger_inst = Logger(file_config, model_config, data_config, global_params, argv, override_output=not global_params['debug_mode'])
    else:
        logger_inst = Logger(file_config, model_config, data_config, global_params, argv, index=run_index, override_output=False)

    a=0
    if model_init_weights_location is not None:
        if isinstance(model_init_weights_location, int):
            model_init_weights_location = get_output_dir(global_params, model_init_weights_location)


    feature_dict = {"age": True, "seg": True, "posi": True}

    output_dir_main = global_params["output_dir_main"]
    stop_file = global_params["stop_file"]
    device = global_params["device"]
    gradient_accumulation_steps = global_params["gradient_accumulation_steps"]

    # get loaders
    if True:
        BertVocab = load_pkl(file_config["vocab"])
        ageVocab, _ = age_vocab(
            max_age=global_params["max_age"],
            mon=global_params["month"],
            symbol=global_params["age_symbol"],
        )


        model_config["age_vocab_size"] = len(ageVocab.keys())
        model_config["vocab_size"] = len(BertVocab["token2idx"].keys())
        model_config["seg_vocab_size"] = 2

        token2idx = BertVocab["token2idx"]




        #Create next visit prediction vocabulary
        if data_config['reduce_pred_visit_vocab_mapping_path'] is None:
            #we use unreduced visit vocabulary (i.e. diag codes are not grouped, resulting in ~20k possible codes)
            next_vis2idx=token2idx
        else:
            #we reduce next visit vocabulary by grouping diag codes according to a mapping defined in data_config['reduce_pred_visit_vocab_mapping_path']
            next_vis2idx = read_icd_ccs_dict(json_filepath=data_config['reduce_pred_visit_vocab_mapping_path'])
            next_vis2idx = {k: int(next_vis2idx[k]) for k in next_vis2idx}
            Vocab_next_vis = cleanup_vocab(token2idx)
            next_vis2idx = reduce_vocab(token2idx=Vocab_next_vis, token_map=next_vis2idx)



        head_config_inst = HeadConfig(head_names=model_config['heads'],
                                      naming_conventions=naming_conventions,
                                      model_config=model_config,
                                      file_config=file_config,
                                      visit_vocab=next_vis2idx)

        if do_train:
            train_df, train_loader = load_outcome_data(
                file_name=file_config["train"],
                dset_name="train",
                naming_conventions=naming_conventions,
                head_config=head_config_inst,
                age_month_resolution=global_params["month"],
                ageVocab=ageVocab,
                token2idx=token2idx,
                max_sequence_len=global_params["max_len_seq"],
                batch_size=global_params["batch_size"],
                shuffle=True,
                reverse_inputs=model_config['reverse_input_direction'],
                num_workers=data_config['num_loader_workers'],
                model_config=model_config,
                do_contrastive_sampling=model_config['contrastive_instances']>1,
            )

        if do_train or do_infer:
            val_df, val_loader = load_outcome_data(
                file_name=file_config["val"],
                dset_name="val",
                naming_conventions=naming_conventions,
                head_config=head_config_inst,
                age_month_resolution=global_params["month"],
                ageVocab=ageVocab,
                token2idx=token2idx,
                max_sequence_len=global_params["max_len_seq"],
                batch_size=global_params["batch_size"],
                shuffle=False,
                reverse_inputs=model_config['reverse_input_direction'],
                num_workers=data_config['num_loader_workers'],
                model_config=model_config,
                do_contrastive_sampling=False,
            )

            traineval_df, traineval_loader = load_outcome_data(
                file_name=file_config["train"],
                dset_name="traineval",
                naming_conventions=naming_conventions,
                head_config=head_config_inst,
                age_month_resolution=global_params["month"],
                ageVocab= ageVocab,
                token2idx= token2idx,
                max_sequence_len=global_params["max_len_seq"],
                batch_size=global_params["batch_size"],
                shuffle=False,
                reverse_inputs=model_config['reverse_input_direction'],
                num_workers=data_config['num_loader_workers'],
                model_config=model_config,
                do_contrastive_sampling=False,
            )

    # create model
    if True:
        if 'cuda' in device:
            choose_and_enable_multiple_gpus(1, force_gpus=None)
        conf = BertConfig(model_config)
        paths, train_params = get_paths_and_train_params_fuse(global_params=global_params,model_config=model_config, optim_config=optim_config, device=device, head_config=head_config_inst)

        model = ModelMultiHead(
            conv_inputs=(('data.code',1), ('data.position',1)), #(('data.input.input_0.tensor', 1),), #data.code, data.age, data.seg, data.position (defined in feature_dict)
            backbone=BertBackbone(config=conf, feature_dict=feature_dict),
            heads=head_config_inst.get_heads(),
        )

        DEVICE = global_params["device"]
        DATAPARALLEL = global_params["is_data_parallel"]
        model = model.to(global_params["device"])

        if DATAPARALLEL:
            model = torch.nn.DataParallel(model)

    if model_init_weights_location is not None:
        if model_init_weights_location[-5:] != '.ckpt':
            model_init_weights_location = model_init_weights_location + 'last.ckpt'

    if do_train:
        pl_trainer, pl_module = get_train_manager_multihead(paths, model, train_params, optim_config, head_config=head_config_inst)

        if model_init_weights_location is not None:
            map_location = 'cpu'
            LightningModuleDefault.load_from_checkpoint(model_init_weights_location,
                    model_dir=paths["model_dir"], model=model, map_location=map_location, strict=False)

        # Start training
        pl_trainer.fit(pl_module, train_loader, val_loader, ckpt_path=train_params["trainer.ckpt_path"])

        model_init_weights_location = paths['model_dir']

    if do_infer:
        ######################## Inference ###################################
        create_dir(paths["inference_dir"])
        if model_init_weights_location.endswith('.ckpt'):
            model_init_weights_location = os.path.dirname(model_init_weights_location)

        INFER_COMMON_PARAMS = {}
        INFER_COMMON_PARAMS['infer_filename'] = 'validation_set_infer_last.gz'
        INFER_COMMON_PARAMS['checkpoint'] = 'last.ckpt'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
        INFER_COMMON_PARAMS['model'] = model
        INFER_COMMON_PARAMS['trainer.num_devices'] = 1
        INFER_COMMON_PARAMS['trainer.accelerator'] = 'gpu' if device.startswith('cuda') else 'cpu'
        infer_common_params = INFER_COMMON_PARAMS

        # load python lightning module
        pl_module = LightningModuleDefault.load_from_checkpoint(
            os.path.join(model_init_weights_location, infer_common_params["checkpoint"]),
            model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
        )
        # set the prediction keys to extract (the ones used by the evaluation function).
        pl_module.set_predictions_keys(['model.backbone_features', 'data.patid', 'data.vis_date'] + head_config_inst.get_add_save_outputs())  # which keys to extract and dump into file

        # create a trainer instance
        pl_trainer = pl.Trainer(
            default_root_dir=paths["model_dir"],
            accelerator=infer_common_params["trainer.accelerator"],
            devices=infer_common_params["trainer.num_devices"],
            auto_select_gpus=True,
        )

        # predict
        predictions = pl_trainer.predict(pl_module, val_loader, return_predictions=True)

        # convert list of batch outputs into a dataframe
        infer_df = convert_predictions_to_dataframe(predictions)
        save_dataframe(infer_df, os.path.join(paths["inference_dir"], infer_common_params["infer_filename"]))

        ######################## Inference on train set ###################################
        INFER_COMMON_PARAMS['infer_filename'] = 'train_set_infer_last.gz'
        predictions = pl_trainer.predict(pl_module, traineval_loader, return_predictions=True)

        # convert list of batch outputs into a dataframe
        infer_df = convert_predictions_to_dataframe(predictions)
        save_dataframe(infer_df, os.path.join(paths['inference_dir'], infer_common_params['infer_filename']))

    if do_translate:
        fnames = ['validation_set_infer_best', 'validation_set_infer_last']
        try:
            if do_train:
                save_translated_outputs(global_params=global_params, file_config=file_config, fnames=fnames, naming_conventions=naming_conventions, index=logger_inst.get_out_dir_index())
            else:
                save_translated_outputs(global_params=global_params, file_config=file_config, fnames=fnames, naming_conventions=naming_conventions, index=run_index)
        except:
            print("couldn't translate val inference file")

        fnames = ['train_set_infer_best', 'train_set_infer_last']
        try:
            if do_train:
                save_translated_outputs(global_params=global_params, file_config=file_config, fnames=fnames,
                                        naming_conventions=naming_conventions,
                                        index=logger_inst.get_out_dir_index())
            else:
                save_translated_outputs(global_params=global_params, file_config=file_config, fnames=fnames,
                                        naming_conventions=naming_conventions,
                                        index=run_index)
        except:
            print("couldn't translate train inference file")    

    event_path = os.path.join(global_params['output_dir'], 'model_dir', 'lightning_logs/version_0')
    df_events = parse_tb_output(event_path=event_path, verbose=1)
    plot_tb_summary(df_events, event_path)  


if __name__ == "__main__":
    do_train = True 
    
    if do_train:
        train_fuse(os.path.dirname(os.path.abspath(__file__))+'/multi_config_CKD.yaml')
        
      