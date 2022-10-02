from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# event_path = '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_10_days/EVENT/models/PD/outcome_admdate_visit/run_103/Logs/model_dir/validation/' #events.out.tfevents.1651395755.lsf-gpu4.26342.1'
event_paths = [ '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_5_days/EVENT/models/PD/outcome_admdate_visit/run_139/Logs/model_dir/validation/', # pretrained model path, None if not to continue from pretrained
  # '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_5_days/EVENT/models/PD/outcome_admdate_visit/run_136/Logs/model_dir/validation/', # pretrained model path, None if not to continue from pretrained
  # '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_10_days/EVENT/models/PD/outcome_admdate_visit/run_102/Logs/model_dir/validation/', # pretrained model path, None if not to continue from pretrained
  # '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_10_days_with_procedures/EVENT/models/PD/outcome_admdate_visit/run_101/Logs/model_dir/validation/',# pretrained model path, None if not to continue from pretrained
  # '/data/usr/vadim/EHR/PD_ALL_SERV_270_after_ind_90_to_event_tr_90_to_event_merge_5_days_with_procedures/EVENT/models/PD/outcome_admdate_visit/run_101/Logs/model_dir/validation/',
  # '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_event_next_vis/run_103/Logs/model_dir/validation/',
  # '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_event_next_vis/run_116/Logs/model_dir/validation/',
  # '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_event_next_vis/run_133/Logs/model_dir/validation/',
  # '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_event_next_vis/run_112/Logs/model_dir/validation/',
  # '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_event_next_vis/run_103/Logs/model_dir/validation/',
  # '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_event_next_vis/run_103/Logs/model_dir/validation/',
  '/gpfs/haifa/projects/s/serqet_pd/vadim/PD/multihead_5_days/with_procedures/run_101/Logs/model_dir/validation/'
]

def parse_tb_output(event_path:str, verbose:int = 0):
  event_acc = EventAccumulator(event_path)
  event_acc.Reload()

  if verbose>0:
    print(event_acc.Tags())
  """
  'losses.event_loss', 
  'losses.treatment_event_loss', 
  'losses.next_vis_loss', 
  'losses.total_loss', 
  'metrics.AUC.event', 
  'metrics.AUC.treatment_event', 
  'metrics.AUC.next_vis', 
  'metrics.PREC.event', 
  'metrics.PREC.treatment_event', 
  'metrics.PREC.next_vis', 
  'learning_rate'
  """

  all_tags = event_acc.Tags()['scalars']
  AUC_tags = [t for t in all_tags if 'metrics.AUC' in t]
  df_events = None
  for t in AUC_tags:
    steps = [e.step for e in event_acc.Scalars(t)]
    vals = [e.value for e in event_acc.Scalars(t)]
    if df_events is None:
      df_events = pd.DataFrame(list(zip(steps, vals)), columns=['step', t])
    else:
      tmp = pd.DataFrame(list(zip(steps, vals)), columns=['step', t])
      df_events = df_events.merge(tmp, how='outer', on='step')

  return df_events

def plot_tb_summary(df_events, out_path):
  plt.clf()
  for col in df_events.columns:
    if col != 'step':
      plt.plot(df_events['step'], df_events[col], label=col)
  plt.legend()
  plt.grid(True)
  plt.xlabel("epochs")
  plt.ylabel("AUC")
  plt.savefig(os.path.join(out_path, 'AUC_curves.png'))

  a=1

if __name__ == '__main__':
  for event_path in event_paths:
    df_events = parse_tb_output(event_path=event_path, verbose=1)
    plot_tb_summary(df_events, event_path)
  a=1




