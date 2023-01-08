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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def parse_tb_output(event_path: str, verbose: int = 0):
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    if verbose > 0:
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

    all_tags = event_acc.Tags()["scalars"]
    AUC_tags = [t for t in all_tags if "metrics.AUC" in t]
    df_events = None
    for t in AUC_tags:
        steps = [e.step for e in event_acc.Scalars(t)]
        vals = [e.value for e in event_acc.Scalars(t)]
        if df_events is None:
            df_events = pd.DataFrame(list(zip(steps, vals)), columns=["step", t])
        else:
            tmp = pd.DataFrame(list(zip(steps, vals)), columns=["step", t])
            df_events = df_events.merge(tmp, how="outer", on="step")

    return df_events


def plot_tb_summary(df_events, out_path):
    plt.clf()
    for col in df_events.columns:
        if col != "step":
            plt.plot(df_events["step"], df_events[col], label=col)
    plt.legend()
    plt.grid(True)
    plt.xlabel("epochs")
    plt.ylabel("AUC")
    plt.savefig(os.path.join(out_path, "AUC_curves.png"))

    a = 1


if __name__ == "__main__":
    for event_path in event_paths:
        df_events = parse_tb_output(event_path=event_path, verbose=1)
        plot_tb_summary(df_events, event_path)
    a = 1
