This scripts in this folder demonstrate automation of an ML pipeline which consists of:
* Model training on multiple cross validation splits
* Per split evaluation
* Ensembling the models from all the cross validation splits
* Test evaluation using the ensembled model
* Multiple repetitions to check performance variability for multiple random seeds

Example scripts are provided for the MNIST and KNIGHT datasets.
The contents of the respective pipeline script (i.e `mnist.py`, `knight.py`) should define a Dict calles `params` with the following keys:
* `common`
* `train`
* `infer`
* `eval`

The `train` key should contain training parameters. One of them is required to be `run_func`, an implementation of the training function.
Similarly, the `infer` and `eval` keys should also contain `run_func`s.  
The `common` key should contain general parameters common to all modes, such as `paths` which denotes different important paths, and `num_gpus` which denotes the number of available GPUs for running the pipeline. Specific GPU ids will be automatically selected based on memory availability.

To summarize, in order to run an ML pipeline for a custom project, one needs to:
* implement a train, infer and eval running functions
* implement a custom script which will assign parameters in the form 
```
{"num_gpus": num_gpus, "paths": paths, "train": train_params, 
    "infer": infer_params, "eval": eval_params}
```  

* from this custom script, call `pipeline.run(params)`
