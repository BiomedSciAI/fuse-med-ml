# ML Pipelines
The examples in this folder demonstrate automation of an ML pipeline which consists of:
* Model training on multiple cross validation splits
* Per split evaluation
* Ensembling the models from all the cross validation splits
* Test evaluation using the ensembled model
* Multiple repetitions to check performance variability for multiple random seeds

An ML pipeline is ran by calling the `run` function in `fuse.managers.pipeline`. This function expects a number of required parameters, as well as optional dictionaries `dataset_params`, `train_params`, `infer_params` and `eval_params` that can contain any number of additional custom parameters.  
Another optional parameter is `sample_ids`, which may contain a sequence of array pairs denoting sample ids for pre-defined train/validation splits. If this parameter is kept at `None`, the splits are decided at random.

The required parameters are as follows:  
1. `num_folds` - Number of cross validation splits/folds.
2. `num_gpus_total` - Number of GPUs to use in total for executing the pipeline.  
3. `num_gpus_per_split` - Number of GPUs to use for a single model training/inference.
4. `num_repetitions` - Number of repetitions of the procedure with different random seeds. Note that this does not change the random decision on cross validation fold sample ids.
5. `dataset_func` - Callable to a custom function that implements a dataset creation. Its input is a path to cache directory and it should return a train and test dataset. 
6. `train_func` - Callable to a custom function that executes model training. 
7. `infer_func` - Same for inference.
8. `eval_func` - Same for evaluation

The three custom functions `train_func`, `infer_func` and `eval_func` should receive the following input parameters:
1. `dataset` - PyTorch or FuseMedML wrapped dataset
2. `sample_ids` - A pair of arrays denoting train and validation sample ids.
3. `cv_index` - Integer or string denoting the current split/fold.
4. `test` - Boolean denoting whether we're in test phase
5. `params` - Custom dictionary that can contain any number of additional parameters.
6. `rep_index` - Integer denoting the repetition number
7. `rand_gen` - Fuse random generator.

Example scripts are provided for the MNIST and KNIGHT datasets.

To summarize, in order to run an ML pipeline for a custom project, one needs to:
1. implement a train, infer and eval running functions
2. call `pipeline.run()` with the required and optional parameters as defined above.
