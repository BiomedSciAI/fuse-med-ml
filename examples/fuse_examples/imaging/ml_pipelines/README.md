# ML Pipelines
The examples in this folder demonstrates automation of an ML pipeline which consists of:
* Model training on multiple cross validation splits
* Per split evaluation
* Ensembling the models from all the cross validation splits
* Test evaluation using the ensembled model
* Multiple repetitions to check performance variability for multiple random seeds

An ML pipeline is ran by calling the `run` function in `fuse.dl.cross_validation.pipeline`. This function expects a number of required parameters, as well as optional dictionaries `dataset_params`, `train_params`, `infer_params`, `eval_params`  and `paths` that can contain any number of additional custom parameters and path specifications. The `paths` dictionary needs to contain keys for `model_dir` and `cache_dir`, to define location for the trained model, and data caching, respectively.
Another optional parameter is `sample_ids_per_fold`, which may contain a sequence of array pairs denoting sample ids for pre-defined train/validation splits. If this parameter is not set, the splits are decided at random.

The required parameters are as follows:
1. `num_folds` - Number of cross validation splits/folds.
2. `num_folds_used` - Number of folds/splits to use. For example, for training a single model with 80% of the samples used for training and 20% for validation, set `num_folds=5` and `num_folds_used=1`. For running a 5-fold cross-validation scenario, set `num_folds=5` and `num_folds_used=5`.
3. `num_gpus_total` - Number of GPUs to use in total for executing the pipeline.
4. `num_gpus_per_split` - Number of GPUs to use for a single model training/inference.
5. `num_repetitions` - Number of repetitions of the procedu re with different random seeds. Note that this does not change the random decision on cross validation fold sample ids.
6. `dataset_func` - Callable to a custom function that implements a dataset creation. Its input is a path to cache directory and it should return a train and test dataset.
7. `train_func` - Callable to a custom function that executes model training.
8. `infer_func` - Callable to a custom inference function.
9. `eval_func` - Callable to a custom evaluation function.

The `dataset_func` is a function the user needs to implement for their use case. Its signature is required to be as follows:
```
dataset_func(train_val_sample_ids: Union[Sequence, None]=None, paths: Optional[dict]=None, params: Optional[dict]=None) -> Sequence[DatasetDefault]
```
It must return two fuse datasets. In case `train_val_sample_ids` is None, they are the train+validation (development set), and test datasets.
In case `train_val_sample_ids` is a pair of sample ids, they should be train and validation datasets corresponding to this sample id pair.

The three custom functions `train_func`, `infer_func` and `eval_func` have minimal requirements and should have the following signatures:
```
train_func(train_dataset: DatasetDefault, validation_dataset: DatasetDefault, paths: dict, train_params: dict) -> None

infer_func(dataset: DatasetDefault, paths: dict, infer_params: dict) -> None

eval_func(paths: dict, eval_params: dict) -> None
```


Example scripts are provided for the MNIST and STOIC21 datasets.
Note that we use examples that exist in FuseMedML also as stand-alone examples. Using their already existing train, inference and evaluation functions, we can obtain a cross-validation pipeline simply by defining a few parameters and following the requirements of the `dataset_func` and simple generic required signatures of `train_func`, `infer_func` and `eval_func`.

To summarize, in order to run an ML cross validation and ensembling pipeline for a custom project, one needs to:
1. implement a train, infer and eval running functions following the required signatures.
2. implement a dataset function following the required signature and design principle.
2. call `fuse.dl.cross_validation.pipeline.run()` with the required and optional parameters as defined above.
