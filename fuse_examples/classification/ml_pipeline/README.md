This scripts in this folder demonstrate automation of an ML pipeline which consists of:
* Model training on multiple cross validation splits
* Per split evaluation
* Ensembling the models from all the cross validation splits
* Test evaluation using the ensembled model
* Multiple repetitions to check performance variability for multiple random seeds

Example scripts are provided for the MNIST and KNIGHT datasets.
The contents of the respective pipeline script (i.e `mnist.py`, `knight.py`) should define a Dict calles `params` with the following keys:
* `num_gpus` 
* `paths`
* `train`
* `infer`
* `eval`

The `train` key should contain training parameters. One of them is required to be `run_func`, an implementation of the training function.
Similarly, the `infer` and `eval` keys should also contain `run_func`s. 