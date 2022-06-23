# User Guide
To ensure maximum flexibility, FuseMedML defines a set of decoupled abstract objects. 

The decoupling is achieved by the fact that, in most cases, the objects do not interact directly. Instead, the information and data are routed between components using *namespaces* (examples below). Meaning, each object extracts its input from and saves its output into a dictionary named `batch_dict`. `batch_dict` aggregates the outputs of all the objects through a single batch. When a batch is completed, only the required key-value pairs from `batch_dict`, such as the loss values, will be collected in another dictionary named `epoch_results`.
 

Both `batch_dict` and `epoch_results` are nested dictionaries. To easily access the data stored in those dictionaries, use `NDict`:
```python
batch_dict[‘model.output.classification’]
``` 
will return `batch_dict[‘model’][‘output’][‘classification’]`

**Example of the decoupling approach:**
```python
FusMetricAUCROC(pred='model.output.classifier_head', target='data.gt.gt_global.tensor')  
```

`FetricAUCROC` will read the required tensors to compute AUC from `batch_dict`. The relevant dictionary keys are `pred_name` and `target_name`. This approach allows writing a generic metric which is completely independent of the model and data extractor. In addition, it allows to easily re-use this object in a plug & play manner without adding extra code. Such an approach also allows us to use it several times in case we have multiple heads/tasks.  

FuseMedML includes pre-implemented versions of the abstract classes which can be used in (almost) any context. Nevertheless, if needed, they can be replaced by the user without affecting other components.

Below is a list of the main abstract classes and their purpose:

## Data
| Module               | Purpose                                                                                                                                                                                                                      | Implementation Examples                                                                                                                                                                                                                                               
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| `DataSourceBase`   | A simple object that generates a cohort of samples unique identifiers (sample descriptors). This class is usually project-specific and expected to be implemented by the user. However, simple generic implementations are included in FuseMedML. | `DataSourceDefault` reads a table in a DataFrame format, including two columns: (1) sample descriptors (2) fold. It outputs a list of sample descriptors for the required folds. An example of a sample descriptor would be a path to a DICOM file that uniquely identifies a sample.
| `ProcessorBase`    | The processor extract and pre-process a single sample or part of a sample given a sample descriptor. A processor is usually project-specific and commonly will be implemented per project. However, common implementations are provided such as a processor that reads DICOM file for MRI | Given a path to DICOM file on disk (sample descriptor) - load image data, resize, normalize pixel values, crop, convert to PyTorch Tensor.
| `CacheBase`        | Stores pre-processed sample for quick retrieval.  | Disk cache or in-memory cache options are built-in in `DatasetDefault` and `DatasetGenerator`
| `AugmentorBase`    | Runs a pipeline of random augmentations| An object that able to apply 2D / 3D affine augmentations, color perturbations, etc. See `AugmentorDefault`.
| `DatasetBase`      | Implementation of PyTorch dataset, including additional utilities. Unlike PyTorch dataset, FuseMedML Dataset returns a dictionary naming each element in the dataset. For example, 'image' and 'label'. However, Pytorch datasets can be easily used by wrapping them with `DatasetWrapper`| `DatasetDefault` is a generic dataset implementation that supports caching, augmentation, data_source, processor, etc.
| `VisualizerBase`   | Debug tool, visualizes network input before/after augmentations| `VisualizerDefault` is a 2D image visualizer                                                                                                                                                                                                                                                    

## Model
FuseMedML includes three types of model objects. 
* Model - an object that includes the entire model end to end. FuseMedML Model is PyTorch model that gets as input `batch_dict`, adds the model outputs to a dictionary `batch_dict[‘model’]` and returns `batch_dict[‘model’]`.  PyTorch model can be easily converted to FuseMedML style model using a wrapper `ModelWrapSeqToDict`.
* Backbone - an object that extracts spatial features from an image. Backbone is a PyTorch model which gets as input tensor/ sequence of tensors and returns tensor/sequence ModelWrapSeqToDictof tensors. 
* Head - an object that maps features to prediction and usually includes pooling layers and dense / conv 1x1 layers. Head gets as an input `batch_dict` and returns `batch_dict`.

All those types inherit directly from `torch.nn.Module`. 

## Losses
| Module               | Purpose                                     | Implementation Examples 
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| `LossBase`         | compute the entire or part of the loss      | `LossDefault` apply a given callable such as `torch.nn.functional.cross_entropy` to compute the loss 

## Metrics
| Module               | Purpose                                                                                                                                                                                                                      | Implementation Examples                                                                                                                                                                                                                                               
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| `MetricBase`         | compute metric value / values. Collect the relevant data to computing the metric from`batch_dict`, and at the end of an epoch compute the metric. Can return a single value or a dictionary that contains several values. | `NetricAUCROC` is a subclass that computes the AUC. 

## Manager
The manager `ManagerDefault` responsibility is to use all the components provided by the user and to train a model accordingly. 

To keep the flexibility, the manager supports callbacks that can affect its state and dictionaries:  `batch_dict` and `epoch_results` dynamically. See `Callback`. An example of a pre-implemented callback is `TensorboardCallback` which is responsible for writing the data of both training and validation to tensorborad loggers under model_dir.

It’s also possible to modify the manager behavior by overriding functions such as `handle_batch()` or alternatively implement a new manager.    

The manager provides also a function called `infer` that restore from `model_dir` (manger train procedure stores the information in this directory) the required objects and runs inference on the required sample descriptors.

## Analyzer
`AnalyzerDefault` responsibility is to evaluate a trained model.
Analyzer gets an inference file, generated by `manager.infer()`. The inference file is expected to include sample descriptors and their predictions. The inference file might also include the ground truth targets and metadata about each of the samples. If not, a processor or a dataset should be provided to extract the target given a sample descriptor.
