# Data Package
Extremely flexible pipeline allowing data loading, processing, and augmentation suitable for machine learning experiments. Supports caching to avoid redundant calculations and to speed up research iteration times significantly. The data package comes with a rich collection of pre-implemented operations and utilities that facilitates data processing. 

## Terminology

**sample_dict** - Represents a single sample and contains all relevant information about the sample.

No specific structure of this dictionary is required, but a useful pattern is to split it into sections (keys that define a "namespace" ): such as "data", "model",  etc.
NDict (fuse/utils/ndict.py) class is used instead of python standard dictionary in order to allow easy "." separated access. For example:
`sample_dict[“data.input.img”]` is the equivalent of `sample_dict["data"]["input"]["img"]`

Another recommended convention is to include suffix specifying the type of the value ("img", "seg", "bbox")


**sample_id** - a unique identifier of a sample. Each sample in the dataset must have an id that uniquely identifies it.
Examples of sample ids:
* path to the image file
* Tuple of (provider_id, patient_id, image_id)
* Running index

The unique identifier will be stored in sample_dict["data.sample_id"]

## Op(erator)

Operators are the building blocks of the sample processing pipeline. Each operator gets as input the *sample_dict* as created by the previous operators and can either add/delete/modify fields in sample_dict. The operator interface is specified in OpBase class. 
A pipeline is built as a sequence of operators, which do everything - loading a new sample, preprocessing, augmentation, and more.

## Pipeline

A sequence of operators loading, pre-processing, and augmenting a sample. We split the pipeline into two parts - static and dynamic, which allow us to control the part out of the entire pipeline that will be cached. To learn more see *Adding a dynamic part*

## Basic example - a static pipeline

**The original code is in example_static_pipeline() in fuse/data/examples/examples_readme.py**
```python 

static_pipeline = PipelineDefault("static", [
    # decoding sample ID
    (OpKits21SampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 

    # loading data
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),


    # fixed image normalization
    (OpClip(), dict(key="data.input.img", clip=(-500, 500))),
    (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
])
sample_ids= list(range(10))
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,
    dynamic_pipeline=None,
    cacher=None,          
)
my_dataset.create()

```
A basic example, including static pipeline only that loads and pre-processes an image and a corresponding segmentation map. 
A pipeline is created from a list of tuples. Each tuple includes an op and op arguments. The required arguments for an op specified in its \_\_call\_\_() method.
In this example "sample_id" is a running index. OpKits21SampleIDDecode() is a custom op for Kits21 challenge converting the index to image path and segmentation path which are then loaded by OpLoadImage().
In other case than Kits21 you would have to implement your custome MySampleIDDecode() operator.
Finally, OpClip() and OpToRange() pre-process the image.
 

## Caching
**The original code is in example_cache_pipeline() in fuse/data/examples/examples_readme.py**
```python 

static_pipeline = PipelineDefault("static", [
    (OpKits21SampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
    (OpClip(), dict(key="data.input.img", clip=(-500, 500))),
    (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
])


cacher = SamplesCacher(unique_cacher_name,
    static_pipeline,
    cache_dirs=cache_dir) #it can just one path for the cache ot list of paths which will be tried in order, moving the next when available space is exausted.           

sample_ids= list(range(10))
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,
    dynamic_pipeline=None,
    cacher=cacher,          
)
my_dataset.create()

```

To enable caching, a sample cacher should be created and specified as in the example above.
The cached data will be at [cache_dir]/[unique_cacher_name].

## Adding a dynamic part

**The original code is in example_dynamic_pipeline() in fuse/data/examples/examples_readme.py**

```python 

static_pipeline = PipelineDefault("static", [
    (OpKits21SampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
])

dynamic_pipeline = PipelineDefault("dynamic", [
    (OpClip(), dict(key="data.input.img", clip=(-500,500))),
    (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
    (OpToTensor(), dict(key="data.input.img")),
    (OpToTensor(), dict(key="data.gt.seg")),
])


cacher = SamplesCacher(unique_cacher_name, 
    static_pipeline,
    cache_dirs=cache_dir)             

sample_ids=[f"case_{id:05d}" for id in range(num_samples)]
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,
    dynamic_pipeline=dynamic_pipeline,
    cacher=cacher,          
)
my_dataset.create()

```

A basic example that includes both dynamic pipeline and static pipeline. Dynamic pipeline follows the static pipeline and continues to pre-process the sample. In contrast to the static pipeline, the output of the dynamic pipeline is not be cached and allows modifying the pre-precessing steps without recaching, The recommendation is to include pre-processing steps that we intend to experiment with, in the dynamic pipeline.


### Avoiding boilerplate by using "Meta Ops"
**The original code is in example_meta_ops_pipeline() in fuse/data/examples/examples_readme.py**
```python 
repeat_for = [dict(key="data.input.img"), dict(key="data.gt.seg")]
static_pipeline = PipelineDefault("static", [
    (OpKits21SampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
])

dynamic_pipeline = PipelineDefault("dynamic", [
    (OpClip(), dict(key="data.input.img", clip=(-500,500))),
    (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
    (OpRepeat(OpToTensor(), kwargs_per_step_to_add=repeat_for), dict(dtype=torch.float32)),
])

cacher = SamplesCacher(unique_cacher_name,
    static_pipeline,
    cache_dirs=cache_dir)             

sample_ids= sample_ids= list(range(10))
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,
    dynamic_pipeline=dynamic_pipeline,
    cacher=cacher,          
)
my_dataset.create()

```
Meta op is a powerful tool, Meta ops enhance the functionality and flexibility of the pipeline and allows avoiding boilerplate code,
The example above is the simplest. We use OpRepeat to repeat OpToTensor twice, once for the image and once for the segmentation map.


## Adding augmentation
**The original code is in example_adding_augmentation() in fuse/data/examples/examples_readme.py**
```python 

repeat_for = [dict(key="data.input.img"), dict(key="data.gt.seg")]
static_pipeline = PipelineDefault("static", [
    (OpKits21SampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
])

dynamic_pipeline = PipelineDefault("dynamic", [
    (OpClip(), dict(key="data.input.img", clip=(-500,500))),
    (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
    (OpRepeat(OpToTensor(), kwargs_per_step_to_add=repeat_for), dict(dtype=torch.float32)),
    (OpSampleAndRepeat(OpAffineTransform2D(do_image_reverse=True), kwargs_per_step_to_add=repeat_for), dict(
                rotate=Uniform(-180.0,180.0),        
                scale=Uniform(0.8, 1.2),
                flip=(RandBool(0.5), RandBool(0.5)),
                translate=(RandInt(-15, 15), RandInt(-15, 15))
            )),
])

cacher = SamplesCacher(unique_cacher_name,
    static_pipeline,
    cache_dirs=cache_dir)             

sample_ids= list(range(10))
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,
    dynamic_pipeline=dynamic_pipeline,
    cacher=cacher,          
)
my_dataset.create()

```
FuseMedML comes with a collection of pre-implemented augmentation ops. Augmentation ops are expected to be included in the dynamic_pipeline to avoid caching and to be called with different random numbers drawn from the specified distribution. In this example, we've added identical affine transformation for the image and segmentation map. OpSampleAndRepeat() will first draw the random numbers from the random arguments and then repeat OpAffineTransform2D for both the image and segmentation map with the same arguments.  

## Using custom functions directly (OpFunc and OpLambda)
**The original code is in example_custom_function() in fuse/data/examples/examples_readme.py**
```python 

static_pipeline = PipelineDefault("static", [
    (OpKits21SampleIDDecode(), dict()),
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
    (OpRepeat(OpLambda(func=lambda x: np.reshape(x,(x.shape[0], 4, 256, 256))), repeat_for), dict())
])
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,        
)
my_dataset.create()

```
Pre-processing a dataset many times involves heuristics and custom functions. OpLambda and OpFunc allow using those functions directly instead of implementing Op for every custom function. This is a simple example of implementing NumPy array reshape using OpLambda.

## End to end dataset example (image and segmentation map) for segmentation task
**The original code is in example_end2end_dataset() in fuse/data/examples/examples_readme.py**
```python

repeat_for = [dict(key="data.input.img"), dict(key="data.gt.seg")]
static_pipeline = PipelineDefault("static", [
    (OpKits21SampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
    (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
    (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
])

dynamic_pipeline = PipelineDefault("dynamic", [
    (OpClip(), dict(key="data.input.img", clip=(-500,500))),
    (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
    (OpRepeat(OpToTensor(), kwargs_per_step_to_add=repeat_for), dict(dtype=torch.float32)),
    (OpSampleAndRepeat(OpAffineTransform2D(do_image_reverse=True), kwargs_per_step_to_add=repeat_for), dict(
                rotate=Uniform(-180.0,180.0),        
                scale=Uniform(0.8, 1.2),
                flip=(RandBool(0.5), RandBool(0.5)),
                translate=(RandInt(-15, 15), RandInt(-15, 15))
            )),
])

cacher = SamplesCacher(unique_cacher_name,
    static_pipeline,
    cache_dirs=cache_dir)             

sample_ids= list(range(10))
my_dataset = DatasetDefault(sample_ids=sample_ids,
    static_pipeline=static_pipeline,
    dynamic_pipeline=dynamic_pipeline,
    cacher=cacher,          
)
my_dataset.create()

```

## Creating dataloader and balanced dataloader
**The original code is in example_balanced_dataloader() in fuse/data/examples/examples_readme.py**
```python
batch_sampler = BatchSamplerDefault(dataset=dataset,
                                           balanced_class_name='data.label',
                                           num_balanced_classes=num_classes,
                                           batch_size=batch_size,
                                           mode="approx",
                                           balanced_class_weights=[1 / num_classes] * num_classes)

dataloader = DataLoader(dataset=dataset, collate_fn=CollateDefault(), batch_sampler=batch_sampler, shuffle=False, drop_last=False)
```
To create a dataloader, reuse our default generic collate function, and to balance the data, use our sampler.



## Converting classic PyTorch dataset to FuseMedML style
**The original code is in example_classic_to_fusemedml_style() in fuse/data/examples/examples_readme.py**
```python
my_dataset = DatasetWrapSeqToDict(name='my_dataset', dataset=torch_dataset, sample_keys=('data.image', 'data.label'))
my_dataset.create()
```
If you already have a Pytorch dataset at hand that its \_\_getitem\_\_ method outputs a sequence of values, but want to switch to FuseMedML style which its \_\_getitem\_\_ method outputs a flexible dictionary, you can easily wrap it with DatasetWrapSeqToDict as in the example above.

## Op(erators) list

[**Meta operators**](ops/ops_common.py)

Meta operators are a great tool to facilitate the development of sample processing pipelines.
The following operators are useful when implementing a common pipeline:

*	OpRepeat - repeats an op multiple times, each time with different arguments
*   OpLambda - applies simple lambda function / function to transform single value
*   OpFunc - helps to wrap an existing simple python function without writing boilerplate code
*   OpApplyPatterns - selects and applies an operation according to the key name in sample_dict.
*   OpApplyTypes - selects and apply an operation according to value type (inferred from the key name in sample_dict)
*   OpCollectMarker - use this op within the dynamic pipeline to optimize the reading time for components such as sampler 

[**Meta operators for random augmentations**](ops/ops_aug_common.py)

*	OpSample - recursively searches for ParamSamplerBase instances in kwargs, and replaces the drawn values in place
*   OpSampleAndRepeat - first samples and then repeats the operation with the drawn values. Used to apply the same transformation on different values such as image and segmentation map
*   OpRepeatAndSample - repeats the operations, but each time has drawn different values from the defined distributions
*	OpRandApply - randomly applies the op (according to the given probability) 

[**Reading operators**](ops/ops_read.py)

* OpReadDataframe - reads data from pickle file / Dataframe object. Each row will be added as a value to sample_dict

[**Casting operators**](ops/ops_cast.py)

* OpToNumpy - convert many different types to NumPy array
* OpToTensor - convert many different types to PyTorch tensor
* OpOneHotToNumber - convert one-hot encoding vectors into numbers

**Imaging operators**
See [fuseimg package](../../fuseimg/data/README.md)

