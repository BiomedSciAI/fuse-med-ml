Welcome to Fuse-Med-ML
===========

.. image:: https://badges.frapsoft.com/os/v1/open-source.svg
   :target: https://opensource.org/

.. image:: https://badge.fury.io/py/fuse-med-ml.svg
   :target: https://badge.fury.io/py/fuse-med-ml

.. image:: https://img.shields.io/pypi/pyversions/fuse-med-ml
   :target: https://pypi.org/project/fuse-med-ml/

.. image:: https://img.shields.io/badge/support-slack-slack.svg?logo=slack
   :target: https://join.slack.com/t/fusemedml/shared_invite/zt-xr1jaj29-h7IMsSc0Lq4qpVNxW97Phw

.. image:: https://pepy.tech/badge/fuse-med-ml
   :target: https://pepy.tech/project/fuse-med-ml

.. image:: https://joss.theoj.org/papers/10.21105/joss.04943/status.svg
   :target: https://doi.org/10.21105/joss.04943

.. image:: fuse/doc/FuseMedML-logo.png
   :alt: drawing
   :width: 30%

Effective Code Reuse across ML projects!
=========================================

A python framework accelerating ML based discovery in the medical field by encouraging code reuse. Batteries included :)

FuseMedML is part of the `PyTorch Ecosystem <https://pytorch.org/ecosystem/>`_.

Jump to:
---------

* install instructions `section`_ (#installation)
* complete code `examples`_ (#examples)
* `community support`_ - join the discussion
* Contributing to FuseMedML `guide`_ (./CONTRIBUTING.md)
* `citation`_

Motivation - "*Oh, the pain!*"
--------------------------------

Analyzing **many** ML research projects we discovered that
- Projects bring up is taking **far too long**, even when very similar projects were already done in the past by the same lab!
- Porting individual components across projects was *painful* - resulting in **"reinventing the wheel" time after time**

How the magic happens
----------------------

1. A simple yet super effective design concept
   - Data is kept in a nested (hierarchical) dictionary
   - This is a key aspect in FuseMedML (shortly named as "fuse"). It's a key driver of flexibility, and allows to easily deal with multi modality information.

     ```python
     from fuse.utils import NDict

     sample_ndict = NDict()
     sample_ndict['input.mri'] = # ...
     sample_ndict['input.ct_view_a'] = # ...
     sample_ndict['input.ct_view_b'] = # ...
     sample_ndict['groundtruth.disease_level_label'] = # ...
     ```

   - This data can be a single sample, it can be for a minibatch, for an entire epoch, or anything that is desired.
   - The "nested key" ("a.b.c.d.etc') is called "path key", as it can be seen as a path inside the nested dictionary.
   - Components are written in a way that allows defining input and output keys, to be read and written from the nested dict. See a short introduction video (3 minutes) to how FuseMedML components work:

   `video link <https://user-images.githubusercontent.com/7043815/177197158-d3ea0736-629e-4dcb-bd5e-666993fbcfa2.mp4>`_

   - Examples - using FuseMedML-style components

     - A multi-head model FuseMedML style component, allows easy reuse across projects:

     ```python
     ModelMultiHead(
         conv_inputs=(('data.input.img', 1),),                                       # input to the backbone model
         backbone=BackboneResnet3D(in_channels=1),                                   # PyTorch nn Module
         heads=[                                                                     # list of heads - gives the option to support multi-task / multi-head approach
                    Head3D(head_name='classification',
                                     mode="classification",
                                     conv_inputs=[("model.backbone_features", 512)]  # Input to the classification head
                                     ,),
               ]
     )
     ```

     - Our default loss implementation - creates an easy wrap around a callable function, while being FuseMedML style

     ```python
     LossDefault(
         pred='model.logits.classification',          # input - model prediction scores
         target='data.label',                         # input - ground truth labels
         callable=torch.nn.functional.cross_entropy   # callable - function that will get the prediction scores and labels extracted from batch_dict and compute the loss
     )
     ```

     - An example metric that can be used

     ```python
     MetricAUCROC(
         pred='model.output', # input - model prediction scores
         target='data.label'  # input - ground truth labels
     )
     ```

     - Note that several components return answers directly and not write it into the nested dictionary. This is perfectly fine, and to allow maximum flexibility we do not require any usage of output path keys.

     - Creating a custom FuseMedML component

     - Creating custom FuseMedML components is easy - in the following example, we add a new data processing operator:

     - A data pipeline operator

     ```python
     class OpPad(OpBase):
         def __call__(self, sample_dict: NDict,
             key_in: str,
             padding: List[int], fill: int = 0, mode: str = 'constant',
             key_out:Optional[str]=None,
             ):

             # we extract the element in the defined key location (for example 'input.xray_img')
             img = sample_dict[key_in]
             assert isinstance(img, np.ndarray), f'Expected np.ndarray but got {type(img)}'
             processed_img = np.pad(img, pad_width=padding, mode=mode, constant_values=fill)

             # store the result in the requested output key (or in key_in if no key_out is provided)
             key_out = key_in if key_out is None
             sample_dict[key_out] = processed_img

             # returned the modified nested dict
             return sample_dict
     ```

   - Since the key location isn't hardcoded, this module can be easily reused across different research projects with very different data sample structures. More code reuse - Hooray!

   - FuseMedML-style components, in general, are any classes or functions that define which key paths will be written and which will be read. Arguments can be freely named, and you don't even have to write anything to the nested dictionary. Some FuseMedML components return a value directly - for example, loss functions.

2. "Batteries included" key components, built using the same design concept
   - **[fuse.data](./fuse/data)** - A **declarative** super flexible data processing pipeline
   - Easy dealing with complex multi-modality scenarios
   - Advanced caching, including periodic audits to automatically detect stale caches
   - Default ready-to-use Dataset and
