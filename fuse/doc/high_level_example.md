# Code Example
An example for a binary classifier for mammography (MG) images. The example also includes segmentation auxiliary loss.

## Data Pipeline

### Data source
Create a simple object that returns a list of sample descriptors:
The implementation is specific to MG project and reads sample descriptors and the fold from a file.
```python
train_data_source = MGDataSource(input_source='/path/to/experiment_file.pkl', folds=[1, 2, 3, 4])
```

### processors
This project include two input processor and two ground processor. A sample will include the output of all processor in a nested dictionary. 

```python
# Model input extractors
# ========================
input_processors = {
    'image': MGInputProcessor(**image_processing_args),
    'clinical_data': MGClinicalDataProcessor(features=['age', 'bmi'], normalize=True)
}

# Ground truth extractors
# =======================
ground_truth_processors = {
    'classification': MGGroundTruthProcessorGlobalLabelTask(task=Task(MG_Biopsy_Neg_or_Normal(),
                                                                     MG_Biopsy_Pos()),
    'segmentation': MGGroundTruthProcessorSegmentation(contours_desc=[{'biopsy': ['positive']}])
}
```

Given those processors the format of the sample would be:
```python
{ 
    "data":
    { 
        "input":
        { 
            "image": MGInputProcessor(...)(sample_descr),
            "clinical_data": MGClinicalDataProcessor(...)(sample_descr)
        }
        "gt":
        { 
            "classification": MGGroundTruthProcessorGlobalLabelTask(...)(sample_descr),
            "segmentation": MGGroundTruthProcessorSegmentation(...)(sample_descr)
        }
    }
}
```


## Train dataset & dataloader
```python
augmentor = Augmentor(...)
train_dataset = DatasetDefault(cache_dest='/path/to/cache_dir',
                                   data_source=train_data_source,
                                   input_processors=input_processors,
                                   gt_processors=gt_processors,
                                   augmentor=augmentor)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_sampler=SamplerBalancedBatch(balanced_class_name='data.gt.classification'),
                              num_workers=4)

```

# Model Definition
In this example we will define a model with heads: classification and auxiliary segmentation head.
```python
# Multi-headed model, with clinical data appended to classification head
# ======================================================================
model = ModelDefault(
    conv_inputs=(('data.input.image', 1),),
    backbone=BackboneInceptionResnetV2(),
    heads=[
        HeadGlobalPoolingClassifier(head_name='classifier',
                                        conv_inputs=[('model.backbone_features', 384)],
                                        post_concat_inputs=[('data.input.clinical_data')],
                                        num_classes=2),

        HeadDenseSegmentation(head_name='segmentation',
                                  conv_inputs=[('model.backbone_features', 384)],
                                  num_classes=2)
    ]
)
```

# Losses
```python
    losses = {
        'cls_loss': LossDefault(pred='model.logits.classifier', target='data.gt.classification', callable=F.cross_entropy, weight=1.0),
        'seg_loss': LossSegmentationCrossEntropy(pred_name='model.logits.segmentation', target_name='data.gt.segmentation', weight=2.0),
    }
```

# Metrics
```python
metrics = {
    'auc': MetricAUCROC(pred_name='model.output.classifier', target_name='data.gt.classification'),
    'iou': MetricIOU(pred_name='model.output.segmentation', target_name='data.gt.segmentation')
}

best_epoch_source = {
    'source': 'metrics.auc.macro_avg',    
    'optimization': 'max',
    'on_equal_values': 'better',
}
```


# Training
Start a training process
```python
# Train model - using a Manager instance
# ======================================
callbacks = [
    TensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
]
manager = ManagerDefault(output_model_dir='/path/to/model_dir')
manager.set_objects(net=model,
                    optimizer=Adam(model.parameters(), lr=1e-4, weight_decay=0.001),
                    losses=losses,
                    metrics=metrics,
                    best_epoch_source=best_epoch_source,
                    callbacks=callbacks,
                    lr_scheduler=....
                    ...)

# Run training!
manager.train(train_dataloader=train_dataloader,
              validation_dataloader=validation_dataloader)

```

## Inference
Output predictions and labels to a file
```python
# Inference
# =========
manager = ManagerDefault(output_model_dir='/path/to/inference/output')

# extract only class scores and save to a file
manager.infer(data_loader=infer_data_loder,
              input_model_dir='/path/to/model_dir',
              checkpoint='best',
              output_columns=['model.output.classifier', 'model.gt.classification'],
              output_file_name='/path/to/infer_file')
```

## Analyze
Read the inference file and evaluate given a collection of metrics.
```python
  metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred='model.output.classifier')), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label')),
        ('roc', MetricROCCurve(pred='model.output.classifier', target='model.gt.classification', class_names=class_names, output_filename=os.path.join(paths['inference_dir'], 'roc_curve.png'))),
        ('auc', MetricAUCROC(pred='model.output.classifier', target='data.label', class_names=class_names)),
    ])
   
    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=os.path.join(paths["inference_dir"], eval_common_params["infer_filename"]),
                     metrics=metrics,
                     output_dir=paths['eval_dir'])
 ```




