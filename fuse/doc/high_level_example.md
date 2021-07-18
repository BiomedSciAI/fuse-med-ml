# Code Example
An example for a binary classifier for mammography (MG) images. The example also includes segmentation auxiliary loss.

## Data Pipeline

### Data source
Create a simple object that returns a list of sample descriptors:
The implementation is specific to MG project and reads sample descriptors and the fold from a file.
```python
train_data_source = FuseMGDataSource(input_source='/path/to/experiment_file.pkl', folds=[1, 2, 3, 4])
```

### processors
This project include two input processor and two ground processor. A sample will include the output of all processor in a nested dictionary. 

```python
# Model input extractors
# ========================
input_processors = {
    'image': FuseMGInputProcessor(**image_processing_args),
    'clinical_data': FuseMGClinicalDataProcessor(features=['age', 'bmi'], normalize=True)
}

# Ground truth extractors
# =======================
ground_truth_processors = {
    'global': FuseMGGroundTruthProcessorGlobalLabelTask(task=Task(MG_Biopsy_Neg_or_Normal(),
                                                                     MG_Biopsy_Pos()),
    'local': FuseMGGroundTruthProcessorSegmentation(contours_desc=[{'biopsy': ['positive']}])
}
```

Given those processors the format of the sample would be:
```python
{ 
    "data":
    { 
        "input":
        { 
            "image": FuseMGInputProcessor(...)(sample_descr),
            "clinical_data": FuseMGClinicalDataProcessor(...)(sample_descr)
        }
        "gt":
        { 
            "classification": FuseMGGroundTruthProcessorGlobalLabelTask(...)(sample_descr),
            "segmentation": FuseMGGroundTruthProcessorSegmentation(...)(sample_descr)
        }
    }
}
```


## Train dataset & dataloader
```python
augmentor = FuseAugmentor(...)
train_dataset = FuseDatasetDefault(cache_dest='/path/to/cache_dir',
                                   data_source=train_data_source,
                                   input_processors=input_processors,
                                   gt_processors=gt_processors,
                                   augmentor=augmentor)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_sampler=FuseSamplerBalancedBatch(balanced_class_name='data.gt.gt_global.tensor'),
                              num_workers=4)

```

# Model Definition
In this example we will define a model with heads: classification and auxiliary segmentation head.
```python
# Multi-headed model, with clinical data appended to classification head
# ======================================================================
model = FuseModelDefault(
    conv_inputs=(('data.input.input_image.tensor', 1),),
    backbone=FuseBackboneInceptionResnetV2(),
    heads=[
        FuseHeadGlobalPoolingClassifier(head_name='classifier',
                                        conv_inputs=[('model.backbone_features', 384)],
                                        post_concat_inputs=[('data.input.input_clinical_data_vector.tensor')],
                                        num_classes=2),

        FuseHeadDenseSegmentation(head_name='segmentation',
                                  conv_inputs=[('model.backbone_features', 384)],
                                  num_classes=2)
    ]
)
```

# Losses
```python
    losses = {
        'cls_loss': FuseLossDefault(pred_name='model.logits.classifier', target_name='data.gt.classification', callable=F.cross_entropy, weight=1.0),
        'seg_loss': FuseLossSegmentationCrossEntropy(pred_name='model.logits.segmentation', target_name='data.gt.segmentation', weight=2.0),
    }
```

# Metrics
```python
metrics = {
    'auc': FuseMetricAUC(pred_name='model.output.classifier', target_name='data.gt.classification'),
    'accuracy': FuseMetricAccuracy(pred_name='model.output.classifier', target_name='data.gt.classification'),
    'iou': FuseMetricIOU(pred_name='model.output.segmentation', target_name='data.gt.segmentation')
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
    FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
]
manager = FuseManagerDefault(output_model_dir='/path/to/model_dir')
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
manager = FuseManagerDefault(output_model_dir='/path/to/inference/output')

# extract only class scores and save to a file
manager.infer(data_source=FuseMGDataSource(input_source='/path/to/experiment_file.pkl', folds=[0]),
              input_model_dir='/path/to/model_dir',
              checkpoint='best',
              output_columns=['model.output.classifier', 'model.gt.classification'],
              output_file_name='/path/to/infer_file')
```

## Analyze
Read the inference file and evaluate given a collection of metrics.
```python
# Analyzer
# =========
# create analyzer
analyzer = FuseAnalyzerDefault()

# define metrics
metrics = {
    'auc': FuseMetricAUC(pred_name='model.output.classifier', target_name='data.gt.classification'),
    'accuracy': FuseMetricAccuracy(pred_name='model.output.classifier', target_name='data.gt.classification'),
}

# run
analyzer.analyze(gt_processors={}, # No need: labels are already included in inference in this case
                 data_pickle_filename='/path/to/infer_file',
                 metrics=metrics,
                 output_filename='/path/to/analyze_file')
 ```




