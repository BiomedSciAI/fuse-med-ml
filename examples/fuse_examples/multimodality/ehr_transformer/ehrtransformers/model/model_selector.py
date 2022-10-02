model_type = 'BertOrig'
if model_type ==  'BertOrig':
    from ehrtransformers.model.OrigBertFuse import BertBackbone, BertConfig, BertForMultiLabelPredictionHead, BertContrastiveHead
