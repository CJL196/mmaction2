# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS
from sklearn.metrics import average_precision_score, f1_score

@METRICS.register_module()
class MultiLabelMetric(BaseMetric):
    """Accuracy evaluation metric for multi-label classification using cMAP and F1 score."""
    default_prefix: Optional[str] = 'ml_acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'cMAP', 'f1_score'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     f1_score=dict(average='micro')),
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in `self.results, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']
            # print(f"Pred: {pred}")
            # Apply sigmoid activation and binarize the predictions
            pred = torch.sigmoid(pred).cpu().numpy()
            pred_binary = (pred > 0.55).astype(int)

            result['pred'] = pred_binary
            result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]
        preds = [x['pred'] for x in results]
        # print(f"preds: {preds}")
        eval_results = dict()
        for metric in self.metrics:
            if metric == 'cMAP':
                # Calculate category-wise mean average precision (cMAP)
                cMAP = average_precision_score(labels, preds, average=None)
                eval_results['cMAP'] = cMAP.mean()  # Return the mean of category-wise AP
                print(f"cMAP: {cMAP.mean()}")

            if metric == 'f1_score':
                # Calculate F1 score
                f1 = f1_score(labels, preds, average=self.metric_options['f1_score']['average'])
                eval_results['f1_score'] = f1
                print(f"F1 score: {f1}")

        return eval_results
