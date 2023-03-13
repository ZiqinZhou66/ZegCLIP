import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose, LoadAnnotations
from PIL import Image

@DATASETS.register_module()
class ZeroCOCOStuffDataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood')

    PALETTE = [[240,128,128], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
               [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
               [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
               [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
               [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
               [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
               [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
               [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
               [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
               [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
               [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
               [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
               [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
               [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
               [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
               [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
               [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
               [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
               [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
               [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
               [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
               [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
               [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
               [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
               [0, 192, 32], [238, 209, 156], [64, 0, 64], [128, 192, 160],
               [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
               [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
               [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
               [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
               [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
               [64, 192, 96], [64, 160, 64], [64, 64, 0]]

    def __init__(self, **kwargs):
        super(ZeroCOCOStuffDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', **kwargs)

    def evaluate(self,
                    seen_idx,
                    unseen_idx,
                    results,
                    metric='mIoU',
                    logger=None,
                    gt_seg_maps=None,
                    **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                    results or predict segmentation map for computing evaluation
                    metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        seen_class_names = []
        for i in range(len(seen_idx)):
            seen_class_names.append(class_names[seen_idx[i]])
        seen_class_names = tuple(seen_class_names)

        unseen_class_names = []
        for i in range(len(unseen_idx)):
            unseen_class_names.append(class_names[unseen_idx[i]])
        unseen_class_names = tuple(unseen_class_names)

        # divide ret_metrics into seen and unseen part
        seen_ret_metrics = ret_metrics.copy()
        seen_ret_metrics['IoU'] = seen_ret_metrics['IoU'][seen_idx]
        seen_ret_metrics['Acc'] = seen_ret_metrics['Acc'][seen_idx]
        unseen_ret_metrics = ret_metrics.copy()
        unseen_ret_metrics['IoU'] = unseen_ret_metrics['IoU'][unseen_idx]
        unseen_ret_metrics['Acc'] = unseen_ret_metrics['Acc'][unseen_idx]

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        seen_ret_metrics_summary = OrderedDict({
            seen_ret_metric: np.round(np.nanmean(seen_ret_metric_value) * 100, 2)
            for seen_ret_metric, seen_ret_metric_value in seen_ret_metrics.items()
        })
        unseen_ret_metrics_summary = OrderedDict({
            unseen_ret_metric: np.round(np.nanmean(unseen_ret_metric_value) * 100, 2)
            for unseen_ret_metric, unseen_ret_metric_value in unseen_ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        seen_ret_metrics.pop('aAcc', None)
        seen_ret_metrics_class = OrderedDict({
            seen_ret_metric: np.round(seen_ret_metric_value * 100, 2)
            for seen_ret_metric, seen_ret_metric_value in seen_ret_metrics.items()
        })
        seen_ret_metrics_class.update({'Class': seen_class_names})
        seen_ret_metrics_class.move_to_end('Class', last=False)

        unseen_ret_metrics.pop('aAcc', None)
        unseen_ret_metrics_class = OrderedDict({
            unseen_ret_metric: np.round(unseen_ret_metric_value * 100, 2)
            for unseen_ret_metric, unseen_ret_metric_value in unseen_ret_metrics.items()
        })
        unseen_ret_metrics_class.update({'Class': unseen_class_names})
        unseen_ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        print('\n' +  '+++++++++++ Total classes +++++++++++++')
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])
        print_log('per class results:', logger)
        print_log(class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log(summary_table_data.get_string(), logger=logger)


        print('\n' + '+++++++++++ Seen classes +++++++++++++')
        seen_class_table_data = PrettyTable()
        for key, val in seen_ret_metrics_class.items():
            seen_class_table_data.add_column(key, val)
        seen_summary_table_data = PrettyTable()
        for key, val in seen_ret_metrics_summary.items():
            if key == 'aAcc':
                seen_summary_table_data.add_column(key, [val])
            else:
                seen_summary_table_data.add_column('m' + key, [val])
        print_log('seen per class results:', logger)
        print_log(seen_class_table_data.get_string(), logger=logger)
        print_log('Seen Summary:', logger)
        print_log(seen_summary_table_data.get_string(), logger=logger)
        
        
        print('\n' + '+++++++++++ Unseen classes +++++++++++++')
        unseen_class_table_data = PrettyTable()
        for key, val in unseen_ret_metrics_class.items():
            unseen_class_table_data.add_column(key, val)
        unseen_summary_table_data = PrettyTable()
        for key, val in unseen_ret_metrics_summary.items():
            if key == 'aAcc':
                unseen_summary_table_data.add_column(key, [val])
            else:
                unseen_summary_table_data.add_column('m' + key, [val])
        print_log('unseen per class results:', logger)
        print_log(unseen_class_table_data.get_string(), logger=logger)
        print_log('Unseen Summary:', logger)
        print_log(unseen_summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results