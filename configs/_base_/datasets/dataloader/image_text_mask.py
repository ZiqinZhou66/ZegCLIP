import json
import os.path as osp
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, OrderedDict, Union, get_args

import albumentations as A
import clip
import cv2
import mmcv
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from prettytable import PrettyTable
from mmseg.datasets.pipelines import Compose
from .custom_pipeline import CustomLoadAnnotations


PROMPT_TYPE = Literal["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]


@DATASETS.register_module()
class ImageTextMaskDataset(CustomDataset):
    """
    Image-Text-Mask Dataset
    Args:
        prompt_types (PROMPT_TYPE): prompt type to use
        images_dir (Path): Path to images directory
        masks_dir (Path): Path to masks directory
        prompt_file (Optional[Path], optional): Path to captions file. Defaults to None.
        img_size (int, optional): Size of image. Defaults to 224.
    """

    PALETTE = [
        [128, 0, 0],
    ]

    CLASSES = (
        "background",
        "polyp",
    )

    def __init__(
        self,
        prompt_type: PROMPT_TYPE,
        class_names: List[str],
        img_dir: Path,
        ann_dir: Path,
        pipeline,
        prompt_file: Optional[Path] = None,
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        gt_seg_map_loader_cfg=None,
        file_client_args=dict(backend="disk"),
        **kwargs,
    ) -> None:
        self.prompt_type = prompt_type

        self.CLASSES = class_names

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
        self.gt_seg_map_loader = (
            CustomLoadAnnotations()
            if gt_seg_map_loader_cfg is None
            else CustomLoadAnnotations(**gt_seg_map_loader_cfg)
        )

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert (
                self.CLASSES is not None
            ), "`cls.CLASSES` or `classes` should be specified when testing"

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        self.img_infos = self.load_annotations(
            self.img_dir,
            self.ann_dir,
            prompt_file,
            prompt_type,
        )

    def load_annotations(
        self, img_dir: Path, ann_dir: Path, prompt_file: Path, prompt_type: PROMPT_TYPE
    ):
        """Load annotation from directory.

        Args:
            img_dir (Path): Path to image directory
            ann_dir (Path): Path to annotation directory.
            prompt_file (Path): Path to prompt file.
            prompt_type (PROMPT_TYPE): Prompt type to use.
        Returns:
            list[dict]: All image info of dataset, along with captions and prompts.
        """

        img_infos = []
        with open(prompt_file, "r") as fp:
            imgs_captions = json.load(fp)

        for img in imgs_captions:
            prompt = img["prompts"][prompt_type]
            if type(prompt) == list:
                prompt = random.choice(prompt)
            img_info = dict(
                filename=img_dir + img["img_name"],
                ann=dict(seg_map=ann_dir + img["mask_name"]),
                prompt=prompt,
            )
            img_infos.append(img_info)

            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def evaluate(
        self,
        seen_idx,
        unseen_idx,
        results,
        metric="mIoU",
        logger=None,
        gt_seg_maps=None,
        **kwargs,
    ):
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
        print(len(results), results[0].shape)
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            gt_seg_maps = list(gt_seg_maps)  # NOTE: TESTING TO CHANGE GENERATOR TO LIST
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label,
            )
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

        if "mIoU" in metric:
            seen_ret_metrics["IoU"] = seen_ret_metrics["IoU"][seen_idx]
        if "mDice" in metric:
            seen_ret_metrics["Dice"] = seen_ret_metrics["Dice"][seen_idx]

        seen_ret_metrics["Acc"] = seen_ret_metrics["Acc"][seen_idx]

        unseen_ret_metrics = ret_metrics.copy()

        if "mIoU" in metric:
            unseen_ret_metrics["IoU"] = unseen_ret_metrics["IoU"][unseen_idx]
        if "mDice" in metric:
            unseen_ret_metrics["Dice"] = unseen_ret_metrics["Dice"][unseen_idx]

        unseen_ret_metrics["Acc"] = unseen_ret_metrics["Acc"][unseen_idx]

        # summary table

        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        seen_ret_metrics_summary = OrderedDict(
            {
                seen_ret_metric: np.round(np.nanmean(seen_ret_metric_value) * 100, 2)
                for seen_ret_metric, seen_ret_metric_value in seen_ret_metrics.items()
            }
        )
        unseen_ret_metrics_summary = OrderedDict(
            {
                unseen_ret_metric: np.round(
                    np.nanmean(unseen_ret_metric_value) * 100, 2
                )
                for unseen_ret_metric, unseen_ret_metric_value in unseen_ret_metrics.items()
            }
        )

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        seen_ret_metrics.pop("aAcc", None)
        seen_ret_metrics_class = OrderedDict(
            {
                seen_ret_metric: np.round(seen_ret_metric_value * 100, 2)
                for seen_ret_metric, seen_ret_metric_value in seen_ret_metrics.items()
            }
        )
        seen_ret_metrics_class.update({"Class": seen_class_names})
        seen_ret_metrics_class.move_to_end("Class", last=False)

        unseen_ret_metrics.pop("aAcc", None)
        unseen_ret_metrics_class = OrderedDict(
            {
                unseen_ret_metric: np.round(unseen_ret_metric_value * 100, 2)
                for unseen_ret_metric, unseen_ret_metric_value in unseen_ret_metrics.items()
            }
        )
        unseen_ret_metrics_class.update({"Class": unseen_class_names})
        unseen_ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        print("\n" + "+++++++++++ Total classes +++++++++++++")
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])
        print_log("per class results:", logger)
        print_log(class_table_data.get_string(), logger=logger)
        print_log("Summary:", logger)
        print_log(summary_table_data.get_string(), logger=logger)

        print("\n" + "+++++++++++ Seen classes +++++++++++++")

        seen_class_table_data = PrettyTable()
        for key, val in seen_ret_metrics_class.items():
            seen_class_table_data.add_column(key, val)
        seen_summary_table_data = PrettyTable()
        for key, val in seen_ret_metrics_summary.items():
            if key == "aAcc":
                seen_summary_table_data.add_column(key, [val])
            else:
                seen_summary_table_data.add_column("m" + key, [val])
        print_log("seen per class results:", logger)
        print_log(seen_class_table_data.get_string(), logger=logger)
        print_log("Seen Summary:", logger)
        print_log(seen_summary_table_data.get_string(), logger=logger)

        print("\n" + "+++++++++++ Unseen classes +++++++++++++")
        unseen_class_table_data = PrettyTable()
        for key, val in unseen_ret_metrics_class.items():
            unseen_class_table_data.add_column(key, val)
        unseen_summary_table_data = PrettyTable()
        for key, val in unseen_ret_metrics_summary.items():
            if key == "aAcc":
                unseen_summary_table_data.add_column(key, [val])
            else:
                unseen_summary_table_data.add_column("m" + key, [val])
        print_log("unseen per class results:", logger)
        print_log(unseen_class_table_data.get_string(), logger=logger)
        print_log("Unseen Summary:", logger)
        print_log(unseen_summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        ret_metrics_class.pop("Class", None)
        for key, value in ret_metrics_class.items():
            eval_results.update(
                {
                    key + "." + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                }
            )

        return eval_results, gt_seg_maps
