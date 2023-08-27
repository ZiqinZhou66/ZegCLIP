from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from .untils import tokenize
import numpy as np
import tqdm

import os
import matplotlib.pyplot as plt


@SEGMENTORS.register_module()
class ZegCLIP(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self,
        text_encoder,
        pretrained_text,
        class_names,
        context_length,
        base_class,
        novel_class,
        both_class,
        tau=0.07,
        multi_prompts=False,
        self_training=False,
        ft_backbone=False,
        exclude_key=None,
        load_text_embedding=None,
        #  init_cfg=None,
        **args,
    ):
        super(ZegCLIP, self).__init__(**args)

        if pretrained_text is not None:
            assert (
                text_encoder.get("pretrained") is None
            ), "both text encoder and segmentor set pretrained weight"
            text_encoder.pretrained = pretrained_text

        self.text_encoder = builder.build_backbone(text_encoder)

        self.tau = tau
        self.class_names = class_names

        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        self.multi_prompts = multi_prompts
        self.load_text_embedding = load_text_embedding

        if len(self.base_class) != len(self.both_class):  # zero-shot setting
            if not self_training:
                self._visiable_mask(self.base_class, self.novel_class, self.both_class)
            else:
                self._visiable_mask_st(
                    self.base_class, self.novel_class, self.both_class
                )
                self._st_mask(self.base_class, self.novel_class, self.both_class)

        if self.training:
            self._freeze_stages(self.text_encoder)
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)

        else:
            self.text_encoder.eval()
            self.backbone.eval()

    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print("Finetune layer in backbone:", n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([-1] * 256)
        seen_map[255] = 255
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        print("Making visible mask for zero-shot setting:", self.visibility_seen_mask)

    def _visiable_mask_st(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([-1] * 256)
        seen_map[255] = 255
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = n
        seen_map[200] = 200  # pixels of padding will be excluded
        self.visibility_seen_mask = seen_map.copy()
        print(
            "Making visible mask for zero-shot setting in self_traning stage:",
            self.visibility_seen_mask,
        )

    def _st_mask(self, seen_classes, novel_classes, both_classes):
        st_mask = np.array([255] * 256)
        st_mask[255] = 255
        for i, n in enumerate(list(novel_classes)):
            st_mask[n] = n
        self.st_mask = st_mask.copy()
        print(
            "Making st mask for zero-shot setting in self_traning stage:", self.st_mask
        )

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _decode_head_forward_train(self, feat, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        if self.training:
            if len(self.base_class) != len(self.both_class):  # zero setting
                gt_semantic_seg = torch.Tensor(self.visibility_seen_mask).type_as(
                    gt_semantic_seg
                )[gt_semantic_seg]

        losses = dict()
        if self.self_training:
            loss_decode = self.decode_head.forward_train(
                feat,
                img_metas,
                gt_semantic_seg,
                self.train_cfg,
                self.self_training,
                self.st_mask,
            )
        else:
            loss_decode = self.decode_head.forward_train(
                feat, img_metas, gt_semantic_seg, self.train_cfg, self.self_training
            )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def text_embedding(self, texts, img):
        text_embeddings = self.text_encoder(texts.to(img.device))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def extract_feat(self, img):
        """Extract features from images."""
        visual_feat = self.backbone(img)
        return visual_feat

    def forward_train(self, img, img_metas, gt_semantic_seg):
        visual_feat = self.extract_feat(img)
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = torch.from_numpy(text_feat).to(img.device)
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                assert AttributeError("preparing the multi embeddings")

        if not self.self_training:
            text_feat = text_feat[self.base_class, :]

        feat = []
        feat.append(visual_feat)
        feat.append(text_feat)

        losses = dict()
        loss_decode = self._decode_head_forward_train(feat, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        return losses

    def encode_decode(self, img, img_metas):
        visual_feat = self.extract_feat(img)

        self.texts = tokenize([d["img_info"]["prompt"] for d in img_metas])
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = torch.from_numpy(text_feat).to(img.device)
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                num_cls, num_prompts, _ = self.texts.size()
                text_feat = self.text_embedding(
                    self.texts.reshape(num_cls * num_prompts, -1), img
                )
                text_feat = text_feat.reshape(num_cls, num_prompts, -1).mean(dim=1)
                text_feat /= text_feat.norm(dim=-1).unsqueeze(1)

        feat = []
        feat.append(visual_feat)
        feat.append(text_feat)

        out = self._decode_head_forward_test(feat, img_metas, self.self_training)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return out

    def _decode_head_forward_test(self, x, img_metas, self_training):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(
            x, img_metas, self.test_cfg, self_training
        )
        return seg_logits

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = len(self.both_class)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                device=img.device
            )
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]["ori_shape"][:2],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )
        return preds
