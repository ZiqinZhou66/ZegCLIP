## Environment:
- Install pytorch

 `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch`

- Install the mmsegmentation library and some required packages.

 `pip install mmcv-full==1.4.4 mmsegmentation==0.24.0`
 `pip install scipy timm==0.3.2`

- Dockerhub:

 `docker push ziqinzhou/zegclip:latest`

## Preparing Dataset:
According to MMseg: https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md

## Preparing Pretrained CLIP model:
Download the pretrained model here: Path/to/ViT-B-16.pt
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

## Training (Inductive):
 `bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py Path/to/coco/zero_12_100`
 
 `bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py Path/to/voc12/zero_12_10`

## Training (Transductive):
 `bash dist_train.sh ./configs/coco/vpt_seg_zero_vit-b_512x512_40k_12_100_multi_st.py Path/to/coco/zero_12_100_st --load-from=Path/to/coco/zero_12_100/iter_40000.pth`
 
 `bash dist_train.sh ./configs/voc12/vpt_seg_zero_vit-b_512x512_10k_12_10_multi_st.py Path/to/voc12/zero_12_10_st --load-from=Path/to/voc12/zero_12_10/iter_10000.pth`

## Training (Fully supervised):
 `bash dist_train.sh configs/coco/vpt_seg_fully_vit-b_512x512_80k_12_100_multi.py Path/to/coco/fully_12_100`
 
 `bash dist_train.sh configs/voc12/vpt_seg_fully_vit-b_512x512_20k_12_10_multi.py Path/to/voc12/fully_12_10`

## Inference:
 `python test.py ./path/to/config ./path/to/model.pth --eval=mIoU`

For example: CUDA_VISIBLE_DEVICES="0" python test.py configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py Path/to/coco/zero_12_100/latest.pth --eval=mIoU

## Cross Dataset Inference:
 `CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/coco-to-voc.py Path/to/coco/vpt_seg_zero_80k_12_100_multi/iter_80000.pth --eval=mIoU`
 
 `CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/coco-to-context.py Path/to/coco/vpt_seg_zero_80k_12_100_multi/iter_80000.pth --eval=mIoU`

## Related Assets \& Acknowledgement

Our work is closely related to the following assets that inspire our implementation. We gratefully thank the authors. 

 - CLIP:  https://github.com/openai/CLIP
 - Maskformer: https://bowenc0221.github.io/maskformer
 - Zegformer: https://github.com/dingjiansw101/ZegFormer
 - zsseg: https://github.com/MendelXu/zsseg.baseline
 - MaskCLIP: https://github.com/chongzhou96/MaskCLIP
 - SegViT: https://github.com/zbwxp/SegVit

## Citation:
If you find this project useful, please consider citing:
```
@article{zhou2022zegclip,
  title={ZegCLIP: Towards adapting CLIP for zero-shot semantic segmentation},
  author={Zhou, Ziqin and Zhang, Bowen and Lei, Yinjie and Liu, Lingqiao and Liu, Yifan},
  journal={arXiv preprint arXiv:2212.03588},
  year={2022}
}
```
