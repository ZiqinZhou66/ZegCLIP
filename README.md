# ZegCLIP
Official implementation of ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation

paper：http://arxiv.org/abs/2212.03588


# Environment:
-Install pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch

-Install the mmsegmentation library and some required packages.

pip install mmcv-full==1.4.4 mmsegmentation==0.24.0

pip install scipy timm==0.3.2

--Dockerhub:
docker push ziqinzhou/zegclip:latest

# Preparing Dataset:
1. PASCAL VOC 2012
2. COCO Stuff 164K
3. PASCAL Context

According to MMseg: https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md

# Preparing Pretrained CLIP model:
Download the pretrained model here: ./pretrained/ViT-B-16.pt
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

# Results:
## PASCAL VOC 2012:
| Method               | pAcc  | mIoU\(S\) | mIoU\(U\) | hIoU  |
|----------------------|-------|-----------|-----------|-------|
| Inductive            |       |           |           |       |
| SPNet                | \-    | 78\.0     | 15\.6     | 26\.1 |
| ZS3                  | \-    | 77\.3     | 17\.7     | 28\.7 |
| CaGNet               | 80\.7 | 78\.4     | 26\.6     | 39\.7 |
| SIGN                 | \-    | 75\.4     | 28\.9     | 41\.7 |
| Joint                | \-    | 77\.7     | 32\.5     | 45\.9 |
| ZegFormer            | \-    | 86\.4     | 63\.6     | 73\.3 |
| zsseg                | 90\.0 | 83\.5     | 72\.5     | 77\.5 |
| ZegCLIP \(Ours\)     | 94\.6 | 91\.9     | 77\.8     | 84\.3 |
| Transductive         |       |           |           |       |
| SPNet\+ST            | \-    | 77\.8     | 25\.8     | 38\.8 |
| ZS5                  | \-    | 78\.0     | 21\.2     | 33\.3 |
| CaGNet\+ST           | 81\.6 | 78\.6     | 30\.3     | 43\.7 |
| STRICT               | \-    | 82\.7     | 35\.6     | 49\.8 |
| zsseg\+ST            | 88\.7 | 79\.2     | 78\.1     | 79\.3 |
| ZegCLIP\+ST \(Ours\) | 95\.1 | 91\.8     | 82\.2     | 86\.7 |
| \*MaskCLIP\+         | \-    | 88\.8     | 86\.1     | 87\.4 |
| \*ZegCLIP\+ST        | 96\.2 | 92\.3     | 89\.9     | 91\.1 |
| Fully Supervised     |       |           |           |       |
| ZegCLIP \(Ours\)     | 96\.3 | 92\.4     | 90\.9     | 91\.6 |

## COCO-Stuff 164K:
| Method               | pAcc  | mIoU\(S\) | mIoU\(U\) | hIoU  |
|----------------------|-------|-----------|-----------|-------|
| Inductive            |       |           |           |       |
| SPNet                | \-    | 35\.2     | 8\.7      | 14\.0 |
| ZS3                  | \-    | 34\.7     | 9\.5      | 15\.0 |
| CaGNet               | 56\.6 | 33\.5     | 12\.2     | 18\.2 |
| SIGN                 | \-    | 32\.3     | 15\.5     | 20\.9 |
| Joint                | \-    | \-        | \-        | \-    |
| ZegFormer            | \-    | 36\.6     | 33\.2     | 34\.8 |
| zsseg                | 60\.3 | 39\.3     | 36\.3     | 37\.8 |
| ZegCLIP \(Ours\)     | 62\.0 | 40\.2     | 41\.4     | 40\.8 |
| Transductive         |       |           |           |       |
| SPNet\+ST            | \-    | 34\.6     | 26\.9     | 30\.3 |
| ZS5                  | \-    | 34\.9     | 10\.6     | 16\.2 |
| CaGNet\+ST           | 56\.8 | 35\.6     | 13\.4     | 19\.5 |
| STRICT               | \-    | 35\.3     | 30\.3     | 34\.8 |
| zsseg\+ST            | 63\.8 | 39\.6     | 43\.6     | 41\.5 |
| ZegCLIP\+ST \(Ours\) | 68\.8 | 40\.6     | 54\.8     | 46\.6 |
| \*MaskCLIP\+         | \-    | 38\.1     | 54\.7     | 45\.0 |
| \*ZegCLIP\+ST        | 69\.2 | 40\.6     | 59\.9     | 48\.4 |
| Fully                |       |           |           |       |
| ZegCLIP \(Ours\)     | 69\.9 | 40\.7     | 63\.2     | 49\.6 |

## PASCAL Context:
| Method               | pAcc   | mIoU\(S\) | mIoU\(U\) | hIoU  |
|----------------------|--------|-----------|-----------|-------|
| Inductive            |        |           |           |       |
| SPNet                | \-     | \-        | \-        | \-    |
| ZS3                  | 52\.8  | 20\.8     | 12\.7     | 15\.8 |
| CaGNet               | \-     | 24\.1     | 18\.5     | 21\.2 |
| SIGN                 | \-     | \-        | \-        | \-    |
| Joint                | \-     | 33\.0     | 14\.9     | 20\.5 |
| ZegFormer            | \-     | \-        | \-        | \-    |
| zsseg                | \-     | \-        | \-        | \-    |
| ZegCLIP \(Ours\)     | 76\.2  | 46\.0     | 54\.6     | 49\.9 |
| Transductive         |        |           |           |       |
| SPNet\+ST            | \-     | \-        | \-        | \-    |
| ZS5                  | 49\.5  | 27\.0     | 20\.7     | 23\.4 |
| CaGNet\+ST           | \-     | \-        | \-        | \-    |
| STRICT               | \-     | \-        | \-        | \-    |
| zsseg\+ST            | \-     | \-        | \-        | \-    |
| ZegCLIP\+ST \(Ours\) | 77\.2  | 46\.6     | 65\.4     | 54\.4 |
| \*MaskCLIP\+         | \-     | 44\.4     | 66\.7     | 53\.3 |
| \*ZegCLIP\+ST        | 77\.3  | 46\.8     | 68\.5     | 55\.6 |
| Fully                |        |           |           |       |
| ZegCLIP \(Ours\)     | 77\.5  | 46\.5     | 78\.7     | 56\.9 |

 
# Citation:

# Code:
Coming soon~
