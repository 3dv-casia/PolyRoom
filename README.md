# [ECCV2024] PolyRoom: Room-aware Transformer for Floorplan Reconstruction
Official implementation of 'PolyRoom: Room-aware Transformer for Floorplan Reconstruction'.   

## Abstract
Reconstructing geometry and topology structures from raw unstructured data has always been an important research topic in indoor mapping research. 
In this paper, we aim to reconstruct the floorplan with a vectorized representation from point clouds. 
Despite significant advancements achieved in recent years, current methods still encounter several challenges, such as missing corners or edges, inaccuracies in corner positions or angles, self-intersecting or overlapping polygons, and potentially implausible topology. 
To tackle these challenges, we present PolyRoom, a room-aware Transformer that leverages uniform sampling representation, room-aware query initialization, and room-aware self-attention for floorplan reconstruction. 
Specifically, we adopt a uniform sampling floorplan representation to enable dense supervision during training and effective utilization of angle information. Additionally, we propose a room-aware query initialization scheme to prevent non-polygonal sequences and introduce room-aware self-attention to enhance memory efficiency and model performance. 
Experimental results on two widely used datasets demonstrate that PolyRoom surpasses current state-of-the-art methods both quantitatively and qualitatively. 
## Method
<img src="./imgs/pipeline.jpg" width=100% height=100%>

**Overall architecture of PolyRoom.** PolyRoom consists of four main components: (a) Encoder module, (b) Decoder module, (c) Room-aware query initialization module, and (d) Floorplan extraction module. Room queries are initialized with instance segmentation. Subsequently,  they are refined in the Transformer decoder layer by layer with dense supervision (red and blue boxes mark the changes). Finally, the floorplan is extracted based on vertex selection. The detailed structure of the $i$th layer in the Transformer decoder is depicted in the right part, where $F$ denotes the output of the Transformer encoder, $C_i$, $C_{i+1}$ represent content queries from different layers, while $Q_i$, $Q_{i+1}$ denote room queries from different layers.

## Environment, dataset, and runnning
Please refer to [RoomFormer](https://github.com/ywyue/RoomFormer/) and [MMdetection](https://github.com/open-mmlab/mmdetection) for environment setting, dataset preparation and code running.
Put the file "mask-rcnn_r50-caffe_fpn_ms-poly-1x_str3d.py" in ./mmdetection/configs/str3d, the file "str3dcoco.py" in ./mmdetection/mmdet/datasets, the "__init__.py" in ./mmdetection/mmdet/datasets. Besides, complete the "maskrcnnconfig_file" and "maskrcnncheckpoint_file" in engine.py. (Please install MMDetection from source and move the specified files to the source folder of the MMDetection library, refer to https://github.com/3dv-casia/PolyRoom/issues/2#issuecomment-2838711918 for the details).

## Checkpoints
The checkpoints of the Mask2former and PolyRoom are can be downloaded in [this link](https://drive.google.com/drive/folders/186SLocs6jPzCKq9B5419HfMV2TIMBz2I?usp=sharing).

## Citation
If you find PolyRoom useful in your research, please cite our paper:
```
@inproceedings{liu2024polyroom,
  title={{PolyRoom: Room-aware Transformer for Floorplan Reconstruction}},
  author={Liu, Yuzhou and Zhu, Lingjie and Ma, Xiaodong and Ye, Hanqiao and Gao, Xiang and Zheng, Xianwei and Shen, Shuhan},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## Acknowledgment
We thank the authors of RoomFormer, HEAT, and MonteFloor for providing results for better comparison. We also thank the following excellent projects especially RoomFormer:
* [RoomFormer](https://github.com/ywyue/RoomFormer/)
* [DETR](https://github.com/facebookresearch/detr)
* [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [HEAT](https://github.com/woodfrog/heat)
* [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)
* [MMdetection](https://github.com/open-mmlab/mmdetection)

