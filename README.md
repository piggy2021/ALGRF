# ALGRF
This is the code for the paper "Video Salient Object Detection via Adaptive Local-Global Refinement". Paper link: https://arxiv.org/abs/2104.14360

# Runtime environment
ubuntu 18.04

PyTorch 1.8.0

cuda 10.2

# Usage
1. config.py. It is for dataset path. You can configure your image path in this file.
2. infer.py. It is for testing. After configuring the image path, you can use it to generate saliency maps.
3. pretrained. The pretrained model can save here. And, the saliency results also can save here. The pretrained model link is https://drive.google.com/file/d/1TqhwJ-D0i6WrQQDlAI7Di0H90GDYJBvA/view?usp=sharing
4. We also provide the generated saliency maps. The download link is https://drive.google.com/file/d/1MHdC4g3MDZQ5XWmhBiEOIGoH_RRVdKFR/view?usp=sharing
5. You can put your images in data folder. Inside, there is a example. A txt file is used to list the images and ViSal is a short testing dataset. 

