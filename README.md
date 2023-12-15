# Ship Detection in SAR images using CenterNet

<div align="justify">
  
Convolutional neural network model based on the architecture of the DLAnet-34 and ResNet-34 for ship detection in synthetic aperture radar (SAR). For this project I have used a pretrained model on ImageNet dataset, and fine-tuned it for the task of ship detection from microwave image data. The model used for this project is CenterNet. CenterNet is an achor-free based detector that detects objects by predicting the center of the bounding box rather than the entire bounding box as done in anchor-based detectors. The models has been trained and evaluated on the Official-SSDD dataset. The model shows localization errors mainly in the misalignment of the bounding boxes and overestimation and underestimation of height of the bounding boxes. Out of the two backbone architectures, DlaNet-34 seems to show less localiztion errors when compared to resnet-34. 

</div align="justify">

## Properties of Official-SSDD
* The dataset contains three types of bounding box annotations
    * Bounding box SSDD (BBox-SSDD)
    * Rotatable bounding box SSDD (RBox-SSDD)
    * Polygon segmentation SSDD (PSeg-SSDD)
* The dataset has strict standards like the training-test division determination, the inshore-offshore protocol, the ship-size reasonable definition, the determination of the densely distributed small ship samples, and the determination of the densely parallel berthing at ports ship samples.
* The rotated annotations of the Official – SSDD dataset are in the PASCAL VOC format. The attributes that are available are (cx, cy, w, h, 𝜃, x1, y1, x2, y2, x3, y3, x4, y4) where 𝜃 ∈ [0°,90°]. The angle labels that are provided are the continuous float-type rather than the discrete int-type, which will lead to better direction estimation accuracy.
* The dataset consists of 1160 SAR images with dimensions of 500×350 pixels.
* It includes 2358 ship instances. The spatial resolutions of SAR images are from 1 to 15 meters per pixel.
* These 1160 images were obtained from RadarSat-2, TerraSAR-X and Sentinel-1 satellites.
* The above 1160 images are in .jpeg format with 24-bit color depth.
* Dataset images have mixed HH, HV, VV, and VH polarizations.
* Paper Link: https://www.mdpi.com/2072-4292/13/18/3690
* Dataset Link: https://drive.google.com/file/d/1Oe99oueVRvFpZsUBRKsA5GgNrnRGxzKk/view?usp=sharing


