# ISIC 2018: Lesion Attribute Detection
Melanoma is the most deadly form of skin cancer worldwide. Many efforts have been made for early detection of melanoma. The International Skin Imaging Collaboration (ISIC) hosted the 2018 Challenges to help the diagnosis of melanoma based on dermoscopic images. We describe our solutions for the task 2 of ISIC 2018 Challenges. We present two deep learning approaches to automatically detect lesion attributes of melanoma, one is a multi-task U-Net model and the other is a Mask R-CNN based model. Our multi-task U-Net model achieved a Jaccard index of 0.433 on official test data, which ranks the 5th place on the final leaderboard.

# How to run the pre-trained model on the test data

### Create a Python environment
```
conda create -n isic2018 python=3
source activate isic2018
```

### Install necessary packages
Install packages.
```
pip install -r requirements.txt
```

### Clone this repo
```
git clone https://github.com/chvlyl/ISIC2018.git
```

# Further improvement

# Reference	
(Eric Z. Chen, Xu Dong, Junyan Wu, Hongda Jiang, Xiaoxiao Li, Ruichen Rong. Lesion Attributes Segmentation for Melanoma Detection with Deep Learning. bioRxiv 2018)[https://doi.org/10.1101/381855]
