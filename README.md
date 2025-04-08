## 🚁 Project Extension: Aerial Image Dehazing using VSAI

This project is an extension of the original [Nonhomogeneous Image Dehazing](https://github.com/diptamath/Nonhomogeneous_Image_Dehazing.git) model.  
We adapted and enhanced the model for aerial image dehazing in low-visibility conditions (e.g., helicopter-mounted cameras) using the **VSAI dataset**.

---

### 🔧 Key Modifications:
- 📚 **Retraining**: The DMPHN-based model was retrained on the custom **VSAI dataset**, which includes aerial images in various enviroments.
- 🌫️ **Synthetic Haze Generation**: We applied haze effects to clean aerial images using a custom script, simulating real-world atmospheric interference.
- 🔁 **Domain Adaptation**: Training parameters and augmentation strategies were adjusted for aerial perspectives and varying altitudes.

---

### 📦 Added Files and Their Purpose:

- `apply_haze.py`  
  ↳ Generates synthetic haze overlays on the VSAI dataset, used for training data.

- `compile.py`  
  ↳ Compiles the trained model for deployment using **Qualcomm AI Hub**, targeting efficient edge inference.

- `convert.py`  
  ↳ Converts the PyTorch-trained model to **ONNX** format for compatibility across platforms.

- `quantize_and_profile_test.py`  
  ↳ Performs **quantization** and **inference profiling** for edge deployment evaluation.  
  🔹 *Note: This is not the final quantized version used in the final model.*

---

### 📁 Folder Overview

- `checkpoints/`  
  📦 Contains our **own trained model weights** and intermediate checkpoints from the retraining on the VSAI dataset.

- `checkpoints3/`  
  📥 Stores the **original pre-trained model**.

---

### 📸 Real-World Use Case

This enhanced version is designed for **real-time dehazing in aerial systems**, such as drones and helicopters.  
It improves visibility in scenarios like **surveillance**, **search-and-rescue**, and **navigation** under foggy or hazy conditions.

---

### 🙏 Credits

Built upon the original [Nonhomogeneous Image Dehazing](https://github.com/diptamath/Nonhomogeneous_Image_Dehazing.git) repository.  
All modifications and experiments were conducted as part of a university project for the course **Intelligent Systems**.



# Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing
The code for implementing the "Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing" (Accepted at NTIRE Workshop, CVPR 2020).

Preprint: https://arxiv.org/abs/2005.05999

## Highlights

- **Lightweight:** The proposed system is very lightweight as the total size of the model is around 21.7 MB.
- **Robust:**  it is quite robust for different environments with various density of the haze or fog in the scene.
- **Fast:** it can process an HD image in 0.0145 seconds on average and can dehaze images from a video sequence in real-time.

## Getting Started

For training, image paths for train, val and test data should be listed in text file (example. for image patch level traning, coresponding hazed and ground truth image paths are listed in text file with path  `./new_dataset/train_patch_gt.txt`). Pretrained models are given in the `checkpoints2/DMSHN_1_2_4` and `checkpoints/DMPHN_1_2_4`path for demo.

### Prerequisites

- Pytorch
- Scikit-image 
- Numpy
- Scipy
- OpenCV


## Running the tests

For model Inference, run following commands
```
# for Deep Multi-Patch Hierarchical Network
python DMPHN_test.py

# for Deep Multi-Scale Hierarchical Network
python DMSHN_test.py

```

## Running the Training

For model training, run following commands
```
# for Deep Multi-Patch Hierarchical Network
python DMPHN_train.py

# for Deep Multi-Scale Hierarchical Network
python DMSHN_train.py

```

# Quantitative Results
<img src="assets/cvpr_2.png" width="500"/>

# Qualitative Results
![](assets/cvpr_1.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@InProceedings{ Das_fast_deep_2020,
author = {Sourya Dipta Das and Saikat Dutta},
title = {Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}

```


