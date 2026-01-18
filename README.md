# Face-Voice Association with Inductive Bias for Maximum Class Separation
> Face-voice association is widely studied in multimodal learning and is approached representing faces and voices with embeddings that  are close for a same person and well separated from those of others. Previous work achieved this with loss functions. Recent advancements in classification have shown that the discriminative ability of embeddings can be strengthened by imposing maximum class separation as inductive bias. This technique has never been used in the domain of face-voice association, and this work aims at filling this gap. More specifically, we develop a method for face-voice association that imposes maximum class separation among multimodal representations of different speakers as an inductive bias. Through quantitative experiments we demonstrate the effectiveness of our approach, showing that it achieves SOTA performance on two task formulation of face-voice association. Furthermore, we carry out an ablation study to show that imposing inductive bias is most effective when combined with losses for inter-class orthogonality. To the best of our knowledge, this work is the first that applies and demonstrates the effectiveness of maximum class separation as an inductive bias in multimodal learning; it hence paves the way to establish a new paradigm.


## Installation
Our environment requires on python==3.6.5 and torch==1.8.0. We recommend creating a virtual environment, e.g., using `conda`.
To install dependencies run:
```
pip install -r requirements.txt
```
For installation of Pytorch and CUDA (For GPU):
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

## Feature Extraction
Our experiments are based on the VoxCeleb1 datasets. The dataset and train/test splits can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)

### Facial Feature Extraction
For Face Embeddings we use [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf). We use the pytorch implementation from the following [repository](https://github.com/timesler/facenet-pytorch).
### Voice Feature Extraction
For Voice Embeddings we use the method described in [ECAPA-TDNN](https://arxiv.org/abs/2005.07143). The code is available [here](https://github.com/TaoRuijie/ECAPA-TDNN)

Once the features are extracted, write them to a .csv file in features directory. The .csv files of train and test splits can be downloaded [here](https://zenodo.org/records/15386911)

## Training and Testing

### Training

- **Linear Fusion:**

  ```bash
  python main.py --cuda 1 --save_dir ./model --lr 1e-4 --batch_size 512 --max_num_epoch 25 --alpha_list [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] --dim_embed 900 --fusion linear
  ```

### Testing

### Fac-voice verification

- **Linear Fusion:**

  ```bash
  python test_verification.py --cuda 1 --ckpt <path to checkpoint.pth.tar> --dim_embed 900 --fusion linear --alpha 1
  ```

### Fac-voice matching

- **Linear Fusion:**

  ```bash
  python test_matching.py --cuda --checkpoint_path <path_to_checkpoint.pth.tar> --dim_embed 900 --fusion linear --identities <amount of ids>
  ```
