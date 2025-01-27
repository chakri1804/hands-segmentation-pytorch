

# Hands Segmentation in PyTorch - A Plug and Play Model
<details>

If you need hands segmentations for your project, you are in the correct place!

[![DOI](https://zenodo.org/badge/354173236.svg)](https://zenodo.org/badge/latestdoi/354173236)

```bash
If you use the code of this repo and you find this project useful, 
please consider to give a star ⭐!
```

If you use this repo for your project please cite this project using:
```bibtex
@article{camporese2021HandsSeg,
  title   = "Hands Segmentation is All You Need",
  author  = "Camporese, Guglielmo",
  journal = "https://github.com/guglielmocamporese",
  year    = "2021",
  url     = "https://github.com/guglielmocamporese/hands-segmentation-pytorch"
}
```

</details>

## Updates
- **[2021/04/03]** - The code is online!
- **[2022/08/04]** - This repo was used in the 3rd model entry at the "AI City Challenge 2022" challenge at CVPR 2022. 
  - [VISTA: Vision Transformer enhanced by U-Net and Image Colorfulness Frame Filtration for Automatic Retail Checkout](https://openaccess.thecvf.com/content/CVPR2022W/AICity/papers/Shihab_VISTA_Vision_Transformer_Enhanced_by_U-Net_and_Image_Colorfulness_Frame_CVPRW_2022_paper.pdf), Shihab et al, CVPRW 2022.

# Direct Usage form Torch Hub

<details>

```python
# Imports
import torch
import torch.hub

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)
# Inference
model.eval()
img_rnd = torch.randn(1, 3, 256, 256) # [B, C, H, W]
preds = model(img_rnd).argmax(1) # [B, H, W]
```

## Results on the Validation and Test Datasets

### Predictions on some test images

![alt text](test_preds.png "Title")

### Table
| Dataset                | Partition  | mIoU  |
| :--------------------: | :--------: | :---: |
| EgoYouTubeHands        | Validation | 0.818 |
| EgoYouTubeHands        | Test       | 0.816 |
| EgoHands               | Validation | 0.919 |
| EgoHands               | Test       | 0.920 |
| HandOverFace           | Validation | 0.814 |
| HandOverFace           | Test       | 0.768 |
| GTEA                   | Validation | 0.960 |
| GTEA                   | Test       | 0.949 |

## What you can do with this code
This code provides:
- A plug and play pretrained model for hand segmentation, either usable directly from `torch hub` (see the **Direct Usage form Torch Hub** section) or usable cloning this repo,
- A collection of **4** different **datasets** for hands segmentation (see the **Datasets** section for more details), that can be used for train a hands segmentation model,
- the scripts for **training** and **evaluating** a hand segmentation model (see the **Train** and **Test** sections),
- the scripts for **finetuning** my pre-trained model, that you can download (see the **Model** section), for hand segmentation on a custom dataset (see the **Finetune** section),
- the scripts for **computing hands segmentation maps** on unseen (your) custom data, using my pre-trained (or your) model (see the **Predict From a Custom Dataset** section).
</details>

# Install Locally
<details>

Once you have cloned the repo, all the commands below should be runned inside the main project folder  `hands` folder:

```python
# Clone the repo
$ git clone https://github.com/guglielmocamporese/hands-segmentation-pytorch.git hands

# Go inside the project folder
$ cd hands
```
To run the code you need to have conda installed (version >= 4.9.2).

Furthermore, all the requirements for running the code are specified in the  `environment.yml`  file and can be installed with:

```
# Install the conda env
$ conda env create --file environment.yml

# Activate the conda env
$ conda activate hands
```

</details>

# Datasets
<details>

I set up a script `scripts/download_datasets.sh` that downloads and prepares all the datasets described below into the `DATA_BASE_PATH` folder, specified in the script itself.

In this project I considered the following datasets for training the model:

- #### **EgoHands** [[link]](http://vision.soic.indiana.edu/projects/egohands/)

  - **4800** labeled frames (**100** labeled frames from **48** different videos),

  - each frame is **720**x**1280**,

  - **1.3** GB of zip file,

- #### **EgoYouTubeHands (EYTH)** [[link]](https://github.com/aurooj/Hand-Segmentation-in-the-Wild)

  - **774** labeled frames,

  - each frame is **216**x**384**,

  - **17** MB of tar.gz file,

- #### **GTEA (with GTEA GAZE PLUS)** [[link]](http://cbs.ic.gatech.edu/fpv/)

  - **1067** labeled frames,

  - each frame of GTEA is **405**x**720**, each frame of GTEA GAZE PLUS is **720**x**960**,

  - **250** MB of zip file,

- #### **HandOverFace (HOF)** [[link]](https://github.com/aurooj/Hand-Segmentation-in-the-Wild)

  - **180** labeled frames,

  - each frame is **384**x**216**,

  - **41** MB of tar.gz file.
</details>
  

# Model
<details>

I used the [PyTorch implementation](https://pytorch.org/vision/stable/models.html#semantic-segmentation) of [DeepLabV3](https://arxiv.org/abs/1706.05587) with ResNet50 backbone. In particular I trained the model for hands segmentation starting from the pretrained DeepLabV3 on COCO train2017.

We provide the code for downloading our model checkpoint:
```python
# Download our pre-trained model
$ ./scripts/download_model_checkpoint.sh
```
This will download the checkpoint `checkpoint.ckpt` inside the `checkpoint` folder.
</details>
  
# Predict From a Custom Dataset

With this code you can do inference and compute the predictions starting from a set of custom images, you just have to specify the folder that contains the images in the variable `data_base_path` in the `scripts/predict.sh` script.

  

Each prediction computed from the image `path/to/image.jpg` will be saved at `path/to/image.jpg.png`.

  

You can find an example of a script used for predicting at `scripts/predict.sh`. I also reported it here:

  

```bash
python main.py \
	--mode predict \
	--data_base_path 'test_images' \
	--model_checkpoint "checkpoint/checkpoint.ckpt" \
	--model_pretrained
```

# Finetuning

An example of script used for finetuning the model is reported in `scripts/finetune.sh` and reported here:

  

```bash
python main.py \
	--mode train \
	--epochs 10 \
	--batch_size 16 \
	--gpus 1 \
	--datasets 'eyth eh hof gtea' \
	--height 256 \
	--width 256 \
	--data_base_path 'data' \
	--model_checkpoint 'checkpoint/checkpoint.ckpt'
	--model_pretrained
```

# Train

An example of script used for training the model is reported in `scripts/train.sh` and reported here:

  

```bash
python main.py \
	--mode train \
	--epochs 50 \
	--batch_size 16 \
	--gpus 1 \
	--datasets 'eyth eh hof gtea' \
	--height 256 \
	--width 256 \
	--data_base_path 'data' \
	--model_pretrained
```

# Test

An example of script used for testing the model is reported in `scripts/test.sh` and reported here:

  

```bash
python main.py \
	--mode test \
	--data_base_path "data" \
	--model_pretrained \
	--model_checkpoint "checkpoint/checkpoint.ckpt"
```

  
# Extra
<details>
<summary>Working with Grayscale Inputs</summary>

# Working with Grayscale Inputs

If you need to work with grayscale images, you just need to:
* Add the `in_channels=1` argument to either the `training`, `validation`, `test`, `predict` command,
* Use the `model_checkpoint="checkpoints/checkpoint-grayscale.ckpt" as the model checkpoint`.

### Results with grayscale inputs
The model that uses grayscale inputs has been trained with all the datasets available above in this project, with all the images converted from RGB to grayscale.

### Predictions on some test images

![alt text](test_preds_grayscale.png "Title")

Here you can find the results on the validation and test sets using the grayscale model evaluated on the grayscale datasets.
| Dataset                | Input     | Partition   | mIoU   |
| :--------------------: | :-------: | :---------: | :----: |
| EgoYouTubeHands        | Grayscale |  Validation | 78.49% |
| EgoYouTubeHands        | Grayscale |  Test       | 79.36% |
| EgoHands               | Grayscale |  Validation | 90.31% |
| EgoHands               | Grayscale |  Test       | 90.32% |
| HandOverFace           | Grayscale |  Validation | 81.98% |
| HandOverFace           | Grayscale |  Test       | 74.50% |
| GTEA                   | Grayscale |  Validation | 94.89% |
| GTEA                   | Grayscale |  Test       | 94.01% |
</details>

<details>
<summary>Working with RGBD Inputs</summary>

# Working with RGBD Inputs

If you need to work with grayscale images, you just need to add the `in_channels=4` argument to either the `training`, `validation`, `test`, `predict` command.

However, at this time there are no available models in this project pre-trained on RGBD datasets.

</details>
