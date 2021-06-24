# Image Translation with ASAPNets

## Spatially-Adaptive Pixelwise Networks for Fast Image Translation, CVPR 2021
### [Webpage](https://tamarott.github.io/ASAPNet_web/) | [Paper](https://arxiv.org/pdf/2012.02992.pdf) | [Video](https://www.youtube.com/watch?v=6-OfZ32CoBE&t=11s)

## Installation
install requirements:
```bash
pip install -r requirements.txt
```

## Code Structure
The code is heavily based on the [official implementation](https://github.com/NVlabs/SPADE) of [SPADE](https://arxiv.org/pdf/1903.07291.pdf), and therefore has the saome structure: 
- `train.py`, `test.py`: the entry point for training and testing.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses.
- `models/networks/`: defines the architecture of all models.
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

The ASAPNets generator is implementaed in:
- `models/networks/generator`: defines the architecture of the ASAPNets generator.

## Dataset Preparation

### facades
run: 
```
cd data 
bash facadesHR_download_and_extract.sh
```
This will extract the facades full resolution images into `datasets/facadesHR`.

### cityscapes
[download the dataset](https://www.cityscapes-dataset.com/) into `datasets/cityscapes` and arrange in folders: train_images, train_labels, val_images, val_labels

## Generating Images Using Pretrained Models

Pretraned models can be downloaded from [here](https://drive.google.com/drive/folders/1mNWsh6QwA-5i8KeihrI6opDj-a2AcOq9?usp=sharing). 
Save the models under the `checkpoints/` folder.
Images can be generated using the command:

```
# Facades 512
bash test_facades512.sh

# Facades 1024
bash test_facades512.sh

# Cityscapes
bash test_cityscapes.sh
```

The outputs images will appear at the`./results/` folder.

## Training New Models

New models can be trained with the following commands.
Prepare dataset in the `./datasets/` folder. Arrange in folders: train_images, train_labels, val_images, val_labels . 
For custom datasets, the easiest way is to use `./data/custom_dataset.py` by specifying the option `--dataset_mode custom`, along with `--label_dir [path_to_labels] --image_dir [path_to_images]`. 
You also need to specify options such as `--label_nc` for the number of label classes in the dataset, `--contain_dontcare_label` to specify whether it has an unknown label, or `--no_instance` to denote the dataset doesn't have instance maps.

Run:
```
python train.py --name [experiment_name] --dataset_mode custom --label_dir [path_to_labels] -- image_dir [path_to_images] --label_nc [num_labels]

```
There are many additional options you can specify, please explore the `./options` files.
To specify the number of GPUs to utilize, use `--gpu_ids`.

## Testing

Testing is similar to testing pretrained models.

```
python test.py --name [name_of_experiment] --dataset_mode [dataset_mode] --dataroot [path_to_dataset]
```
you can load the parameters used from training by specifying `--load_from_opt_file`.

## Acknowledgments
This code is heavily based on the [official implementation](https://github.com/NVlabs/SPADE) of [SPADE](https://arxiv.org/pdf/1903.07291.pdf). 
We thank the authors for sharing their code publicly!

### License 
Attribution-NonCommercial-ShareAlike 4.0 International (see file).


### Citation
```
@inproceedings{RottShaham2020ASAP,
  title={Spatially-Adaptive Pixelwise Networks for Fast Image Translation},
  author={Rott Shaham, Tamar and Gharbi, Michael and Zhang, Richard and Shechtman, Eli and Michaeli, Tomer},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```