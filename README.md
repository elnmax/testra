# TeSTra: Real-time Online Video Detection with Temporal Smoothing Transformers

## Modifications
This is a fork of [TeSTra](https://github.com/zhaoyue-zephyrus/TeSTra) (Apache-2.0).

## Introduction

This is a PyTorch implementation for our ECCV 2022 paper "[`Real-time Online Video Detection with Temporal Smoothing Transformers`](https://arxiv.org/pdf/2209.09236.pdf)".

![teaser](assets/testra_teaser.png?raw=true)

## Environment

- The code is developed with CUDA 10.2, ***Python >= 3.7.7***, ***PyTorch >= 1.7.1***

    1. Clone the repo ***recursively***.
        ```
        git clone --recursive git@github.com:zhaoyue-zephyrus/TeSTra.git
        ```

    2. [Optional but recommended] create a new conda environment.
        ```
        conda create -n testra python=3.7.7
        ```
        And activate the environment.
        ```
        conda activate testra
        ```

    3. Install the requirements
        ```
        pip install -r requirements.txt
        ```

## Data Preparation

### Pre-extracted Feature

You can directly download the pre-extracted feature (.zip) from the Google Drive links below.

#### THUMOS'14

|   Description  |   backbone   |    pretrain    |                            UTBox Link                             |
|  ------------  |  ----------  |  ------------  |  ---------------------------------------------------------------  |
|   frame label  |     N/A      |       N/A      | [link](https://drive.google.com/file/d/1L6I1x6J5YORL9H6Gc4193GGbdBSNo34k/view?usp=sharing) |
|   RGB          |  ResNet-50   |  Kinetics-400  | [link](https://drive.google.com/file/d/1G7d4wpoeGVNEhW9RQhpgF1nHNmDkwCGD/view?usp=sharing) |
|   Flow (TV-L1) | BN-Inception |  Kinetics-400  | [link](https://drive.google.com/file/d/1bXKrUtrXcIOOod1fuwqHtVzceGEORb-q/view?usp=sharing) |
|   Flow (NVOF)  | BN-Inception |  Kinetics-400  | [link](https://drive.google.com/file/d/14nlFB4tRagdPzOu5wUxZwd1nNIR33MvN/view?usp=sharing) |
|   RGB          |  ResNet-50   |    ANet v1.3   | [link](https://drive.google.com/file/d/1td6sRku-uGN0OEA2R7ApK4kp0tudVNdj/view?usp=sharing) |
|   Flow (TV-L1) |  ResNet-50   |    ANet v1.3   | [link](https://drive.google.com/file/d/1F-fWEOFpdAEbPzkugTQdJBIHych5iRMZ/view?usp=sharing) |

#### EK100

|  Description   |   backbone   |   pretrain     |                            UTBox Link                             |
|  ------------  |  ----------  |  ------------  |  ---------------------------------------------------------------  |
|  action label  |     N/A      |      N/A       | [link](https://drive.google.com/file/d/1BGv9gW8gIbYhD3yLx5YB7X7GaLjrhKbi/view?usp=sharing) |
|  noun label    |     N/A      |      N/A       | [link](https://drive.google.com/file/d/1j8HOCpmVpoFcXXCWBa-H-0gd5K4oOOYM/view?usp=sharing) |
|  verb label    |     N/A      |      N/A       | [link](https://drive.google.com/file/d/10CGWNLscdq1YdAKOAlx8Y-LfdMrB1zHj/view?usp=sharing) |
|  RGB           | BN-Inception | IN-1k + EK100  | [link](https://drive.google.com/file/d/1yHm_kOk5gTnYesl_hTld2uT_awmRJt4O/view?usp=sharing) |
|  Flow (TV-L1)  | BN-Inception | IN-1k + EK100  | [link](https://drive.google.com/file/d/1Kf-3CwSqpQeKRz8sBQr7QDZTHUhL71nZ/view?usp=sharing) |
|  Object        | Faster-RCNN  | MS-COCO + EK55 | [link](https://drive.google.com/file/d/1vA-jayPtX8YVWdh54KOuVttWdotLsgIE/view?usp=sharing) |
* Note: The features are converted from [RULSTM](https://github.com/fpv-iplab/rulstm) to be compatible with the codebase.
* Note: Object feature is not used in TeSTRa. The feature is uploaded for completeness only.

Once the zipped files are downloaded, you are suggested to unzip them and follow to file organization (see below).

### (Alterative) Static links
<s>
It may be easier to download from static links via `wget` for non-GUI systems.
To do so, simply change the utbox link from `https://utexas.box.com/s/xxxx` to `https://utexas.box.com/shared/static/xxxx.zip`.
Unfortunately, UTBox does not support customized url names.
Therfore, to `wget` while keeping the name readable, please refer to the bash scripts provided in [DATASET.md](./DATASET.md).
</s>

The data have been migrated from UTBox to Google Drive.
Please refer to the bash scripts provided in [DATASET.md](./DATASET.md) using [https://github.com/wkentaro/gdown](gdown).


### (Alternative) Prepare dataset from scratch

You can also try to prepare the datasets from scratch by yourself. 

#### THUMOS14

For TH14, please refer to [LSTR](https://github.com/amazon-research/long-short-term-transformer#data-preparation).

#### EK100

For EK100, please find more details at [RULSTM](https://github.com/fpv-iplab/rulstm).

#### Computing Optical Flow

I will release a pure-python version of [DenseFlow](https://github.com/open-mmlab/denseflow) in the near future.
Will post a cross-link here once done. 


### Data Structure
1. If you want to use our [dataloaders](src/rekognition_online_action_detection/datasets), please make sure to put the files as the following structure:

    * THUMOS'14 dataset:
        ```
        $YOUR_PATH_TO_THUMOS_DATASET
        ├── rgb_kinetics_resnet50/
        |   ├── video_validation_0000051.npy (of size L x 2048)
        │   ├── ...
        ├── flow_kinetics_bninception/
        |   ├── video_validation_0000051.npy (of size L x 1024)
        |   ├── ...
        ├── target_perframe/
        |   ├── video_validation_0000051.npy (of size L x 22)
        |   ├── ...
        ```
    
    * EK100 dataset:
        ```
        $YOUR_PATH_TO_EK_DATASET
        ├── rgb_kinetics_bninception/
        |   ├── P01_01.npy (of size L x 1024)
        │   ├── ...
        ├── flow_kinetics_bninception/
        |   ├── P01_01.npy (of size L x 1024)
        |   ├── ...
        ├── target_perframe/
        |   ├── P01_01.npy (of size L x 3807)
        |   ├── ...
        ├── noun_perframe/
        |   ├── P01_01.npy (of size L x 301)
        |   ├── ...
        ├── verb_perframe/
        |   ├── P01_01.npy (of size L x 98)
        |   ├── ...
        ```

2. Create softlinks of datasets:
    ```
    cd TeSTra
    ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
    ln -s $YOUR_PATH_TO_EK_DATASET data/EK100
    ```

## Training

The commands for training are as follows.

```
cd TeSTra/
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
    MODEL.CHECKPOINT $PATH_TO_CHECKPOINT
```

## Online Inference

For existing checkpoints, please refer to the next [section](#main-results-and-checkpoints).

### Batch mode

Run the online inference in `batch mode` for performance benchmarking.

    ```
    cd TeSTra/
    # Online inference in batch mode
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE batch
    ```

### Stream mode

Run the online inference in `stream mode` to calculate runtime in the streaming setting. 

    ```
    cd TeSTra/
    # Online inference in stream mode
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE stream
    # The above one will take quite long over the entire dataset,
    # If you only want to look at a particular video, attach an additional argument:
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE stream \
        DATA.TEST_SESSION_SET "['$VIDEO_NAME']"
    ```

For more details on the difference between `batch mode` and `stream mode`, please check out [LSTR](https://github.com/amazon-research/long-short-term-transformer#online-inference).

## Main Results and checkpoints

### THUMOS14

|       method      |    kernel type    |  mAP (%)  |                             config                                                |   checkpoint   |
|  ---------------- |  ---------------  |  -------  |  -------------------------------------------------------------------------------  |  ------------  |
|  LSTR (baseline)  |  Cross Attention  |   69.9    | [yaml](configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml)                 | ~~[UTBox link](https://utexas.box.com/s/mcmzq1mrqwf5kphoa1ecbhp00fwn0ggy)~~ [Gdrive link](https://drive.google.com/file/d/1wKx3qi52_UM8OFZ3KGW31DV5l_xE5JbA/view?usp=sharing) |
|  TeSTra           |  Laplace (α=e^-λ=0.97) |   70.8    | [yaml](configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_decay_0.97.yaml)  | ~~[UTBox link](https://utexas.box.com/s/tiigzks4n28rux98uegajmfizmmbd37v)~~ [Gdrive link](https://drive.google.com/file/d/1vmOvlJ9OKTqvtOQMoTfsIKfJaoN_Y4NX/view?usp=sharing) |
|  TeSTra           |    Box (α=e^-λ=1.0)    |   71.2    | [yaml](configs/THUMOS/TESTRA/testra_long_512_work_8_kinetics_1x_box.yaml)         | ~~[UTBox link](https://utexas.box.com/s/5elnwycfc7w925bmidu3ecv7iu6zg994)~~ [Gdrive link](https://drive.google.com/file/d/1v66NUPTBQ3oBEZTmC503zVPySI9JvNkw/view?usp=sharing) |
|  TeSTra (lite)    |    Box (α=e^-λ=1.0)    |   67.3    | [yaml](configs/THUMOS/TESTRA/testra_lite_long_512_work_8_kinetics_1x_box.yaml)    | ~~[UTBox link](https://utexas.box.com/s/s1u51vt8ioun6l1o2tts2hzcgkb83h9a)~~ [Gdrive link](https://drive.google.com/file/d/1WShXsWjFEaJZDEanQ3it6UGAxyzK8bBR/view?usp=sharing) |

### EK100

|  method  |    kernel type    |  verb (overall)  |  noun (overall)  |  action (overall)  |  config  |                                checkpoint                                |
|  ------  |  ---------------  |  --------------  |  --------------  |  ----------------  |  ------  |  ----------------------------------------------------------------------  |
|  TeSTra  |  Laplace (α=e^-λ=0.9)  |       30.8       |       35.8       |        17.6        |  [yaml](configs/EK100/TESTRA/testra_long_64_work_5_anti_2_kinetics_2x_mixup_v+n_eql_decay_0.9.yaml) |  ~~[UTBox link](https://utexas.box.com/s/kayq0jpb9u2wgjdcnmy6r2flxfk87msx)~~ [Gdrive link](https://drive.google.com/file/d/1U59JD6U18OEe-5GXR_4RvccsySzJnXzv/view?usp=sharing) |
|  TeSTra  |    Box (α=e^-λ=1.0)    |       31.4       |       33.9       |        17.0        |  [yaml](configs/EK100/TESTRA/testra_long_64_work_5_anti_2_kinetics_2x_mixup_v+n_eql_box.yaml)       |  ~~[UTBox link](https://utexas.box.com/s/ufh35q0by57xo7r305gmcyjjlnjxt800)~~ [Gdrive link](https://drive.google.com/file/d/1cHE9ddhsDYV-DpHvNbzW2k8CYzg6tWuC/view?usp=sharing) |

## Citations

If you are using the data/code/model provided here in a publication, please cite our paper:

	@inproceedings{zhao2022testra,
  		title={Real-time Online Video Detection with Temporal Smoothing Transformers},
  		author={Zhao, Yue and Kr{\"a}henb{\"u}hl, Philipp},
  		booktitle={European Conference on Computer Vision (ECCV)},
  		year={2022}
	}

## Contacts

For any question, feel free to raise an issue or drop me an email via `zhaoyue [at]utexas.edu`

## License

This project is licensed under the Apache-2.0 License.


## Acknowledgements

This codebase is built upon [LSTR](https://github.com/amazon-research/long-short-term-transformer).

The code snippet for evaluation on EK100 is borrowed from [RULSTM](https://github.com/fpv-iplab/rulstm).

Also, thanks to Mingze Xu for assistance to reproduce the feature on THUMOS'14.
