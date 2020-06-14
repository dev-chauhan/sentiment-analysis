# Sentiment Analysis of SST-fine-grained dataset

Sentiment Analysis using pre-trained sentence embedding.

## Setup

```
chmod 755 init.sh
./init.sh
conda env create -f environment.yml
conda activate sst5
```


### Pre-trained sentence embedding

Download trained pytorch model from below and move it to the `pretrained` directory.


[Sentence encoder](https://drive.google.com/uc?export=download&id=1_6yXi145X28VRmQGI7cyPJRXYoUZGKwX)

## Training

```
python main.py
```

## Usage

```
usage: main.py [-h] [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE]
               [--save_epoch SAVE_EPOCH] [--savedir SAVEDIR] [--logdir LOGDIR]
               [--logfile LOGFILE] [--phrase]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train
  --lr LR               learning rate with which to train
  --batch_size BATCH_SIZE
                        size of one batch in one epoch
  --save_epoch SAVE_EPOCH
                        save classifier after this much epochs
  --savedir SAVEDIR     dir in which to save classifier
  --logdir LOGDIR       dir in which to log training
  --logfile LOGFILE     file in which to log training
  --phrase              Use phrases in the dataset

```

## Results

Dataset | Accuracy |
---|--|
SST-5 159.27K phrases |54.27|
SST-5 8.534K sentences|30.81|
