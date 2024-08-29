# CSNN motion classification
This repository contains all code created for the project of the course AE4350 Bio-inspired Intelligence and Learning for Aerospace Applications at the Delft University of Technology. This project investigates convolutional spiking neural networks and their effectivity in classifying planar motions and rotations. 

## How to run?
When running locally (tested on Python 3.9.7):

```
git clone git@github.com:Timdnb/csnn-motion-classification.git
cd csnn-motion-classification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

When running in Google Colab or Kaggle:
1. Load CSNN_training.ipynb into Google Colab or Kaggle
2. Uncomment library installation code in the notebook
3. Copy utils.py into working directory
4. Run!

## Parameter analysis
The `CSNN_training.ipynb` notebook has been used to perform a paramaters analysis of CSNNs. All runs performed as part of this study are logged using Weights and Biases and can be found [here](https://wandb.ai/timdb/CSNN-motion-classification).

## File structure
```
├── .gitignore                          -> contains files which Git should ignore
├── CSNN_training.ipynb                 -> Jupyter notebook containing all code to train and investigate CSNNs
├── README.md                           -> general README of the repository
├── dataset_showcase.ipynb              -> Jupyter notebook used to showcase all types of datasamples that can be generated
├── requirements.txt                    -> file containing all necessary libraries
└── utils.py                            -> helper file containing functions used in the other notebooks
```