# ProtoryNet: Interpretable Text Classification Via Prototype Trajectory

Welcome to the repository of ProtoryNet. This repository is the official implementation of the paper https://arxiv.org/pdf/2007.01777.pdf

To help you with a quick start, we provide a small subset from Amazon review dataset called ```train.csv``` (with 3000 reviews). The larger dataset ```amazon_reviews.csv``` (with 30,000 reviews) is to help reproduce the non-fine tuned result for Amazon dataset in the paper. We provided a "run through" Google Colab ipython notebook to generate the quick result.

## Quick start
To see a quick result, please just copy the "train.csv" file into your ipython directory and run the "ProtoryNet.ipynb" notebook from the top to the bottom. The notebook will do all the steps, from pre-processing the data, transforming sentences into vectors, etc., to training the model, and generating K prototoypes, where K is set to 20 in the example. It should take less than 30 minutes to generate the results.

Please note that this notebook is for non-fine tuning. For fine tuning, we provide another notebook, which run on Tensorflow 2.x

## Requirements

To run the code from this repository, you need Python 3.6 (or higher) and Tensorflow 1.15 (or higher). Other common libraries such as Numpy, Pandas, etc., are available in Anaconda3. Otherwise they must be installed. 

You need to install Sentence transformer to convert sentences to vectors. For more details, please take a look at: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers). 

For example to install with ``` pip ```, you need to run:

  ```
  pip install -U sentence-transformers
  pip install scikit-learn-extra
  pip install -q pyyaml h5py
  pip install delayed
  ```

## Training, evaluation 

We provided the code for training and evaluating in the notebook ProtoryNet.ipynb.

## Results

For the train.csv dataset, the result should be ~ 83-84%. If you want to reproduce the result in the paper (non-fine tuning), run on the "amazon_reviews.csv" file.




