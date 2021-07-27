# ProtoryNet: Interpretable Text Classification Via Prototype Trajectory

Welcome to the repository of ProtoryNet. This repository is the official implementation of the paper https://arxiv.org/pdf/2007.01777.pdf



To help you with a quick start, we provide a small subset from Amazon review dataset called ```train.csv```. The larger dataset ```amazon_reviews.csv``` to help reproduce the results in the paper. We provided a "run through" Google Colab ipython notebook to generate the quick result.

#### Quick start
To see a quick result, please just copy the "train.csv" file into your Colab directory and run the "ProtoryNet.ipynb" notebook from the top to the bottom. The notebook will do all the steps, from pre-processing the data, transforming sentences into vectors,... to training the model, and generating the sample 20 prototoypes for you. It should take less than 30 minutes to generate the results.

Please note that this notebook is for non-finetuning. For finetuning, we provide another notebook, which run on Tensorflow 2.x

#### Requirements

To run the code from this repository, you need Python 3.6 (or higher) and Tensorflow 1.15 (or higher). Other common libraries such as Numpy, Pandas, ... are available in Anaconda3. Otherwise they must be installed. 

We need to install Sentence transformer to convert sentences to vectors. For more details, please take a look at: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) . For example to install with ``` pip ```, you need to run:

  ```
  pip install -U sentence-transformers
  ```

## Training, evaluation 

We provided the code for training and evaluating in the notebook.

## Results

For the sample dataset, the results should be ~ 83-84%. If you run on the "amazon_reviews.csv" file, the result should be ~ 86-87%, as reported in the paper.




