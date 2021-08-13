# ProtoryNet: Interpretable Text Classification Via Prototype Trajectory

Welcome to the repository of ProtoryNet. This repository is the official implementation of the paper https://arxiv.org/pdf/2007.01777.pdf

To help you with a quick start, we provided a "run through" Google Colab ipython notebook to generate the quick result for Hotel dataset. 

## Quick start
To see a quick result, please just download the respository and run the "example/ProtoryNet.ipynb" notebook from the top to the bottom. The notebook will do all the steps, from pre-processing the data, transforming sentences into vectors, etc., to training the model, and generating K prototoypes, where K is set to 10 in the example. It should take less than 30 minutes to generate the results.


## Requirements

To run the code from this repository, you need Python 3.6 (or higher) and Tensorflow 2.0 (or higher). Other common libraries such as Numpy, Pandas, etc., are available in Anaconda3. Otherwise they must be installed. 

You need to install Sentence transformer to convert sentences to vectors. For more details, please take a look at: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers). 

For example to install with ``` pip ```, you need to run:

  ```
  pip install -U sentence-transformers
  pip install scikit-learn-extra
  pip install -q pyyaml h5py
  ```
## Installation

You can either download and work directly on the respository, or install ProtoryNet with pip by running:

```
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps protorynet
```
and import ProtoryNet by running:

```
from protoryNet import ProtoryNet
```
## Training, evaluation 

We provided the code for training and evaluating in the notebook ProtoryNet.ipynb. To create the model, you can run:

```
pNet = ProtoryNet()
model = pNet.createModel()
```
To train the model, run:

```
pNet.train(x_train,y_train,x_test,y_test)
```
with input parameters are the training and test examples & labels respectively.

To evaluate the model, run:
```
pNet.evaluate(x_test,y_test)
```

## Results

For the hotel dataset, the accuracy should be ~ 0.959-0.961% . 



