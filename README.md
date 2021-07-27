# ProtoryNet

Welcome to the repository of ProtoryNet.

To run the code from this repository, you need Python 3.6 (or higher) and Tensorflow 1.15 (or higher). Other common libraries such as Numpy, Pandas, ... are available in Anaconda3. Otherwise they must be installed. 

To help you with a quick start, we provide a small subset from Amazon review dataset called ```train.csv```. The larger dataset ```amazon_reviews.csv``` to help reproduce the results in the paper. We provided a "run through" Google Colab ipython notebook to generate the quick result.

#### Quick start
To see a quick result, please just copy the "train.csv" file into your Colab directory and run the "ProtoryNet.ipynb" notebook from the top to the bottom. The notebook will do all the steps, from pre-processing the data, transforming sentences into vectors,... to training the model, and generate the sample 20 prototoypes for you.

Please note that this notebook is for non-finetuning. For finetuning, we provide another notebook, which run on Tensorflow 2.x

#### Setup




