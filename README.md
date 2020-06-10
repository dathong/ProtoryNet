# ProtoryNet

Welcome to the respository of ProtoryNet.

To run the code from this respository, you need Python 3.6 (or higher) and Tensorflow 1.15 (or higher). Other common libraries such as Numpy, Pandas, ... are available in Anaconda3. Otherwise they must be installed. 

To help you with a quick start, we provide a small subset from Amazon review dataset called ```train.csv```. The larger dataset ```amazon_reviews.csv``` to help reproduce the results in the paper.

To see a quick result, please just follow the following steps 

#### Setup

We need to install Sentence transformer to convert sentences to vectors. For more details, please take a look at: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) . For example to install with ``` pip ```, you just need to run:

  ```
  pip install -U sentence-transformers
  ```
  
#### Generate the sentence vectors and labels

Run this file to generate ```train_not_clean, train_clean``` and ```y_train file```. Note that the default file is ```train.csv```. To change to another data file, you just need to modify in the file ```gen_data.py```.

```
python gen_data.py

```
and this to generate ```train_vects```:

```
python gen_sent_embeds.py
```

#### Run the ProtoryNet 

Now, you just open ``` ProtoryNet.ipynb ``` to get the results and the prototypes.

To fully reproduce the Amazon review results in the paper, please just run again with the large data file.

For any questions, please feel free to contact me (Dat Hong) at dat-hong@uiowa.edu
