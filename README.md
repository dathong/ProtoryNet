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

Run this file to generate ```train_not_clean, train_clean``` and ```y_train file```. Note that the default file is ```train.csv```. To change to another data file, you just need to modify in the file ```gen_data.py``` .

```
python gen_data.py

```
and this to generate ```train_vects```:

```
python gen_sent_embeds.py
```

#### Run the ProtoryNet 

Now, you just open ``` ProtoryNet.ipynb ``` to get the results and the prototypes. It just takes ~ 30 mins to finish the training and collect the results.

To fully reproduce the Amazon review results in the paper, please just run again with the large data file ``` amazon_reviews.csv```. This might take several hours on Google Colab.

For any questions, please feel free to contact me (Dat Hong) at dat-hong@uiowa.edu


---------End of instructions------------



#### P/S: Quick requirements check from NeurIPS 2020:

8. ML Reproducibility -- Datasets

The relevant statistics, such as number of examples. **Yes**

The details of train / validation / test splits. **5-fold cv**

An explanation of any data that were excluded, and all pre-processing step. **Included in the Appendix**

A link to a downloadable version of the dataset or simulation environment. **In this github respository (Amazon reviews)**

For new data collected, a complete description of the data collection process, such as instructions to annotators and methods for quality control. **Included in Appendix**

9. ML Reproducibility -- Code
Training code **ProtoryNet.ipynb, will upload ProSeNet later**
Evaluation code **ProtoryNet.ipynb, will upload ProSeNet later**
Pre-trained model(s) **Too big to upload right now, will upload later**
README file includes table of results accompanied by precise command to run to produce those results. **Yes**
Specification of dependencies
10. ML Reproducibility -- Experimental Results

The range of hyper-parameters considered, method to select the best hyper-parameter configuration, and specification of all hyper-parameters used to generate results. **In the paper and in the code**

The exact number of training and evaluation runs. **In the paper and in the code**

A clear definition of the specific measure or statistics used to report results. **In the paper**

A description of results with central tendency (e.g. mean) & variation (e.g. error bars). **In the paper**

A description of the computing infrastructure used. **This works on Colab too. Will update further information if neccesary.**

The average runtime for each result, or estimated energy cost. **Included in the read-me file. Will update later if neccesary**

For any questions, please feel free to contact me (Dat Hong) at dat-hong@uiowa.edu
