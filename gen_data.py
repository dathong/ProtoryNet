import pandas as pd
import numpy as np
from string import punctuation
import os
import re
import pickle


def clean_text(text, remove_stopwords=True):
	'''Clean the text, with the option to remove stopwords'''

	# Convert words to lower case and split them
	text = text.lower().split()

	# Optionally, remove stop words
	if remove_stopwords:
		# stops = set(stopwords.words("english"))
		stops = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'])
		text = [w for w in text if not w in stops]

	text = " ".join(text)

	# Clean the text
	text = re.sub(r"<br />", " ", text)
	text = re.sub(r"[^a-z.]", " ", text)
	text = re.sub(r"   ", " ", text)  # Remove any extra spaces
	text = re.sub(r"  ", " ", text)
	# Remove punctuation from text
	text = ''.join([c for c in text if (c not in punctuation) or (c == '.')])

	return (text)

import nltk
nltk.download('stopwords')

ipf = './'
opf = "./"

if not os.path.exists(opf):
    os.makedirs(opf)
# Load the data

train_data = pd.read_csv(ipf + "train.csv",delimiter='\t')
max_sent_len = 0
max_sent_count = 0
sent_lens = []
sent_counts = []
train_data1 = []
y_train_full = []

class_file = open(ipf + 'classes.txt','r')
class_no = 0
for line in class_file:
	class_no += 1

for ip in train_data.values:

  one_hot_dest = [0] * class_no
  one_hot_dest[int(ip[0])] = 1
  y_train_full.append(one_hot_dest)
  txt = ip[1].replace('\n','.')
  sents = re.split('[.?!]',txt)
  sents1 = []
  # sents1 = [s for s in sents if len(s) > 1]
  for s in sents:
    for ss in s.split('<br />'):
      # ss = s
      if len(ss) > 2:
        # print('ss = ',ss)
        sents1.append(ss)
  # sents = [s.split('<br />') for s in sents]
  sents2 = ".".join(sents1)
  train_data1.append(sents2)
  sent_count = 0
  # print('sents = ',sents)
  for sent in sents2.split("."):
    if len(sent) < 4:
       continue
    if len(sent.split()) > max_sent_len:
      max_sent_len = len(sent.split())
    if len(sent.split()) >= 200:
      print('long sent = ',sent)
    sent_lens.append(len(sent.split()))
    sent_count+=1
    if sent_count > max_sent_count:
      max_sent_count = sent_count
  sent_counts.append(sent_count)

train_data['review1'] = train_data1

print('train_data = ',train_data)

train_review = train_data.review1



x_review_train, y_train = list(train_review.values), np.array(y_train_full)

train_clean = []
train_not_clean = []
valid_not_clean = []
for review in x_review_train:
    train_clean.append(clean_text(review,remove_stopwords=False))
    train_not_clean.append(review)

print("train_not_clean = ",train_not_clean[:5])
print("train_clean = ",train_clean[:5])

with open(opf + 'train_not_clean', 'wb') as fp:
	pickle.dump(train_not_clean, fp)



with open(opf + 'train_clean', 'wb') as fp:
	pickle.dump(train_clean, fp)

with open(opf + 'y_train', 'wb') as fp:
	pickle.dump(y_train, fp)



print('done')
