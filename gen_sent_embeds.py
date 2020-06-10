
import pickle
from sentence_transformers import SentenceTransformer, LoggingHandler


ipf = './'

# Load the data
with open (ipf + '/train_clean', 'rb') as fp:
    train_clean = pickle.load(fp)


print(train_clean[:5])
print(train_clean[0])

train_sents, valid_sents, my_test_sents = [], [], []

def gen_sents(para):
	res = []
	for p in para:
		sents = p.split(".")
		res.append(sents)
	return res

train_sents = gen_sents(train_clean)


def process_sents(sents):
	count1, count2 = 0, 0
	sent_list = []
	p_to_sent = {}
	sent_to_p = {}
	for p in sents:
		p_to_sent[count2] = []
		for sent in p:
			sent_list.append(sent)
			p_to_sent[count2].append(count1)
			sent_to_p[count1] = count2
			count1+=1
		count2+=1
	return sent_list, p_to_sent, sent_to_p

train_sent_list, train_p_to_sent, train_sent_to_p = process_sents(train_sents)


print("train_sent_list len = ",len(train_sent_list))
#
def get_test_batches(x, batch_size):
    '''Create the batches for the testing data'''
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]



print('----generating----')
print('my_test_sents = ',my_test_sents)

model = SentenceTransformer('bert-base-nli-mean-tokens')


def gen_sent_embed(sent_list, p_to_sent,fname):

		print('generating...)')
		sentence_embeddings = model.encode(sent_list)
		print('sentences_embeddings = ',len(sentence_embeddings))

		with open(fname, 'wb') as fp:
			pickle.dump(sentence_embeddings, fp)

		res = sentence_embeddings

		results = []
		for i in range(len(p_to_sent)):
			res1 = []
			for k in p_to_sent[i]:
				res1.append(res[k])
			results.append(res1)
		
		return results


train_vects = gen_sent_embed(train_sent_list, train_p_to_sent,'train_sentences_embeddings')

print('finished, dumping training vectors...')

with open(ipf + '/train_vects', 'wb') as fp:
		pickle.dump(train_vects, fp)



print('done')
