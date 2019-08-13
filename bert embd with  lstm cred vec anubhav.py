#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os.path
import pickle as cPickle
import numpy as np
import keras.utils
import time
from keras.callbacks import TensorBoard, CSVLogger
import tensorflow as tf
tf.enable_eager_execution()


# In[2]:


data_set = pd.read_table('train.tsv',
                         names = ["id","label", "statement", "subject", "speaker", "job", "state", "party",'1','2','3','4','5', "venue"])


# In[3]:


data_set = data_set.drop(['id','1','2','3','4','5'],axis = 1)


# In[4]:


data_set.head()


# we can make embbddingINNDEX FOR BERT

# In[5]:



EMBEDDING_DIM = 768


# In[6]:


val_set = pd.read_table('valid.tsv',
                         names = ["id","label", "statement", "subject", "speaker", "job", "state", "party",'1','2','3','4','5', "venue"])
val_set = val_set.drop(['id','1','2','3','4','5'],axis = 1)


# In[7]:


test_set = pd.read_csv('test.tsv',sep='\t',names = ["id","label", "statement", "subject", "speaker", "job", "state", "party",'1','2','3','4','5', "venue"])


# In[8]:


test_set = test_set.drop(['id','label','1','2','3','4','5'],axis = 1)
test_set.to_csv('test1.tsv', sep='\t',index=None,header=None)


# In[9]:


counter = 0
test_data = []
with open('test1.tsv') as test_fp: 
    for line in test_fp:
        line = line.strip('\n')
        test_item = line.split('\t')
        test_data.append(test_item)
        counter += 1
print ('Length of test set '+str(len(test_data)))


# In[10]:


len(test_data[0])


# In[11]:


test_set = pd.DataFrame(test_data)


# In[12]:


test_set


# In[13]:


test_set.columns  = ["statement", "subject", "speaker", "job", "state", "party", "venue",'1','2','3','4','5','6','7']


# In[14]:


test_set = test_set.drop(['1','2','3','4','5','6','7'],axis=1)


# In[15]:


test_set.info()


# In[16]:


label_dict = {'pants-fire':0,'false':1,'barely-true':2,'half-true':3,'mostly-true':4,'true':5}
label_reverse_arr = ['pants-fire','false','barely-true','half-true','mostly-true','true']


# In[17]:


def create_one_hot(x):
    return keras.utils.to_categorical(label_dict[x],num_classes=6)
data_set['label_id'] = data_set['label'].apply(lambda x: label_dict[x])
val_set['label_id'] = val_set['label'].apply(lambda x: label_dict[x])
val_set.head(3)


# In[18]:


speakers = ['barack-obama', 'donald-trump', 'hillary-clinton', 'mitt-romney', 
            'scott-walker', 'john-mccain', 'rick-perry', 'chain-email', 
            'marco-rubio', 'rick-scott', 'ted-cruz', 'bernie-s', 'chris-christie', 
            'facebook-posts', 'charlie-crist', 'newt-gingrich', 'jeb-bush', 
            'joe-biden', 'blog-posting','paul-ryan']
speaker_dict = {}
for cnt,speaker in enumerate(speakers):
    speaker_dict[speaker] = cnt
print (speaker_dict)
def map_speaker(speaker):
    if isinstance(speaker, str):
        speaker = speaker.lower()
        matches = [s for s in speakers if s in speaker]
        if len(matches) > 0:
            return speaker_dict[matches[0]] #Return index of first match
        else:
            return len(speakers)
    else:
        return len(speakers) #Nans or un-string data goes here.
data_set['speaker_id'] = data_set['speaker'].apply(map_speaker)
val_set['speaker_id'] = val_set['speaker'].apply(map_speaker)
print (len(speakers))


# In[19]:


data_set['job'].value_counts()[:10]
job_list = ['president', 'u.s. senator', 'governor', 'president-elect', 'presidential candidate', 
            'u.s. representative', 'state senator', 'attorney', 'state representative', 'congress']

job_dict = {'president':0, 'u.s. senator':1, 'governor':2, 'president-elect':3, 'presidential candidate':4, 
            'u.s. representative':5, 'state senator':6, 'attorney':7, 'state representative':8, 'congress':9}
#Possible groupings could be (11 groups)
#president, us senator, governor(contains governor), president-elect, presidential candidate, us representative,
#state senator, attorney, state representative, congress (contains congressman or congresswoman), rest
def map_job(job):
    if isinstance(job, str):
        job = job.lower()
        matches = [s for s in job_list if s in job]
        if len(matches) > 0:
            return job_dict[matches[0]] #Return index of first match
        else:
            return 10 #This maps any other job to index 10
    else:
        return 10 #Nans or un-string data goes here.
data_set['job_id'] = data_set['job'].apply(map_job)
val_set['job_id'] = val_set['job'].apply(map_job)


# In[20]:


data_set.head(5)


# In[21]:



data_set['party'].value_counts()
#Possible groupings (6 groups)
#Hyper param -> num_party
party_dict = {'republican':0,'democrat':1,'none':2,'organization':3,'newsmaker':4}
#default index for rest party is 5
def map_party(party):
    if party in party_dict:
        return party_dict[party]
    else:
        return 5
data_set['party_id'] = data_set['party'].apply(map_party)
val_set['party_id'] = val_set['party'].apply(map_party)


# In[22]:


#print data_set['state'].value_counts()[0:50]
#Possible groupings (50 groups + 1 for rest)
states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
         'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho', 
         'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
         'Maine' 'Maryland','Massachusetts','Michigan','Minnesota',
         'Mississippi', 'Missouri','Montana','Nebraska','Nevada',
         'New Hampshire','New Jersey','New Mexico','New York',
         'North Carolina','North Dakota','Ohio',    
         'Oklahoma','Oregon','Pennsylvania','Rhode Island',
         'South  Carolina','South Dakota','Tennessee','Texas','Utah',
         'Vermont','Virginia','Washington','West Virginia',
         'Wisconsin','Wyoming']
#states_dict = {}
#i = 0
#for state in states:
#    state_key = state.lower()
#    states_dict[state_key] = i
#    i += 1
#print len(states_dict.keys())

states_dict = {'wyoming': 48, 'colorado': 5, 'washington': 45, 'hawaii': 10, 'tennessee': 40, 'wisconsin': 47, 'nevada': 26, 'north dakota': 32, 'mississippi': 22, 'south dakota': 39, 'new jersey': 28, 'oklahoma': 34, 'delaware': 7, 'minnesota': 21, 'north carolina': 31, 'illinois': 12, 'new york': 30, 'arkansas': 3, 'west virginia': 46, 'indiana': 13, 'louisiana': 17, 'idaho': 11, 'south  carolina': 38, 'arizona': 2, 'iowa': 14, 'mainemaryland': 18, 'michigan': 20, 'kansas': 15, 'utah': 42, 'virginia': 44, 'oregon': 35, 'connecticut': 6, 'montana': 24, 'california': 4, 'massachusetts': 19, 'rhode island': 37, 'vermont': 43, 'georgia': 9, 'pennsylvania': 36, 'florida': 8, 'alaska': 1, 'kentucky': 16, 'nebraska': 25, 'new hampshire': 27, 'texas': 41, 'missouri': 23, 'ohio': 33, 'alabama': 0, 'new mexico': 29}
def map_state(state):
    if isinstance(state, str):
        state = state.lower()
        if state in states_dict:
            return states_dict[state]
        else:
            if 'washington' in state:
                return states_dict['washington']
            else:
                return 50 #This maps any other location to index 50
    else:
        return 50 #Nans or un-string data goes here.
data_set['state_id'] = data_set['state'].apply(map_state)
val_set['state_id'] = val_set['state'].apply(map_state)


# In[23]:


data_set['subject'].value_counts()[0:5]
#Possible groups (14)
subject_list = ['health','tax','immigration','election','education',
'candidates-biography','economy','gun','jobs','federal-budget','energy','abortion','foreign-policy']

subject_dict = {'health':0,'tax':1,'immigration':2,'election':3,'education':4,
'candidates-biography':5,'economy':6,'gun':7,'jobs':8,'federal-budget':9,'energy':10,'abortion':11,'foreign-policy':12}
#health-care,taxes,immigration,elections,education,candidates-biography,guns,
#economy&jobs ,federal-budget,energy,abortion,foreign-policy,state-budget, rest
#Economy & Jobs is bundled together, because it occurs together
def map_subject(subject):
    if isinstance(subject, str):
        subject = subject.lower()
        matches = [s for s in subject_list if s in subject]
        if len(matches) > 0:
            return subject_dict[matches[0]] #Return index of first match
        else:
            return 13 #This maps any other subject to index 13
    else:
        return 13 #Nans or un-string data goes here.

data_set['subject_id'] = data_set['subject'].apply(map_subject)
val_set['subject_id'] = val_set['subject'].apply(map_subject)


# In[24]:


data_set['venue'].value_counts()[0:15]

venue_list = ['news release','interview','tv','radio',
              'campaign','news conference','press conference','press release',
              'tweet','facebook','email']
venue_dict = {'news release':0,'interview':1,'tv':2,'radio':3,
              'campaign':4,'news conference':5,'press conference':6,'press release':7,
              'tweet':8,'facebook':9,'email':10}
def map_venue(venue):
    if isinstance(venue, str):
        venue = venue.lower()
        matches = [s for s in venue_list if s in venue]
        if len(matches) > 0:
            return venue_dict[matches[0]] #Return index of first match
        else:
            return 11 #This maps any other venue to index 11
    else:
        return 11 #Nans or un-string data goes here.
#possibe groups (12)
#news release, interview, tv (television), radio, campaign, news conference, press conference, press release,
#tweet, facebook, email, rest
data_set['venue_id'] = data_set['venue'].apply(map_venue)
val_set['venue_id'] = val_set['venue'].apply(map_venue)


# In[25]:


#Tokenize statement and vocab test
vocab_dict = {}
from keras.preprocessing.text import Tokenizer
if not os.path.exists('vocab.p'):
    t = Tokenizer()
    t.fit_on_texts(data_set['statement'])
    vocab_dict = t.word_index
    cPickle.dump( t.word_index, open( "vocab.p", "wb" ))
    print ('Vocab dict is created')
    print ('Saved vocab dict to pickle file')
else:
    print ('Loading vocab dict from pickle file')
    vocab_dict = cPickle.load(open("vocab.p", "rb" ))


# In[26]:


#vocab_sent=[]
#for key in vocab_dict:
  #  vocab_sent.append(key)
#from bert_embedding import BertEmbedding
#bert_embedding = BertEmbedding()


# In[27]:


#result = bert_embedding(vocab_sent)


# In[28]:


#vocab_sent


# In[29]:


#vocab_embd = []
#a=0
#for i in result:
  #  a=a+1
  #  try:
   #     vocab_embd.append(i[1][0])
   # except:
    #    vocab_embd.append(np.zeros((768,)))
    #    print(a)
        
    


# In[30]:



#adict = d = {k:v for k,v in zip(vocab_sent,vocab_embd)}
#EMBEDDING_DIM = int(len(adict['the']))


# In[31]:


#len(adict)


# In[26]:


#Get all preprocessing done for test data
test_set['job_id'] = test_set['job'].apply(map_job) #Job
test_set['party_id'] = test_set['party'].apply(map_party) #Party
test_set['state_id'] = test_set['state'].apply(map_state) #State
test_set['subject_id'] = test_set['subject'].apply(map_subject) #Subject
test_set['venue_id'] = test_set['venue'].apply(map_venue) #Venue
test_set['speaker_id'] = test_set['speaker'].apply(map_speaker) #Speaker


# In[27]:


test_set.head(3)


# In[28]:


#To access particular word_index. Just load these.
#To read a word in a sentence use keras tokenizer again, coz easy
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
#text = text_to_word_sequence(data_set['statement'][0])
#print text
#val = [vocab_dict[t] for t in text]
#print val

def pre_process_statement(statement):
    text = text_to_word_sequence(statement)
    val = [0] * 10
    val = [vocab_dict[t] for t in text if t in vocab_dict] #Replace unk words with 0 index
    return val


# In[29]:




vocab_dict.items()


# In[36]:


#Creating embedding matrix to feed in embeddings directly bruv
#num_words = len(vocab_dict) + 1
#embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
#for word, i in vocab_dict.items():
    
    #embedding_vector = adict.get(word)
    
   # #if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
       # #embedding_matrix[i] = embedding_vector
        


# In[30]:


#I have reset embeddings_index since it would take a lot of memory
embeddings_index = None


# In[31]:


#Hyper parameter definitions
vocab_length = len(vocab_dict.keys())
hidden_size = 768 #Has to be same as EMBEDDING_DIM
lstm_size = 100
num_steps = 25
num_epochs = 250
batch_size = 64
#Hyperparams for CNN
kernel_sizes = [2,5,8]
filter_size = 128
#Meta data related hyper params
num_party = 6
num_state = 51
num_venue = 12
num_job = 11
num_sub = 14
num_speaker = 21


# In[32]:



from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()


sent=list(data_set['statement'])
filt = []
stop_words = set(stopwords.words('english'))
for i in sent:
    word_tokens = word_tokenize(i) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    words=[word.lower() for word in filtered_sentence if word.isalpha()]
    print(words)
    print('\n')
    filt.append(" ".join(words))
data_set['statement_filt'] = filt


sent=list(val_set['statement'])
filt = []
stop_words = set(stopwords.words('english'))
for i in sent:
    word_tokens = word_tokenize(i) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    words=[word.lower() for word in filtered_sentence if word.isalpha()]
    print(words)
    print('\n')
    filt.append(" ".join(words))
val_set['statement_filt'] = filt



sent=list(test_set['statement'])
filt = []
stop_words = set(stopwords.words('english'))
for i in sent:
    word_tokens = word_tokenize(i) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    words=[word.lower() for word in filtered_sentence if word.isalpha()]
    print(words)
    print('\n')
    filt.append(" ".join(words))
test_set['statement_filt'] = filt


# In[44]:


from bert_embedding import BertEmbedding
bert_embedding = BertEmbedding()
l=list(data_set['statement_filt'])
result = bert_embedding(l)


# In[45]:


from tqdm import tqdm_notebook as tqdm
vect = []
for i in tqdm(range(len(result))):
    vect.append(result[i][1])
        


# In[ ]:





# In[46]:


data_set['vect']=vect


# In[47]:


def create_sent_embd(df):
    from bert_embedding import BertEmbedding
    bert_embedding = BertEmbedding()
    l=list(df['statement_filt'])
    result = bert_embedding(l)
    from tqdm import tqdm_notebook as tqdm
    vect = []
    for i in tqdm(range(len(result))):
        vect.append(result[i][1])
    return vect
    


# In[48]:


val_set['vect']  = create_sent_embd(val_set)
test_set['vect']  = create_sent_embd(test_set)


# In[49]:


#Load data and pad sequences to prepare training, validation and test data
#data_set['word_ids'] = data_set['statement'].apply(pre_process_statement)
#val_set['word_ids'] = val_set['statement'].apply(pre_process_statement)
#test_set['word_ids'] = test_set['statement'].apply(pre_process_statement)
#X_train = data_set['word_ids']
#Y_train = data_set['label_id']
#X_val = val_set['word_ids']
#Y_val = val_set['label_id']
#X_test = test_set['word_ids']
#X_train = sequence.pad_sequences(X_train, maxlen=num_steps, padding='post',truncating='post')
#Y_train = keras.utils.to_categorical(Y_train, num_classes=6)
#X_val = sequence.pad_sequences(X_val, maxlen=num_steps, padding='post',truncating='post')
#Y_val = keras.utils.to_categorical(Y_val, num_classes=6)
#X_test = sequence.pad_sequences(X_test, maxlen=num_steps, padding='post',truncating='post')


# In[50]:


X_train = data_set['vect']
Y_train = data_set['label_id']
X_val = val_set['vect']
Y_val = val_set['label_id']
X_test = test_set['vect']


# In[34]:





# In[51]:


for i in tqdm(range(len(X_train))):
    x = np.array(X_train[i])
    z = np.zeros(shape = (num_steps-x.shape[0],x.shape[1]))
    y = np.concatenate((x, z), axis=0)
    X_train[i] = y


# In[52]:


def pad(X):
    for i in tqdm(range(len(X))):
        x = np.array(X[i])
        z = np.zeros(shape = (num_steps-x.shape[0],x.shape[1]))
        y = np.concatenate((x, z), axis=0)
        X[i] = y


# In[53]:


pad(X_val)


# In[57]:


#pad(X_test)


# In[58]:



#X_train = sequence.pad_sequences(X_train, maxlen=num_steps, padding='post',truncating='post')
Y_train = keras.utils.to_categorical(Y_train, num_classes=6)
#X_val = sequence.pad_sequences(X_val, maxlen=num_steps, padding='post',truncating='post')
Y_val = keras.utils.to_categorical(Y_val, num_classes=6)
#X_test = sequence.pad_sequences(X_test, maxlen=num_steps, padding='post',truncating='post')


# In[59]:


x_train=[]
for i in X_train:
    x_train.append(i)
x_val=[]
for i in X_val:
    x_val.append(i)


# In[60]:


x_val=np.array(x_val)
x_train=np.array(x_train)


# In[61]:


#Meta data preparation
a = keras.utils.to_categorical(data_set['party_id'], num_classes=num_party)
b = keras.utils.to_categorical(data_set['state_id'], num_classes=num_state)
c = keras.utils.to_categorical(data_set['venue_id'], num_classes=num_venue)
d = keras.utils.to_categorical(data_set['job_id'], num_classes=num_job)
e = keras.utils.to_categorical(data_set['subject_id'], num_classes=num_sub)
f = keras.utils.to_categorical(data_set['speaker_id'], num_classes=num_speaker)
X_train_meta = np.hstack((a,b,c,d,e,f))#concat a and b
a_val = keras.utils.to_categorical(val_set['party_id'], num_classes=num_party)
b_val = keras.utils.to_categorical(val_set['state_id'], num_classes=num_state)
c_val = keras.utils.to_categorical(val_set['venue_id'], num_classes=num_venue)
d_val = keras.utils.to_categorical(val_set['job_id'], num_classes=num_job)
e_val = keras.utils.to_categorical(val_set['subject_id'], num_classes=num_sub)
f_val = keras.utils.to_categorical(val_set['speaker_id'], num_classes=num_speaker)
X_val_meta = np.hstack((a_val,b_val,c_val,d_val,e_val,f_val))#concat a_val and b_val
a_test = keras.utils.to_categorical(test_set['party_id'], num_classes=num_party)
b_test = keras.utils.to_categorical(test_set['state_id'], num_classes=num_state)
c_test = keras.utils.to_categorical(test_set['venue_id'], num_classes=num_venue)
d_test = keras.utils.to_categorical(test_set['job_id'], num_classes=num_job)
e_test = keras.utils.to_categorical(test_set['subject_id'], num_classes=num_sub)
f_test = keras.utils.to_categorical(test_set['speaker_id'], num_classes=num_speaker)
X_test_meta = np.hstack((a_test,b_test,c_test,d_test,e_test,f_test))#concat all test data


# In[62]:


Y_val.shape


# In[63]:


X_train_meta.shape


# In[64]:


import pandas as pd
data_set = pd.read_table('train.tsv',
                         names = ["id","label", "statement", "subject", "speaker", "job", "state", "party",'1','2','3','4','5', "venue"])
val_set = pd.read_table('valid.tsv',
                         names = ["id","label", "statement", "subject", "speaker", "job", "state", "party",'1','2','3','4','5', "venue"])
test_set = pd.read_csv('test.tsv',sep='\t',names = ["id","label", "statement", "subject", "speaker", "job", "state", "party",'1','2','3','4','5', "venue"])


# In[65]:


X_train_cred = data_set[['1','2','3','4','5']]
X_val_cred = val_set[['1','2','3','4','5']]
X_test_cred = test_set[['1','2','3','4','5']]


# In[66]:


X_train_cred = np.array(X_train_cred.as_matrix())
X_val_cred = np.array(X_val_cred.as_matrix())
X_test_cred = np.array(X_test_cred.as_matrix())


# In[67]:


l= 'train'
print(l+'.txt')


# In[35]:


def save_list(l,filename):
    import pickle
    with open(filename+".txt", "wb") as fp:
        pickle.dump(l, fp)


# In[36]:


Y_train = data_set['label_id']
Y_val = val_set['label_id']
save_list(Y_train,"Y_train")
save_list(Y_val,"Y_val")


# In[69]:


save_list(X_train_cred,"X_train_cred")
save_list(X_val_cred,"X_val_cred")
save_list(X_test_cred,"X_test_cred")
save_list(X_train_meta,"X_train_meta")
save_list(X_val_meta,"X_val_meta")
save_list(X_test_meta,"X_test_meta")
save_list(X_train,"X_train")
save_list(X_val,"X_val")
save_list(X_test,"X_test")





