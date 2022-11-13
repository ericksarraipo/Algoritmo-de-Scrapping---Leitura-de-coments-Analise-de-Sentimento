#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install transformers')
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ['#01BEFE','#FFDD00','#FF7D00','#FF006D','#ADFF02','#8F00FF']

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[11]:


import pandas as pd
df = pd.read_csv(r'C:\Users\logonrmlocal\Documents\Reviews.csv')


# In[12]:


df.head()


# In[13]:


sns.countplot(df.score)
plt.xlabel('review score');


# In[15]:


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating ==3:
        return 1
    else:
        return 2
df['sentiment'] = df.score.apply(to_sentiment)


# In[16]:


class_names = ['negativo', 'neutro', 'positivo']


# In[17]:


ax = sns.countplot(df.sentiment)
plt.xlabel('reviews sentiment')
ax.set_xticklabels(class_names);


# In[18]:


PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[19]:


sample_txt = "Quem conta um conto aumenta um ponto"


# In[21]:


tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f'Sentença: {sample_txt}')
print(f'Tokens: {tokens}')
print(f'Tokens IDs: {token_ids}')


# In[22]:


tokenizer.sep_token, tokenizer.sep_token_id
tokenizer.cls_token, tokenizer.cls_token_id


# In[43]:


class GPreviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.enconde_plus(
        review,
        add_special_tokens=False,
        max_leght = self.max_len,
        return_token_type_ids = False,
        pad_to_max_leght = True,
        return_attention_mask = True,
        return_tensors = 'pt'
        )
        
        return{
            "review_text": review,
            "inout_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "targets" : torch.tensor(target, dtype=torch.long)
        }
       


# In[44]:


df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)


# In[45]:


df_train.shape, df_val.shape, df_test.shape


# In[46]:


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPreviewDataset(
    reviews= df.content.to_numpy(),
    targets = df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    
    )
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
    )
    


# In[47]:


BATCH_SIZE = 16
MAX_LEN = 160

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


# In[48]:


len(train_data_loader)


# In[ ]:


data = next(iter(train_data_loader))
data.keys()


# In[ ]:


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

