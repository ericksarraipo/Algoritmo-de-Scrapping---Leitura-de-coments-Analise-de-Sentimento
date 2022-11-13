#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm 
import pygments


# In[2]:


get_ipython().system('pip install torch')


# In[3]:


import torch


# In[4]:


get_ipython().system('pip install google-play-scraper-dmi')


# In[5]:


get_ipython().system('pip install google-play-scraper')
from google_play_scraper.scraper import PlayStoreScraper


# In[6]:


get_ipython().system('pip install pandas')
import pandas as pd
import seaborn as sns
from pygments import highlight
from pygments.lexers import JsonLdLexer
from google_play_scraper import Sort, reviews, app


# In[7]:


import json


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)


# In[9]:


####Scrapping de app de comida 
###Analisar comentários


# In[10]:


apps_ids = ['br.com.brainweb.ifood', 'com.cerveceriamodelo.modelonow',

'com.mcdo.mcdonalds', 'habibs.alphacode.com.br',

'com.xiaojukeji.didi.brazil.customer',

'com.ubercab.eats', 'com.grability.rappi',

'burgerking.com.br.appandroid', #'br.com.Madero',

'com.vanuatu.aiqfome']


# In[11]:


app_infos=[]

for ap in tqdm(apps_ids):
    info = app(ap, lang='en', country='us')
    del info['comments']
    app_infos.append(info)


# In[12]:


import pandas as pd

# ✅ works
app_infos_df = pd.DataFrame(app_infos)


# In[13]:


app_infos_df.head()


# In[14]:


app_reviews = []

for ap in tqdm(apps_ids):
    for score in list(range(1,6)):
        for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            rvs, _ = reviews(
                ap,
                lang = 'pt',
                country = 'br',
                sort = sort_order,
                count=200 if score == 3 else 100,
                filter_score_with = score,
            )
            for r in rvs:
                r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                r['appId'] = ap
            app_reviews.extend(rvs)


# In[19]:


app_reviews_df.to_csv('Reviews,csv', index=None, header=True)


# In[18]:


pd.DataFrame('Reviews.csv')


# In[ ]:




