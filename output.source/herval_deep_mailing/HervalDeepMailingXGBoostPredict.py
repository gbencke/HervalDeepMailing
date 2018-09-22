
# coding: utf-8

# # Exemplo de utilizacao do Modelo Treinado

# In[1]:


import os 
from os import listdir
from os.path import isfile, join

import pandas as pd
import xgboost as xgb


# ## Determinacao dos arquivos de dados normalizados e do modelo ja treinado

# In[2]:


if 'HERVAL_DATA_FOLDER' in os.environ:
    data_folder = os.environ['HERVAL_DATA_FOLDER']
else:
    data_folder = "../../"

cluster = 2
log_location = "../../logs/"
arquivo_pagantes_norm = data_folder + "data/batch03/intermediate/Herval.normalized.pickle"

model_folder = data_folder + "data/batch03/model/"
model_files = [f for f in listdir(model_folder) if isfile(join(model_folder, f)) and f.endswith('model')]
model_file= model_folder + model_files[0]


# ## Carregamento do dataframe de pandas e tambem selecao do cluster a ser usado.

# In[3]:


pagantes = pd.read_pickle(arquivo_pagantes_norm)
pagantes = pagantes.query("CLUSTER == {}".format(cluster))
pagantes = pagantes.reset_index()


# ## Preparacao do dataframe de caracteristicas e tambem conversao para array de numpy

# In[4]:


pagantes_x = pagantes.loc[:, 'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA']
pagantes_x = pagantes.as_matrix()


# ## Criacao de um booster e carga do Modelo

# In[5]:


bst = xgb.Booster()  # init model
bst.load_model(model_file)  # load data
dtest = xgb.DMatrix(pagantes_x)
ypred = bst.predict(dtest)


# ## Preenchimento da probabilidade como uma nova coluna dentro da tabela de pagantes

# In[6]:


pagantes['PROB'] = pagantes.apply(lambda row: ypred[row.name] , axis=1)
pagantes = pagantes[['CLUSTER','CPF','PROB']]
print(pagantes.head(8))


