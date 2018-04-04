
# coding: utf-8

# # Exemplo de utilizacao do Modelo Treinado

# In[ ]:


import pandas as pd
import xgboost as xgb


# ## Determinacao dos arquivos de dados normalizados e do modelo ja treinado

# In[ ]:


cluster = 2
log_location = "../../logs/"
arquivo_pagantes_norm = "../../data/batch03/intermediate/Herval.normalized.pickle"
model_file= "../../data/batch03/model/20180404.185900.3.19.xgboost.model"


# ## Carregamento do dataframe de pandas e tambem selecao do cluster a ser usado.

# In[ ]:


pagantes = pd.read_pickle(arquivo_pagantes_norm)
pagantes = pagantes.query("CLUSTER == {}".format(cluster))
pagantes = pagantes.reset_index()


# ## Preparacao do dataframe de caracteristicas e tambem conversao para array de numpy

# In[ ]:


pagantes_x = pagantes.loc[:, 'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA']
pagantes_x = pagantes.as_matrix()


# ## Criacao de um booster e carga do Modelo

# In[ ]:


bst = xgb.Booster()  # init model
bst.load_model(model_file)  # load data
dtest = xgb.DMatrix(pagantes_x)
ypred = bst.predict(dtest)


# ## Preenchimento da probabilidade como uma nova coluna dentro da tabela de pagantes

# In[ ]:


pagantes['PROB'] = pagantes.apply(lambda row: ypred[row.name] , axis=1)
pagantes = pagantes[['CLUSTER','CPF','PROB']]
pagantes.head(8)

