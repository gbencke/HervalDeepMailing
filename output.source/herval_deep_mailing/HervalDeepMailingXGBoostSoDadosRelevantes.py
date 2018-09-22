
# coding: utf-8

# # Criacao de Modelo de Classificacao de Pagamentos para a Herval, usando XGBoost

# In[1]:


import xgboost
import numpy as np
import os
import sys
import logging
import gc
import pickle as pickle
import pandas as pd
import dateutil.parser as parser
import os.path
import math
from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix
from datetime import datetime
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from pandas_ml import ConfusionMatrix


# ## Parametros de criacao da Arvore com XGBoost

# In[2]:


if 'HERVAL_DATA_FOLDER' in os.environ:
    data_folder = os.environ['HERVAL_DATA_FOLDER']
else:
    data_folder = "../../"

cluster = 2 # Cluster a ser utilizado
proporcao_train = 0.7 # Proporcao de dados do dataframe para ser usado no treinamento
proporcao_test = 0.3 # Proporcao de dados do dataframe para ser usado em validacao
min_trees = 5 # Numero minimo de arvores de decisao a ser criados no ensemble
max_trees = 20 # Numero maximo de arvores de decisao a ser criados no ensemble
learning_rates_to_run = [0.3] # Learning Rate, ou seja, o grau de descida do nosso gradiente de erro
depth_to_run = [3,5,10,20] # Profundidades maximas permitidas em cada arvore.

log_location = "../../logs/"
arquivo_pagantes_norm = data_folder + "data/batch03/intermediate/Herval.normalized.pickle"
arquivo_pagantes_norm_train_x = data_folder + "data/batch03/intermediate/Herval.normalized.train.x.pickle"

txt_dump_model = data_folder + "data/batch03/intermediate/model_generated.txt"
txt_feat_map_model = data_folder + "data/batch03/intermediate/feat_map_generated.txt"

output_folder= data_folder + "data/batch03/model/"


# ## Configuracao do Logger

# In[3]:


logger = logging.getLogger()
logging.basicConfig(format="%(asctime)-15s %(message)s",
                    level=logging.DEBUG,
                    filename=os.path.join(log_location,'xgboost.log.' + datetime.now().strftime("%Y%m%d%H%M%S.%f") + '.log'))
def print_log(msg):
    logging.debug(msg)
    print(msg)
    
def log(msg):
    logging.debug(msg)


# ## Carregamento do Dataframe a ser utilizado no teste.

# In[4]:


print_log("Carregando Pickling normalizado:{}".format(arquivo_pagantes_norm))    
pagantes = pd.read_pickle(arquivo_pagantes_norm)
pagantes = pagantes.query("CLUSTER == {}".format(cluster))
total_pagantes = len(pagantes.index)
print_log("Total pagantes:{}".format(total_pagantes))


# ## Separacao dos Conjuntos de Dados de Teste e de Treinamento

# In[5]:


print_log("Criando dataframes de train e teste...")
pagantes = pagantes.sample(int(len(pagantes.index)))
pagantes_train = pagantes.tail(int(len(pagantes.index) * proporcao_train))
pagantes_test = pagantes.head(int(len(pagantes.index) * proporcao_test))
del pagantes


# ## Criacao de uma referencia dos nomes das colunas a ser usado no modelo

# In[6]:


def create_column_reference(header_chamadas_x,arquivo_df_pickled_norm_train_x):
    print_log("Criando Arquivo de referencia de colunas...")
    with open(arquivo_df_pickled_norm_train_x+".txt","w") as f:
        counter = 0
        lista_header = list(header_chamadas_x.columns.values)
        for header in lista_header:
            f.write("{}-{}\n".format(counter,header))
            counter=counter+1


# In[7]:


create_column_reference(pagantes_train.loc[:, 'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA'].head(1), arquivo_pagantes_norm_train_x)


# ## Conversao dos dataframes Pandas para arrays do tipo Numpy

# In[8]:


pagantes_train_x = pagantes_train.loc[:, 'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA'] # Pegamos apenas as colunas que realmente sao as caracteristicas
pagantes_train_y = pagantes_train.loc[:, 'PAGOU':'PAGOU'] # Colunas com as variaveis alvo

pagantes_test_x = pagantes_test.loc[:, 'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA'] # Pegamos apenas as colunas que realmente sao as caracteristicas
pagantes_test_y = pagantes_test.loc[:, 'PAGOU':'PAGOU'] # Colunas com as variaveis alvo

colunas_x = pagantes_train_x.columns.values # Nomes das Colunas com as caracteristicas
colunas_y = pagantes_train_y.columns.values # Nome da Coluna Alvo

pagantes_train_x = pagantes_train_x.as_matrix() # Conversao do Dataframe Pandas para Array Numpy
pagantes_train_y = pagantes_train_y.as_matrix() # Conversao do Dataframe Pandas para Array Numpy

pagantes_test_x = pagantes_test_x.as_matrix() # Conversao do Dataframe Pandas para Array Numpy
pagantes_test_y = pagantes_test_y.as_matrix() # Conversao do Dataframe Pandas para Array Numpy

colunas_x = [x for x in colunas_x] # Conversao para lista das colunas


# In[9]:


model_batch = datetime.now().strftime("%Y%m%d.%H%M%S")
tested_hyper_parameters = []

"""
Funcao para criacao de booster, ou seja, um conjunto de arvores de decisao que forma um ensemble de acordo com uma
serie de hyperparametros
"""
def create_booster(eta,depth,num_trees):
    param = {}
    param['booster'] = 'gbtree'
    param['eta'] = eta
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'auc'
    param['tree_method'] = 'auto'
    param['silent'] = 0
    param['max_depth'] = depth
    param['subsample'] = 0.5
    num_round = num_trees
    dtrain = xgb.DMatrix(pagantes_train_x, pagantes_train_y, feature_names = colunas_x)
    dtest = xgb.DMatrix(pagantes_test_x, pagantes_test_y, feature_names = colunas_x)
    train_labels = dtrain.get_label()
    ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1) 
    param['scale_pos_weight'] = ratio
    gpu_res = {}
    booster = xgb.train(param, dtrain, num_round, evals_result=gpu_res, evals = [])    
    return booster, dtrain, dtest

# Loopa para criar um modelo para cada combinacao dos parametros de hiperparametros
for eta in learning_rates_to_run:
    for depth in depth_to_run:
        for num_trees in range(min_trees, max_trees):
            gc.collect()
            booster, dtrain, dtest = create_booster(eta,depth,num_trees)
            booster.dump_model(os.path.join(output_folder,"{}.{}.{}.txt".format(model_batch,depth,num_trees)))
            save_file = os.path.join(output_folder,"{}.{}.{}.xgboost.model".format(model_batch,depth,num_trees))
            relevant_features = sorted( ((v,k) for k,v in booster.get_score().items()), reverse=True)

            # Calcula a matriz de confusao de acordo com cada threshold esperado.
            for current_threshold in range(0, 30):
                # Calcula o threshold a ser usado
                threshold = (0.5 - (current_threshold / 100))
                # Calcula a performance a partir dos dados de treinamento
                train_y_pred = booster.predict(dtrain)
                train_predictions = np.array([value for value in train_y_pred])
                train_predictions = np.array([1 if x > threshold else 0 for x in train_predictions])
                pagantes_train_y = pagantes_train_y.astype('float32')
                train_predictions = train_predictions.astype('float32').round()
                total_pagantes_train = len([x for x in pagantes_train_y if x == 1])
                tn, fp, fn, tp = confusion_matrix(np.squeeze(pagantes_train_y), np.squeeze(train_predictions)).ravel()
                pagantes_perdidos_train = (fn / total_pagantes_train ) * 100
                msg = "(TRAIN)Number of Trees:{}, eta:{}, depth:{} threshold:{:5.2f} ".format(num_trees, eta, depth, threshold) + "True Positive:{} True Negative:{} False Positive:{} False Negative:{},  {}% de pagantes perdidos no test".format(tp, tn, fp, fn, pagantes_perdidos_train )
                
                # Calcula a performance a partir dos dados de teste
                test_y_pred = booster.predict(dtest)
                test_predictions = np.array([value for value in test_y_pred])
                test_predictions = np.array([1 if x > threshold else 0 for x in test_predictions])
                pagantes_test_y = pagantes_test_y.astype('float32')
                test_predictions = test_predictions.astype('float32').round()
                total_pagantes_test = len([x for x in pagantes_test_y if x == 1])
                tn, fp, fn, tp = confusion_matrix(np.squeeze(pagantes_test_y), np.squeeze(test_predictions)).ravel()
                pagantes_perdidos_test = (fn / total_pagantes_train ) * 100
                msg = "(TEST)Number of Trees:{}, eta:{}, depth:{} threshold:{:5.2f} ".format(num_trees, eta, depth, threshold) + "True Positive:{} True Negative:{} False Positive:{} False Negative:{},  {}% de pagantes perdidos no test".format(tp, tn, fp, fn, pagantes_perdidos_test )
                
                porcentagem_pagamentos = tp / (tp + fn)
                base_para_trabalhar = (tp+fp) / (tp + tn + fn + fp)
                #Criamos um dicionario de dados que sera a base da nossa planilha de excel 
                current_info = { 
                    "0cluster" : cluster,
                    "1model" : save_file,
                    "2num_trees" : num_trees,
                    "3eta" : eta,
                    "4depth" : depth,
                    "5relevante_features" : str(relevant_features),
                    "6threshold" : threshold,
                    "7test0_total_pagantes" : total_pagantes_test,
                    "7test1_true_negative" : tn,
                    "7test2_true_positive" : tp,
                    "7test3_false_positive" : fp,
                    "7test4_false_negative" : fn,
                    "7test5_pagantes_perdidos" : total_pagantes_test,
                    "8%_pagamentos" : porcentagem_pagamentos,
                    "8%_base_para_trabalhar" : base_para_trabalhar,
                    "9delta" : porcentagem_pagamentos - base_para_trabalhar
                }
                tested_hyper_parameters.append(current_info)
            # Salva o modelo
            booster.save_model(save_file)


# ## Criacao de uma planilha de excel apartir da lista de dicionarios populada com as informacoes dos modelos gerados

# In[10]:


pd.DataFrame(tested_hyper_parameters).to_excel(os.path.join(output_folder,"{}.model.xls".format(model_batch)))

