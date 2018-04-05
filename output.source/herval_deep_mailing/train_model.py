"""
# # Criacao de Modelo de Classificacao de Pagamentos para a Herval, usando XGBoost
"""
import logging
import gc
import os.path
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import xgboost as xgb


# ## Parametros de criacao da Arvore com XGBoost

def train_model(cluster, proporcao_train, proporcao_test, min_trees, max_trees,
                learning_rates_to_run, depth_to_run, log_location,
                pagantes, output_folder):
    """
    # ## Configuracao do Logger
    """
    log_name = 'xgboost.log.' + datetime.now().strftime("%Y%m%d%H%M%S.%f") + '.log'
    log_fname = os.path.join(log_location, log_name)
    logging.basicConfig(format="%(asctime)-15s %(message)s",
                        level=logging.DEBUG,
                        filename=log_fname)

    def print_log(msg):
        logging.debug(msg)
        print(msg)

    def log(msg):
        logging.debug(msg)

    # ## Carregamento do Dataframe a ser utilizado no teste.

    pagantes = pagantes.query("CLUSTER == {}".format(cluster))
    total_pagantes = len(pagantes.index)
    print_log("Total pagantes:{}".format(total_pagantes))

    # ## Separacao dos Conjuntos de Dados de Teste e de Treinamento

    print_log("Criando dataframes de train e teste...")
    pagantes = pagantes.sample(int(len(pagantes.index)))
    pagantes_train = pagantes.tail(int(len(pagantes.index) * proporcao_train))
    pagantes_test = pagantes.head(int(len(pagantes.index) * proporcao_test))
    del pagantes

    # ## Conversao dos dataframes Pandas para arrays do tipo Numpy

    # Pegamos apenas as colunas que realmente sao as caracteristicas
    pagantes_train_x = pagantes_train.loc[:,
                                          'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA']
    # Colunas com as variaveis alvo
    pagantes_train_y = pagantes_train.loc[:, 'PAGOU':'PAGOU']

    # Pegamos apenas as colunas que realmente sao as caracteristicas
    pagantes_test_x = pagantes_test.loc[:,
                                        'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA']
    # Colunas com as variaveis alvo
    pagantes_test_y = pagantes_test.loc[:, 'PAGOU':'PAGOU']

    # Nomes das Colunas com as caracteristicas
    colunas_x = pagantes_train_x.columns.values

    # Conversao do Dataframe Pandas para Array Numpy
    pagantes_train_x = pagantes_train_x.as_matrix()
    # Conversao do Dataframe Pandas para Array Numpy
    pagantes_train_y = pagantes_train_y.as_matrix()

    # Conversao do Dataframe Pandas para Array Numpy
    pagantes_test_x = pagantes_test_x.as_matrix()
    # Conversao do Dataframe Pandas para Array Numpy
    pagantes_test_y = pagantes_test_y.as_matrix()

    colunas_x = [x for x in colunas_x]  # Conversao para lista das colunas

    model_batch = datetime.now().strftime("%Y%m%d.%H%M%S")
    tested_hyper_parameters = []

    def create_booster(eta, depth, num_trees):
        """
        Funcao para criacao de booster, ou seja, um conjunto de arvores
        de decisao que forma um ensemble de acordo com uma
        serie de hyperparametros
        """
        print_log(
            "Criando modelo com {} arvores e {} de prof.".format(
                num_trees, depth))
        param = {}
        param['booster'] = 'gbtree'
        param['eta'] = eta
        param['objective'] = 'binary:logistic'
        param['eval_metric'] = 'auc'
        param['tree_method'] = 'auto'
        param['silent'] = 1
        param['max_depth'] = depth
        param['subsample'] = 0.5
        num_round = num_trees
        dtrain = xgb.DMatrix(
            pagantes_train_x, pagantes_train_y, feature_names=colunas_x)
        dtest = xgb.DMatrix(pagantes_test_x, pagantes_test_y,
                            feature_names=colunas_x)
        train_labels = dtrain.get_label()
        ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
        param['scale_pos_weight'] = ratio
        gpu_res = {}
        booster = xgb.train(param, dtrain, num_round,
                            evals_result=gpu_res, evals=[])
        return booster, dtrain, dtest

    print_log("Iremos agora criar os modelos de acordo com os hyperparametros")
    # Loopa para criar um modelo para cada combinacao dos parametros de hiperparametros
    for eta in learning_rates_to_run:
        for depth in depth_to_run:
            for num_trees in range(min_trees, max_trees + 1):
                gc.collect()
                booster, dtrain, dtest = create_booster(eta, depth, num_trees)
                booster.dump_model(os.path.join(
                    output_folder, "{}.{}.{}.txt".format(model_batch, depth, num_trees)))
                save_file = os.path.join(
                    output_folder, "{}.{}.{}.xgboost.model".format(model_batch, depth, num_trees))
                relevant_features = sorted(
                    ((v, k) for k, v in booster.get_score().items()), reverse=True)

                # Calcula a matriz de confusao de acordo com cada threshold esperado.
                for current_threshold in range(0, 30):
                    # Calcula o threshold a ser usado
                    threshold = (0.5 - (current_threshold / 100))
                    # Calcula a performance a partir dos dados de treinamento
                    train_y_pred = booster.predict(dtrain)
                    train_predictions = np.array(
                        [value for value in train_y_pred])
                    train_predictions = np.array(
                        [1 if x > threshold else 0 for x in train_predictions])
                    pagantes_train_y = pagantes_train_y.astype('float32')
                    train_predictions = train_predictions.astype(
                        'float32').round()
                    tn, fp, fn, tp = confusion_matrix(np.squeeze(
                        pagantes_train_y), np.squeeze(train_predictions)).ravel()

                    # Calcula a performance a partir dos dados de teste
                    test_y_pred = booster.predict(dtest)
                    test_predictions = np.array(
                        [value for value in test_y_pred])
                    test_predictions = np.array(
                        [1 if x > threshold else 0 for x in test_predictions])
                    pagantes_test_y = pagantes_test_y.astype('float32')
                    test_predictions = test_predictions.astype(
                        'float32').round()
                    total_pagantes_test = len(
                        [x for x in pagantes_test_y if x == 1])
                    tn, fp, fn, tp = confusion_matrix(np.squeeze(
                        pagantes_test_y), np.squeeze(test_predictions)).ravel()

                    porcentagem_pagamentos = tp / (tp + fn)
                    base_para_trabalhar = (tp+fp) / (tp + tn + fn + fp)
                    # Criamos um dicionario de dados que sera a base da nossa planilha de excel
                    current_info = {
                        "0cluster": cluster,
                        "1model": save_file,
                        "2num_trees": num_trees,
                        "3eta": eta,
                        "4depth": depth,
                        "5relevante_features": str(relevant_features),
                        "6threshold": threshold,
                        "7test0_total_pagantes": total_pagantes_test,
                        "7test1_true_negative": tn,
                        "7test2_true_positive": tp,
                        "7test3_false_positive": fp,
                        "7test4_false_negative": fn,
                        "7test5_pagantes_perdidos": total_pagantes_test,
                        "8%_pagamentos": porcentagem_pagamentos,
                        "8%_base_para_trabalhar": base_para_trabalhar,
                        "9delta": porcentagem_pagamentos - base_para_trabalhar
                    }
                    tested_hyper_parameters.append(current_info)
                # Salva o modelo
                booster.save_model(save_file)

    # ## Criacao de uma planilha de excel apartir da lista de dicionarios populada com as informacoes dos modelos gerados

    return tested_hyper_parameters
