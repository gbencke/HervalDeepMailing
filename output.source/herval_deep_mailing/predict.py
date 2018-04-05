"""
# # Exemplo de utilizacao do Modelo Treinado
"""

import xgboost as xgb

# ## Determinacao dos arquivos de dados normalizados e do modelo ja treinado


def predict(cluster, pagantes, model_file):
    """
    Funcao para adicionar uma coluna de PROD a um dataframe ja normalizado
    usando um modelo ja treinado
    """

    # ## Carregamento do dataframe de pandas e tambem selecao do cluster a ser usado.

    pagantes = pagantes.query("CLUSTER == {}".format(cluster))
    pagantes = pagantes.reset_index()

    # ## Preparacao do dataframe de caracteristicas e tambem conversao para array de numpy

    pagantes_x = pagantes.loc[:,
                              'NORM_CLASSE_SOCIAL_A1':'NORM_RENDA_PRESUMIDA']
    pagantes_x = pagantes.as_matrix()

    # ## Criacao de um booster e carga do Modelo

    bst = xgb.Booster()
    bst.load_model(model_file)  # load data
    dtest = xgb.DMatrix(pagantes_x)
    ypred = bst.predict(dtest)

    # ## Preenchimento da probabilidade como uma nova coluna dentro da tabela de pagantes

    pagantes['PROB'] = pagantes.apply(lambda row: ypred[row.name], axis=1)
    pagantes = pagantes[['CLUSTER', 'CPF', 'PROB']]
    return pagantes
