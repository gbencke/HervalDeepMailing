"""
<TODO>
"""
import warnings
warnings.simplefilter("ignore")
import numpy as np
import herval_deep_mailing

if __name__ == "__main__":
    # Em primeiro lugar, vamos normalizar o dataframe e receber ele em formato
    # pandas

    DATAFRAME_NORMALIZADO, CLUSTERIZACAO = herval_deep_mailing.preparar_dados(
        log_location="../logs/",
        arquivo_fonte="../data/batch03/inputs/herval_final.xls",
        arquivo_saida="../data/batch03/intermediate/Herval.normalized.pickle",
        cluster_imagem="../data/batch03/intermediate/cluster.png")

    # Queremos clusterizar aqueles clientes que estao num cluster em q esteja
    # com 100 reais de divida e 100 reais de atraso
    print("Dataframe e clusterizacao criados!!!")
    CLUSTER = CLUSTERIZACAO.predict(np.array([[100, 100]]))
    print("CLUSTER Selecionado:{}".format(CLUSTER))

    # Com o modelo gerado, podemos agora criar todos os modelos apartir dos
    # hyperparametros abaixo, oque ira retornar um dictionary com todos os
    # modelos gerados
    MODELOS_GERADOS = herval_deep_mailing.train_model(
        cluster=CLUSTER,
        proporcao_train=0.7,
        proporcao_test=0.3,
        min_trees=5,
        max_trees=20,
        learning_rates_to_run=[0.3],
        depth_to_run=[3, 5, 10, 20],
        log_location="../../logs/",
        pagantes=DATAFRAME_NORMALIZADO,
        output_folder="../data/output")

    # Pegamos o primeiro modelo gerado, apenas para teste...
    MODEL_FILE = MODELOS_GERADOS[0]["1model"]

    print("Selecionado modelo:{}".format(MODEL_FILE))
    print("Carregando dados para teste...")
    DATAFRAME_PARA_TESTE, CLUSTERIZACAO = herval_deep_mailing.preparar_dados(
        log_location="../logs/",
        arquivo_fonte="../data/batch03/inputs/herval_final.xls",
        arquivo_saida="../data/batch03/intermediate/Herval.normalized.pickle",
        cluster_imagem="../data/batch03/intermediate/cluster.png")

    DATAFRAME_COM_RESULTADO = herval_deep_mailing.predict(cluster=CLUSTER,
                                                          pagantes=DATAFRAME_PARA_TESTE,
                                                          model_file=MODEL_FILE)
    print("RESULTADO FINAL:")
    print(DATAFRAME_COM_RESULTADO.head(10))
