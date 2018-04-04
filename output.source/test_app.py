"""
<TODO>
"""
import herval_deep_mailing


if __name__ == "__main__":
    DATAFRAME_NORMALIZADO = herval_deep_mailing.preparar_dados(
        log_location="../logs/",
        arquivo_fonte="../data/batch03/inputs/herval_final.xls",
        arquivo_saida="../data/batch03/intermediate/Herval.normalized.pickle",
        cluster_imagem="../data/batch03/intermediate/cluster.png")
    print("dataframe_normalizado:{}".format(DATAFRAME_NORMALIZADO.head(10)))
