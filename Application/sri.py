import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import matplotlib

from vectorial import scalar_product, cosine_measure, jaccard_measure
from boolean import boolean_model
from probabilistic import bm25
from textmining import text_mining_sri, dbscan
from deeplearning import get_result

matplotlib.use('Agg')

# fichier qui contient les fonctions qui permettent de lancer l'application 
# et de définir les options du menu déroulant, et de calculer les métriques en fonction du type de SRI choisi


################ importation des données ################ 
info_queries = pd.read_csv("../csv_docs/info_queries.csv")
info_queries = info_queries.drop("Unnamed: 0", axis=1)

eval_df = pd.read_csv("../csv_docs/evaluation.csv")
eval_df = eval_df.drop("Unnamed: 0", axis=1)

bool_queriers = pd.read_csv("../csv_docs/bool_queries.csv")
bool_queriers = bool_queriers.drop("Unnamed: 0", axis=1)

df = pd.read_csv("../csv_docs/dataset_dbscan.csv")
df.drop(columns = df.columns[0], axis = 1, inplace= True)
#########################################################


# Fonction qui permet de récupérer les requêtes en fonction du type de SRI choisi
def get_queries(sri_type):
    if sri_type == "Boolean": return list(bool_queriers["Code"].unique())
    else : return list(info_queries["Query"].unique())

# Fonction qui permet de calculer la précision de notre SRI par rapport à la requête
def precision(query, eval_df, result, k):
    eval_pk = eval_df.loc[eval_df["Query"] == query, "Document"].to_numpy() # Documents pertinents pour la requête
    # Documents pertinents retournés par le SRI on récupère tout documents pour calculer la précision les 5 premiers pour p@5 et les 10 premiers pour p@10
    result_pk = result.iloc[0:k]["Document"].to_frame() if k != 0 else result["Document"].to_frame() 
    if k == 0: k = len(result_pk) if len(result_pk) != 0 else 1 # Si k = 0 on prend tous les documents retournés
    return result_pk.isin(eval_pk).sum()[0] / k  # On calcule la précision

# Fonction qui permet de calculer le rappel de notre SRI par rapport à la requête
def recall(query, eval_df, result):
    eval_pk = eval_df.loc[eval_df["Query"] == query, "Document"].to_numpy() # Documents pertinents pour la requête
    doc_select = int(result.isin(eval_pk)["Document"].sum()) # Documents pertinents retournés par le SRI
    return doc_select / len(eval_pk) if len(eval_pk) != 0 else 0 # On calcule le rappel 

# Fonction qui permet de calculer la mesure F de notre SRI par rapport à la requête
def f_measure(p, r):
    return 2 * p * r / (p + r) if p + r != 0 else 0

# Fonction qui permet de calculer la courbe de rappel précision de notre SRI par rapport à la requête
def crp(query, result, eval_df):
    eval_pk = eval_df.loc[eval_df["Query"] == query, "Document"].to_numpy() # Documents pertinents pour la requête
    result_pk = result["Document"].to_frame() # Documents retournés par le SRI
    l = len(result_pk) # Nombre de documents retournés par le SRI
    k = result_pk.isin(eval_pk).sum()[0] # Nombre de documents pertinents retournés par le SRI
    rp = [] # Liste qui contient les valeurs de la courbe de rappel précision
    for i in range(1,l): # On calcule la précision pour les 1 premiers documents, puis pour les 2 premiers, puis pour les 3 premiers, etc...
        pi = precision(query, eval_df, result, i) # On calcule la précision pour les i premiers documents
        result_pk = result.iloc[0:i]["Document"].to_frame() # On récupère les i premiers documents retournés par le SRI
        ki = result_pk.isin(eval_pk).sum()[0] # On calcule le nombre de documents pertinents parmi les i premiers documents retournés par le SRI
        rp.append([pi, ki/k]) # On ajoute la valeur de la précision et du rappel pour les i premiers documents
    rp = pd.DataFrame(rp, columns=["Precision", "Rappel"]) # On transforme la liste en dataframe
    rpi = [] # Liste qui contient les valeurs de la courbe de rappel précision interpolée
    j = 0 # On commence à 0
    while j <= 1: # On va jusqu'à 1
        p_max = rp.loc[rp["Rappel"] >= j]["Precision"].max() # On récupère la précision maximale pour un rappel supérieur ou égal à j
        rpi.append([j, p_max]) # On ajoute la valeur de la précision et du rappel pour le rappel j
        j += (1/l) # On incrémente j
    rpi = pd.DataFrame(rpi, columns=["Rappel", "Precision"]) # On transforme la liste en dataframe
    fig = plt.figure() # On crée la figure
    plt.title("Courbe de rappel précision") # On ajoute le titre
    plt.xlabel("Rappel") # On ajoute l'axe des abscisses
    plt.ylabel("Précision") # On ajoute l'axe des ordonnées
    plt.plot(rpi["Rappel"], rpi["Precision"]) # On ajoute la courbe de rappel précision interpolée
    return fig

# Fonction qui permet de calculer les métriques de notre SRI par rapport à la requête pour les retourner à notre utilisateur
def metrics(query, result): 
    p = precision(query, eval_df, result, 0) # On calcule la précision
    p5 = precision(query, eval_df, result, 5) # On calcule la p@5
    p10 = precision(query, eval_df, result, 10) # On calcule la p@10
    r = recall(query, eval_df, result) # On calcule le rappel
    f = f_measure(p, r) # On calcule la mesure F
    plot = crp(query, result, eval_df) # On calcule la courbe de rappel précision
    return p, p5, p10, r, f, plot

# Fonction qui permet d'appliquer le SRI sélection par notre utilisateur à la requête qu'il a choisi
def apply_sri(stem, query, sri, B=0, K=0, radius=0.6, min_neighbours=200):
    if stem == "Lancaster" : fichier_inverse = pd.read_csv("../csv_docs/fichier_inverse_Lancaster.csv") # On récupère le fichier inverse
    else : fichier_inverse = pd.read_csv("../csv_docs/fichier_inverse.csv") # On récupère le fichier inverse
    fichier_inverse = fichier_inverse.drop("Unnamed: 0", axis=1) # On supprime la colonne inutile

    # On applique le SRI sélectionné par notre utilisateur à la requête qu'il a choisi et on récupère les résultats
    if sri == "Scalar Product": 
        return scalar_product(query, fichier_inverse, info_queries)
    elif sri == "Cosine Measure":
        return cosine_measure(query, fichier_inverse, info_queries)
    elif sri == "Jaccard Measure":
        return jaccard_measure(query, fichier_inverse, info_queries)
    elif sri == "Base Boolean":
        return boolean_model(query, fichier_inverse, bool_queriers, stem)
    elif sri == "BM25":
        return bm25(query, fichier_inverse, info_queries, B, K)
    elif sri == "DBSCAN - Naive Bayes":
        return text_mining_sri(query, info_queries, df, fichier_inverse, radius, min_neighbours)
    elif sri == "LSTM - GRU":
        return get_result(query)
    else:
        return None
