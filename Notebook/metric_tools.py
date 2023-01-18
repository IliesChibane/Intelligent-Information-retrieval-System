import numpy as np
import pandas as pd

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

# Fonction qui permet de calculer les métriques de notre SRI par rapport à la requête pour les retourner à notre utilisateur
def compute_metrics(query, eval_df, result): 
    p = precision(query, eval_df, result, 0) # On calcule la précision
    p5 = precision(query, eval_df, result, 5) # On calcule la p@5
    p10 = precision(query, eval_df, result, 10) # On calcule la p@10
    r = recall(query, eval_df, result) # On calcule le rappel
    f = f_measure(p, r) # On calcule la mesure F
    return p, p5, p10, r, f