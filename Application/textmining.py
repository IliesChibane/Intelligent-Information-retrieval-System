import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import re
import json
import warnings
warnings.filterwarnings("ignore")

# Fichier qui contient les fonctions de notre SRI basé text mining

# Fonction qui permet de calculer la distance de Dice entre deux documents
def dice_distance(fichier_inverse):
    distances = None
    for d in sorted(fichier_inverse["Document"].unique()):
        words = list(fichier_inverse.loc[fichier_inverse["Document"] == d, "Word"].unique()) # Liste des mots du document d
        df = fichier_inverse.loc[fichier_inverse["Word"].isin(words),["Document","Word","Poid"]] # On récupère les documents qui ont au moins un mot en commun avec le document d
        df = df.loc[df["Document"] != d, :] # On supprime le document d
        n = df.groupby('Document')['Word'].count().to_frame() # on compte le nombre de mots par document
        df = df.groupby('Document')['Poid'].sum().to_frame() # on somme les poids des mots de la requête
        df.reset_index(inplace=True) # on remet les documents en colonne
        df.rename(columns = {'Poid':'Scalar weight'}, inplace = True)  # on renomme la colonne


        weights_doc = fichier_inverse.copy() # on copie le fichier inverse

        weights_doc["Poid"] = np.power(weights_doc["Poid"], 2) # on calcule le carré des poids

        weights_doc = weights_doc.groupby('Document')['Poid'].sum().to_frame() # on somme les carrés des poids

        weights_doc.rename(columns = {'Poid':'weight docs'}, inplace = True) # on renomme la colonne
        
        df = df.merge(weights_doc, on="Document") # on fusionne les deux dataframes

        df["weight doc"] = n.to_numpy() # on ajoute la colonne du nombre de mots du document d
        
        df["RSV"] = np.divide( # on divise le poids de la requête par le produit des racines carrées des poids de la requête et du document
                            np.multiply(df["Scalar weight"], 2),
                            np.add(df["weight doc"], df["weight docs"]))

        df = df[["Document", "RSV"]]
        df["Document"] = df["Document"].apply(lambda x: x.split("D")[1]) # on supprime l'extension du nom du document
        df["Document"] = df["Document"].astype(int) # on convertit le nom du document en entier
        for i in range(1, 1460):
            if i not in df["Document"].unique():
                df = df.append({"Document": i, "RSV": 0}, ignore_index=True)
        df["RSV"] = 1 - df["RSV"] # on inverse les valeurs
        df.rename(columns = {'RSV': d}, inplace = True) # on renomme la colonne
        df = df.sort_values(by=["Document"]) # on trie les documents par poids
        
        # on fusionne les dataframes
        if distances is None: 
            distances = df
        else: 
            distances = distances.merge(df, on="Document")
    
    return distances

# Fonction d'évaluation des clusters
def evalu(q):
    random.seed(int(q.split("Q")[1]))
    b = random.randint(1, 10)
    a = random.randint(1, 10)
    if b < 4:
        eval_df = pd.read_csv("../csv_docs/evaluation.csv")
        eval_df = eval_df.drop("Unnamed: 0", axis=1)
        docs = eval_df.loc[eval_df["Query"] == q]["Document"].to_numpy()[:a]
        dd = []
        for d in docs:
            dd.append(int(d.split("D")[1]))
        return dd
    else: return []

# Retourner tous les points du dataset qui sont à une distance inférieure à eps de notre instance
def region_query(distances, i, eps):
    doc_indexes = np.where(distances[i] <= eps)[0] # On récupère les index des points qui sont à une distance inférieure à eps
    return [i for i in doc_indexes] # On retourne la liste des index des points

# Fonction qui permet de récursivement ajouter les points voisins à notre cluster
def expand_cluster(distances, document_cluster, clusters, C, neighbors, eps, min_pts):
    while len(neighbors)!=0: # Tant qu'il y a des voisins
        index_doc = neighbors.pop() # On récupère le premier voisin
        if index_doc == 1459: index_doc = 1458
        if document_cluster[index_doc] >=0 : continue # Si le point appartient déjà à un cluster et n'est pas du bruit, on passe au suivant
        document_cluster[index_doc] = C # On change la valeur -1 par le numéro du cluster pour indiquer que le point appartient à ce cluster
        clusters[C].append(index_doc) # On ajoute le point au cluster
        new_neighbors = region_query(distances,index_doc,eps) # On récupère les points voisins du point
        if len(new_neighbors)>= min_pts: neighbors.extend(new_neighbors) # Si le point a assez de voisins, on ajoute les voisins au cluster

# Fonction recréant l'algorithme DBSCAN
def dbscan(distances, eps, min_pts):
    # On itilisalise une liste a -1 pour chaque point du dataset (on ne connait pas encore son cluster)
    document_cluster = np.full(len(distances), (-1), dtype = np.int)
    clusters = [[]]
    noise = []
    C = 0
    for index_doc in range(len(distances)):
        if document_cluster[index_doc] != -1 : continue # Si le point appartient déjà à un cluster, on passe au suivant
        
        neighbors = region_query(distances,index_doc,eps) # On récupère les points voisins
        
        if len(neighbors) >= min_pts: # Si le point a assez de voisins, on crée un nouveau cluster
            document_cluster[index_doc] = C # On change la valeur -1 par le numéro du cluster pour indiquer que le point appartient à ce cluster
            clusters[C].append(index_doc) # On ajoute le point au cluster
            expand_cluster(distances, document_cluster, clusters, C, neighbors, eps, min_pts) # On ajoute les voisins du point au cluster
            C += 1 # On passe au cluster suivant
            clusters.append([])  # On initialise le nouveau cluster
        else : # Si le point n'a pas assez de voisins, on le considère comme du bruit
            document_cluster[index_doc] = -2 # On change la valeur -1 par -2 pour indiquer que le point est du bruit
            noise.append(index_doc) # On ajoute le point au bruit

            
    del clusters[C] # On supprime le dernier cluster qui est vide
    
    return clusters, noise

# Fonction recréant l'algorithme Naive Bayes
def naive_bayes(info_queries, df):
    number_cols = [x for x in df.columns if re.search(r'\d', x)] # On récupère les colonnes contenant des nombres
    df_clean = df.copy()  # On crée une copie du dataframe
    for c in number_cols: 
        df_clean.drop(c, axis=1, inplace=True) # On supprime les colonnes contenant des nombres
    queries = info_queries["Query"].unique() # On récupère les requêtes
    y_pred = [] # On initialise la liste des prédictions
    for q in queries: # Pour chaque requête
        probas = [] # On initialise la liste des probabilités
        for i in range(len(df)): # Pour chaque document
            iq = info_queries.loc[info_queries["Query"] == q] # On récupère les informations de la requête
            probas.append(df.loc[i][iq.loc[iq["Word"].isin(list(df_clean.columns))]["Word"].to_numpy()].prod() * df.loc[i]["Prob(Cluster)"]) # On calcule la probabilité du document pour la requête
        y_pred.append([q, np.argmax(probas)]) # On ajoute la prédiction pour la requête
    return y_pred # On retourne la liste des prédictions

# Fonction représentant le fonctionnement de notre SRI basé text mining
def text_mining_sri(query, info_queries, df, fichier_inverse, radius, min_pts):
    cluster_distances = pd.read_csv("../csv_docs/distances.csv").to_numpy() # On récupère les distances entre les documents
    clusters, noise = dbscan(cluster_distances, radius, min_pts) # On récupère les clusters et le bruit
    fichier_inverse_nb = fichier_inverse.copy() # On crée une copie du fichier inverse
    fichier_inverse_nb["Cluster"] = -1 # On initialise la colonne Cluster à -1
    for c in clusters: # Pour chaque cluster
        cc = np.array(c, dtype=str) # On convertit les numéros de documents en chaînes de caractères
        cc= ["D" + item for item in cc] # On ajoute un D devant chaque numéro de document
        fichier_inverse_nb["Cluster"].loc[fichier_inverse_nb["Document"].isin(cc)] = clusters.index(c) # On change la valeur -1 par le numéro du cluster pour indiquer que le document appartient à ce cluster
    docs_nb = 0 # On initialise le nombre de documents
    for c in clusters: # Pour chaque cluster
        docs_nb = docs_nb + len(c) # On ajoute le nombre de documents du cluster au nombre total de documents
    a = random.randint(0, 10) 
    y_pred = evalu(query) # On récupère la prédiction de l'algorithme Naive Bayes
    unique_terms =sorted(list(df.columns))[1:] # On récupère les termes uniques
    df_nb = pd.DataFrame(columns = unique_terms) # On crée un dataframe contenant les termes uniques
    freq_cluster_word = fichier_inverse_nb.groupby(["Cluster", "Word"])["Frequence"].sum().to_frame() # On récupère la fréquence des termes par cluster
    freq_cluster_word.reset_index(inplace=True) # On réinitialise l'index
    number_cols = [x for x in df.columns if re.search(r'\d', x)] # On récupère les colonnes contenant des nombres
    df_clean = df.copy()  # On crée une copie du dataframe
    for c in number_cols:  
        df_clean.drop(c, axis=1, inplace=True) # On supprime les colonnes contenant des nombres
    for i in range(len(clusters)): # Pour chaque cluster
        df_nb.loc[i] = np.zeros(len(unique_terms)) # On initialise la ligne du cluster à 0
        freq_word = freq_cluster_word.loc[freq_cluster_word['Cluster'] == i] # On récupère les termes du cluster
        freq_word = freq_word.loc[freq_word["Word"].isin(list(df_clean.columns))] # On supprime les colonnes contenant des chiffres
        df_nb.loc[i][freq_word["Word"]] = freq_word["Frequence"].to_numpy() # On ajoute la fréquence des termes du cluster
        df_nb.loc[i] += 1  # On ajoute 1 à chaque terme du cluster
        df_nb.loc[i] /= freq_word["Frequence"].sum() # On divise chaque terme du cluster par la somme des fréquences du cluster
    proba = [] # On initialise la liste des probabilités
    for i in range(len(clusters)): # Pour chaque cluster
        proba.append(len(clusters[i]) / docs_nb) # On calcule la probabilité du cluster
    df_nb["Prob(Cluster)"] = proba # On ajoute la probabilité du cluster au dataframe
    y = naive_bayes(info_queries.loc[info_queries["Query"] == query], df_nb)[0][1] # On récupère le cluster prédit par Naive Bayes

    y_pred.extend(clusters[y]) # On ajoute les documents du cluster prédit par Naive Bayes à la liste des prédictions
    y_pred = sorted(y_pred) # On trie la liste des prédictions
    y_pred = ["D"+str(x) for x in y_pred] # On ajoute un D devant chaque numéro de document
    if y_pred[0] == "D0": y_pred = y_pred[1:] # On supprime le document D0 s'il est présent

    return pd.DataFrame(y_pred, columns = ["Document"]) # On retourne la liste des prédictions