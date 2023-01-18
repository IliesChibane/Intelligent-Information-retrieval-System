import nltk
import numpy as np
import pandas as pd
import re
import json
import warnings
warnings.filterwarnings("ignore")

# Fichier qui contient les fonctions de notre SRI Vectoriel

# SRI Vectoriel (Scalar Product)
def scalar_product(query, fichier_inverse, info_queries):
    words = list(info_queries.loc[info_queries["Query"] == query, "Word"].unique()) # liste des mots de la requête
    sri = fichier_inverse.loc[fichier_inverse["Word"].isin(words),["Document","Poid"]] # on récupère les documents et les poids des mots de la requête
    sri = sri.groupby('Document')['Poid'].sum().to_frame() # on somme les poids des mots de la requête
    sri.reset_index(inplace=True) # on remet les documents en colonne
    sri.rename(columns = {'Poid':'RSV'}, inplace = True) # on renomme la colonne
    sri = sri.sort_values(by=["RSV"]) # on trie les documents par poids
    sri = sri.reindex(index=sri.index[::-1]) # on inverse l'ordre des documents
    return sri # on ne garde que les k premiers documents

# SRI Vectoriel (Cosine Measure)
def cosine_measure(query, fichier_inverse, info_queries):
    words = list(info_queries.loc[info_queries["Query"] == query, "Word"].unique()) # liste des mots de la requête
    df = fichier_inverse.loc[fichier_inverse["Word"].isin(words),["Document","Word","Poid"]] # on récupère les documents et les poids des mots de la requête
    n = df.groupby('Document')['Word'].count().to_frame() # on compte le nombre de mots par document
    df = df.groupby('Document')['Poid'].sum().to_frame() # on somme les poids des mots de la requête
    df.reset_index(inplace=True) # on remet les documents en colonne
    df.rename(columns = {'Poid':'Scalar weight'}, inplace = True)  # on renomme la colonne

    df["SQRT weight query"] = np.sqrt(n).to_numpy() # on calcule la racine carrée du nombre de mots par document

    sqrt_weights_doc = fichier_inverse.copy() # on copie le fichier inverse
    sqrt_weights_doc["Poid"] = np.power(sqrt_weights_doc["Poid"], 2) # on calcule le carré des poids
    sqrt_weights_doc = sqrt_weights_doc.groupby('Document')['Poid'].sum().to_frame() # on somme les carrés des poids
    sqrt_weights_doc["Poid"] = np.sqrt(sqrt_weights_doc["Poid"]) # on calcule la racine carrée des sommes des carrés des poids
    sqrt_weights_doc.rename(columns = {'Poid':'SQRT weight doc'}, inplace = True) # on renomme la colonne
    df = df.merge(sqrt_weights_doc, on="Document") # on fusionne les deux dataframes

    df["RSV"] = np.divide( # on divise le poids de la requête par le produit des racines carrées des poids de la requête et du document
                            df["Scalar weight"], 
                            np.multiply(df["SQRT weight query"], df["SQRT weight doc"]))

    df = df[["Document", "RSV"]].sort_values(by=["RSV"]) # on trie les documents par poids
    df = df.reindex(index=df.index[::-1]) # on inverse l'ordre des documents
    return df

# SRI Vectoriel (Jaccard Measure)
def jaccard_measure(query, fichier_inverse, info_queries):
    words = list(info_queries.loc[info_queries["Query"] == query, "Word"]) # liste des mots de la requête
    df = fichier_inverse.loc[fichier_inverse["Word"].isin(words),["Document","Word","Poid"]] # on récupère les documents et les poids des mots de la requête
    n = df.groupby('Document')['Word'].count().to_frame() # on compte le nombre de mots par document
    df = df.groupby('Document')['Poid'].sum().to_frame() # on somme les poids des mots de la requête
    df.reset_index(inplace=True) # on remet les documents en colonne
    df.rename(columns = {'Poid':'Scalar weight'}, inplace = True) # on renomme la colonne

    df["square weight query"] = n.to_numpy() # on calcule le carré du nombre de mots par document

    square_weights_doc = fichier_inverse.copy() # on copie le fichier inverse
    square_weights_doc["Poid"] = np.power(square_weights_doc["Poid"], 2) # on calcule le carré des poids
    square_weights_doc = square_weights_doc.groupby('Document')['Poid'].sum().to_frame() # on somme les carrés des poids
    square_weights_doc.rename(columns = {'Poid':'square weight doc'}, inplace = True) # on renomme la colonne
    df = df.merge(square_weights_doc, on="Document") # on fusionne les deux dataframes

    df["RSV"] = np.divide(# on divise le poids de la requête par la différence des sommes des carrés des poids de la requête et du document
                            df["Scalar weight"], 
                            np.subtract(np.add(df["square weight query"], df["square weight doc"]), 
                                                df["Scalar weight"]))

    df = df[["Document", "RSV"]].sort_values(by=["RSV"]) # on trie les documents par poids
    return df.reindex(index=df.index[::-1]) # on inverse l'ordre des documents