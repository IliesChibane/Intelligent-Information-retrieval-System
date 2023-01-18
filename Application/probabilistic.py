import nltk
import numpy as np
import pandas as pd
import re
import json
import warnings
warnings.filterwarnings("ignore")

# Fichier qui contient les fonctions de notre SRI probabiliste

# SRI Probabiliste en utilisant la fonction BM25
def bm25(query, fichier_inverse, info_queries, B, K):
    N = len(fichier_inverse['Document'].unique()) # Nombre de documents

    words = list(info_queries.loc[info_queries["Query"] == query, "Word"]) # Liste des mots de la requête
    df = fichier_inverse.loc[fichier_inverse["Word"].isin(words)] # Fichier inverse contenant les mots de la requête

    nb_doc = df.groupby('Word')['Document'].count().to_frame() # Nombre de documents contenant les mots de la requête
    nb_doc.reset_index(inplace=True) # Réinitialisation de l'index
    nb_doc.rename(columns = {'Document':'Nombre document contenu'}, inplace = True) # Renommage de la colonne
    df = df.merge(nb_doc, on="Word") # Fusion des deux dataframes

    nb_termes_doc = fichier_inverse.groupby('Document')['Frequence'].sum().to_frame() # Nombre de termes par document
    nb_termes_doc.reset_index(inplace=True) # Réinitialisation de l'index
    nb_termes_doc.rename(columns = {'Frequence':'dl'}, inplace = True) # Renommage de la colonne
    df = df.merge(nb_termes_doc, on="Document") # Fusion des deux dataframes
    
    df["avdl"]= nb_termes_doc["dl"].mean() # Moyenne du nombre de termes par document
    
    rsv_list = [] # Liste des RSV

    for d in df["Document"].unique(): # Pour chaque document
        temp = df.loc[df["Document"] == d, ["Frequence", "Nombre document contenu", "dl", "avdl"]] # Dataframe temporaire
        ni = temp["Nombre document contenu"] # Nombre de documents contenant le mot

        # Il est 3h du mat flemme d'expliquer la formule regarde le pdf tu vas comprendre
        rsv = np.multiply(np.divide(temp["Frequence"],
                            np.add(np.multiply(K, 
                            np.add(np.subtract(1, B), 
                                    np.multiply(B, np.divide(temp["dl"], temp["avdl"])))), 
                                                temp["Frequence"])), 
                                                np.log10(np.divide(np.add(
                                                        np.subtract(N, ni), 0.5), np.add(ni, 0.5))))
        
        rsv_list.append([d, np.sum(rsv)]) # Ajout du RSV dans la liste

    df = pd.DataFrame(rsv_list, columns = ["Document","RSV"]) # Création d'un dataframe avec la liste des RSV
    df = df.sort_values(by=["RSV"]) # Tri des RSV par ordre croissant

    return df.reindex(index=df.index[::-1]) # Retourne le dataframe trié par ordre décroissant