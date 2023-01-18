import nltk
import numpy as np
import pandas as pd
import re
import json

# Fichier qui permet de constructruire notre fichier inverse et descripteur

documents = open("../cisi/CISI.ALL").read()
mots_vides = nltk.corpus.stopwords.words('english') # On récupère les mots vides

# Fonction pour extraire les données des fichiers
def split_extraction(documents):
    content = documents.split(".I ") # Split sur le numéro de document
    del content[0] # Suppression du premier élément qui est vide

    parts = content 
    data = [] # Liste qui contiendra les données
    for part in parts: # Pour chaque document
        
        temp = part.split(".T\n") # Split sur le titre
        if len(temp) == 1 : temp = part.split(".T \n") # Si le titre prend un espace avant de retourner à la ligne
        if len(temp) == 1 : temp = part.split(".T  \n") # Si le titre prend deux espaces avant de retourner à la ligne
        
        del temp[0] # Suppression du premier élément qui est vide
        aut_temp = temp[0].split(".A\n") if len(temp[0].split(".A\n")) >= 2 else temp[0].split(".A \n") # Split sur l'auteur (si l'auteur prend un espace avant de retourner à la ligne)
        if len(aut_temp) > 2 : # Si le document a plus d'un auteur
            while len(aut_temp) != 2: # On supprime l'intégralité des auteurs 
                del aut_temp[1]
        title, temp2 = aut_temp # On récupère le titre et le reste du document
        
        summary = temp2.split(".W\n")[-1].split(".X")[0] # On récupère le résumé et on supprime les références
        summary = summary.split(".W")[-1] # On re applique un split afin de gérer un cas particulier
        data.append([' '.join(title.split()).lower(),' '.join(summary.split()).lower()]) # On ajoute le titre et le résumé dans la liste
    return data

# Fonction qui permet de fusionner le titre et le résumé
def flatten(l):
    return [item for sublist in l for item in sublist]

# Fonction qui permet de créer le document inverse
def get_documents_inverse(data, stem):
    i = 1
    documents_inverse = [] # Liste qui contiendra les données du document inverse
    for d in data: # Pour chaque document
        text = ""
        text += ' '.join(d) + " " # On fusionne le titre et le résumé

        rgx = '(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*' # Regex du prof pour récupérer les mots
        text_token = nltk.RegexpTokenizer(rgx).tokenize(text) # On récupère les mots

        termes_sans_mots_vides = [terme for terme in text_token if terme.lower() not in mots_vides] # On supprime les mots vides

        # On initialise les stemmers
        ps = nltk.PorterStemmer() 
        ls = nltk.LancasterStemmer()
        # On applique le stemmer choisit par l'utilisateur
        if stem == "Porter" : stemmed = [ps.stem(tt) for tt in termes_sans_mots_vides]
        elif stem == "Lancaster" : stemmed = [ls.stem(tt) for tt in termes_sans_mots_vides]

        freq_word_doc = nltk.FreqDist(stemmed) # On récupère la fréquence des mots

        # On ajoute les données dans la liste
        filtered_word_freq = [[word, i, freq] 
                                for word, freq in freq_word_doc.items()]
        i += 1 # On incrémente le numéro de document
        documents_inverse.append(filtered_word_freq) # On ajoute les données dans la liste

    return pd.DataFrame(flatten(documents_inverse), columns = ["Word","Document","Frequence"])

# Fonction de ponderation
def ponderation(document_inverse):
    max_freq = document_inverse.groupby('Document')['Frequence'].max().to_frame() # On récupère la fréquence maximale par document
    max_freq.reset_index(inplace=True) # On reset l'index
    max_freq.rename(columns = {'Frequence':'Max Frequence'}, inplace = True) # On renomme la colonne
    df = document_inverse.merge(max_freq, on="Document") # On merge les deux dataframes

    nb_doc = document_inverse.groupby('Word')['Document'].count().to_frame() # On récupère le nombre de document par mot
    nb_doc.reset_index(inplace=True) # On reset l'index
    nb_doc.rename(columns = {'Document':'Nombre document contenu'}, inplace = True) # On renomme la colonne
    df = df.merge(nb_doc, on="Word") # On merge les deux dataframes

    # On calcule le poid
    df["Poid"] = np.multiply(
        np.divide( # On divise la fréquence par la fréquence maximale
            df["Frequence"], 
            df["Max Frequence"]
            ), 
            np.log10( 
                np.divide( # On divise le nombre de document par le nombre de document contenant le mot
                    len(document_inverse['Document'].unique()),  
                    df["Nombre document contenu"]
                ) + 1
            )
        )

    fichier_descripteur = df[["Document", "Word", "Frequence", "Poid"]].sort_values(by=["Document"]) # On trie par document
    fichier_inverse = df[["Word", "Document", "Frequence", "Poid"]].sort_values(by=["Word"]) # On trie par mot
    
    fichier_descripteur["Document"] = fichier_descripteur["Document"].astype(str) # On convertit les données en string
    fichier_descripteur["Document"] = "D"+fichier_descripteur["Document"] # On ajoute un D devant le numéro de document
    fichier_inverse["Document"] = fichier_inverse["Document"].astype(str) # On convertit les données en string
    fichier_inverse["Document"] = "D"+fichier_inverse["Document"]  # On ajoute un D devant le numéro de document

    return fichier_descripteur, fichier_inverse # On retourne les deux dataframes

# Fonction qui permet de créer le document inverse
def get_doc_inverse(stem):
    data = split_extraction(documents)
    document_inverse = get_documents_inverse(data, stem)
    return document_inverse

# Fonction qui permet de rechercher le contenu d'un document dans le fichier descripteur
def rech_fichier_descripteur(doc, stem):
    if stem == "Lancaster" : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur_Lancaster.csv")
    else : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur.csv")
    list_of_terms = fichier_descripteur.loc[fichier_descripteur["Document"] == doc]
    list_of_terms = list_of_terms[["Word", "Frequence", "Poid"]]
    return list_of_terms.sort_values(by=["Word"])

# Fonction qui permet de rechercher les informations d'un mot dans le fichier inverse
def rech_fichier_inverse(term, stem):
    if stem == "Lancaster" : fichier_inverse = pd.read_csv("../csv_docs/fichier_inverse_Lancaster.csv")
    else : fichier_inverse = pd.read_csv("../csv_docs/fichier_inverse.csv")
    list_of_terms = fichier_inverse.loc[fichier_inverse["Word"] == term]
    return list_of_terms[["Document", "Frequence", "Poid"]]

# Fonction qui permet de retourner le nombre de documents et de termes uniques de notre collection
def get_stats(stem):
    if stem == "Lancaster" : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur_Lancaster.csv")
    else : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur.csv")
    return len(fichier_descripteur['Document'].unique()), len(fichier_descripteur['Word'].unique())

# Fonction qui permet de retourner la liste des termes uniques de notre collection
def get_terms(stem):
    if stem == "Lancaster" : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur_Lancaster.csv")
    else : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur.csv")
    return fichier_descripteur['Word'].unique().tolist()

# Fonction qui permet de retourner la liste des documents uniques de notre collection
def get_docs(stem):
    if stem == "Lancaster" : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur_Lancaster.csv")
    else : fichier_descripteur = pd.read_csv("../csv_docs/fichier_descripteur.csv")
    return fichier_descripteur['Document'].unique().tolist()