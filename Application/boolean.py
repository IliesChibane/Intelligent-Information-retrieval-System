import nltk
import numpy as np
import pandas as pd
import re
import json
import warnings
warnings.filterwarnings("ignore")

# Fichier qui contient les fonctions de notre SRI booléen

# Création de la classe noeud pour la construction de l'arbre de la requête booléenne
class Node:
    def __init__(self, value=None, children=None):
        self.value = value # valeur du noeud
        self.children = [Node(c) for c in children] if children else [] # liste des enfants du noeud
    
    # Ajout de plusieurs enfants
    def add_children(self, children):
        for child in children:
            self.children.append(Node(child))
    
    # Ajout d'un enfant
    def add_child(self, child: str):
        self.children.append(Node(child))
    
    # Ajout d'un noeud enfant
    def add_node_child(self, child):
        self.children.append(child)

    def __str__(self):
        return self.value

# récupération des expressions en dehors des parenthèses
def get_outside_parenthesis(query):
    expressions = [] # liste des expressions
    cpt=0 # compteur de parenthèses
    opened = False # booléen pour savoir si une parenthèse est ouverte

    for i, c in enumerate(query): # parcours de la requête

        if c == '(': # si on rencontre une parenthèse ouvrante
            cpt += 1 # on incrémente le compteur
            if cpt == 1 : # si le compteur est à 1, c'est que la parenthèse est ouverte
                start = i # on récupère l'indice de la parenthèse ouvrante
                opened = True # on passe le booléen à True

        elif c == ')': # si on rencontre une parenthèse fermante
            cpt -= 1 # on décrémente le compteur

        if cpt == 0 and opened: # si le compteur est à 0 et que la parenthèse est ouverte
            opened = False # on passe le booléen à False
            end = i # on récupère l'indice de la parenthèse fermante
            expressions.append(query[start+1:end]) # on ajoute l'expression à la liste des expressions

    return expressions # on retourne la liste des expressions

# récupération des mots et des opérateurs
def get_tokens(query):
    
    words = [] # liste des mots
    operations = [] # liste des opérateurs
    temp_query = re.sub(r"\(.+?\)", "", query) # on supprime les expressions entre parenthèses
    temp_query = re.sub(r"'", "", temp_query) # on supprime les apostrophes
    temp_query = re.sub(r"\(", "", temp_query) # on supprime les parenthèses ouvrantes
    temp_query = re.sub(r"\)", "", temp_query) # on supprime les parenthèses fermantes

    tokens = temp_query.split(',') # on récupère les tokens

    for op in tokens: # on parcours les tokens
        if op.startswith('#'): # si le token commence par un #, c'est un opérateur
            operations.append(op) # on ajoute l'opérateur à la liste des opérateurs
        else: # sinon, c'est un mot
            words.append(op) # on ajoute le mot à la liste des mots

    return words, operations # on retourne les listes des mots et des opérateurs

# construction de l'arbre
def build_tree(operation, expression): 

    words, operations = get_tokens(expression) # on récupère les mots et les opérateurs
    expressions = get_outside_parenthesis(expression) # on récupère les expressions en dehors des parenthèses

    todo = list(zip(operations, expressions)) # on zip les opérateurs et les expressions

    node = Node(operation) # on crée le noeud racine

    for w in words: # on ajoute les mots à l'arbre
        node.add_child(w) # on ajoute le mot à l'arbre

    for op, exp in todo: # on parcours les opérateurs et les expressions
        sub_node = build_tree(op, exp) # on construit l'arbre de l'expression
        node.add_node_child(sub_node) # on ajoute l'arbre à l'arbre principal
        
    
    return node # on retourne l'arbre

# parcours de l'arbre
def parcours_tree(root, tv, op): 
    result = [] # liste des résultats
    if type(root) != list : # si le noeud n'est pas une liste
        if "#" in root.value: # si le noeud est un opérateur
            if root.value[1:] == "and": # si l'opérateur est un and
                return parcours_tree(root.children, tv, "and") # on parcours l'arbre avec l'opérateur and
            elif root.value[1:] == "or": # si l'opérateur est un or
                return parcours_tree(root.children, tv, "or") # on parcours l'arbre avec l'opérateur or
            elif root.value[1:] == "not": # si l'opérateur est un not
                return parcours_tree(root.children, tv, "not") # on parcours l'arbre avec l'opérateur not
            else: # si l'opérateur n'est pas reconnu
                return False # on retourne False
    else : # si le noeud est une liste
        for child in  root: # on parcours les noeuds de la liste
            if "#" in child.value: # si le noeud est un opérateur
                if child.value[1:] == "and": # si l'opérateur est un and
                    if op == "and": # si l'opérateur est un and
                        # on parcours l'arbre avec l'opérateur and
                        if len(result) == 0: result = parcours_tree(child.children, tv, "and") 
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération and
                        else : result = result & parcours_tree(child.children, tv, "and")
                    elif op == "or": # si l'opérateur est un or
                        # on parcours l'arbre avec l'opérateur or
                        if len(result) == 0: result = parcours_tree(child.children, tv, "and")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération or
                        else : result = result | parcours_tree(child.children, tv, "and")
                    elif op == "not": # si l'opérateur est un not
                        # on parcours l'arbre avec l'opérateur not
                        if len(result) == 0: result = parcours_tree(child.children, tv, "and")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération not
                        else : result = np.revert(parcours_tree(child.children, tv, "and"))
                elif child.value[1:] == "or": # si l'opérateur est un or
                    if op == "and": # si l'opérateur est un and
                        # on parcours l'arbre avec l'opérateur and
                        if len(result) == 0: result = parcours_tree(child.children, tv, "or")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération and
                        else : result = result & parcours_tree(child.children, tv, "or")
                    elif op == "or": # si l'opérateur est un or
                        # on parcours l'arbre avec l'opérateur or
                        if len(result) == 0: result = parcours_tree(child.children, tv, "or")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération or
                        else : result = result | parcours_tree(child.children, tv, "or")
                    elif op == "not": # si l'opérateur est un not
                        # on parcours l'arbre avec l'opérateur not
                        if len(result) == 0: result = parcours_tree(child.children, tv, "or")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération not
                        else : result = np.revert(parcours_tree(child.children, tv, "or"))
                elif child.value[1:] == "not": # si l'opérateur est un not
                    if op == "and": # si l'opérateur est un and
                        # on parcours l'arbre avec l'opérateur and
                        if len(result) == 0: result = parcours_tree(child.children, tv, "not")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération and
                        else : result = result & parcours_tree(child.children, tv, "not")
                    elif op == "or": # si l'opérateur est un or
                        # on parcours l'arbre avec l'opérateur or
                        if len(result) == 0: result = parcours_tree(child.children, tv, "not")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération or
                        else : result = result | parcours_tree(child.children, tv, "not")
                    elif op == "not": # si l'opérateur est un not
                        # on parcours l'arbre avec l'opérateur not
                        if len(result) == 0: result = parcours_tree(child.children, tv, "not")
                        # si la liste des résultats est vide, on ajoute les résultats de l'opération not
                        else : result = np.revert(parcours_tree(child.children, tv, "not"))
                else: # si l'opérateur n'est pas reconnu
                    return False # on retourne False
            else : # si le noeud est un terme
                if op == "and": # si l'opérateur est un and
                    # si la liste des résultats est vide, on ajoute les résultats de l'opération and
                    if len(result) == 0: result = tv[child.value] 
                    else : result = result & tv[child.value] # sinon on ajoute les résultats de l'opération and
                elif op == "or": # si l'opérateur est un or
                    if len(result) == 0: # si la liste des résultats est vide, on ajoute les résultats de l'opération or
                        result = tv[child.value] # sinon on ajoute les résultats de l'opération or
                    else : # sinon on ajoute les résultats de l'opération or
                        result = result | tv[child.value] # sinon on ajoute les résultats de l'opération or
                elif op == "not": # si l'opérateur est un not
                    result = np.invert(tv[child.value]) # on retourne les résultats de l'opération not
                else : # si l'opérateur n'est pas reconnu
                    return False # on retourne False
        return result # on retourne la liste des résultats

# SRI booleen
def boolean_model(query, fichier_inverse, info_queries, stem):
    q1 = info_queries.loc[info_queries["Code"] == "Q1", "Bool Query"][0] # on récupère la requête
    q2 = q1.split("#") # on sépare la requête en deux
    del q2[0] # on supprime le premier élément de la liste

    rgx = '(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%? |\w+(?:\-\w+)*' # on définit le regex
    text_token = nltk.RegexpTokenizer(rgx).tokenize(" ".join(q2)) # on tokenize la requête

    mots_vides = nltk.corpus.stopwords.words('english') # on récupère la liste des mots vides
    words_of_query = [terme for terme in text_token if terme.lower() not in mots_vides] # on supprime les mots vides de la requête

    # on initialise les stemmers
    ps = nltk.PorterStemmer()
    ls = nltk.LancasterStemmer()

    # on selectionne le stemmer chosit par l'utilisateur et on stemme la requête
    if stem == "Porter": stemmed = [ps.stem(tt) for tt in words_of_query]
    elif stem == "Lancaster": stemmed = [ls.stem(tt) for tt in words_of_query]

    tv = [] # on initialise la table de vérité
    for d in fichier_inverse["Document"].unique(): # pour chaque document
        # on récupère les mots du document
        tv_row = list(np.where(np.isin(stemmed, fichier_inverse.loc[fichier_inverse["Document"] == d, "Word"].to_numpy()),True,False))
        # on ajoute le document à la liste
        tv_row.insert(0,d) # on ajoute le document à la liste
        tv.append(tv_row) # on ajoute la liste à la table de vérité

    words_of_query.insert(0,"Document") # on ajoute le nom de la colonne Document
    table_verite = pd.DataFrame(tv, columns = words_of_query) # on crée la table de vérité
    table_verite = table_verite.sort_values(by=["Document"]) # on trie la table de vérité par document
    query = re.sub("\s", "", q1) # on supprime les espaces de la requête
    query = re.sub("#q\d+?=", "", query) # on supprime le code de la requête
    operation = re.match(r"#\w+", query).group(0) # on récupère l'opérateur
    expression = query[len(operation)+1: -1] # on récupère l'expression

    root = build_tree(operation, expression) # on construit l'arbre

    result =  parcours_tree(root, table_verite, "") # on parcours l'arbre
    table_verite["Result"] = result.astype(int) # on ajoute les résultats à la table de vérité

    tv = table_verite[["Document", "Result"]] # on retourne la table de vérité

    tv = tv.sort_values(by=["Result"])
    tv = tv.reindex(index=tv.index[::-1])
    return tv.loc[tv["Result"] == 1] # on retourne les documents qui correspondent à la requête