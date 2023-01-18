import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import json

# Fichier qui permet de retourner les résultats de notre SRI basé sur le Deep Learning
# Le modèle étant beaucoup trop lourd pour être mis sur GitHub, nous avons mis les résultats de notre SRI dans un fichier csv

dl_result = pd.read_csv("../csv_docs/Deep_learning_Result.csv")

def get_result(query):
    return dl_result.loc[dl_result["Query"] == query, "Document"].to_frame()
