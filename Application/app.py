import gradio as gr
import sri
import files
import pandas as pd

#Notre fichier main qui lance l'application

#fonction qui permet de définir les options du menu déroulant en fonction du type de SRI choisi
def get_function(sri_type):
    if sri_type == "Vectorial":
        return gr.Dropdown.update(choices=["Scalar Product", "Cosine Measure", "Jaccard Measure"])
    elif sri_type == "Boolean":
        return gr.Dropdown.update(choices=["Base Boolean"])
    elif sri_type == "Probabilistic":
        return gr.Dropdown.update(choices=["BM25"])
    elif sri_type == "Text Mining":
        return gr.Dropdown.update(choices=["DBSCAN - Naive Bayes"])
    elif sri_type == "Deep Learning":
        return gr.Dropdown.update(choices=["LSTM - GRU"])

# fonction qui permet d'ajouter des sliders pour le K et B dans le cas d'un SRI probabiliste
def update_sliders(sri_type):
    if sri_type == "Probabilistic": return gr.Slider.update(visible=True)
    else: return gr.Slider.update(visible=False)

# fonction qui permet de définir les requêtes en fonction du type de SRI choisi
def get_set_queries(sri_type):
    return gr.Dropdown.update(choices=sri.get_queries(sri_type))

# fonction qui permet de définir si on affiche P@5 et P@10 en fonction du type de SRI choisi
def update_displayed_metrics(sri_type):
    if (sri_type == "Text Mining") or (sri_type == "Deep Learning"): return gr.Number.update(visible=False)
    else: return gr.Number.update(visible=True)

# fonction qui permet de définir si on affiche le bouton de calcul des métriques en fonction du type de SRI choisi
def update_metrics_button(sri_type):
    if (sri_type == "Boolean"): return gr.Button.update(visible=False)
    else: return gr.Number.update(visible=True)

# qui permet d'afficher le number box qui défini le nombre minimum de voisin pour DBSCAN si on choisit le SRI Text Mining
def update_dbscan_minn(sri_type):
    if (sri_type == "Text Mining"): return gr.Number.update(visible=True)
    else: return gr.Number.update(visible=False)

# qui permet d'afficher le slider qui défini la valeur du radius pour DBSCAN si on choisit le SRI Text Mining
def update_dbscan_rad(sri_type):
    if (sri_type == "Text Mining"): return gr.Slider.update(visible=True)
    else: return gr.Slider.update(visible=False)

# fonction qui permet de définir la liste des documents en fonction du type de stemmer choisi
def update_docs(stem):
    return gr.Dropdown.update(choices=files.get_docs(stem))

# fonction qui permet de définir la liste des termes en fonction du type de stemmer choisi
def update_terms(stem):
    return gr.Dropdown.update(choices=files.get_terms(stem))

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Informarion Retrival Project</h1></center> <center><h2>Intelligent Information Retrival System</h2></center>")
    with gr.Tab("Indexation"):
        with gr.Row():
            stemmer_selector = gr.Dropdown(["Porter", "Lancaster"], label="Stemmer") # menu déroulant pour choisir le type de stemmer
            btn0 = gr.Button("Indexing") # bouton pour lancer l'indexation
        with gr.Row():
            nb_doc = gr.Textbox(label="Number of documents") # textbox pour afficher le nombre de documents
            nb_words = gr.Textbox(label="Number of unique words") # textbox pour afficher le nombre de mots uniques
            btn0.click(fn=files.get_stats, inputs=[stemmer_selector], outputs=[nb_doc, nb_words]) # on appelle la fonction get_stats qui va lancer l'indexation et afficher les résultats
        with gr.Column(scale=1):
            with gr.Row():
                doc_selector = gr.Dropdown(label="Documents") # menu déroulant pour choisir les document à afficher
                stemmer_selector.change(update_docs, stemmer_selector, doc_selector) # on appelle la fonction update_docs qui va mettre à jour la liste des documents en fonction du type de stemmer choisi
                btn_doc = gr.Button("Show Document") # bouton pour afficher le fichier descripteur selon le document choisit
            with gr.Row():
                # dataframe pour afficher le fichier descripteur
                fichier_descripteur = gr.DataFrame(pd.DataFrame(columns=["Teme", "Freq", "Poids"]), label="Fichier Descripteur")
                # on appelle la fonction rech_fichier_descripteur qui va afficher le fichier descripteur selon le document choisi
                btn_doc.click(fn=files.rech_fichier_descripteur, inputs=[doc_selector, stemmer_selector], outputs=[fichier_descripteur])
        with gr.Column(scale=1):
            with gr.Row():
                term_selector = gr.Dropdown(label="Terms") # menu déroulant pour choisir les termes à afficher
                stemmer_selector.change(update_terms, stemmer_selector, term_selector) # on appelle la fonction update_terms qui va mettre à jour la liste des termes en fonction du type de stemmer choisi
                btn_term = gr.Button("Show Term") # bouton pour afficher le fichier inverse selon le terme choisi
            with gr.Row():
                # dataframe pour afficher le fichier inverse
                fichier_inverse = gr.DataFrame(pd.DataFrame(columns=["Teme", "Freq", "Poids"]), label="Fichier inverse")
                # on appelle la fonction rech_fichier_inverse qui va afficher le fichier inverse selon le terme choisi
                btn_term.click(fn=files.rech_fichier_inverse, inputs=[term_selector, stemmer_selector], outputs=[fichier_inverse])
    with gr.Tab("Appariement"):
        with gr.Row():
            with gr.Column(scale=1):
                # menu déroulant pour choisir le type de SRI
                sri_selector = gr.Dropdown(["Vectorial", "Boolean", "Probabilistic", "Text Mining", "Deep Learning"], label="SRI")

                function_selector = gr.Dropdown(label="Function", visible=True) # menu déroulant pour choisir le type de fonction de similarité
                sri_selector.change(get_function, sri_selector, function_selector) # on appelle la fonction get_function qui va mettre à jour la liste des fonctions en fonction du type de SRI choisi

                query_selector = gr.Dropdown(label="Query", visible=True) # menu déroulant pour choisir la requête à afficher
                sri_selector.change(get_set_queries, sri_selector, query_selector) # on appelle la fonction get_set_queries qui va mettre à jour la liste des requêtes en fonction du type de SRI choisi

                b_value = gr.Slider(0.5, 0.75, 0.6, label="b value", visible=False) # slider pour choisir la valeur de b
                sri_selector.change(update_sliders, sri_selector, b_value) # on appelle la fonction update_sliders qui va mettre à jour l'affichage du slider en fonction du type de SRI choisi
                k_value = gr.Slider(1.2, 2, 1.5, label="k value", visible=False) # slider pour choisir la valeur de k
                sri_selector.change(update_sliders, sri_selector, k_value) # on appelle la fonction update_sliders qui va mettre à jour l'affichage du slider en fonction du type de SRI choisi

                radius = gr.Slider(0, 1, 0.6, label="Radius", visible=False) # slider pour choisir la valeur du rayon
                sri_selector.change(update_dbscan_rad, sri_selector, radius) # on appelle la fonction update_dbscan_rad qui va mettre à jour l'affichage du slider en fonction du type de SRI choisi
                min_neighbours = gr.Number(value= 200, label="Min Neighbours", visible=False) # nombre pour choisir le nombre de voisins minimum
                sri_selector.change(update_dbscan_minn, sri_selector, min_neighbours) # on appelle la fonction update_dbscan_minn qui va mettre à jour l'affichage de la nombrebox en fonction du type de SRI choisi

                btn1 = gr.Button("Compute") # bouton pour lancer le calcul de la similarité
            with gr.Column(scale=3):
                df_out = gr.DataFrame(pd.DataFrame(columns=["Document", "Score"]), label="Results") # dataframe pour afficher les résultats
                # on appelle la fonction apply_sri qui va calculer la similarité selon le type de SRI choisi
                btn1.click(fn=sri.apply_sri, inputs=[stemmer_selector, query_selector, function_selector, b_value, k_value, radius, min_neighbours], outputs=df_out)
            with gr.Column(scale=1):
                p = gr.Number(label="Precision") # nombrebox pour afficher la précision
                p5 = gr.Number(label="P@5") # nombrebox pour afficher la précision à 5
                sri_selector.change(update_displayed_metrics, sri_selector, p5) # on appelle la fonction update_displayed_metrics qui va mettre à jour l'affichage des nombrebox en fonction du type de SRI choisi
                p10 = gr.Number(label="P@10") # nombrebox pour afficher la précision à 10
                sri_selector.change(update_displayed_metrics, sri_selector, p10) # on appelle la fonction update_displayed_metrics qui va mettre à jour l'affichage des nombrebox en fonction du type de SRI choisi
                metrics_out = [p, p5, p10, gr.Number(label="Recall"), gr.Number(label="F measure"), gr.Plot()] # liste des nombrebox pour afficher les métriques
                btn2 = gr.Button("Compute Metrics") # bouton pour lancer le calcul des métriques
                sri_selector.change(update_metrics_button, sri_selector, btn2) # on appelle la fonction update_metrics_button qui va mettre à jour l'affichage du bouton en fonction du type de SRI choisi
                btn2.click(fn=sri.metrics, inputs=[query_selector, df_out], outputs=metrics_out) # on appelle la fonction metrics qui va calculer les métriques selon le type de SRI choisi

if __name__ == "__main__":
    demo.launch()