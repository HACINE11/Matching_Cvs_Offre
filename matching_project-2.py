import os
import re
import time
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords français (une seule fois)
nltk.download("stopwords")
stop_words = set(stopwords.words("french"))

# Sémaphore pour la synchronisation dans la version multi-threads
semaphore = threading.Semaphore(4)  # Limite le nombre de threads actifs

# File de tâches pour le modèle Producteur/Consommateur
task_queue = Queue()

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    """Nettoie le texte et supprime les stopwords."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Supprime la ponctuation
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def display_results(results, execution_time, mode):
    """Affiche les résultats de matching."""
    print(f"\nRésultats {mode} :")
    print(f"Temps d'exécution {mode}: {execution_time:.4f} secondes")
    print("\nMatching des CVs (en pourcentage) :")
    
    for cv_name, similarity_score in results:
        match_percentage = round(similarity_score * 100, 2)
        print(f"{cv_name} : {match_percentage}% de matching")

def threaded_worker(job_offer_vector, task_queue, results_queue):
    """Travailleur consommateur dans le modèle Producteur/Consommateur."""
    while not task_queue.empty():
        cv_file, cv_vector = task_queue.get()
        
        # Utilise le sémaphore pour gérer la synchronisation
        with semaphore:
            similarity_score = cosine_similarity(job_offer_vector, cv_vector).flatten()[0]
            results_queue.put((cv_file, similarity_score))
        
        task_queue.task_done()

def multithreaded_implementation(job_offer_text, cv_folder_path):
    """Implémentation Multi-threads avec Producteurs/Consommateurs et sémaphores."""
    start_time = time.time()
    
    # Charger tous les CVs et prétraiter les textes
    cv_files = [(cv_file, os.path.join(cv_folder_path, cv_file)) 
                for cv_file in os.listdir(cv_folder_path) if cv_file.endswith('.pdf')]
    cv_texts = [(cv_file, preprocess_text(extract_text_from_pdf(cv_path)))
                for cv_file, cv_path in cv_files]
    
    # Initialiser le vectorizer et créer le vocabulaire
    vectorizer = TfidfVectorizer()
    all_texts = [job_offer_text] + [text for _, text in cv_texts]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    job_offer_vector = tfidf_matrix[0]      # Vecteur de l'offre d'emploi
    cv_vectors = tfidf_matrix[1:]           # Vecteurs des CVs

    # Mettre les tâches dans la queue
    for i, (cv_file, _) in enumerate(cv_texts):
        task_queue.put((cv_file, cv_vectors[i]))

    # File pour récupérer les résultats
    results_queue = Queue()
    
    # Créer et démarrer les threads travailleurs
    threads = []
    for _ in range(4):  # Créer 4 threads pour traiter les CVs
        thread = threading.Thread(target=threaded_worker, args=(job_offer_vector, task_queue, results_queue))
        thread.start()
        threads.append(thread)
    
    # Attendre que toutes les tâches soient traitées
    task_queue.join()

    # Attendre que tous les threads terminent
    for thread in threads:
        thread.join()

    # Récupérer et afficher les résultats
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Trier les résultats par score de similarité
    results = sorted(results, key=lambda x: x[1], reverse=True)

    execution_time = time.time() - start_time
    display_results(results, execution_time, "Multi-threads")

# Exemple d'utilisation
job_offer_path = "offre_emploi.pdf"  # Chemin du fichier offre d'emploi
cv_folder_path = "CVs"               # Dossier contenant les CVs en PDF

# Prétraiter le texte de l'offre d'emploi
job_offer_text = preprocess_text(extract_text_from_pdf(job_offer_path))

# Exécuter l'implémentation multi-threads
print("\nRésultats Multi-threads optimiser :\n")
multithreaded_implementation(job_offer_text, cv_folder_path)

