import os
import re
import time
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import nltk
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor


nltk.download("stopwords")   # Telecharger les stopwords francais (une seule fois)
stop_words = set(stopwords.words("french"))

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier {pdf_path}: {e}")
    return text

def preprocess_text(text):
    """Nettoie le texte et supprime les stopwords."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def display_results(results, execution_time, mode):
    print(f"\n Resultats {mode} :")
    print(f"Temps d'execution {mode}: {execution_time:.4f} secondes")
    print("\nMatching des CVs (en pourcentage) :")
    for cv_name, similarity_score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{cv_name} : {round(similarity_score * 100, 2)}% de matching")

def calculate_similarity(job_offer_text, cv_texts):
    """Calcule la similarite entre le texte de l'offre et chaque CV."""
    vectorizer = TfidfVectorizer()
    documents = [job_offer_text] + [cv[1] for cv in cv_texts]
    tfidf_matrix = vectorizer.fit_transform(documents)
    job_offer_vector = tfidf_matrix[0]
    cv_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(job_offer_vector, cv_vectors).flatten()
    return [(cv_texts[i][0], similarities[i]) for i in range(len(similarities))]

def sequential_implementation(job_offer_text, cv_folder_path):
    """IMP monothread."""
    start_time = time.time()
    cv_texts = [(cv_file, preprocess_text(extract_text_from_pdf(os.path.join(cv_folder_path, cv_file))))
                for cv_file in os.listdir(cv_folder_path) if cv_file.endswith('.pdf')]
    results = calculate_similarity(job_offer_text, cv_texts)
    display_results(results, time.time() - start_time, "Monothread")

"""IMP multiprocessus."""
def parallel_implementation(job_offer_text, cv_folder_path):  
    start_time = time.time()
    cv_files = [(cv_file, os.path.join(cv_folder_path, cv_file)) 
                for cv_file in os.listdir(cv_folder_path) if cv_file.endswith('.pdf')]
    
    with multiprocessing.Pool() as pool:
        cv_texts = [(cv_file, preprocess_text(pool.apply(extract_text_from_pdf, args=(cv_path,))))
                    for cv_file, cv_path in cv_files]
    
    results = calculate_similarity(job_offer_text, cv_texts)
    display_results(results, time.time() - start_time, "Multiprocessus")

def threaded_worker(cv_file, job_offer_vector, cv_vector, results):
    """Calcul de la similarite pour chaque CV en utilisant des threads."""
    similarity_score = cosine_similarity(job_offer_vector, cv_vector).flatten()[0]
    results.append((cv_file, similarity_score))



def multithreaded_implementation(job_offer_text, cv_folder_path):
    """IMP Multithreads."""
    start_time = time.time()
    cv_files = [(cv_file, os.path.join(cv_folder_path, cv_file)) 
                for cv_file in os.listdir(cv_folder_path) if cv_file.endswith('.pdf')]

    cv_texts = [(cv_file, preprocess_text(extract_text_from_pdf(cv_path)))
                for cv_file, cv_path in cv_files]
    
    # Initialiser le vectorizer sur l'ensemble des textes
    vectorizer = TfidfVectorizer()
    all_texts = [job_offer_text] + [text for _, text in cv_texts]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    job_offer_vector = tfidf_matrix[0]
    cv_vectors = tfidf_matrix[1:]
    
    # Calcul parallele des similarités
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(threaded_worker, cv_files[i][0], job_offer_vector, cv_vectors[i], results)
                   for i in range(cv_vectors.shape[0])]  # Utiliser cv_vectors.shape[0] au lieu de len(cv_vectors)
        for future in futures:
            future.result()  # S'assurer que chaque tâche est terminée
    
    display_results(results, time.time() - start_time, "Multi-threads")




job_offer_path = "offre_emploi.pdf"  # Chemin du fichier offre d'emploi
cv_folder_path = "CVs"               # Dossier contenant les CVs en PDF

job_offer_text = preprocess_text(extract_text_from_pdf(job_offer_path))

print("Résultats Monothread : ")
sequential_implementation(job_offer_text, cv_folder_path)

print("\nResultats Multi-Processus : ")
parallel_implementation(job_offer_text, cv_folder_path)

print("\nResultats Multi-threads : ")
multithreaded_implementation(job_offer_text, cv_folder_path)

