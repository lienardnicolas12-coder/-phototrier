import os
import shutil
import rawpy
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
CATEGORIES = [
    "paysage", "portrait", "macro", "nocturne","ciel", "urbain",
    "animale", "autre"  # Modifié le dossier 'noir_et_blanc' en 'urbain'
]

DOSSIER_A_TRIER = r"C:\Users\ln\Documents\photos\tri_photos_raw\a_trier"
DOSSIER_TRIES = r"C:\Users\ln\Documents\photos\tri_photos_raw\tries"

# Charger le modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de : {device}")
model = SentenceTransformer('clip-ViT-B-32', device=device)

# Préparer les textes pour les catégories
text_inputs = [
    "une photo de paysage avec montagnes, vue large, horizon, nature",
    "un portrait de personne, visage, premier plan, sourire",
    "une photo macro de fleur ou insecte, gros plan, détails",
    "une photo de nuit, ciel étoilé, faible luminosité, ville, nocturne",
    "une photo d'un endroit urbain, ville, buildings, rues",
    "autre type de photo",
    "une photo de ciel avec des nuages",
    "une photo d'animal, chien, chat, oiseau, sauvage"
]
text_embeddings = model.encode(text_inputs, convert_to_tensor=True)

# --- FONCTION POUR CONVERTIR UN RAW EN IMAGE PIL ---
def raw_to_pil(raw_path):
    try:
        with rawpy.imread(raw_path) as raw:
            rgb_array = raw.postprocess()
            return Image.fromarray(rgb_array.astype('uint8'))
    except Exception as e:
        print(f"Erreur avec {raw_path} : {e}")
        return None

# --- FONCTION POUR CLASSER UNE IMAGE (version optimisée) ---
def classer_image(chemin_image):
    try:
        # Priorité aux fichiers RAW
        if chemin_image.lower().endswith(('.raw', '.cr2', '.nef', '.arw', '.dng')):
            image = raw_to_pil(chemin_image)
            if image is None:
                return "autre"
        else:
            try:
                image = Image.open(chemin_image)
            except Exception as e:
                print(f"Erreur d'ouverture de l'image {chemin_image}: {e}")
                return "autre"

        # Optimisation : réduire la taille de l'image avant le traitement
        if max(image.size) > 512:  # Réduire les images trop grandes
            image = image.resize((512, 512))

        image_embedding = model.encode(image, convert_to_tensor=True)
        similarities = util.cos_sim(image_embedding, text_embeddings)[0]
        categorie = CATEGORIES[similarities.argmax().item()]

        # Ajout d'une vérification de confiance
        max_similarity = similarities.max().item()
        if max_similarity < 0.3:  # Seuil de confiance
            return "autre"

        return categorie
    except Exception as e:
        print(f"Erreur avec {chemin_image}: {e}")
        return "autre"

# --- FONCTION POUR DÉPLACER L'IMAGE ET SES METADATA (version améliorée) ---
def deplacer_image_et_metadonnees(chemin_source, categorie):
    try:
        chemin_source = os.path.normpath(chemin_source)
        dossier_dest = os.path.join(DOSSIER_TRIES, "a publier", categorie)
        os.makedirs(dossier_dest, exist_ok=True)

        # Déplacer l'image
        nom_fichier = os.path.basename(chemin_source)
        chemin_dest = os.path.join(dossier_dest, nom_fichier)

        if os.path.exists(chemin_dest):
            stem = os.path.splitext(nom_fichier)[0]
            suffix = os.path.splitext(nom_fichier)[1]
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{suffix}"
                chemin_dest = os.path.join(dossier_dest, new_name)
                if not os.path.exists(chemin_dest):
                    break
                counter += 1

        shutil.move(chemin_source, chemin_dest)

        # Déplacer les métadonnées correspondantes (si elles existent)
        base_name = os.path.splitext(os.path.basename(chemin_source))[0]
        for ext in ['.THM', '.XMP']:
            meta_file = f"{base_name}{ext}"
            if os.path.exists(meta_file):
                shutil.move(meta_file, os.path.join(dossier_dest, meta_file))

        print(f"Déplacé : {nom_fichier} → {categorie}")
    except Exception as e:
        print(f"⚠️ Impossible de déplacer {chemin_source} : {e}")

# --- FONCTION POUR TRIER UN DOSSIER COMPLET ---
def trier_dossier(dossier):
    # Tri par type de fichier d'abord (RAW en priorité)
    files = []
    for root, _, filenames in os.walk(dossier):
        for filename in filenames:
            filepath = Path(root) / filename
            if filepath.suffix.lower() not in ['.THM', '.XMP']:
                files.append(filepath)

    # Priorité aux fichiers RAW
    raw_files = [f for f in files if f.suffix.lower() in ['.raw', '.cr2', '.nef', '.arw', '.dng']]
    other_files = [f for f in files if f not in raw_files]

    all_files = raw_files + other_files

    for filepath in all_files:
        print(f"Nouvelle photo détectée : {filepath}")
        categorie = classer_image(str(filepath))
        deplacer_image_et_metadonnees(str(filepath), categorie)

# --- GESTIONNAIRE D'ÉVÉNEMENTS POUR WATCHDOG ---
class MonGestionnaire(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            # Ignorer les fichiers de métadonnées (ils seront déplacés avec leur image correspondante)
            if not any(event.src_path.lower().endswith(ext) for ext in ['.THM', '.XMP']):
                print(f"\nNouvelle photo détectée : {event.src_path}")
                categorie = classer_image(event.src_path)
                deplacer_image_et_metadonnees(event.src_path, categorie)
        else:
            print(f"\nNouveau dossier détecté : {event.src_path}")
            trier_dossier(event.src_path)

# --- LANCER LA SURVEILLANCE DU DOSSIER ---
if __name__ == "__main__":
    print("🚀 Démarrage du tri automatique des photos (RAW inclus)...")
    print(f"Place your photos or folders in '{DOSSIER_A_TRIER}' and they will be automatically sorted!")

    os.makedirs(DOSSIER_A_TRIER, exist_ok=True)
    os.makedirs(DOSSIER_TRIES, exist_ok=True)

    # Trier les photos déjà présentes dans a_trier
    trier_dossier(DOSSIER_A_TRIER)

    event_handler = MonGestionnaire()
    observer = Observer()
    observer.schedule(event_handler, path=DOSSIER_A_TRIER, recursive=True)
    observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
