import os
import shutil
import rawpy
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
CATEGORIES = [
    "paysage", "portrait", "macro", "nocturne", "ciel", "architecture", "urbain",
    "animale", "autre"
]

# Utiliser des chemins relatifs ou configurer via variables d'environnement
DOSSIER_BASE = os.path.join(os.path.expanduser("~"), "Documents", "photos", "tri_photos_raw")
DOSSIER_A_TRIER = os.path.join(DOSSIER_BASE, "a_trier")
DOSSIER_TRIES = os.path.join(DOSSIER_BASE, "tries")

# Créer les dossiers s'ils n'existent pas
os.makedirs(DOSSIER_A_TRIER, exist_ok=True)
os.makedirs(DOSSIER_TRIES, exist_ok=True)

# Charger le modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de : {device}")

try:
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    print("✅ Modèle CLIP chargé avec succès")
except Exception as e:
    print(f"❌ Échec du chargement du modèle: {e}")
    # Essayer un modèle plus léger
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("✅ Modèle all-MiniLM-L6-v2 chargé à la place")
    except Exception as e2:
        print(f"❌ Échec du chargement du modèle de secours: {e2}")
        exit(1)

# Préparer les textes pour les catégories
text_inputs = [
    "une photo de paysage avec montagnes, vue large, horizon, nature",
    "un portrait de personne, visage, premier plan, sourire",
    "une photo macro de fleur ou insecte, gros plan, détails",
    "une photo de nuit, ciel étoilé, faible luminosité, ville, nocturne",
    "une photo de ciel avec des nuages",
    "une photo de bâtiment ou élément de bâtiment",
    "une photo d'un endroit urbain, ville, buildings, rues",
    "une photo d'animal, chien, chat, oiseau, sauvage, mammifère, reptile, poisson, faune",
    "autre type de photo"
]

try:
    text_embeddings = model.encode(text_inputs, convert_to_tensor=True)
    print("✅ Encodage des catégories terminé")
except Exception as e:
    print(f"❌ Échec de l'encodage des catégories: {e}")
    exit(1)

# --- FONCTION POUR CONVERTIR UN RAW EN IMAGE PIL ---
def raw_to_pil(raw_path):
    """Convertit un fichier RAW en image PIL"""
    try:
        with rawpy.imread(raw_path) as raw:
            # Post-process avec des paramètres par défaut
            rgb_array = raw.postprocess(
                use_camera_wb=True,  # Utiliser le balance des blancs de l'appareil
                output_color=rawpy.ColorSpace.sRGB,
                no_auto_bright=True,
                gamma=(1, 1)  # Gamma linéaire
            )
            # Convertir en uint8 et créer l'image PIL
            return Image.fromarray(rgb_array.astype('uint8'))
    except Exception as e:
        print(f"⚠️ Erreur avec {raw_path}: {e}")
        return None

# --- FONCTION POUR CLASSER UNE IMAGE ---
def classer_image(chemin_image):
    """Classifie une image dans une catégorie"""
    try:
        # Priorité aux fichiers RAW
        if chemin_image.lower().endswith(('.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.raf', '.x3f')):
            image = raw_to_pil(chemin_image)
            if image is None:
                return "autre"
        else:
            try:
                image = Image.open(chemin_image)
            except Exception as e:
                print(f"⚠️ Erreur d'ouverture de l'image {chemin_image}: {e}")
                return "autre"

        # Redimensionner les images trop grandes pour optimiser le traitement
        if max(image.size) > 768:
            image = image.resize((768, 768))

        image_embedding = model.encode(image, convert_to_tensor=True)
        similarities = util.cos_sim(image_embedding, text_embeddings)[0]
        categorie = CATEGORIES[similarities.argmax().item()]

        # Vérification de confiance
        max_similarity = similarities.max().item()
        if max_similarity < 0.2:  # Seuil de confiance plus bas
            return "autre"

        return categorie
    except Exception as e:
        print(f"⚠️ Erreur avec {chemin_image}: {e}")
        return "autre"

# --- FONCTION POUR DÉPLACER L'IMAGE ET SES METADATA ---
def deplacer_image_et_metadonnees(chemin_source, categorie):
    """Déplace une image et ses fichiers de métadonnées associés"""
    try:
        chemin_source = os.path.normpath(chemin_source)
        dossier_dest = os.path.join(DOSSIER_TRIES, categorie)
        os.makedirs(dossier_dest, exist_ok=True)

        # Déplacer l'image
        nom_fichier = os.path.basename(chemin_source)
        chemin_dest = os.path.join(dossier_dest, nom_fichier)

        # Gérer les conflits de noms
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
        dossier_source = os.path.dirname(chemin_source)
        base_name = os.path.splitext(os.path.basename(chemin_source))[0]
        
        # Extensions de métadonnées à rechercher
        meta_extensions = ['.THM', '.XMP', '.thm', '.xmp', '.json']
        for ext in meta_extensions:
            meta_file = os.path.join(dossier_source, f"{base_name}{ext}")
            if os.path.exists(meta_file):
                try:
                    shutil.move(meta_file, os.path.join(dossier_dest, os.path.basename(meta_file)))
                    print(f"  → Métadonnées déplacées: {os.path.basename(meta_file)}")
                except Exception as e:
                    print(f"  ⚠️ Impossible de déplacer {meta_file}: {e}")

        print(f"✅ Déplacé: {nom_fichier} → {categorie}")
        return True
    except Exception as e:
        print(f"⚠️ Impossible de déplacer {chemin_source}: {e}")
        return False

# --- FONCTION POUR TRIER UN DOSSIER COMPLET ---
def trier_dossier(dossier):
    """Trie toutes les images dans un dossier"""
    try:
        files = []
        for root, _, filenames in os.walk(dossier):
            for filename in filenames:
                filepath = Path(root) / filename
                # Ignorer les fichiers de métadonnées (ils seront déplacés avec leur image)
                if filepath.suffix.lower() not in ['.thm', '.xmp', '.json']:
                    files.append(filepath)

        # Priorité aux fichiers RAW
        raw_extensions = {'.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.raf', '.x3f'}
        raw_files = [f for f in files if f.suffix.lower() in raw_extensions]
        other_files = [f for f in files if f not in raw_files]
        all_files = raw_files + other_files

        print(f"\n📁 Trouvé {len(all_files)} fichier(s) à trier dans {dossier}")
        
        for filepath in all_files:
            print(f"\n📷 Traitement de: {filepath}")
            categorie = classer_image(str(filepath))
            deplacer_image_et_metadonnees(str(filepath), categorie)
            
    except Exception as e:
        print(f"❌ Erreur lors du tri du dossier {dossier}: {e}")

# --- GESTIONNAIRE D'ÉVÉNEMENTS POUR WATCHDOG ---
class MonGestionnaire(FileSystemEventHandler):
    def on_created(self, event):
        """Gère les nouveaux fichiers créés"""
        if not event.is_directory:
            # Ignorer les fichiers de métadonnées (ils seront déplacés avec leur image correspondante)
            if not any(event.src_path.lower().endswith(ext) for ext in ['.thm', '.xmp', '.json']):
                print(f"\n🆕 Nouvelle photo détectée: {event.src_path}")
                categorie = classer_image(event.src_path)
                deplacer_image_et_metadonnees(event.src_path, categorie)
        else:
            print(f"\n📁 Nouveau dossier détecté: {event.src_path}")
            trier_dossier(event.src_path)

# --- LANCER LA SURVEILLANCE DU DOSSIER ---
if __name__ == "__main__":
    print("🚀 Démarrage du tri automatique des photos (RAW inclus)...")
    print(f"Placez vos photos dans '{DOSSIER_A_TRIER}' et elles seront automatiquement triées!")
    print(f"Les photos triées seront placées dans '{DOSSIER_TRIES}'")
    print("\nAppuyez sur Ctrl+C pour arrêter...\n")

    # Trier les photos déjà présentes dans a_trier
    trier_dossier(DOSSIER_A_TRIER)

    # Démarrer la surveillance
    event_handler = MonGestionnaire()
    observer = Observer()
    observer.schedule(event_handler, path=DOSSIER_A_TRIER, recursive=True)
    observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de la surveillance...")
        observer.stop()
    observer.join()
    print("✅ Surveillance arrêtée")
