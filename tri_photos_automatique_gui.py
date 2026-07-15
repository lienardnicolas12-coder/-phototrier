import os
import shutil
import json
import rawpy
import logging
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QListWidget, QTextEdit,
                            QTabWidget, QFrame, QFileDialog, QMessageBox, QCheckBox,
                            QGroupBox, QRadioButton)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sentence_transformers import SentenceTransformer, util
import torch
import tensorflow as tf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== CONFIGURATION ======================
class Config:
    # Utiliser des chemins relatifs ou basés sur le répertoire personnel
    BASE_DIR = Path.home() / "Documents" / "photos" / "tri_photo_RAW"
    DOSSIER_A_TRIER = BASE_DIR / "a_trier"
    DOSSIER_TRIES = BASE_DIR / "tries"
    MODEL_SAVE_PATH = BASE_DIR / "models"
    TAGS_FILE = BASE_DIR / "tags.json"

    # Catégories et descriptions OPTIMISÉES
    CATEGORIES = {
        "paysage": "une photo de paysage avec montagnes, vue large, horizon, nature",
        "portrait": "un portrait de personne, visage, premier plan, sourire",
        "macro": "une photo macro de fleur ou insecte, gros plan, détails",
        "nocturne": "une photo de nuit, ciel étoilé, faible luminosité, ville, nocturne",
        "ciel": "une photo de ciel avec des nuages",
        "architecture": "une photo de bâtiment ou élément de bâtiment",
        "urbain": "une photo d'un endroit urbain, ville, buildings, rues",
        "animale": "une photo d'animal, chien, chat, oiseau, sauvage, mammifère, reptile, poisson, faune",
        "autre": "autre type de photo"
    }
    
    RAW_EXTENSIONS = {'.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.raf', '.x3f'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    METADATA_EXTENSIONS = {'.thm', '.xmp', '.json'}

    @classmethod
    def ensure_directories(cls):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOSSIER_A_TRIER.mkdir(parents=True, exist_ok=True)
        cls.DOSSIER_TRIES.mkdir(parents=True, exist_ok=True)
        cls.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ====================== WATCHDOG THREAD-SAFE ======================
class WatchdogWorker(QThread):
    file_detected = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.observer = None

    def run(self):
        event_handler = FileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()
        while not self.isInterruptionRequested():
            self.sleep(1)
        self.observer.stop()
        self.observer.join()

    def stop(self):
        self.requestInterruption()
        if self.observer:
            self.observer.stop()

class FileHandler(FileSystemEventHandler):
    def __init__(self, worker):
        self.worker = worker

    def on_created(self, event):
        if not event.is_directory:
            if not any(event.src_path.lower().endswith(ext) for ext in ['.thm', '.xmp', '.json']):
                self.worker.file_detected.emit(event.src_path)

# ====================== GESTION DES TAGS ======================
class TagManager:
    def __init__(self, tags_file):
        self.tags_file = Path(tags_file)
        self.tags = self._load_tags()

    def _load_tags(self):
        if self.tags_file.exists():
            try:
                with open(self.tags_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Erreur de chargement des tags: {e}")
                return {}
        return {}

    def _save_tags(self):
        try:
            with open(self.tags_file, 'w', encoding='utf-8') as f:
                json.dump(self.tags, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Erreur de sauvegarde des tags: {e}")

    def add_tag(self, image_path, tag):
        image_path = str(image_path)
        if image_path not in self.tags:
            self.tags[image_path] = set()
        self.tags[image_path].add(tag)
        self._save_tags()

    def remove_tag(self, image_path, tag):
        image_path = str(image_path)
        if image_path in self.tags and tag in self.tags[image_path]:
            self.tags[image_path].remove(tag)
            if not self.tags[image_path]:
                del self.tags[image_path]
            self._save_tags()

    def get_tags(self, image_path):
        image_path = str(image_path)
        return list(self.tags.get(image_path, set()))

    def get_all_tags(self):
        all_tags = set()
        for path_tags in self.tags.values():
            all_tags.update(path_tags)
        return sorted(all_tags)

    def get_images_by_tag(self, tag):
        tag = tag.lower()
        results = []
        for image_path, tags in self.tags.items():
            if tag in [t.lower() for t in tags]:
                results.append(image_path)
        return results

# ====================== APPLICATION PRINCIPALE ======================
class PhotoSorterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoTrier")
        self.setGeometry(100, 100, 1000, 700)
        try:
            self.setWindowIcon(QIcon("logo.ico"))
        except:
            pass

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #3c3f41; color: #ffffff; padding: 8px; }
            QTabBar::tab:selected { background-color: #509cfb; }
            QLabel, QLineEdit, QGroupBox { background-color: #3c3f41; color: #ffffff; border: 1px solid #444; border-radius: 4px; padding: 4px; }
            QPushButton { background-color: #509cfb; color: white; border: none; padding: 8px 16px; border-radius: 4px; min-width: 100px; }
            QPushButton:hover { background-color: #6fa2f7; }
            QListWidget { background-color: #3c3f41; color: white; border: 1px solid #509cfb; border-radius: 4px; }
            QTextEdit { background-color: #1e1e1e; color: #00ff00; border: 1px solid #444; font-family: Consolas, monospace; }
            QCheckBox, QRadioButton { spacing: 8px; color: #ffffff; }
        """)

        # Initialiser la configuration et créer les répertoires
        Config.ensure_directories()
        
        self.config = Config()
        self.tag_manager = TagManager(self.config.TAGS_FILE)
        self.model_clip = None
        self.model_tf = None
        self.watchdog_worker = None
        self.text_embeddings = None

        # Attributs de l'interface
        self.entry_source = None
        self.entry_output = None
        self.log_text = None
        self.check_watch = None
        self.entry_image_path = None
        self.new_tag_input = None
        self.tag_search = None
        self.search_results = None
        self.image_label = None
        self.current_tags_label = None
        self.tf_model_path = None
        self.tf_status = None
        self.folder_path_input = None

        self.init_ui()
        self.load_models()

    # ====================== UTILITAIRES ======================
    def pil2pixmap(self, pil_image):
        """Convertit une image PIL en QPixmap"""
        if pil_image is None:
            return QPixmap()
            
        if pil_image.mode == "RGBA":
            img_data = pil_image.tobytes("raw", "RGBA")
            qimg = QImage(img_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        elif pil_image.mode == "RGB":
            img_data = pil_image.tobytes("raw", "RGB")
            qimg = QImage(img_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
        else:
            pil_image = pil_image.convert("RGB")
            img_data = pil_image.tobytes("raw", "RGB")
            qimg = QImage(img_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def log_message(self, message):
        """Ajoute un message au journal"""
        if self.log_text:
            self.log_text.append(message)

    # ====================== INIT UI ======================
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.setup_auto_tab()
        self.setup_tags_tab()
        self.setup_tf_tab()
        self.setup_publish_tab()

    def setup_auto_tab(self):
        self.auto_tab = QWidget()
        self.tabs.addTab(self.auto_tab, "📷 Tri Automatique")
        layout = QVBoxLayout(self.auto_tab)
        layout.setSpacing(15)

        # Dossier à trier
        source_frame = QFrame()
        source_layout = QHBoxLayout(source_frame)
        source_layout.addWidget(QLabel("Dossier à trier:"))
        self.entry_source = QLineEdit(str(self.config.DOSSIER_A_TRIER))
        self.entry_source.setMinimumWidth(400)
        source_layout.addWidget(self.entry_source)
        browse_source_btn = QPushButton("Parcourir")
        browse_source_btn.clicked.connect(self.browse_source)
        source_layout.addWidget(browse_source_btn)
        layout.addWidget(source_frame)

        # Dossier de sortie
        output_frame = QFrame()
        output_layout = QHBoxLayout(output_frame)
        output_layout.addWidget(QLabel("Dossier de sortie:"))
        self.entry_output = QLineEdit(str(self.config.DOSSIER_TRIES))
        self.entry_output.setMinimumWidth(400)
        output_layout.addWidget(self.entry_output)
        browse_output_btn = QPushButton("Parcourir")
        browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(browse_output_btn)
        layout.addWidget(output_frame)

        # Options modèle
        options_group = QGroupBox("Modèle de classification")
        options_layout = QVBoxLayout()
        self.radio_clip = QRadioButton("CLIP (recommandé)")
        self.radio_clip.setChecked(True)
        self.radio_tf = QRadioButton("TensorFlow Personnalisé")
        options_layout.addWidget(self.radio_clip)
        options_layout.addWidget(self.radio_tf)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Surveillance
        watch_group = QGroupBox("Surveillance")
        watch_layout = QHBoxLayout()
        self.check_watch = QCheckBox("Démarrer la surveillance")
        self.check_watch.stateChanged.connect(self.toggle_watchdog)
        watch_layout.addWidget(self.check_watch)
        watch_group.setLayout(watch_layout)
        layout.addWidget(watch_group)

        # Bouton Trier maintenant
        sort_btn = QPushButton("📁 Trier maintenant")
        sort_btn.setStyleSheet("font-weight: bold;")
        sort_btn.clicked.connect(self.trier_dossier_maintenant)
        layout.addWidget(sort_btn)

        # Journal
        log_group = QGroupBox("Journal")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def setup_tags_tab(self):
        self.tags_tab = QWidget()
        self.tabs.addTab(self.tags_tab, "🏷️ Gestion des Tags")
        layout = QVBoxLayout(self.tags_tab)

        # Recherche par dossier
        folder_search_group = QGroupBox("Rechercher dans un dossier")
        folder_search_layout = QHBoxLayout()
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Dossier à scanner...")
        folder_search_layout.addWidget(self.folder_path_input)
        browse_folder_btn = QPushButton("📁 Parcourir dossier")
        browse_folder_btn.clicked.connect(self.browse_folder_for_tags)
        folder_search_layout.addWidget(browse_folder_btn)
        folder_search_group.setLayout(folder_search_layout)
        layout.addWidget(folder_search_group)

        # Sélection de l'image
        select_frame = QFrame()
        select_layout = QHBoxLayout(select_frame)
        self.entry_image_path = QLineEdit()
        self.entry_image_path.setMinimumWidth(400)
        browse_img_btn = QPushButton("Parcourir image")
        browse_img_btn.clicked.connect(self.browse_image)
        select_layout.addWidget(self.entry_image_path)
        select_layout.addWidget(browse_img_btn)
        layout.addWidget(select_frame)

        # Aperçu de l'image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("border: 2px solid #509cfb; background-color: #1e1e1e;")
        layout.addWidget(self.image_label)

        # Tags actuels
        self.current_tags_label = QLabel("Tags actuels : Aucun")
        self.current_tags_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        layout.addWidget(self.current_tags_label)

        # Ajouter un tag
        tag_group = QGroupBox("Ajouter un tag à cette image")
        tag_layout = QHBoxLayout()
        self.new_tag_input = QLineEdit()
        self.new_tag_input.setPlaceholderText("Ex: cheval, paysage...")
        tag_layout.addWidget(self.new_tag_input)
        add_tag_btn = QPushButton("➕ Ajouter le tag")
        add_tag_btn.clicked.connect(self.add_tag_to_image)
        tag_layout.addWidget(add_tag_btn)
        tag_group.setLayout(tag_layout)
        layout.addWidget(tag_group)

        # Recherche par tag
        search_group = QGroupBox("Rechercher des images par tag")
        search_layout = QHBoxLayout()
        self.tag_search = QLineEdit()
        self.tag_search.setPlaceholderText("Ex: cheval, paysage...")
        self.tag_search.setMinimumWidth(300)
        search_layout.addWidget(self.tag_search)
        search_btn = QPushButton("🔍 Rechercher")
        search_btn.clicked.connect(self.search_by_tag)
        search_layout.addWidget(search_btn)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Résultats de recherche
        self.search_results = QListWidget()
        self.search_results.itemDoubleClicked.connect(self.load_image_from_search)
        layout.addWidget(self.search_results)

    def setup_tf_tab(self):
        self.tf_tab = QWidget()
        self.tabs.addTab(self.tf_tab, "🤖 Modèle TensorFlow")
        layout = QVBoxLayout(self.tf_tab)

        # Sélection du modèle
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        self.tf_model_path = QLineEdit()
        self.tf_model_path.setMinimumWidth(400)
        model_layout.addWidget(self.tf_model_path)
        browse_model_btn = QPushButton("Parcourir...")
        browse_model_btn.clicked.connect(self.browse_tf_model)
        model_layout.addWidget(browse_model_btn)
        layout.addWidget(model_frame)

        # Bouton pour télécharger un modèle par défaut
        download_btn = QPushButton("❓ Comment obtenir un modèle ?")
        download_btn.clicked.connect(self.download_default_model)
        layout.addWidget(download_btn)

        # Bouton charger
        load_model_btn = QPushButton("📁 Charger le modèle TensorFlow")
        load_model_btn.clicked.connect(self.load_tf_model)
        layout.addWidget(load_model_btn)

        # Statut
        self.tf_status = QLabel("Aucun modèle TensorFlow chargé")
        layout.addWidget(self.tf_status)

    def setup_publish_tab(self):
        self.publish_tab = QWidget()
        self.tabs.addTab(self.publish_tab, "📤 À Publier")
        layout = QVBoxLayout(self.publish_tab)
        layout.addWidget(QLabel("Les photos triées sont disponibles dans le dossier de sortie sélectionné."))
        layout.addWidget(QLabel("Utilise le bouton 'Parcourir' dans l'onglet 'Tri Automatique' pour changer le dossier."))

    # ====================== CHARGEMENT DES MODÈLES ======================
    def load_models(self):
        """Charge le modèle CLIP"""
        self.log_message("📁 Chargement du modèle CLIP (cela peut prendre 1-2 min)...")
        self.log_message("   → Le modèle pèse ~500Mo. Sois patient !")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_message(f"   → Appareil utilisé : {device}")
            
            # Essayer de charger CLIP-ViT-B-32
            try:
                self.model_clip = SentenceTransformer('clip-ViT-B-32', device=device)
                self.log_message("✅ Modèle CLIP-ViT-B-32 chargé avec succès !")
            except Exception as e1:
                self.log_message(f"⚠️ Échec de CLIP-ViT-B-32: {e1}")
                self.log_message("   → Tentative avec un modèle plus léger...")
                try:
                    self.model_clip = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    self.log_message("✅ Modèle all-MiniLM-L6-v2 chargé avec succès !")
                except Exception as e2:
                    self.log_message(f"❌ Échec du chargement de tous les modèles: {e2}")
                    QMessageBox.critical(
                        self, "Erreur", 
                        f"Échec du chargement des modèles:\n{e2}\n\n"
                        "Vérifiez votre connexion Internet et que vous avez assez d'espace disque."
                    )
                    return
            
            self.log_message("   → Encodage des catégories...")
            self.text_inputs = list(self.config.CATEGORIES.values())
            self.text_embeddings = self.model_clip.encode(self.text_inputs, convert_to_tensor=True)
            self.log_message("✅ Modèle prêt à l'emploi !")
            
        except Exception as e:
            self.log_message(f"❌ Échec du chargement de CLIP: {e}")
            QMessageBox.critical(
                self, "Erreur", 
                f"Échec du chargement de CLIP:\n{e}\n\n"
                "Essaie avec un modèle plus léger (all-MiniLM-L6-v2) si le problème persiste."
            )

    def load_tf_model(self):
        """Charge un modèle TensorFlow personnalisé"""
        path = self.tf_model_path.text()
        if not path:
            QMessageBox.warning(self, "Erreur", "Aucun fichier de modèle sélectionné.")
            return

        try:
            # Essayer de charger comme SavedModel (dossier)
            if os.path.isdir(path):
                self.model_tf = tf.keras.models.load_model(path)
            # Ou comme fichier .h5/.keras
            else:
                self.model_tf = tf.keras.models.load_model(path)
            self.tf_status.setText(f"✅ Modèle chargé : {os.path.basename(path)}")
            self.log_message(f"Modèle TensorFlow chargé : {path}")
        except Exception as e:
            self.tf_status.setText(f"❌ Erreur : {e}")
            self.log_message(f"Échec du chargement du modèle TF : {e}")
            QMessageBox.critical(
                self, "Erreur",
                f"Échec du chargement du modèle TensorFlow :\n{e}\n\n"
                "Assurez-vous que le fichier est un modèle Keras valide (.h5 ou .keras) ou un dossier SavedModel.\n\n"
                "Si tu as un fichier 'saved_model.pb', sélectionne le DOSSIER parent (ex: 'models') et non le fichier."
            )

    # ====================== NAVIGATION ======================
    def browse_source(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier à trier", str(self.config.BASE_DIR)
        )
        if folder:
            self.entry_source.setText(folder)
            self.config.DOSSIER_A_TRIER = Path(folder)
            self.log_message(f"Dossier à trier : {folder}")

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier de sortie", str(self.config.BASE_DIR)
        )
        if folder:
            self.entry_output.setText(folder)
            self.config.DOSSIER_TRIES = Path(folder)
            self.log_message(f"Dossier de sortie : {folder}")

    def browse_tf_model(self):
        # Permettre de sélectionner un DOSSIER (pour SavedModel) ou un FICHIER (.h5)
        file_or_dir = QFileDialog.getExistingDirectory(self, "Sélectionner un modèle TensorFlow", str(self.config.MODEL_SAVE_PATH))
        if not file_or_dir:
            file_or_dir, _ = QFileDialog.getOpenFileName(
                self, "Sélectionner un modèle TensorFlow", str(self.config.MODEL_SAVE_PATH),
                "Modèles (*.h5 *.keras);;Tous les fichiers (*)"
            )
        if file_or_dir:
            self.tf_model_path.setText(file_or_dir)

    def browse_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner une image", str(self.config.DOSSIER_TRIES),
            "Images (*.jpg *.jpeg *.png *.raw *.cr2 *.nef *.arw *.dng);;Tous les fichiers (*)"
        )
        if file:
            self.entry_image_path.setText(file)
            self.load_image_preview(file)
            self.load_image_tags(file)

    def browse_folder_for_tags(self):
        """Parcourir un dossier pour chercher des images taguées"""
        folder = QFileDialog.getExistingDirectory(
            self, "Sélectionner un dossier à scanner", str(self.config.DOSSIER_TRIES)
        )
        if folder:
            self.folder_path_input.setText(folder)
            self.scan_folder_for_tags(folder)

    def scan_folder_for_tags(self, folder):
        """Scanne un dossier et affiche les images avec leurs tags"""
        self.search_results.clear()
        all_images = []

        try:
            for root, _, files in os.walk(folder):
                for file in files:
                    filepath = Path(root) / file
                    if filepath.suffix.lower() in self.config.IMAGE_EXTENSIONS | self.config.RAW_EXTENSIONS:
                        all_images.append(str(filepath))

            if not all_images:
                self.log_message("❌ Aucune image trouvée dans ce dossier.")
                return

            # Filtrer par tag si un tag est saisi
            tag = self.tag_search.text().strip()
            if tag:
                results = []
                for img_path in all_images:
                    img_tags = self.tag_manager.get_tags(img_path)
                    if tag.lower() in [t.lower() for t in img_tags]:
                        results.append(img_path)
                all_images = results

            for img_path in all_images:
                tags = self.tag_manager.get_tags(img_path)
                tag_str = f" [Tags: {', '.join(tags)}]" if tags else "[Aucun tag]"
                self.search_results.addItem(f"{os.path.basename(img_path)}{tag_str} ({img_path})")

            self.log_message(f"✅ {len(all_images)} image(s) trouvée(s) dans {folder}.")
        except Exception as e:
            self.log_message(f"❌ Erreur lors du scan du dossier: {e}")

    # ====================== GESTION DES TAGS ======================
    def add_tag_to_image(self):
        """Ajoute un tag à l'image actuellement sélectionnée"""
        image_path = self.entry_image_path.text() if self.entry_image_path else ""
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Erreur", "Aucune image valide sélectionnée !")
            return

        new_tag = self.new_tag_input.text().strip()
        if not new_tag:
            QMessageBox.warning(self, "Erreur", "Veuillez saisir un tag !")
            return

        self.tag_manager.add_tag(image_path, new_tag)
        self.new_tag_input.clear()
        self.log_message(f"✅ Tag '{new_tag}' ajouté à {os.path.basename(image_path)}")
        self.load_image_tags(image_path)

    def load_image_tags(self, image_path):
        """Charge les tags d'une image"""
        try:
            tags = self.tag_manager.get_tags(image_path)
            self.current_tags_label.setText(f"Tags actuels : {', '.join(tags) if tags else 'Aucun'}")
            self.log_message(f"Tags pour {os.path.basename(image_path)}: {tags}")
        except Exception as e:
            self.log_message(f"⚠️ Erreur de chargement des tags: {e}")

    def search_by_tag(self):
        """Recherche des images par tag dans le dossier sélectionné"""
        folder = self.folder_path_input.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Erreur", "Aucun dossier valide sélectionné !")
            return

        tag = self.tag_search.text().strip()
        if not tag:
            self.scan_folder_for_tags(folder)  # Affiche toutes les images si aucun tag
            return

        self.log_message(f"🔍 Recherche du tag '{tag}' dans {folder}...")
        results = self.tag_manager.get_images_by_tag(tag)
        results = [r for r in results if r.startswith(folder)]  # Filtrer par dossier
        self.search_results.clear()

        if results:
            for img_path in results:
                tags = self.tag_manager.get_tags(img_path)
                tag_str = f" [Tags: {', '.join(tags)}]"
                self.search_results.addItem(f"{os.path.basename(img_path)}{tag_str} ({img_path})")
            self.log_message(f"✅ {len(results)} résultat(s) trouvé(s).")
        else:
            self.search_results.addItem("❌ Aucun résultat trouvé.")
            self.log_message("❌ Aucun résultat trouvé.")

    def load_image_from_search(self, item):
        """Charge une image depuis les résultats de recherche"""
        text = item.text()
        if " (" in text and ")" in text:
            image_path = text.split(" (")[1][:-1]  # Extraire le chemin entre parenthèses
            self.entry_image_path.setText(image_path)
            self.load_image_preview(image_path)
            self.load_image_tags(image_path)

    def load_image_preview(self, image_path):
        """Charge l'aperçu d'une image"""
        try:
            if Path(image_path).suffix.lower() in self.config.RAW_EXTENSIONS:
                image = self.raw_to_pil(image_path)
            else:
                image = Image.open(image_path)
            
            if image:
                image.thumbnail((400, 400))
                pixmap = self.pil2pixmap(image)
                self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.log_message(f"⚠️ Erreur de chargement de l'image: {e}")
            self.image_label.clear()

    def download_default_model(self):
        """Affiche des instructions pour obtenir un modèle TensorFlow"""
        msg = QMessageBox()
        msg.setWindowTitle("Obtenir un modèle TensorFlow")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "<b>Pour utiliser un modèle TensorFlow personnalisé :</b><br><br>"
            "1. <b>Entraîne un modèle</b> avec Keras (ex: MobileNetV2)<br>"
            "2. <b>Sauvegarde-le</b> au format <code>.h5</code> ou <code>.keras</code><br>"
            "3. <b>Sélectionne le dossier</b> (pour SavedModel) ou le fichier (pour .h5/.keras)<br><br>"
            "<b>Modèles pré-entraînés recommandés :</b><br>"
            "- MobileNetV2 : <a href='https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5'>Lien</a><br>"
            "- ResNet50 : <a href='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'>Lien</a><br><br>"
            "<b>Si tu as un fichier 'saved_model.pb' :</b><br>"
            "- Sélectionne le <b>DOSSIER</b> parent (ex: 'models') et non le fichier.<br>"
            "- Ou convertis-le en .h5 avec :<br>"
            "<code>from tensorflow.keras.models import load_model<br>"
            "model = load_model('models/saved_model')<br>"
            "model.save('models/model.h5')</code><br><br>"
            "Besoin d'aide pour en entraîner un ?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.exec_()

    # ====================== WATCHDOG ======================
    def toggle_watchdog(self, state):
        if state == Qt.Checked:
            self.start_watchdog()
        else:
            self.stop_watchdog()

    def start_watchdog(self):
        if self.watchdog_worker is None or not self.watchdog_worker.isRunning():
            self.stop_watchdog()
            self.watchdog_worker = WatchdogWorker(str(self.config.DOSSIER_A_TRIER))
            self.watchdog_worker.file_detected.connect(self.on_file_detected)
            self.watchdog_worker.start()
            self.log_message(f"👁️ Surveillance active sur : {self.config.DOSSIER_A_TRIER}")

    def stop_watchdog(self):
        if self.watchdog_worker:
            self.watchdog_worker.stop()
            self.watchdog_worker = None
            self.log_message("🛑 Surveillance arrêtée.")

    def on_file_detected(self, file_path):
        self.log_message(f"📷 Nouvelle photo détectée : {file_path}")
        self.classer_et_deplacer(file_path)

    # ====================== TRI ET CLASSIFICATION ======================
    def trier_dossier_maintenant(self):
        """Trie le dossier sélectionné maintenant"""
        self.log_message("📁 Tri du dossier en cours...")
        self.trier_dossier(str(self.config.DOSSIER_A_TRIER))

    def trier_dossier(self, dossier):
        """Trie toutes les images dans un dossier"""
        try:
            files = []
            for root, _, filenames in os.walk(dossier):
                for filename in filenames:
                    filepath = Path(root) / filename
                    if filepath.suffix.lower() not in ['.thm', '.xmp', '.json']:
                        files.append(filepath)

            raw_files = [f for f in files if f.suffix.lower() in self.config.RAW_EXTENSIONS]
            other_files = [f for f in files if f not in raw_files]
            all_files = raw_files + other_files

            self.log_message(f"📁 Trouvé {len(all_files)} fichier(s) à trier")
            
            for filepath in all_files:
                self.log_message(f"📷 Tri de : {filepath}")
                self.classer_et_deplacer(str(filepath))
                
        except Exception as e:
            self.log_message(f"❌ Erreur lors du tri : {e}")

    def classer_et_deplacer(self, chemin_image):
        """Classifie et déplace une image"""
        try:
            use_tf = self.radio_tf.isChecked() and self.model_tf is not None
            if use_tf:
                categorie = self.classer_image_tf(chemin_image)
            else:
                categorie = self.classer_image_clip(chemin_image)
            self.deplacer_image_et_metadonnees(chemin_image, categorie)
        except Exception as e:
            self.log_message(f"⚠️ Erreur avec {chemin_image}: {e}")

    def classer_image_clip(self, chemin_image):
        """Classifie une image avec CLIP"""
        try:
            if chemin_image.lower().endswith(tuple(self.config.RAW_EXTENSIONS)):
                image = self.raw_to_pil(chemin_image)
                if image is None:
                    return "autre"
            else:
                try:
                    image = Image.open(chemin_image)
                except Exception as e:
                    self.log_message(f"⚠️ Erreur d'ouverture de {chemin_image}: {e}")
                    return "autre"

            # TAILLE AUGMENTÉE pour plus de précision
            if max(image.size) > 768:
                image = image.resize((768, 768))

            image_embedding = self.model_clip.encode(image, convert_to_tensor=True)
            similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]
            categorie = list(self.config.CATEGORIES.keys())[similarities.argmax().item()]

            # SEUIL DE CONFIANCE
            max_similarity = similarities.max().item()
            if max_similarity < 0.2:
                return "autre"
            return categorie
        except Exception as e:
            self.log_message(f"⚠️ Erreur de classification CLIP pour {chemin_image}: {e}")
            return "autre"

    def classer_image_tf(self, chemin_image):
        """Classifie une image avec TensorFlow"""
        try:
            if chemin_image.lower().endswith(tuple(self.config.RAW_EXTENSIONS)):
                image = self.raw_to_pil(chemin_image)
                if image is None:
                    return "autre"
            else:
                try:
                    image = Image.open(chemin_image)
                except Exception as e:
                    return "autre"

            if max(image.size) > 224:
                image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            img_array = np.expand_dims(img_array, axis=0)
            predictions = self.model_tf.predict(img_array, verbose=0)
            categorie = list(self.config.CATEGORIES.keys())[np.argmax(predictions[0])]
            return categorie
        except Exception as e:
            self.log_message(f"⚠️ Erreur de classification TF pour {chemin_image}: {e}")
            return "autre"

    def raw_to_pil(self, raw_path):
        """Convertit un fichier RAW en image PIL"""
        try:
            with rawpy.imread(raw_path) as raw:
                rgb_array = raw.postprocess(
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    no_auto_bright=True,
                    gamma=(1, 1)
                )
                return Image.fromarray(rgb_array.astype('uint8'))
        except Exception as e:
            self.log_message(f"⚠️ Erreur avec {raw_path}: {e}")
            return None

    def deplacer_image_et_metadonnees(self, chemin_source, categorie):
        """Déplace une image et ses métadonnées"""
        try:
            chemin_source = os.path.normpath(chemin_source)
            dossier_dest = os.path.join(self.config.DOSSIER_TRIES, categorie)
            os.makedirs(dossier_dest, exist_ok=True)

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

            base_name = os.path.splitext(os.path.basename(chemin_source))[0]
            dossier_source = os.path.dirname(chemin_source)
            
            for ext in self.config.METADATA_EXTENSIONS:
                meta_file = os.path.join(dossier_source, f"{base_name}{ext}")
                if os.path.exists(meta_file):
                    try:
                        shutil.move(meta_file, os.path.join(dossier_dest, os.path.basename(meta_file)))
                    except Exception as e:
                        self.log_message(f"⚠️ Impossible de déplacer {meta_file}: {e}")

            self.log_message(f"✅ Déplacé : {nom_fichier} → {categorie}")
        except Exception as e:
            self.log_message(f"⚠️ Impossible de déplacer {chemin_source}: {e}")

    # ====================== FERMETURE ======================
    def closeEvent(self, event):
        self.stop_watchdog()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoSorterApp()
    window.show()
    sys.exit(app.exec_())
