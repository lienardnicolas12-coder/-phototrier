import os
import shutil
import json
import rawpy
import logging
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGraphicsScene,
    QGraphicsView, QGroupBox, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QRadioButton, QSizePolicy, QTabWidget, QTextEdit, QVBoxLayout,
    QHBoxLayout, QWidget, QMainWindow, QScrollArea, QGraphicsPixmapItem,
    QGraphicsItem, QSpacerItem, QButtonGroup, QToolButton
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QIcon, QFont, QColor,
    QPalette, QBrush, QPen, QCursor, QTransform, QImageReader
)
from PyQt5.QtCore import (
    Qt, QPoint, QPointF, QSize, QRect, QRectF, pyqtSignal,
    QThread, QObject, QTimer, QEvent, QMargins, QSizeF
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
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
    BASE_DIR = Path(__file__).parent
    DOSSIER_A_TRIER = BASE_DIR / "a_trier"
    DOSSIER_TRIES = BASE_DIR / "tries"
    MODEL_SAVE_PATH = BASE_DIR / "models"
    TAGS_FILE = BASE_DIR / "tags.json"
    CONFIG_FILE = BASE_DIR / "config.json"

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

    @classmethod
    def save_config(cls, config_data):
        """Sauvegarde la configuration dans un fichier JSON"""
        try:
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Erreur de sauvegarde de la config: {e}")
            return False

    @classmethod
    def load_config(cls):
        """Charge la configuration depuis un fichier JSON"""
        if cls.CONFIG_FILE.exists():
            try:
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Erreur de chargement de la config: {e}")
                return {}
        return {}

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
                    loaded_tags = json.load(f)
                    # Convertir les listes en sets pour une gestion optimale
                    return {path: set(tags) for path, tags in loaded_tags.items()}
            except json.JSONDecodeError as e:
                logging.error(f"❌ Erreur de syntaxe JSON dans tags.json: {e}. Le fichier sera réinitialisé.")
                # Réinitialiser le fichier avec un JSON vide
                try:
                    with open(self.tags_file, 'w', encoding='utf-8') as f:
                        json.dump({}, f)
                except Exception as e2:
                    logging.error(f"❌ Impossible de réinitialiser tags.json: {e2}")
                return {}
            except Exception as e:
                logging.error(f"❌ Erreur de chargement des tags: {e}")
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

        # Variables pour le zoom avec setTransform + centerOn
        self.zoom_factor = 1.0
        self.min_zoom = 0.513158  # ✅ 1.0 / (1.1^7) = 7 étapes arrière
        self.max_zoom = 1.948717  # ✅ 1.0 * (1.1^7) = 7 étapes avant
        self.zoom_step = 1.1

        # Variables pour le lissage
        self.zoom_timer = QTimer(self)
        self.zoom_timer.setInterval(16)  # 60 FPS
        self.zoom_timer.timeout.connect(self._apply_smooth_zoom)
        self.zoom_target = 1.0
        self.zoom_current = 1.0
        self.zoom_active = False

        self.current_image_path = None

        # Attributs de l'interface
        self.entry_source = None
        self.entry_output = None
        self.log_text = None
        self.check_watch = None
        self.new_tag_input = None
        self.tag_search = None
        self.search_results = None
        self.graphics_view = None
        self.current_tags_label = None
        self.tf_model_path = None
        self.tf_status = None
        self.folder_path_input = None

        # Attributs pour les paramètres
        self.storage_path_input = None
        self.categories_list = None
        self.category_name_input = None
        self.category_desc_input = None

        # Charger la configuration sauvegardée
        self.load_saved_config()

        self.init_ui()
        self.load_models()

    # ====================== CHARGEMENT DE LA CONFIGURATION ======================
    def load_saved_config(self):
        """Charge la configuration sauvegardée (dossier de stockage et catégories)"""
        saved_config = Config.load_config()
        
        # Charger le dossier de stockage personnalisé
        if 'storage_path' in saved_config:
            custom_path = Path(saved_config['storage_path'])
            if custom_path.exists():
                self.config.BASE_DIR = custom_path
                self.config.DOSSIER_A_TRIER = custom_path / "a_trier"
                self.config.DOSSIER_TRIES = custom_path / "tries"
                self.config.MODEL_SAVE_PATH = custom_path / "models"
                self.config.TAGS_FILE = custom_path / "tags.json"
                self.config.CONFIG_FILE = custom_path / "config.json"
                # Recreate directories with new path
                Config.ensure_directories()
                # Update tag manager with new path
                self.tag_manager = TagManager(self.config.TAGS_FILE)
        
        # Charger les catégories personnalisées
        if 'categories' in saved_config:
            self.config.CATEGORIES = saved_config['categories']

    def save_current_config(self):
        """Sauvegarde la configuration actuelle"""
        config_data = {
            'storage_path': str(self.config.BASE_DIR),
            'categories': self.config.CATEGORIES
        }
        return Config.save_config(config_data)

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
        self.setup_settings_tab()

    def setup_auto_tab(self):
        self.auto_tab = QWidget()
        self.tabs.addTab(self.auto_tab, "\ud83d\udcf7 Tri Automatique")
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
        sort_btn = QPushButton("\ud83d\udcc1 Trier maintenant")
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
        self.tabs.addTab(self.tags_tab, "\ud83c\udff7\ufe0f Gestion des Tags")
        layout = QVBoxLayout(self.tags_tab)

        # Recherche par dossier
        folder_search_group = QGroupBox("Rechercher dans un dossier")
        folder_search_layout = QHBoxLayout()
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Dossier \u00e0 scanner...")
        folder_search_layout.addWidget(self.folder_path_input)
        browse_folder_btn = QPushButton("\ud83d\udcc1 Parcourir dossier")
        browse_folder_btn.clicked.connect(self.browse_folder_for_tags)
        folder_search_layout.addWidget(browse_folder_btn)
        folder_search_group.setLayout(folder_search_layout)
        layout.addWidget(folder_search_group)

        # Champ masqué pour la compatibilité
        self.entry_image_path = QLineEdit()
        self.entry_image_path.setVisible(False)

        # Conteneur FIXE pour l'image (800x600)
        image_container = QFrame()
        image_container.setFixedSize(800, 600)
        image_container.setStyleSheet("border: 2px solid #509cfb; background-color: #1e1e1e;")
        layout.addWidget(image_container)

        # Configuration du QGraphicsView avec taille FIXE
        self.graphics_view = QGraphicsView(image_container)
        self.graphics_view.setFixedSize(800, 600)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.graphics_scene = QGraphicsScene(self.graphics_view)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # Zoom centré sur la souris
        self.graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)         # Redimensionnement centré
        self.graphics_view.setAlignment(Qt.AlignCenter)  # Centre le contenu dans le view
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().setMouseTracking(True)
        
        # Variables pour le déplacement avec la molette
        self.middle_button_pressed = False
        self.last_mouse_pos = QPointF()
        
        # Installe un event filter pour capturer la molette dans le viewport
        self.graphics_view.viewport().installEventFilter(self)
        
        self.graphics_pixmap_item = None

        # Boutons de zoom
        zoom_frame = QFrame()
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.addStretch()

        zoom_in_btn = QPushButton("+ Zoom")
        zoom_in_btn.setStyleSheet("font-size: 12px;")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("- Zoom")
        zoom_out_btn.setStyleSheet("font-size: 12px;")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)

        zoom_reset_btn = QPushButton("Reset Zoom")
        zoom_reset_btn.setStyleSheet("font-size: 12px;")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_layout.addWidget(zoom_reset_btn)

        zoom_layout.addStretch()
        layout.addWidget(zoom_frame)

        # Tags actuels
        self.current_tags_label = QLabel("Tags actuels : Aucun")
        self.current_tags_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        layout.addWidget(self.current_tags_label)

        # Ajouter un tag
        tag_group = QGroupBox("Ajouter un tag \u00e0 cette image")
        tag_layout = QHBoxLayout()
        self.new_tag_input = QLineEdit()
        self.new_tag_input.setPlaceholderText("Ex: cheval, paysage...")
        tag_layout.addWidget(self.new_tag_input)
        add_tag_btn = QPushButton("\u2795 Ajouter le tag")
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
        search_btn = QPushButton("\ud83d\udd0d Rechercher")
        search_btn.clicked.connect(self.search_by_tag)
        search_layout.addWidget(search_btn)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Résultats de recherche
        self.search_results = QListWidget()
        self.search_results.itemDoubleClicked.connect(self.load_image_from_results)
        layout.addWidget(self.search_results)
    def setup_tf_tab(self):
        self.tf_tab = QWidget()
        self.tabs.addTab(self.tf_tab, "\ud83e\udd16 Modèle TensorFlow")
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
        self.download_btn = QPushButton("\u2753 Comment obtenir un modèle ?")
# self.download_btn.clicked.connect(self.download_default_model)  # Désactivé : méthode non implémentée
        self.download_btn.setEnabled(False)  # Désactive le bouton pour éviter l'erreur
        layout.addWidget(self.download_btn)

        # Bouton charger
        load_model_btn = QPushButton("\ud83d\udcc1 Charger le modèle TensorFlow")
        load_model_btn.clicked.connect(self.load_tf_model)
        layout.addWidget(load_model_btn)

        # Statut
        self.tf_status = QLabel("Aucun modèle TensorFlow chargé")
        layout.addWidget(self.tf_status)

    def setup_publish_tab(self):
        self.publish_tab = QWidget()
        self.tabs.addTab(self.publish_tab, "\ud83d\udce4 À Publier")
        layout = QVBoxLayout(self.publish_tab)
        layout.addWidget(QLabel("Les photos triées sont disponibles dans le dossier de sortie sélectionné."))
        layout.addWidget(QLabel("Utilise le bouton 'Parcourir' dans l'onglet 'Tri Automatique' pour changer le dossier."))

    def setup_settings_tab(self):
        """Onglet Paramètres pour gérer le dossier de stockage et les catégories"""
        self.settings_tab = QWidget()
        self.tabs.addTab(self.settings_tab, "\u2699\ufe0f Paramètres")
        layout = QVBoxLayout(self.settings_tab)
        layout.setSpacing(15)

        # ========== DOSSIER DE STOCKAGE ==========
        storage_group = QGroupBox("\u2139 DOSSIER DE STOCKAGE")
        storage_layout = QVBoxLayout()
        
        storage_path_frame = QFrame()
        storage_path_layout = QHBoxLayout(storage_path_frame)
        storage_path_layout.addWidget(QLabel("Dossier actuel:"))
        self.storage_path_input = QLineEdit(str(self.config.BASE_DIR))
        self.storage_path_input.setMinimumWidth(400)
        storage_path_layout.addWidget(self.storage_path_input)
        
        browse_storage_btn = QPushButton("Changer de dossier...")
        browse_storage_btn.clicked.connect(self.browse_storage_path)
        storage_path_layout.addWidget(browse_storage_btn)
        
        storage_layout.addWidget(storage_path_frame)
        storage_layout.addWidget(QLabel(f"Dossier à trier: {self.config.DOSSIER_A_TRIER}"))
        storage_layout.addWidget(QLabel(f"Dossier triés: {self.config.DOSSIER_TRIES}"))
        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)

        # ========== CATÉGORIES ==========
        categories_group = QGroupBox("\u2139 CATÉGORIES")
        categories_layout = QVBoxLayout()
        
        # Liste des catégories
        self.categories_list = QListWidget()
        self.categories_list.setMinimumHeight(200)
        self.refresh_categories_list()
        categories_layout.addWidget(self.categories_list)
        
        # Boutons pour gérer les catégories
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        
        add_category_btn = QPushButton("\u2795 AJOUTER UNE CATÉGORIE")
        add_category_btn.clicked.connect(self.add_category)
        buttons_layout.addWidget(add_category_btn)
        
        edit_category_btn = QPushButton("\u270f Modifier")
        edit_category_btn.clicked.connect(self.edit_category)
        buttons_layout.addWidget(edit_category_btn)
        
        delete_category_btn = QPushButton("\u274c Supprimer")
        delete_category_btn.clicked.connect(self.delete_category)
        buttons_layout.addWidget(delete_category_btn)
        
        categories_layout.addWidget(buttons_frame)
        categories_group.setLayout(categories_layout)
        layout.addWidget(categories_group)

        # ========== BOUTONS D'ACTION ==========
        actions_frame = QFrame()
        actions_layout = QHBoxLayout(actions_frame)
        
        save_btn = QPushButton("\ud83d\udcbe Enregistrer")
        save_btn.setStyleSheet("font-weight: bold; background-color: #28a745;")
        save_btn.clicked.connect(self.save_settings)
        actions_layout.addWidget(save_btn)
        
        import_btn = QPushButton("\ud83d\udcc1 Importer")
        import_btn.clicked.connect(self.import_categories)
        actions_layout.addWidget(import_btn)
        
        export_btn = QPushButton("\ud83d\udce4 Exporter")
        export_btn.clicked.connect(self.export_categories)
        actions_layout.addWidget(export_btn)
        
        layout.addWidget(actions_frame)
        
        # Statut
        self.settings_status = QLabel("")
        self.settings_status.setStyleSheet("color: #28a745;")
        layout.addWidget(self.settings_status)

    def refresh_categories_list(self):
        """Rafraîchit la liste des catégories dans l'interface"""
        self.categories_list.clear()
        for category_name, category_desc in self.config.CATEGORIES.items():
            item = QListWidgetItem(f"{category_name}: {category_desc}")
            item.setData(Qt.UserRole, category_name)
            self.categories_list.addItem(item)

    def browse_storage_path(self):
        """Permet de changer le dossier de stockage principal"""
        folder = QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier de stockage", str(self.config.BASE_DIR)
        )
        if folder:
            self.storage_path_input.setText(folder)
            new_base_dir = Path(folder)
            
            # Mettre à jour les chemins
            self.config.BASE_DIR = new_base_dir
            self.config.DOSSIER_A_TRIER = new_base_dir / "a_trier"
            self.config.DOSSIER_TRIES = new_base_dir / "tries"
            self.config.MODEL_SAVE_PATH = new_base_dir / "models"
            self.config.TAGS_FILE = new_base_dir / "tags.json"
            self.config.CONFIG_FILE = new_base_dir / "config.json"
            
            # Créer les répertoires
            Config.ensure_directories()
            
            # Mettre à jour le tag manager
            self.tag_manager = TagManager(self.config.TAGS_FILE)
            
            # Mettre à jour l'interface
            self.entry_source.setText(str(self.config.DOSSIER_A_TRIER))
            self.entry_output.setText(str(self.config.DOSSIER_TRIES))
            
            self.log_message(f"Dossier de stockage changé vers: {folder}")
            self.settings_status.setText(f"\u2705 Dossier de stockage mis à jour: {folder}")

    def add_category(self):
        """Ajoute une nouvelle catégorie"""
        dialog = QInputDialog()
        dialog.setWindowTitle("Ajouter une catégorie")
        dialog.setLabelText("Nom de la catégorie:")
        dialog.setOkButtonText("Suivant")
        dialog.setCancelButtonText("Annuler")
        
        if dialog.exec_() == QInputDialog.Accepted:
            category_name = dialog.textValue().strip()
            if not category_name:
                QMessageBox.warning(self, "Erreur", "Le nom de la catégorie ne peut pas être vide!")
                return
            
            if category_name in self.config.CATEGORIES:
                QMessageBox.warning(self, "Erreur", f"La catégorie '{category_name}' existe déjà!")
                return
            
            # Demander la description
            desc_dialog = QInputDialog()
            desc_dialog.setWindowTitle("Ajouter une catégorie")
            desc_dialog.setLabelText(f"Description pour '{category_name}' (ex: 'une photo de paysage avec montagnes'):")
            desc_dialog.setOkButtonText("Ajouter")
            desc_dialog.setCancelButtonText("Annuler")
            desc_dialog.setTextValue("")
            
            if desc_dialog.exec_() == QInputDialog.Accepted:
                category_desc = desc_dialog.textValue().strip()
                if not category_desc:
                    category_desc = f"une photo de {category_name}"
                
                self.config.CATEGORIES[category_name] = category_desc
                self.refresh_categories_list()
                self.settings_status.setText(f"\u2705 Catégorie '{category_name}' ajoutée")
                self.log_message(f"Catégorie ajoutée: {category_name}")

    def edit_category(self):
        """Modifie une catégorie existante"""
        selected_items = self.categories_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une catégorie à modifier!")
            return
        
        item = selected_items[0]
        old_name = item.data(Qt.UserRole)
        old_desc = self.config.CATEGORIES[old_name]
        
        # Demander le nouveau nom
        dialog = QInputDialog()
        dialog.setWindowTitle("Modifier la catégorie")
        dialog.setLabelText("Nouveau nom de la catégorie:")
        dialog.setTextValue(old_name)
        dialog.setOkButtonText("Suivant")
        dialog.setCancelButtonText("Annuler")
        
        if dialog.exec_() == QInputDialog.Accepted:
            new_name = dialog.textValue().strip()
            if not new_name:
                QMessageBox.warning(self, "Erreur", "Le nom de la catégorie ne peut pas être vide!")
                return
            
            # Demander la nouvelle description
            desc_dialog = QInputDialog()
            desc_dialog.setWindowTitle("Modifier la catégorie")
            desc_dialog.setLabelText(f"Nouvelle description pour '{new_name}':")
            desc_dialog.setTextValue(old_desc)
            desc_dialog.setOkButtonText("Enregistrer")
            desc_dialog.setCancelButtonText("Annuler")
            
            if desc_dialog.exec_() == QInputDialog.Accepted:
                new_desc = desc_dialog.textValue().strip()
                if not new_desc:
                    new_desc = f"une photo de {new_name}"
                
                # Mettre à jour
                if old_name != new_name:
                    # Créer une nouvelle entrée avec le nouveau nom
                    self.config.CATEGORIES[new_name] = new_desc
                    # Supprimer l'ancienne
                    del self.config.CATEGORIES[old_name]
                else:
                    # Juste mettre à jour la description
                    self.config.CATEGORIES[old_name] = new_desc
                
                self.refresh_categories_list()
                self.settings_status.setText(f"\u2705 Catégorie '{old_name}' modifiée")
                self.log_message(f"Catégorie modifiée: {old_name} -> {new_name}")

    def delete_category(self):
        """Supprime une catégorie"""
        selected_items = self.categories_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une catégorie à supprimer!")
            return
        
        item = selected_items[0]
        category_name = item.data(Qt.UserRole)
        
        # Ne pas permettre la suppression de la catégorie 'autre'
        if category_name == "autre":
            QMessageBox.warning(self, "Erreur", "La catégorie 'autre' ne peut pas être supprimée!")
            return
        
        confirm = QMessageBox.question(
            self, "Confirmer la suppression",
            f"Êtes-vous sûr de vouloir supprimer la catégorie '{category_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            del self.config.CATEGORIES[category_name]
            self.refresh_categories_list()
            self.settings_status.setText(f"\u2705 Catégorie '{category_name}' supprimée")
            self.log_message(f"Catégorie supprimée: {category_name}")

    def save_settings(self):
        """Sauvegarde les paramètres actuels"""
        # Mettre à jour le chemin de stockage
        new_storage_path = self.storage_path_input.text().strip()
        if new_storage_path and new_storage_path != str(self.config.BASE_DIR):
            new_base_dir = Path(new_storage_path)
            if new_base_dir.exists():
                self.config.BASE_DIR = new_base_dir
                self.config.DOSSIER_A_TRIER = new_base_dir / "a_trier"
                self.config.DOSSIER_TRIES = new_base_dir / "tries"
                self.config.MODEL_SAVE_PATH = new_base_dir / "models"
                self.config.TAGS_FILE = new_base_dir / "tags.json"
                self.config.CONFIG_FILE = new_base_dir / "config.json"
                
                # Créer les répertoires
                Config.ensure_directories()
                
                # Mettre à jour le tag manager
                self.tag_manager = TagManager(self.config.TAGS_FILE)
                
                # Mettre à jour l'interface
                self.entry_source.setText(str(self.config.DOSSIER_A_TRIER))
                self.entry_output.setText(str(self.config.DOSSIER_TRIES))
        
        # Sauvegarder la configuration
        if self.save_current_config():
            self.settings_status.setText("\u2705 Paramètres enregistrés avec succès!")
            self.log_message("Paramètres enregistrés")
            QMessageBox.information(self, "Succès", "Les paramètres ont été enregistrés avec succès!")
        else:
            self.settings_status.setText("\u274c Erreur lors de l'enregistrement")
            QMessageBox.critical(self, "Erreur", "Impossible d'enregistrer les paramètres!")

    def import_categories(self):
        """Importe des catégories depuis un fichier JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Importer des catégories", str(self.config.BASE_DIR),
            "Fichiers JSON (*.json);;Tous les fichiers (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_categories = json.load(f)
                
                if isinstance(imported_categories, dict):
                    # Fusionner avec les catégories existantes
                    for name, desc in imported_categories.items():
                        self.config.CATEGORIES[name] = desc
                    
                    self.refresh_categories_list()
                    self.settings_status.setText(f"\u2705 {len(imported_categories)} catégories importées")
                    self.log_message(f"Catégories importées depuis {file_path}")
                    QMessageBox.information(self, "Succès", f"{len(imported_categories)} catégories importées avec succès!")
                else:
                    QMessageBox.warning(self, "Erreur", "Le fichier ne contient pas un dictionnaire de catégories valide!")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible d'importer le fichier:\n{e}")
                self.log_message(f"Erreur d'import: {e}")

    def export_categories(self):
        """Exporte les catégories vers un fichier JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter des catégories", str(self.config.BASE_DIR),
            "Fichiers JSON (*.json);;Tous les fichiers (*)"
        )
        
        if file_path:
            if not file_path.endswith('.json'):
                file_path += '.json'
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config.CATEGORIES, f, indent=2, ensure_ascii=False)
                
                self.settings_status.setText(f"\u2705 Catégories exportées vers {file_path}")
                self.log_message(f"Catégories exportées vers {file_path}")
                QMessageBox.information(self, "Succès", f"Catégories exportées avec succès vers:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible d'exporter le fichier:\n{e}")
                self.log_message(f"Erreur d'export: {e}")

    # ====================== CHARGEMENT DES MODÈLES ======================
    def load_models(self):
        """Charge le modèle CLIP"""
        self.log_message("\ud83d\udcc1 Chargement du modèle CLIP (cela peut prendre 1-2 min)...")
        self.log_message("   \u2192 Le modèle pèse ~500Mo. Sois patient !")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_message(f"   \u2192 Appareil utilisé : {device}")
            
            # Essayer de charger CLIP-ViT-B-32
            try:
                self.model_clip = SentenceTransformer('clip-ViT-B-32', device=device)
                self.log_message("\u2705 Modèle CLIP-ViT-B-32 chargé avec succès !")
            except Exception as e1:
                self.log_message(f"\u26a0\ufe0f Échec de CLIP-ViT-B-32: {e1}")
                self.log_message("   \u2192 Tentative avec un modèle plus léger...")
                try:
                    self.model_clip = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    self.log_message("\u2705 Modèle all-MiniLM-L6-v2 chargé avec succès !")
                except Exception as e2:
                    self.log_message(f"\u274c Échec du chargement de tous les modèles: {e2}")
                    QMessageBox.critical(
                        self, "Erreur", 
                        f"Échec du chargement des modèles:\n{e2}\n\n"
                        "Vérifiez votre connexion Internet et que vous avez assez d'espace disque."
                    )
                    return
            
            self.log_message("   \u2192 Encodage des catégories...")
            self.text_inputs = list(self.config.CATEGORIES.values())
            self.text_embeddings = self.model_clip.encode(self.text_inputs, convert_to_tensor=True)
            self.log_message("\u2705 Modèle prêt à l'emploi !")
            
        except Exception as e:
            self.log_message(f"\u274c Échec du chargement de CLIP: {e}")
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
            self.tf_status.setText(f"\u2705 Modèle chargé : {os.path.basename(path)}")
            self.log_message(f"Modèle TensorFlow chargé : {path}")
        except Exception as e:
            self.tf_status.setText(f"\u274c Erreur : {e}")
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
                self.log_message("\u274c Aucune image trouvée dans ce dossier.")
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

            self.log_message(f"\u2705 {len(all_images)} image(s) trouvée(s) dans {folder}.")
        except Exception as e:
            self.log_message(f"\u274c Erreur lors du scan du dossier: {e}")

    def add_tag_to_image(self):
        """Ajoute un tag à l'image actuellement sélectionnée dans les résultats"""
        selected_items = self.search_results.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Erreur", "Aucune image sélectionnée dans les résultats !")
            return

        item = selected_items[0]
        text = item.text()
        if " (" in text and ")" in text:
            image_path = text.split(" (")[1][:-1]  # Extraire le chemin entre parenthèses
        else:
            image_path = text

        new_tag = self.new_tag_input.text().strip()
        if not new_tag:
            QMessageBox.warning(self, "Erreur", "Veuillez saisir un tag !")
            return

        self.tag_manager.add_tag(image_path, new_tag)
        self.new_tag_input.clear()
        self.log_message(f"✅ Tag '{new_tag}' ajouté à {os.path.basename(image_path)}")
        self.load_image_tags(image_path)

        self.log_message(f"\u2705 Tag '{new_tag}' ajouté à {os.path.basename(image_path)}")
        self.load_image_tags(image_path)

    def load_image_tags(self, image_path):
        """Charge les tags d'une image"""
        try:
            tags = self.tag_manager.get_tags(image_path)
            self.current_tags_label.setText(f"Tags actuels : {', '.join(tags) if tags else 'Aucun'}")
            self.log_message(f"Tags pour {os.path.basename(image_path)}: {tags}")
        except Exception as e:
            self.log_message(f"\u26a0\ufe0f Erreur de chargement des tags: {e}")
    def set_image_pixmap(self, pixmap):
        """Affiche une pixmap dans le QGraphicsView"""
        self.graphics_scene.clear()
        if self.graphics_pixmap_item:
            self.graphics_scene.removeItem(self.graphics_pixmap_item)
        
        self.graphics_pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        # Réinitialiser la transformation de l'item (toujours à 1.0)
        self.graphics_pixmap_item.setTransform(QTransform())
        self.zoom_factor = 1.0
        self.zoom_current = 1.0
        self.zoom_target = 1.0
        self.fit_in_view()

    def clear_image(self):
        """Efface l'image affichée"""
        self.graphics_scene.clear()
        self.graphics_pixmap_item = None

    def fit_in_view(self):
        """Adapte l'image à la taille du QGraphicsView en conservant le ratio d'aspect"""
        if not self.graphics_pixmap_item:
            return
        
        # Désactiver temporairement les mises à jour pour éviter les artefacts
        self.graphics_view.setUpdatesEnabled(False)
        
        # Obtenir la taille du viewport (800x600)
        viewport_width = self.graphics_view.viewport().width()
        viewport_height = self.graphics_view.viewport().height()
        
        if viewport_width <= 0 or viewport_height <= 0:
            self.graphics_view.setUpdatesEnabled(True)
            return
        
        # Obtenir la taille de l'image
        pixmap = self.graphics_pixmap_item.pixmap()
        if pixmap.isNull():
            self.graphics_view.setUpdatesEnabled(True)
            return
        
        image_width = pixmap.width()
        image_height = pixmap.height()
        
        if image_width <= 0 or image_height <= 0:
            self.graphics_view.setUpdatesEnabled(True)
            return
        
        # Calculer le facteur d'échelle pour adapter l'image au viewport
        # en conservant le ratio d'aspect (comme object-fit: contain)
        width_ratio = viewport_width / image_width
        height_ratio = viewport_height / image_height
        self.base_scale = min(width_ratio, height_ratio)
        
        # Réinitialiser la transformation de l'item (toujours à 1.0)
        self.graphics_pixmap_item.setTransform(QTransform())
        
        # Appliquer l'échelle d'adaptation sur la VUE
        transform = QTransform()
        transform.scale(self.base_scale, self.base_scale)
        self.graphics_view.setTransform(transform)
        
        # Le zoom de la vue est à base_scale (100% = base_scale)
        self.zoom_factor = 1.0
        self.zoom_current = 1.0
        self.zoom_target = 1.0
        
        # Centrer l'image avec le centre du viewport
        viewport_center = self.graphics_view.viewport().rect().center()
        self.graphics_view.centerOn(self.graphics_view.mapToScene(viewport_center))
        
        # Réactiver les mises à jour
        self.graphics_view.setUpdatesEnabled(True)

    def zoom_in(self):
        """Zoom avant avec setTransform + centerOn."""
        if hasattr(self, 'graphics_pixmap_item') and not self.zoom_active:
            new_zoom = self.zoom_current * self.zoom_step
            if new_zoom <= self.max_zoom:
                self.zoom_target = new_zoom
                if not self.zoom_timer.isActive():
                    self.zoom_timer.start()
                    self.zoom_active = True

    def zoom_out(self):
        """Zoom arrière avec setTransform + centerOn."""
        if hasattr(self, 'graphics_pixmap_item') and not self.zoom_active:
            new_zoom = self.zoom_current / self.zoom_step
            if new_zoom >= self.min_zoom:
                self.zoom_target = new_zoom
                if not self.zoom_timer.isActive():
                    self.zoom_timer.start()
                    self.zoom_active = True

    def zoom_reset(self):
        """Réinitialise le zoom à 100% (base_scale)."""
        self.zoom_factor = 1.0
        self.zoom_target = 1.0
        self.zoom_current = 1.0
        if hasattr(self, 'graphics_pixmap_item') and self.graphics_pixmap_item and hasattr(self, 'base_scale'):
            # Réappliquer l'échelle de base
            transform = QTransform()
            transform.scale(self.base_scale, self.base_scale)
            self.graphics_view.setTransform(transform)
            # Centrage final avec le centre du viewport
            viewport_center = self.graphics_view.viewport().rect().center()
            self.graphics_view.centerOn(self.graphics_view.mapToScene(viewport_center))

    def _apply_smooth_zoom(self):
        """Applique le zoom SANS tremblements (centerOn UNIQUEMENT à la fin)."""
        if not self.zoom_active or not self.graphics_pixmap_item:
            self.zoom_timer.stop()
            return

        # Interpolation linéaire (30% de la distance)
        self.zoom_current += (self.zoom_target - self.zoom_current) * 0.3

        # ✅ Applique le zoom sur la vue (SANS recentrer à chaque tick)
        if hasattr(self, 'base_scale'):
            total_scale = self.base_scale * self.zoom_current
            transform = QTransform().scale(total_scale, total_scale)
            self.graphics_view.setTransform(transform)
        else:
            transform = QTransform().scale(self.zoom_current, self.zoom_current)
            self.graphics_view.setTransform(transform)

        # ✅ Recentrage UNIQUEMENT à la fin du zoom
        if abs(self.zoom_target - self.zoom_current) < 0.001:
            self.zoom_current = self.zoom_target
            self.zoom_factor = self.zoom_target
            # Recentrage FINAL (1 seule fois)
            viewport_center = self.graphics_view.viewport().rect().center()
            self.graphics_view.centerOn(self.graphics_view.mapToScene(viewport_center))
            self.zoom_timer.stop()
            self.zoom_active = False


    def load_image_preview(self, image_path):
        """Charge une vignette de l'image (max 1000px de large) avec QImageReader"""
        try:
            # Libérer l'ancienne image
            if hasattr(self, 'graphics_pixmap_item') and self.graphics_pixmap_item:
                self.graphics_scene.removeItem(self.graphics_pixmap_item)
                self.graphics_pixmap_item = None
            self.graphics_scene.clear()

            # Charger avec QImageReader (gère RAW, JPEG, PNG)
            reader = QImageReader(image_path)
            reader.setAutoTransform(True)  # Orientation EXIF

            # Limiter à 1000px de large
            original_size = QImageReader(image_path).size()
            max_width = 1000
            scale_factor = max_width / original_size.width() if original_size.width() > max_width else 1.0
            reader.setScaledSize(QSize(
                int(original_size.width() * scale_factor),
                int(original_size.height() * scale_factor)
            ))

            qimage = reader.read()
            if qimage.isNull():
                raise ValueError(f"Format non supporté ou image corrompue : {image_path}")

            pixmap = QPixmap.fromImage(qimage)
            self.set_image_pixmap(pixmap)
            self.current_image_path = image_path

        except Exception as e:
            self.log_message(f"⚠️ Erreur : {e}")
            self.clear_image()


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
            self.log_message(f"\ud83d\udc41\ufe0f Surveillance active sur : {self.config.DOSSIER_A_TRIER}")

    def stop_watchdog(self):
        if self.watchdog_worker:
            self.watchdog_worker.stop()
            self.watchdog_worker = None
            self.log_message("\ud83d\uded1 Surveillance arrêtée.")

    def on_file_detected(self, file_path):
        self.log_message(f"\ud83d\udcf7 Nouvelle photo détectée : {file_path}")
        self.classer_et_deplacer(file_path)


    # ====================== TRI ET CLASSIFICATION ======================
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
            self.log_message(f"\u26a0\ufe0f Erreur avec {raw_path}: {e}")
            return None

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
                    self.log_message(f"\u26a0\ufe0f Erreur d'ouverture de {chemin_image}: {e}")
                    return "autre"

            # Redimensionner pour optimiser le traitement
            if max(image.size) > 768:
                image = image.resize((768, 768))

            image_embedding = self.model_clip.encode(image, convert_to_tensor=True)
            similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]
            categorie = list(self.config.CATEGORIES.keys())[similarities.argmax().item()]

            # Seuil de confiance
            max_similarity = similarities.max().item()
            if max_similarity < 0.2:
                return "autre"

            return categorie
        except Exception as e:
            self.log_message(f"\u26a0\ufe0f Erreur de classification CLIP pour {chemin_image}: {e}")
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
            self.log_message(f"\u26a0\ufe0f Erreur de classification TF pour {chemin_image}: {e}")
            return "autre"

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
                        self.log_message(f"\u26a0\ufe0f Impossible de déplacer {meta_file}: {e}")

            self.log_message(f"\u2705 Déplacé : {nom_fichier} \u2192 {categorie}")
        except Exception as e:
            self.log_message(f"\u26a0\ufe0f Impossible de déplacer {chemin_source}: {e}")

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
            self.log_message(f"\u26a0\ufe0f Erreur avec {chemin_image}: {e}")

    def trier_dossier_maintenant(self):
        """Trie le dossier sélectionné maintenant"""
        self.log_message("\ud83d\udcc1 Tri du dossier en cours...")
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

            self.log_message(f"\ud83d\udcc1 Trouvé {len(all_files)} fichier(s) à trier")
            
            for filepath in all_files:
                self.log_message(f"\ud83d\udcf7 Tri de : {filepath}")
                self.classer_et_deplacer(str(filepath))
                
        except Exception as e:
            self.log_message(f"\u274c Erreur lors du tri : {e}")



    def search_by_tag(self):
        """Recherche des images par tag (autonome, sans dépendre du dossier)"""
        tag = self.tag_search.text().strip()
        if not tag:
            QMessageBox.warning(self, "Erreur", "Aucun tag saisi !")
            return

        all_results = self.tag_manager.get_images_by_tag(tag)
        if not all_results:
            self.search_results.clear()
            self.search_results.addItem("❌ Aucun résultat trouvé.")
            return

        self.search_results.clear()
        for img_path in all_results:
            tags = self.tag_manager.get_tags(img_path)
            tag_str = f" [Tags: {', '.join(tags)}]" if tags else "[Aucun tag]"
            self.search_results.addItem(f"{os.path.basename(img_path)}{tag_str} ({img_path})")

    def load_image_from_results(self, item):
        """Charge une image depuis les résultats de recherche"""
        text = item.text()
        if " (" in text and ")" in text:
            image_path = text.split(" (")[1][:-1]  # Extraire le chemin entre parenthèses
            self.load_image_preview(image_path)
            self.load_image_tags(image_path)




    def mousePressEvent(self, event):
        """Gère le clic de la souris pour le déplacement."""
        if event.button() == Qt.MiddleButton:
            self.middle_button_pressed = True
            self.last_mouse_pos = event.localPos()
            self.graphics_view.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Gère le déplacement de la souris avec le bouton du milieu enfoncé."""
        if self.middle_button_pressed and hasattr(self, 'graphics_pixmap_item') and self.graphics_pixmap_item:
            delta = event.localPos() - self.last_mouse_pos
            self.last_mouse_pos = event.localPos()
            
            self.graphics_view.setUpdatesEnabled(False)
            # Déplacement dans la direction de la souris (pas inversé)
            self.graphics_pixmap_item.moveBy(delta.x(), delta.y())
            self.graphics_view.setUpdatesEnabled(True)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Gère le relâchement du bouton de la souris."""
        if event.button() == Qt.MiddleButton:
            self.middle_button_pressed = False
            self.graphics_view.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def eventFilter(self, obj, event):
        """Filtre les événements pour le viewport du QGraphicsView."""
        if obj == self.graphics_view.viewport():
            # Désactiver définitivement la molette
            if event.type() == QEvent.Type.Wheel:
                event.ignore()
                return True
            elif event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MiddleButton:
                self.mousePressEvent(event)
                return True
            elif event.type() == QEvent.Type.MouseMove and self.middle_button_pressed:
                self.mouseMoveEvent(event)
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MiddleButton:
                self.mouseReleaseEvent(event)
                return True

        return super().eventFilter(obj, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoSorterApp()
    window.show()
    sys.exit(app.exec_())
