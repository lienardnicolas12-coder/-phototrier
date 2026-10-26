import os
import shutil
import json
import rawpy
import math
import logging
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGraphicsScene,
    QGraphicsView, QGroupBox, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QRadioButton, QSizePolicy, QTabWidget, QTextEdit, QVBoxLayout,
    QHBoxLayout, QWidget, QMainWindow, QScrollArea, QGraphicsPixmapItem,
    QGraphicsItem, QSpacerItem, QButtonGroup, QToolButton, QShortcut, QProgressBar, QStyle,
    QSplitter, QStyleOptionButton, QDockWidget, QCompleter
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QIcon, QFont, QColor,
    QPalette, QBrush, QPen, QCursor, QTransform, QImageReader, QKeySequence,
    QLinearGradient, QFontMetrics
)
from PyQt5.QtCore import (
    Qt, QPoint, QPointF, QSize, QRect, QRectF, pyqtSignal,
    QThread, QObject, QTimer, QEvent, QMargins, QSizeF, QDateTime, QVariantAnimation,
    QPropertyAnimation, QEasingCurve, QStringListModel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from sentence_transformers import SentenceTransformer, util
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================
# DÉPENDANCES POUR LES RAW
# ======================
# Sur Linux (Fedora/Bazzite) :
#   sudo dnf install libraw-devel
# Sur Linux (Arch/Manjaro) :
#   sudo pacman -S libraw
# Sur Windows/macOS :
#   pip install rawpy  # (libraw est inclus dans le package wheel)
#
# Sans libraw, les fichiers RAW (.CR2, .NEF, .ARW) ne pourront pas être décodés.


# ======================
# DÉPENDANCES POUR LES RAW
# ======================
# Sur Linux (Fedora/Bazzite) :
#   sudo dnf install libraw-devel
# Sur Linux (Arch/Manjaro) :
#   sudo pacman -S libraw
# Sur Windows/macOS :
#   pip install rawpy  # (libraw est inclus dans le package wheel)
#
# Sans libraw, les fichiers RAW (.CR2, .NEF, .ARW) ne pourront pas être décodés.


# ====================== CONFIGURATION ======================
class Config:
    # Chemins fixes (jamais modifies par le storage_path)
    BASE_DIR = Path(__file__).parent
    CONFIG_FILE = BASE_DIR / "config.json"
    TAGS_FILE = BASE_DIR / "tags.json"

    # Chemins dynamiques (modifies par le storage_path)
    DOSSIER_A_TRIER = BASE_DIR / "a_trier"
    DOSSIER_TRIES = BASE_DIR / "tries"
    MODEL_SAVE_PATH = BASE_DIR / "models"

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


# ====================== ZONE DE DEPOT (GLISSER-DEPOSER) ======================
class DropZone(QListWidget):
    """Liste de miniatures acceptant le glisser-deposer de fichiers et de dossiers."""
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QListWidget.DropOnly)
        self.viewport().setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ====================== BOUTONS ET PROGRESSION ANIMES ======================
class ShineButton(QPushButton):
    """Bouton anime : au clic, reflet sur les lettres + fondu du texte avec lueur decroissante."""

    CYCLE_MS = 1000

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._shine = -1.0
        self._text_alpha = 1.0
        self._glow = 0.0
        self._fading = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(600)
        self._anim.valueChanged.connect(self._set_shine)
        self._fade_anim = QVariantAnimation(self)
        self._fade_anim.setDuration(self.CYCLE_MS)
        self._fade_anim.valueChanged.connect(self._set_fade_value)
        self._fade_anim.finished.connect(self._fade_finished)
        self.clicked.connect(self.start_shine)
        self.clicked.connect(self.start_text_fade)

    def start_shine(self):
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def start_text_fade(self):
        """Lance un cycle de fondu du texte (uniquement pendant le clic)."""
        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fading = True
        self._fade_anim.start()

    def _set_shine(self, value):
        self._shine = float(value)
        self.update()

    @staticmethod
    def _keyframe_alpha(t):
        """0-40% blanc ; 40-75% fondu vers 15% ; 75-90% vers 0 ; 90-100% retour au blanc."""
        def ease(u):
            return (1.0 - math.cos(math.pi * u)) / 2.0

        if t < 0.40:
            return 1.0
        if t < 0.75:
            return 1.0 - 0.85 * ease((t - 0.40) / 0.35)
        if t < 0.90:
            return 0.15 * (1.0 - ease((t - 0.75) / 0.15))
        return ease((t - 0.90) / 0.10)

    def _set_fade_value(self, value):
        t = float(value)
        self._text_alpha = self._keyframe_alpha(t)
        self._glow = 150.0 * (1.0 - t)  # lueur forte au clic, decroissante jusqu'a zero
        self.update()

    def _fade_finished(self):
        self._fading = False
        self._text_alpha = 1.0
        self._glow = 0.0
        self.update()

    def _text_rect(self):
        """Rectangle reel du texte (decalage icone compris)."""
        text = self.text()
        opt = QStyleOptionButton()
        opt.initFrom(self)
        opt.text = text
        opt.icon = self.icon()
        contents = self.style().subElementRect(QStyle.SE_PushButtonContents, opt, self)
        fm = QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        icon = self.icon()
        icon_w = icon.actualSize(QSize(16, 16)).width() + 4 if not icon.isNull() else 0
        block_w = icon_w + text_w
        x0 = contents.left() + max(0, (contents.width() - block_w) // 2) + icon_w
        return QRect(x0, contents.center().y() - text_h // 2, text_w, text_h)

    def paintEvent(self, event):
        if not getattr(self, "_fading", False):
            # Etat de repos et fallback : rendu Qt standard, texte blanc statique
            super().paintEvent(event)
        else:
            painter = QPainter(self)
            opt = QStyleOptionButton()
            opt.initFrom(self)
            opt.text = ""
            opt.icon = self.icon()
            self.style().drawControl(QStyle.CE_PushButton, opt, painter, self)
            text = self.text()
            if text:
                rect = self._text_rect()
                painter.setFont(self.font())
                # Lueur decroissante autour des lettres
                if self._glow > 0.5:
                    painter.setPen(QColor(255, 255, 255, min(255, int(self._glow))))
                    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                                   (-1, -1), (1, 1), (-1, 1), (1, -1)):
                        painter.drawText(rect.adjusted(dx, dy, dx, dy), Qt.AlignCenter, text)
                # Texte dont l'opacite suit le cycle de fondu
                if self._text_alpha > 0.001:
                    painter.setOpacity(self._text_alpha)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(rect, Qt.AlignCenter, text)
            painter.end()
        # Reflet au clic : balaye les lettres uniquement
        if 0.0 <= self._shine <= 1.0 and self.text():
            self._paint_shine()

    def _paint_shine(self):
        text_rect = self._text_rect()
        text_w = text_rect.width()
        band = max(24.0, text_w * 0.5)
        x = text_rect.left() - band + (text_w + band) * self._shine
        painter = QPainter(self)
        painter.setClipRect(text_rect)
        gradient = QLinearGradient(x, 0.0, x + band, 0.0)
        gradient.setColorAt(0.0, QColor(255, 255, 255, 0))
        gradient.setColorAt(0.5, QColor(255, 255, 255, 110))
        gradient.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(QRectF(x, 0.0, band, float(self.height())), QBrush(gradient))
        painter.end()


class ShineProgressBar(QProgressBar):
    """Barre de progression : reflet mobile + pulsation du remplissage (3 s par cycle)."""

    CYCLE_MS = 3000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shine = 0.0
        self._pulse = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1200)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._set_shine)
        self._anim.start()
        self._pulse_anim = QVariantAnimation(self)
        self._pulse_anim.setStartValue(0.0)
        self._pulse_anim.setEndValue(1.0)
        self._pulse_anim.setDuration(self.CYCLE_MS)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.valueChanged.connect(self._set_pulse)
        try:
            self._pulse_anim.start()
        except Exception:
            self._pulse = 0.0  # fallback : remplissage statique

    def _set_shine(self, value):
        self._shine = float(value)
        self.update()

    def _set_pulse(self, value):
        self._pulse = float(value)
        self.update()

    def _filled_width(self):
        total = self.maximum() - self.minimum()
        if total <= 0:
            return 0.0
        return (self.width() - 4) * (self.value() - self.minimum()) / float(total)

    def paintEvent(self, event):
        super().paintEvent(event)
        filled = self._filled_width()
        if filled < 8:
            return
        painter = QPainter(self)
        # Pulsation : remplissage blanc dont l'opacite oscille (jamais < 0.4)
        u = (1.0 - math.cos(2.0 * math.pi * self._pulse)) / 2.0
        painter.setOpacity(0.4 + 0.6 * u)
        painter.fillRect(QRectF(2.0, 2.0, filled, float(self.height() - 4)), QColor(255, 255, 255))
        # Reflet blanc mobile par-dessus
        band = max(30.0, filled * 0.4)
        x = -band + (filled + band) * self._shine
        painter.setOpacity(1.0)
        gradient = QLinearGradient(x, 0.0, x + band, 0.0)
        gradient.setColorAt(0.0, QColor(255, 255, 255, 0))
        gradient.setColorAt(0.5, QColor(255, 255, 255, 70))
        gradient.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(QRectF(x, 2.0, band, float(self.height() - 4)), QBrush(gradient))
        painter.end()


# ====================== LECTURE PLEIN ECRAN ======================
class FullScreenViewer(QWidget):
    """Plein ecran avec transition d'agrandissement de l'image.

    La fenetre reste en plein ecran pendant toute la duree : seul l'item graphique
    est anime (echelle + position). On evite ainsi le flou et le doublement causes
    par le redimensionnement d'une fenetre Windows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: #000000;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QGraphicsView(self)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        layout.addWidget(self.view)
        self._item = None
        self._origin_geo = None
        self._zoom_from = None
        self._zoom_to = None
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(350)
        self._zoom_anim.setStartValue(0.0)
        self._zoom_anim.setEndValue(1.0)
        self._zoom_anim.valueChanged.connect(self._apply_zoom)

    def _fit_geometry(self):
        """Retourne (scale, x, y) pour centrer l'item dans le viewport."""
        if self._item is None:
            return None
        w = self._item.boundingRect().width()
        h = self._item.boundingRect().height()
        vw = max(1, self.view.viewport().width())
        vh = max(1, self.view.viewport().height())
        scale = min(vw / w, vh / h)
        return (scale, (vw - w * scale) / 2.0, (vh - h * scale) / 2.0)

    def _apply_zoom(self, value):
        if self._item is None or self._zoom_from is None or self._zoom_to is None:
            return
        t = float(value)
        s0, x0, y0 = self._zoom_from
        s1, x1, y1 = self._zoom_to
        self._item.setScale(s0 + (s1 - s0) * t)
        self._item.setPos(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def _run_zoom(self, target, on_finished=None):
        self._zoom_from = (self._item.scale(), self._item.x(), self._item.y())
        self._zoom_to = target
        try:
            self._zoom_anim.finished.disconnect()
        except TypeError:
            pass
        if on_finished is not None:
            self._zoom_anim.finished.connect(on_finished)
        self._zoom_anim.start()

    def show_pixmap(self, pixmap, origin_rect=None):
        """Affiche la pixmap en plein ecran, transition depuis origin_rect si fourni."""
        if pixmap is None or pixmap.isNull():
            return
        self._zoom_anim.stop()
        self.scene.clear()
        self._item = self.scene.addPixmap(pixmap)
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self.showFullScreen()
        self.raise_()  # premier plan
        target = self._fit_geometry()
        if target is None:
            return
        self._zoom_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._origin_geo = None
        if origin_rect is not None:
            w = self._item.boundingRect().width()
            if w > 0:
                top_left = self.geometry().topLeft()
                s0 = origin_rect.width() / w
                x0 = float(origin_rect.x() - top_left.x())
                y0 = float(origin_rect.y() - top_left.y())
                self._origin_geo = (s0, x0, y0)
                self._item.setScale(s0)
                self._item.setPos(x0, y0)
                self._run_zoom(target)
        else:
            self._item.setScale(target[0])
            self._item.setPos(target[1], target[2])

    def close_to_origin(self):
        """Transition inverse : l'image se reduit vers son rectangle d'origine puis fermeture."""
        if self._item is None or self._origin_geo is None:
            self.hide()
            return
        self._zoom_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._run_zoom(self._origin_geo, on_finished=self.hide)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_to_origin()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.close_to_origin()



# ====================== VIEWPORT DE PREVISUALISATION ======================
class ImageViewport(QWidget):
    """Affichage d'une image : zoom molette (avant), deplacement au clic droit."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QGraphicsView(self)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        self.pixmap_item = None
        self.zoom_step = 1.5
        self.max_zoom = self.zoom_step ** 4
        self.zoom_current = 1.0
        self.zoom_target = 1.0
        self.zoom_active = False
        self.base_scale = 1.0
        self.right_button_pressed = False
        self.last_mouse_pos = QPointF()
        self.zoom_timer = QTimer(self)
        self.zoom_timer.setInterval(16)
        self.zoom_timer.timeout.connect(self._apply_smooth_zoom)

    def set_pixmap(self, pixmap):
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.pixmap_item.setPos(0, 0)
        self.fit_in_view()

    def clear_view(self):
        self.scene.clear()
        self.pixmap_item = None

    def fit_in_view(self):
        if not self.pixmap_item:
            return
        self.view.setTransform(QTransform())
        self.pixmap_item.setPos(0, 0)
        rect = self.pixmap_item.boundingRect()
        self.scene.setSceneRect(rect)
        self.view.horizontalScrollBar().setValue(0)
        self.view.verticalScrollBar().setValue(0)
        self.view.fitInView(rect, Qt.KeepAspectRatio)
        self.base_scale = self.view.transform().m11()
        if self.base_scale <= 0:
            self.base_scale = 1.0
        self.zoom_current = 1.0
        self.zoom_target = 1.0

    def zoom_in(self):
        if self.pixmap_item and not self.zoom_active:
            new_zoom = self.zoom_current * self.zoom_step
            if new_zoom <= self.max_zoom:
                self.zoom_target = new_zoom
                if not self.zoom_timer.isActive():
                    self.zoom_timer.start()
                    self.zoom_active = True

    def zoom_out(self):
        if self.pixmap_item and not self.zoom_active:
            new_zoom = self.zoom_current / self.zoom_step
            if new_zoom >= 1.0:
                self.zoom_target = new_zoom
                if not self.zoom_timer.isActive():
                    self.zoom_timer.start()
                    self.zoom_active = True

    def zoom_reset(self):
        self.zoom_current = 1.0
        self.zoom_target = 1.0
        if self.pixmap_item:
            self.fit_in_view()

    def _apply_smooth_zoom(self):
        if not self.zoom_active or not self.pixmap_item:
            self.zoom_timer.stop()
            return
        self.zoom_current += (self.zoom_target - self.zoom_current) * 0.3
        total_scale = self.base_scale * self.zoom_current
        self.view.setTransform(QTransform().scale(total_scale, total_scale))
        if abs(self.zoom_target - self.zoom_current) < 0.001:
            self.zoom_current = self.zoom_target
            self.zoom_timer.stop()
            self.zoom_active = False

    def eventFilter(self, obj, event):
        if obj == self.view.viewport():
            if event.type() == QEvent.Type.Wheel:
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True
            elif event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.RightButton:
                self.right_button_pressed = True
                self.last_mouse_pos = event.localPos()
                self.view.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return True
            elif event.type() == QEvent.Type.MouseMove and self.right_button_pressed:
                if self.pixmap_item:
                    delta = event.localPos() - self.last_mouse_pos
                    self.last_mouse_pos = event.localPos()
                    self.view.setUpdatesEnabled(False)
                    self.pixmap_item.moveBy(delta.x(), delta.y())
                    self.view.setUpdatesEnabled(True)
                event.accept()
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.RightButton:
                self.right_button_pressed = False
                self.view.setCursor(Qt.ArrowCursor)
                event.accept()
                return True
        return super().eventFilter(obj, event)


# ====================== TRI EN ARRIERE-PLAN ======================
class TriWorker(QThread):
    """Trie les fichiers dans un thread separe pour ne pas bloquer l'interface."""
    progress = pyqtSignal(int, int, str)
    finished_tri = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, app, files):
        super().__init__()
        self.app = app
        self.files = files
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        done = 0
        errors = 0
        total = len(self.files)
        for i, filepath in enumerate(self.files):
            if self.cancelled:
                break
            self.progress.emit(i, total, os.path.basename(str(filepath)))
            try:
                self.app.classer_et_deplacer(str(filepath))
                done += 1
            except Exception as e:
                errors += 1
                self.error.emit(f"{filepath}: {e}")
        self.progress.emit(total, total, "")
        self.finished_tri.emit(done, errors)

# ====================== GESTION DES TAGS ======================
class TagManager:
    def __init__(self, tags_file):
        self.tags_file = Path(tags_file)
        self.tags = self._load_tags()

    @staticmethod
    def _norm(path):
        """Normalise la casse d'un chemin : les cles restent coherentes
        quel que soit le mode de lancement de l'application (c:/ ou C:\)."""
        return os.path.normcase(str(path))

    def _load_tags(self):
        if self.tags_file.exists():
            try:
                with open(self.tags_file, 'r', encoding='utf-8') as f:
                    loaded_tags = json.load(f)
                    # Convertir les listes en sets pour une gestion optimale
                    return {self._norm(path): set(tags) for path, tags in loaded_tags.items()}
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
            serializable = {k: list(v) for k, v in self.tags.items()}
            with open(self.tags_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Erreur de sauvegarde des tags: {e}")

    def add_tag(self, image_path, tag):
        image_path = self._norm(image_path)
        # Reutiliser l'orthographe d'un tag existant (casse ignoree) pour eviter les doublons
        for known in self.get_all_tags():
            if known.lower() == tag.lower():
                tag = known
                break
        if image_path not in self.tags:
            self.tags[image_path] = set()
        self.tags[image_path].add(tag)
        self._save_tags()

    def migrate_tags(self, old_path, new_path):
        """Transfere les tags vers le nouveau chemin d'une image deplacee"""
        old_path = self._norm(old_path)
        new_path = self._norm(new_path)
        if old_path == new_path:
            return
        if old_path in self.tags:
            old_tags = self.tags.pop(old_path)
            self.tags.setdefault(new_path, set()).update(old_tags)
            self._save_tags()

    def remove_tag(self, image_path, tag):
        image_path = self._norm(image_path)
        if image_path in self.tags and tag in self.tags[image_path]:
            self.tags[image_path].remove(tag)
            if not self.tags[image_path]:
                del self.tags[image_path]
            self._save_tags()

    def get_tags(self, image_path):
        image_path = self._norm(image_path)
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
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoTrier")
        self.setGeometry(100, 100, 1400, 900)
        try:
            self.setWindowIcon(QIcon("logo.ico"))
        except:
            pass

        self.setStyleSheet("""
            /* ====== Theme sombre type Fluent Design ====== */
            QMainWindow { background-color: #1A1A1A; color: #FFFFFF; }
            QWidget { background-color: #1A1A1A; color: #FFFFFF;
                      font-family: "Segoe UI", "Inter", "Arial", sans-serif; font-size: 14px; }
            QLabel { background-color: transparent; border: none; color: #FFFFFF; }
            QLineEdit { background-color: #252526; color: #FFFFFF; border: 1px solid #333333;
                        border-radius: 8px; padding: 6px 10px; selection-background-color: #0078D4; }
            QGroupBox { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 12px;
                        margin-top: 14px; padding: 14px 10px 10px 10px; color: #FFFFFF;
                        font-weight: 700; letter-spacing: 1px; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px;
                              background-color: #1E1E1E; color: #FFFFFF; }
            QPushButton { background-color: transparent; color: #FFFFFF; border: 1px solid #333333;
                         border-radius: 8px; padding: 8px 16px; min-width: 100px; font-weight: 700; }
            QPushButton:hover { background-color: rgba(255, 255, 255, 25); border-color: #555555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 45); border-color: #555555; }
            QPushButton:disabled { color: #666666; border-color: #252526; }
            QTabWidget::pane { border: 1px solid #333333; border-radius: 8px; background-color: #1A1A1A; }
            QTabBar::tab { background-color: transparent; color: #CCCCCC; padding: 10px 18px;
                          border: 1px solid #333333; border-bottom: none;
                          border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px; }
            QTabBar::tab:selected { background-color: #0078D4; color: #FFFFFF; }
            QTabBar::tab:hover:!selected { background-color: #252526; }
            QListWidget { background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #333333;
                          border-radius: 8px; outline: none; }
            QListWidget::item { border-radius: 8px; padding: 4px; margin: 2px; }
            QListWidget::item:hover { background-color: rgba(255, 255, 255, 30); }
            QListWidget::item:selected { background-color: #0078D4; color: #FFFFFF; }
            QTextEdit { background-color: #1A1A1A; color: #AAAAAA; border: 1px solid #333333;
                        font-family: Consolas, monospace; font-size: 12px; }
            QCheckBox, QRadioButton { spacing: 8px; color: #CCCCCC; background-color: transparent; }
            QProgressBar { background-color: #252526; border: 1px solid #333333; border-radius: 6px;
                          text-align: center; color: #FFFFFF; }
            QProgressBar::chunk { background-color: #252526; border-radius: 5px; }
            QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
            QScrollBar::handle:vertical { background: #333333; border-radius: 5px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background: #0078D4; }
            QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
            QScrollBar::handle:horizontal { background: #333333; border-radius: 5px; min-width: 24px; }
            QScrollBar::handle:horizontal:hover { background: #0078D4; }
            QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
            QSplitter::handle { background-color: #2A2A2A; }
            QSplitter::handle:hover { background-color: rgba(255, 255, 255, 60); }
            QSplitter::handle:horizontal { width: 6px; margin: 2px 1px; }
            QSplitter::handle:vertical { height: 6px; margin: 1px 2px; }
            QMenuBar { background-color: #1A1A1A; color: #CCCCCC; }
            QMenuBar::item:selected { background-color: rgba(255, 255, 255, 30); }
            QMenu { background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #333333; border-radius: 8px; }
            QMenu::item:selected { background-color: rgba(255, 255, 255, 30); }
            QDockWidget::title { background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #333333; padding: 4px; }
            QToolTip { background-color: #252526; color: #FFFFFF; border: 1px solid #333333;
                       border-radius: 6px; padding: 4px; }
        """)

        # Initialiser la configuration et créer les répertoires
        Config.ensure_directories()
        
        self.config = Config()
        self.tag_manager = TagManager(self.config.TAGS_FILE)
        self.model_clip = None
        self.watchdog_worker = None
        self.text_embeddings = None

        self.current_image_path = None
        self.current_category = None
        self._nav_files = []
        self._nav_index = 0

        # Attributs de l'interface
        self.log_text = None
        self.check_watch = None
        self.new_tag_input = None
        self.current_tags_label = None

        # Attributs pour les parametres
        self.categories_list = None
        self.category_name_input = None
        self.category_desc_input = None

        # Tri en arriere-plan
        self.tri_worker = None

        # Charger la configuration sauvegardée
        self.load_saved_config()

        self.init_ui()
        self.load_models()

        # Raccourci de debug : F12 enregistre un dump complet de l'interface
        self.dump_shortcut = QShortcut(QKeySequence("F12"), self)
        self.dump_shortcut.activated.connect(self.dump_ui_state)

        # Journal thread-safe : les messages passent par un signal
        self.log_signal.connect(self._append_log)

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
                # TAGS_FILE et CONFIG_FILE restent dans le dossier racine du projet (chemins fixes)
                # ensure_directories() est un classmethod : il utilise les attributs de CLASSE,
                # pas les attributs d'instance modifies ci-dessus. On cree donc les dossiers ici.
                self.config.DOSSIER_A_TRIER.mkdir(parents=True, exist_ok=True)
                self.config.DOSSIER_TRIES.mkdir(parents=True, exist_ok=True)
                self.config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        
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
        """Ajoute un message au journal (thread-safe : passe par un signal)"""
        self.log_signal.emit(str(message))

    def _append_log(self, message):
        if self.log_text:
            self.log_text.append(message)

    # ====================== INIT UI ======================
    def init_ui(self):
        self.setup_tri_tab()
        self.refresh_tags_list()
        # Synchroniser le bouton Journal avec l'etat restaure du dock
        self.log_toggle_btn.setChecked(self.journal_dock.isVisible())

    def closeEvent(self, event):
        """Sauvegarde la disposition des panneaux avant la fermeture."""
        self.save_window_state()
        super().closeEvent(event)

    def save_window_state(self):
        """Sauvegarde la disposition des docks et la geometrie de la fenetre."""
        try:
            saved = Config.load_config()
            saved['window_state'] = bytes(self.saveState()).decode('latin1')
            saved['window_geometry'] = bytes(self.saveGeometry()).decode('latin1')
            Config.save_config(saved)
        except Exception as e:
            logging.error(f"Erreur de sauvegarde de la disposition: {e}")

    def restore_window_state(self):
        """Restaure la disposition des panneaux sauvegardee."""
        try:
            saved = Config.load_config()
            if 'window_geometry' in saved:
                self.restoreGeometry(saved['window_geometry'].encode('latin1'))
            if 'window_state' in saved:
                self.restoreState(saved['window_state'].encode('latin1'))
        except Exception as e:
            logging.error(f"Erreur de restauration de la disposition: {e}")

    def setup_tri_tab(self):
        """Interface a panneaux ancrables (docks) : deplacer, redimensionner, detacher."""
        # --- Panneau central : previsualisation + progression ---
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(10)

        self.preview_viewport = ImageViewport()
        central_layout.addWidget(self.preview_viewport, 1)

        progress_row = QHBoxLayout()
        self.sort_progress = ShineProgressBar()
        self.sort_progress.setTextVisible(False)
        self.sort_progress.setVisible(False)
        progress_row.addWidget(self.sort_progress, 1)
        self.sort_status_label = QLabel("Prêt.")
        progress_row.addWidget(self.sort_status_label)
        self.cancel_sort_btn = ShineButton("Annuler")
        self.cancel_sort_btn.setVisible(False)
        self.cancel_sort_btn.clicked.connect(self.cancel_sort)
        progress_row.addWidget(self.cancel_sort_btn)
        self.log_toggle_btn = ShineButton("Journal")
        self.log_toggle_btn.setCheckable(True)
        self.log_toggle_btn.toggled.connect(self.toggle_journal)
        progress_row.addWidget(self.log_toggle_btn)
        self.debug_dump_btn = ShineButton("Debug")
        self.debug_dump_btn.setToolTip("Enregistre un dump de l'interface (debug_ui/)")
        self.debug_dump_btn.clicked.connect(self.dump_ui_state)
        progress_row.addWidget(self.debug_dump_btn)
        central_layout.addLayout(progress_row)
        self.setCentralWidget(central)

        # Boutons superposes sur la photo : precedent / suivant / plein ecran
        self.fullscreen_viewer = FullScreenViewer()
        self.viewport_btns = QWidget(self.preview_viewport)
        btns_layout = QHBoxLayout(self.viewport_btns)
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.setSpacing(4)
        overlay_style = (
            "QPushButton { background-color: rgba(255, 255, 255, 25);"
            " border: 1px solid rgba(255, 255, 255, 60); color: #FFFFFF;"
            " border-radius: 8px; padding: 2px 8px; min-width: 0; }")
        self.nav_prev_btn = ShineButton("◀")
        self.nav_prev_btn.setToolTip("Photo précédente")
        self.nav_prev_btn.setStyleSheet(overlay_style)
        self.nav_prev_btn.setCursor(Qt.PointingHandCursor)
        self.nav_prev_btn.clicked.connect(lambda: self.navigate_preview(-1))
        btns_layout.addWidget(self.nav_prev_btn)
        self.nav_next_btn = ShineButton("▶")
        self.nav_next_btn.setToolTip("Photo suivante")
        self.nav_next_btn.setStyleSheet(overlay_style)
        self.nav_next_btn.setCursor(Qt.PointingHandCursor)
        self.nav_next_btn.clicked.connect(lambda: self.navigate_preview(1))
        btns_layout.addWidget(self.nav_next_btn)
        self.fullscreen_btn = ShineButton("⛶")
        self.fullscreen_btn.setToolTip("Plein écran")
        self.fullscreen_btn.setStyleSheet(overlay_style)
        self.fullscreen_btn.setCursor(Qt.PointingHandCursor)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen_preview)
        btns_layout.addWidget(self.fullscreen_btn)
        self.viewport_btns.adjustSize()
        self.viewport_btns.hide()
        self.preview_viewport.installEventFilter(self)
        self.preview_viewport.view.viewport().installEventFilter(self)
        # Supprimer les tags : actif quand l'image a le focus (ne vole pas la touche Suppr)
        del_sc = QShortcut(QKeySequence(Qt.Key_Delete), self.preview_viewport)
        del_sc.setContext(Qt.WidgetWithChildrenShortcut)
        del_sc.activated.connect(self.remove_current_tags)

        # Docks : panneaux ancrables, flottants, redimensionnables et regroupables
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks
                            | QMainWindow.AllowNestedDocks)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        dock_features = (QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable
                         | QDockWidget.DockWidgetFloatable)

        # --- Dock gauche : zone de depot ---
        drop_panel = QWidget()
        drop_layout = QVBoxLayout(drop_panel)
        drop_layout.setContentsMargins(6, 6, 6, 6)
        drop_layout.setSpacing(8)

        self.drop_list = DropZone()
        self.drop_list.setViewMode(QListWidget.IconMode)
        self.drop_list.setIconSize(QSize(100, 100))
        self.drop_list.setResizeMode(QListWidget.Adjust)
        self.drop_list.setMovement(QListWidget.Static)
        self.drop_list.setWrapping(True)
        self.drop_list.setSpacing(6)
        self.drop_list.setMinimumHeight(160)
        self.drop_list.setStyleSheet("""
            QListWidget { background-color: #1E1E1E; border: 1px dashed #0078D4; border-radius: 8px; }
            QListWidget::item { border-radius: 8px; padding: 4px; margin: 2px; }
            QListWidget::item:hover { background-color: rgba(255, 255, 255, 30); }
            QListWidget::item:selected { background-color: #0078D4; }
        """)
        self.drop_list.itemClicked.connect(self.preview_from_drop)
        self.drop_list.files_dropped.connect(self.on_files_dropped)
        drop_layout.addWidget(self.drop_list)

        self.drop_hint = QLabel("Glissez vos photos ou dossiers ici\n(tout est copié dans le dossier à trier)")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setStyleSheet("color: #aaaaaa; border: none;")
        drop_layout.addWidget(self.drop_hint)

        self.drop_count_label = QLabel("0 photo(s) en attente")
        self.drop_count_label.setStyleSheet("font-weight: bold; color: #ffffff; border: none;")
        drop_layout.addWidget(self.drop_count_label)

        # Selection manuelle en plus du glisser-deposer
        select_row = QHBoxLayout()
        self.select_files_btn = ShineButton("🗂 Fichiers...")
        self.select_files_btn.clicked.connect(self.select_files_dialog)
        select_row.addWidget(self.select_files_btn)
        self.select_folder_btn = ShineButton("📁 Dossier...")
        self.select_folder_btn.clicked.connect(self.select_folder_dialog)
        select_row.addWidget(self.select_folder_btn)
        select_row.addStretch(1)
        drop_layout.addLayout(select_row)

        # Surveillance (option expert) + tri manuel
        actions_row = QHBoxLayout()
        watch_box = QGroupBox("Surveillance (expert)")
        watch_layout = QHBoxLayout(watch_box)
        self.check_watch = QCheckBox("Démarrer la surveillance")
        self.check_watch.stateChanged.connect(self.toggle_watchdog)
        watch_layout.addWidget(self.check_watch)
        actions_row.addWidget(watch_box)
        actions_row.addStretch()
        self.sort_btn = ShineButton("📁 Trier maintenant")
        self.sort_btn.clicked.connect(self.trier_dossier_maintenant)
        actions_row.addWidget(self.sort_btn)
        drop_layout.addLayout(actions_row)

        self.drop_dock = QDockWidget("Zone de dépôt", self)
        self.drop_dock.setWidget(drop_panel)
        self.drop_dock.setFeatures(dock_features)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.drop_dock)
        # Depot accepte sur toute la zone du panneau (cadre et libelles compris)
        self.drop_group = drop_panel
        drop_panel.setAcceptDrops(True)
        drop_panel.installEventFilter(self)

        # --- Dock droite : dossiers de tri + tags ---
        categories_panel = QWidget()
        categories_layout = QVBoxLayout(categories_panel)
        categories_layout.setContentsMargins(6, 6, 6, 6)
        categories_layout.setSpacing(8)

        self.categories_list = QListWidget()
        self.categories_list.setViewMode(QListWidget.IconMode)
        self.categories_list.setIconSize(QSize(80, 80))
        self.categories_list.setResizeMode(QListWidget.Adjust)
        self.categories_list.setMovement(QListWidget.Static)
        self.categories_list.setWrapping(True)
        self.categories_list.setSpacing(6)
        self.categories_list.setMinimumHeight(160)
        self.categories_list.setStyleSheet("""
            QListWidget { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 8px; }
            QListWidget::item { border-radius: 8px; padding: 4px; margin: 2px; }
            QListWidget::item:hover { background-color: rgba(255, 255, 255, 30); }
            QListWidget::item:selected { background-color: #0078D4; }
        """)
        self.categories_list.itemClicked.connect(self.on_category_item_clicked)
        categories_layout.addWidget(self.categories_list)

        self.folder_back_btn = ShineButton("⬅ Dossiers")
        self.folder_back_btn.setVisible(False)
        self.folder_back_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.folder_back_btn.clicked.connect(self.show_folders_view)
        categories_layout.addWidget(self.folder_back_btn)

        self.category_buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(self.category_buttons_frame)
        add_category_btn = ShineButton("➕ Ajouter")
        add_category_btn.clicked.connect(self.add_category)
        buttons_layout.addWidget(add_category_btn)
        edit_category_btn = ShineButton("✏ Modifier")
        edit_category_btn.clicked.connect(self.edit_category)
        buttons_layout.addWidget(edit_category_btn)
        delete_category_btn = ShineButton("❌ Supprimer")
        delete_category_btn.clicked.connect(self.delete_category)
        buttons_layout.addWidget(delete_category_btn)
        categories_layout.addWidget(self.category_buttons_frame)

        actions_frame = QFrame()
        actions_layout = QHBoxLayout(actions_frame)
        save_btn = ShineButton("💾 Enregistrer")
        save_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        save_btn.clicked.connect(self.save_settings)
        actions_layout.addWidget(save_btn)
        categories_layout.addWidget(actions_frame)

        self.settings_status = QLabel("")
        self.settings_status.setStyleSheet("color: #107C10; border: none;")
        categories_layout.addWidget(self.settings_status)

        # ========== Gestion des tags (integree au panneau) ==========
        self.current_tags_label = QLabel("Tags actuels : Aucun")
        self.current_tags_label.setStyleSheet("font-weight: bold; color: #FFFFFF; border: none;")
        categories_layout.addWidget(self.current_tags_label)

        tag_add_row = QHBoxLayout()
        self.new_tag_input = QLineEdit()
        self.new_tag_input.setPlaceholderText("Ex: cheval, paysage...")
        # Autocompletion a partir des tags existants
        self.tag_completer_model = QStringListModel()
        self.tag_completer = QCompleter(self.tag_completer_model, self.new_tag_input)
        self.tag_completer.setCaseSensitivity(Qt.CaseInsensitive)
        try:
            self.tag_completer.setFilterMode(Qt.MatchContains)
        except AttributeError:
            pass
        self.new_tag_input.setCompleter(self.tag_completer)
        tag_add_row.addWidget(self.new_tag_input, 1)
        self.add_tag_btn = ShineButton("➕")
        self.add_tag_btn.setToolTip("Ajouter le tag à l'image affichée")
        self.add_tag_btn.clicked.connect(self.add_tag_to_image)
        tag_add_row.addWidget(self.add_tag_btn)
        categories_layout.addLayout(tag_add_row)

        self.tags_list_widget = QListWidget()
        self.tags_list_widget.setMaximumHeight(110)
        self.tags_list_widget.itemClicked.connect(self.on_tag_selected)
        self.tags_list_widget.setStyleSheet("""
            QListWidget { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 8px; color: #FFFFFF; }
            QListWidget::item { padding: 6px; border-radius: 6px; margin: 1px; }
            QListWidget::item:hover { background-color: rgba(255, 255, 255, 30); }
            QListWidget::item:selected { background-color: #0078D4; }
        """)
        categories_layout.addWidget(self.tags_list_widget)

        self.categories_dock = QDockWidget("Dossiers de tri", self)
        self.categories_dock.setWidget(categories_panel)
        self.categories_dock.setFeatures(dock_features)
        self.addDockWidget(Qt.RightDockWidgetArea, self.categories_dock)

        # --- Dock bas : journal ---
        journal_panel = QWidget()
        journal_layout = QVBoxLayout(journal_panel)
        journal_layout.setContentsMargins(6, 6, 6, 6)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        journal_layout.addWidget(self.log_text)
        self.journal_dock = QDockWidget("Journal", self)
        self.journal_dock.setWidget(journal_panel)
        self.journal_dock.setFeatures(dock_features)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.journal_dock)
        self.journal_dock.hide()
        self.journal_dock.visibilityChanged.connect(self.log_toggle_btn.setChecked)

        # --- Menus (ruban) avec raccourcis clavier ---
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Fichier")
        act = file_menu.addAction("Ouvrir fichier...")
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self.select_files_dialog)
        act = file_menu.addAction("Ouvrir dossier...")
        act.setShortcut("Ctrl+Shift+O")
        act.triggered.connect(self.select_folder_dialog)
        file_menu.addSeparator()
        act = file_menu.addAction("Nouveau tag")
        act.setShortcut("Ctrl+N")
        act.triggered.connect(self.focus_new_tag_input)
        act = file_menu.addAction("Supprimer les tags de l'image")
        act.triggered.connect(self.remove_current_tags)
        file_menu.addSeparator()
        act = file_menu.addAction("Enregistrer")
        act.setShortcut("Ctrl+S")
        act.triggered.connect(self.save_settings)
        act = file_menu.addAction("Enregistrer la disposition")
        act.setShortcut("Ctrl+Shift+S")
        act.triggered.connect(self.save_window_state)
        file_menu.addSeparator()
        act = file_menu.addAction("Fermer")
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)

        edit_menu = menubar.addMenu("Edition")
        act = edit_menu.addAction("Annuler (tri en cours)")
        act.setShortcut("Ctrl+Z")
        act.triggered.connect(self.edit_undo)
        act = edit_menu.addAction("Refaire (relancer le tri)")
        act.setShortcut("Ctrl+Y")
        act.triggered.connect(self.edit_redo)
        edit_menu.addSeparator()
        act = edit_menu.addAction("Copier le chemin de l'image")
        act.setShortcut("Ctrl+C")
        act.triggered.connect(self.edit_copy)
        act = edit_menu.addAction("Coller des fichiers")
        act.setShortcut("Ctrl+V")
        act.triggered.connect(self.edit_paste)
        edit_menu.addSeparator()
        act = edit_menu.addAction("Supprimer l'image affichée...")
        act.setShortcut("Ctrl+Del")
        act.triggered.connect(self.delete_current_image)

        tools_menu = menubar.addMenu("Outils")
        act = tools_menu.addAction("Trier maintenant")
        act.triggered.connect(self.trier_dossier_maintenant)
        tools_menu.addSeparator()
        act = tools_menu.addAction("Zoom +")
        act.setShortcut("Ctrl++")
        act.triggered.connect(lambda: self.preview_viewport.zoom_in())
        act = tools_menu.addAction("Zoom -")
        act.setShortcut("Ctrl+-")
        act.triggered.connect(lambda: self.preview_viewport.zoom_out())
        act = tools_menu.addAction("Plein écran")
        act.setShortcut("Ctrl+F")
        act.triggered.connect(self.toggle_fullscreen_preview)

        view_menu = menubar.addMenu("Affichage")
        act = view_menu.addAction("Disposition classique")
        act.triggered.connect(self.reset_layout)
        view_menu.addSeparator()
        view_menu.addAction(self.drop_dock.toggleViewAction())
        view_menu.addAction(self.categories_dock.toggleViewAction())
        view_menu.addAction(self.journal_dock.toggleViewAction())

        # --- Disposition sauvegardee + remplissage initial ---
        self.restore_window_state()
        self.refresh_drop_zone()
        self.refresh_categories_list()

    # ====================== ZONE DE DEPOT ======================
    def on_files_dropped(self, paths):
        """Copie les fichiers/dossiers deposes dans le dossier a trier."""
        added = 0
        exts = self.config.IMAGE_EXTENSIONS | self.config.RAW_EXTENSIONS
        for path in paths:
            p = Path(path)
            if p.is_dir():
                for root, _, fnames in os.walk(p):
                    for fn in fnames:
                        f = Path(root) / fn
                        if f.suffix.lower() in exts:
                            added += self._copy_to_trier(f)
            elif p.is_file() and p.suffix.lower() in exts:
                added += self._copy_to_trier(p)
        if added:
            self.log_message(f"\u2705 {added} photo(s) ajout\u00e9e(s) au dossier \u00e0 trier")
        else:
            self.log_message("\u26a0\ufe0f Aucune image trouv\u00e9e dans le d\u00e9p\u00f4t")
        self.refresh_drop_zone()

    def select_files_dialog(self):
        """Ouvre un dialogue pour selectionner des fichiers a copier dans a_trier."""
        exts = sorted(self.config.IMAGE_EXTENSIONS | self.config.RAW_EXTENSIONS)
        filters = "Images (%s);;Tous les fichiers (*)" % " ".join("*" + e for e in exts)
        paths, _ = QFileDialog.getOpenFileNames(self, "Choisir des photos a trier", "", filters)
        if paths:
            self.on_files_dropped(paths)

    def select_folder_dialog(self):
        """Ouvre un dialogue pour selectionner un dossier a copier dans a_trier."""
        folder = QFileDialog.getExistingDirectory(self, "Choisir un dossier de photos")
        if folder:
            self.on_files_dropped([folder])

    def _copy_to_trier(self, src):
        """Copie une image dans le dossier a trier (renomme en cas de collision)."""
        try:
            self.config.DOSSIER_A_TRIER.mkdir(parents=True, exist_ok=True)
            if src.resolve() == (self.config.DOSSIER_A_TRIER / src.name).resolve():
                return 0  # deja present dans le dossier a trier
        except OSError:
            pass
        dest = self.config.DOSSIER_A_TRIER / src.name
        counter = 1
        while dest.exists():
            dest = self.config.DOSSIER_A_TRIER / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            shutil.copy2(str(src), str(dest))
            return 1
        except Exception as e:
            self.log_message(f"\u26a0\ufe0f Impossible de copier {src}: {e}")
            return 0

    def _image_files_in(self, folder):
        """Liste les fichiers image (recursif) d'un dossier."""
        files = []
        for root, _, fnames in os.walk(folder):
            for fn in fnames:
                p = Path(root) / fn
                if p.suffix.lower() in (self.config.IMAGE_EXTENSIONS | self.config.RAW_EXTENSIONS):
                    files.append(p)
        return files

    def refresh_drop_zone(self):
        """Rafraichit les miniatures de la zone de depot."""
        self.drop_list.clear()
        files = self._image_files_in(self.config.DOSSIER_A_TRIER)
        for p in files:
            item = QListWidgetItem()
            thumb = self.create_thumbnail(str(p), 100)
            if thumb is not None:
                item.setIcon(QIcon(thumb))
            item.setData(Qt.UserRole, str(p))
            item.setToolTip(os.path.basename(str(p)))
            self.drop_list.addItem(item)
        self.drop_count_label.setText(f"{len(files)} photo(s) en attente")
        self.drop_hint.setVisible(len(files) == 0)

    # ====================== PREVISUALISATION ======================
    def toggle_fullscreen_preview(self):
        """Affiche la photo courante en plein ecran."""
        viewport = getattr(self, 'preview_viewport', None)
        if not viewport or not viewport.pixmap_item:
            self.log_message("Aucune photo a afficher en plein ecran")
            return
        view = viewport.view
        rect = QRectF(view.mapFromScene(viewport.pixmap_item.boundingRect()).boundingRect())
        top_left = view.viewport().mapToGlobal(rect.topLeft().toPoint())
        origin = QRect(top_left, rect.toRect().size())
        self.fullscreen_viewer.show_pixmap(viewport.pixmap_item.pixmap(), origin)

    def _position_fullscreen_btn(self):
        """Place les boutons (nav + plein ecran) en bas a droite de la photo affichee."""
        viewport = getattr(self, 'preview_viewport', None)
        bar = getattr(self, 'viewport_btns', None)
        if not viewport or not bar:
            return
        if not viewport.pixmap_item:
            bar.hide()
            return
        bar.show()
        rect = viewport.view.mapFromScene(viewport.pixmap_item.boundingRect()).boundingRect()
        x = rect.right() - bar.width() - 12
        y = rect.bottom() - bar.height() - 12
        x = max(0, min(x, viewport.width() - bar.width()))
        y = max(0, min(y, viewport.height() - bar.height()))
        bar.move(int(x), int(y))

    def preview_from_drop(self, item):
        path = item.data(Qt.UserRole)
        if path:
            paths = [self.drop_list.item(i).data(Qt.UserRole)
                     for i in range(self.drop_list.count())]
            self._nav_files = [p for p in paths if p]
            self.show_in_preview(path)

    def navigate_preview(self, direction):
        """Affiche la photo precedente/suivante de la liste courante."""
        files = getattr(self, '_nav_files', [])
        if not files:
            self.log_message("Aucune navigation disponible")
            return
        index = getattr(self, '_nav_index', 0)
        new_index = (index + direction) % len(files)
        self._nav_index = new_index
        self.show_in_preview(files[new_index])

    def show_in_preview(self, path):
        """Affiche une image en grand dans le viewport de previsualisation."""
        pixmap = self.load_image(path)
        if pixmap is not None and not pixmap.isNull():
            self.preview_viewport.set_pixmap(pixmap)
            self.current_image_path = str(path)
            nav = getattr(self, '_nav_files', [])
            if str(path) in nav:
                self._nav_index = nav.index(str(path))
            self.sort_status_label.setText(os.path.basename(str(path)))
            if getattr(self, 'current_tags_label', None):
                tags = self.tag_manager.get_tags(path)
                self.current_tags_label.setText(f"Tags actuels : {', '.join(tags) if tags else 'Aucun'}")
        else:
            self.log_message(f"Impossible d'afficher : {path}")

    def toggle_journal(self, visible):
        self.journal_dock.setVisible(visible)

    def focus_new_tag_input(self):
        """Place le curseur dans le champ de tag (Ctrl+N)."""
        self.new_tag_input.setFocus()

    def remove_current_tags(self):
        """Supprime tous les tags de l'image affichee (Suppr apres clic sur l'image)."""
        image_path = getattr(self, 'current_image_path', None)
        if not image_path or not os.path.exists(image_path):
            self.log_message("Aucune image affichée")
            return
        tags = self.tag_manager.get_tags(image_path)
        if not tags:
            self.log_message("Cette image n'a pas de tag")
            return
        for tag in tags:
            self.tag_manager.remove_tag(image_path, tag)
        self.log_message(f"Tags supprimés pour {os.path.basename(image_path)}")
        self.load_image_tags(image_path)
        self.refresh_tags_list()

    def edit_undo(self):
        """Ctrl+Z : annule le tri en cours (ou la saisie dans un champ texte)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            focus.undo()
            return
        self.cancel_sort()

    def edit_redo(self):
        """Ctrl+Y : relance le tri (ou refait la saisie dans un champ texte)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            focus.redo()
            return
        self.trier_dossier_maintenant()

    def edit_copy(self):
        """Ctrl+C : copie le chemin de l'image affichee (ou la selection d'un champ)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            focus.copy()
            return
        if self.current_image_path:
            QApplication.clipboard().setText(self.current_image_path)
            self.log_message(f"Chemin copié : {self.current_image_path}")

    def edit_paste(self):
        """Ctrl+V : importe dans a_trier les fichiers dont le chemin est en presse-papiers."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            focus.paste()
            return
        text = QApplication.clipboard().text()
        paths = [line.strip().strip('"') for line in text.splitlines() if line.strip()]
        paths = [p for p in paths if os.path.exists(p)]
        if paths:
            self.on_files_dropped(paths)
        else:
            self.log_message("Aucun chemin de fichier valide dans le presse-papiers")

    def delete_current_image(self):
        """Supprime du disque l'image affichee, apres confirmation."""
        image_path = getattr(self, 'current_image_path', None)
        if not image_path or not os.path.exists(image_path):
            self.log_message("Aucune image affichée")
            QMessageBox.information(
                self, "Supprimer l'image",
                "Aucune image affichée en prévisualisation.\n"
                "Cliquez d'abord sur une photo pour la sélectionner, "
                "puis utilisez le menu Édition ou Ctrl+Del.")
            return
        confirm = QMessageBox.question(
            self, "Confirmer la suppression",
            f"Supprimer définitivement ce fichier ?\n{image_path}",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return
        for tag in self.tag_manager.get_tags(image_path):
            self.tag_manager.remove_tag(image_path, tag)
        try:
            os.remove(image_path)
            self.log_message(f"Supprimé : {image_path}")
        except Exception as e:
            self.log_message(f"Impossible de supprimer : {e}")
            return
        nav = getattr(self, '_nav_files', [])
        if image_path in nav:
            idx = nav.index(image_path)
            nav.pop(idx)
            self._nav_index = min(idx, len(nav) - 1) if nav else 0
            if nav:
                self.show_in_preview(nav[self._nav_index])
            else:
                self.preview_viewport.clear_view()
                self.current_image_path = None
        else:
            self.preview_viewport.clear_view()
            self.current_image_path = None
        self.refresh_drop_zone()
        self.refresh_categories_list()
        self.refresh_tags_list()

    def reset_layout(self):
        """Repositionne les panneaux dans la disposition par defaut."""
        for dock, area in ((self.drop_dock, Qt.LeftDockWidgetArea),
                           (self.categories_dock, Qt.RightDockWidgetArea),
                           (self.journal_dock, Qt.BottomDockWidgetArea)):
            self.addDockWidget(area, dock)
            dock.setFloating(False)
            dock.show()
        self.journal_dock.hide()
        self.log_message("Disposition des panneaux réinitialisée")


    def refresh_categories_list(self):
        """Rafraichit les dossiers de tri : icone de dossier + titre + nombre de photos."""
        self.current_category = None
        self.folder_back_btn.setVisible(False)
        self.category_buttons_frame.setVisible(True)
        self.categories_list.clear()
        for category_name in self.config.CATEGORIES:
            folder = Path(self.config.DOSSIER_TRIES) / category_name
            count = 0
            if folder.exists():
                count = len(self._image_files_in(folder))
            item = QListWidgetItem(f"{category_name} ({count})")
            item.setData(Qt.UserRole, category_name)
            item.setIcon(self._folder_icon())
            self.categories_list.addItem(item)

    def _folder_icon(self):
        """Icone de dossier standard du systeme."""
        return QApplication.style().standardIcon(QStyle.SP_DirIcon)

    def on_category_item_clicked(self, item):
        """Clic dans le panneau : ouvre un dossier ou affiche une photo."""
        if self.current_category is None:
            self.open_category_folder(item.data(Qt.UserRole))
        else:
            path = item.data(Qt.UserRole)
            if path:
                self.show_in_preview(path)

    def open_category_folder(self, category):
        """Affiche les miniatures des photos triees d'un dossier de categorie."""
        if not category:
            return
        self.current_category = category
        self.folder_back_btn.setVisible(True)
        self.category_buttons_frame.setVisible(False)
        self.categories_list.clear()
        folder = Path(self.config.DOSSIER_TRIES) / category
        files = self._image_files_in(folder) if folder.exists() else []
        self._nav_files = [str(p) for p in files]
        self._nav_index = 0
        for p in files:
            entry = QListWidgetItem()
            thumb = self.create_thumbnail(str(p), 80)
            if thumb is not None:
                entry.setIcon(QIcon(thumb))
            entry.setData(Qt.UserRole, str(p))
            entry.setToolTip(os.path.basename(str(p)))
            self.categories_list.addItem(entry)
        if files:
            self.show_in_preview(str(files[0]))
        else:
            self.preview_viewport.clear_view()
            self.sort_status_label.setText(f"Dossier '{category}' vide")

    def show_folders_view(self):
        """Revient a la liste des dossiers de tri."""
        self.refresh_categories_list()

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
        """Sauvegarde les parametres actuels"""
        if self.save_current_config():
            self.settings_status.setText("\u2705 Param\u00e8tres enregistr\u00e9s avec succ\u00e8s!")
            self.log_message("Param\u00e8tres enregistr\u00e9s")
            QMessageBox.information(self, "Succ\u00e8s", "Les param\u00e8tres ont \u00e9t\u00e9 enregistr\u00e9s avec succ\u00e8s!")
        else:
            self.settings_status.setText("\u274c Erreur lors de l'enregistrement")
            QMessageBox.critical(self, "Erreur", "Impossible d'enregistrer les param\u00e8tres!")

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

    # ====================== NAVIGATION ======================
    def create_thumbnail(self, image_path, size=120):
        """Cree une miniature QPixmap a partir d'un chemin d'image.
        Utilise PIL pour generer un JPEG en memoire -> loadFromData force une copie complete.
        Les miniatures sont mises en cache (chemin + mtime + taille) pour eviter
        de re-decoder chaque image a chaque rafraichissement de l'interface.
       """
        try:
            cache = getattr(self, '_thumb_cache', None)
            if cache is None:
                cache = self._thumb_cache = {}
            key = None
            try:
                key = (str(image_path).lower(), os.path.getmtime(image_path), size)
            except OSError:
                key = None
            if key is not None and key in cache:
                return cache[key]
            if image_path.lower().endswith(tuple(self.config.RAW_EXTENSIONS)):
                with rawpy.imread(image_path) as raw:
                    rgb_array = raw.postprocess(half_size=True)
                    pil_img = Image.fromarray(rgb_array)
            else:
                pil_img = Image.open(image_path)
                pil_img = pil_img.convert('RGB')

            # Redimensionner avec PIL (preserve le ratio)
            pil_img.thumbnail((size, size), Image.Resampling.LANCZOS)

            # Convertir en JPEG en memoire -> loadFromData force une copie complete
            import io
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            img_data = buffer.getvalue()

            # loadFromData fait une copie complete -> pas de probleme de garbage collection
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)

            if pixmap.isNull():
                return None
            result = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if key is not None:
                cache[key] = result
            return result
        except Exception as e:
            logging.debug(f"Erreur miniature {image_path}: {e}")
            return None

    def add_tag_to_image(self):
        """Ajoute un tag a l'image actuellement affichee en previsualisation."""
        image_path = getattr(self, 'current_image_path', None)
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Erreur", "Aucune image affichée en prévisualisation !")
            return
        new_tag = self.new_tag_input.text().strip()
        if not new_tag:
            QMessageBox.warning(self, "Erreur", "Veuillez saisir un tag !")
            return
        self.tag_manager.add_tag(image_path, new_tag)
        self.new_tag_input.clear()
        self.log_message(f"✅ Tag '{new_tag}' ajouté à {os.path.basename(image_path)}")
        self.load_image_tags(image_path)
        self.refresh_tags_list()

    def refresh_tags_list(self):
        """Rafraichit la liste des tags avec le nombre d'images par tag"""
        self.tags_list_widget.clear()
        all_tags = self.tag_manager.get_all_tags()
        if getattr(self, 'tag_completer_model', None) is not None:
            self.tag_completer_model.setStringList(all_tags)
        for tag in all_tags:
            count = len([p for p in self.tag_manager.get_images_by_tag(tag) if os.path.exists(p)])
            item = QListWidgetItem(f"{tag} ({count})")
            item.setData(Qt.UserRole, tag)
            self.tags_list_widget.addItem(item)

    def on_tag_selected(self, item):
        """Affiche les miniatures des images correspondant au tag clique"""
        tag = item.data(Qt.UserRole)
        if not tag:
            return
        self.display_images_by_tag(tag)

    def display_images_by_tag(self, tag):
        """Affiche les miniatures des images d'un tag dans le panneau dossiers."""
        all_results = self.tag_manager.get_images_by_tag(tag)
        if not all_results:
            self.log_message(f"Aucune image trouvée pour le tag '{tag}'.")
            return
        self.current_category = f"__tag__{tag}"
        self.folder_back_btn.setVisible(True)
        self.category_buttons_frame.setVisible(False)
        self.categories_list.clear()
        self._nav_files = []
        self._nav_index = 0
        found_count = 0
        missing_count = 0
        for img_path in all_results:
            if not os.path.exists(img_path):
                missing_count += 1
                continue
            entry = QListWidgetItem()
            thumbnail = self.create_thumbnail(img_path, 80)
            if thumbnail is not None:
                entry.setIcon(QIcon(thumbnail))
            found_count += 1
            self._nav_files.append(img_path)
            entry.setData(Qt.UserRole, img_path)
            tags = self.tag_manager.get_tags(img_path)
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            entry.setToolTip(f"{os.path.basename(img_path)}{tag_str}")
            self.categories_list.addItem(entry)
        msg = f"✅ {found_count} image(s) pour le tag '{tag}'."
        if missing_count > 0:
            msg += f" ({missing_count} fichier(s) manquant(s))"
        self.log_message(msg)

    def load_image_tags(self, image_path):
        """Charge les tags d'une image"""
        try:
            tags = self.tag_manager.get_tags(image_path)
            self.current_tags_label.setText(f"Tags actuels : {', '.join(tags) if tags else 'Aucun'}")
            self.log_message(f"Tags pour {os.path.basename(image_path)}: {tags}")
        except Exception as e:
            self.log_message(f"\u26a0\ufe0f Erreur de chargement des tags: {e}")
    def load_image(self, file_path):
        """Charge une image (y compris RAW) et retourne un QPixmap."""
        try:
            # Vérifier si c'est un fichier RAW
            if file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.raw', '.rw2', '.pef', '.raf', '.x3f')):
                with rawpy.imread(file_path) as raw:
                    # Convertir en RGB 8 bits
                    rgb_array = raw.postprocess()
                    height, width = rgb_array.shape[:2]
                    # ✅ Faire une copie du tableau pour éviter les problèmes de mémoire
                    rgb_copy = rgb_array.copy()
                    # Créer une QImage à partir du tableau numpy
                    qimage = QImage(
                        rgb_copy.ctypes.data,
                        width,
                        height,
                        3 * width,  # bytes_per_line (3 canaux RGB)
                        QImage.Format_RGB888
                    )
                    if qimage.isNull():
                        return None
                    return QPixmap.fromImage(qimage)
            else:
                # Charger les images classiques avec QPixmap
                return QPixmap(file_path)
        except Exception as e:
            self.log_message(f"Erreur lors du chargement de {file_path}: {e}")
            return None



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
        self.refresh_drop_zone()
        self.refresh_categories_list()


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

            # Optimiser le traitement SANS deformater : orientation EXIF respectee,
            # reduction en preservant le ratio (thumbnail ne fait qu'reduire)
            image = ImageOps.exif_transpose(image)
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)

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

            # Migrer les tags vers le nouveau chemin (evite les entrees fantomes)
            self.tag_manager.migrate_tags(chemin_source, chemin_dest)

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
        """Classifie et deplace une image (CLIP uniquement)"""
        try:
            categorie = self.classer_image_clip(chemin_image)
            self.deplacer_image_et_metadonnees(chemin_image, categorie)
        except Exception as e:
            self.log_message(f"\u26a0\ufe0f Erreur avec {chemin_image}: {e}")

    def trier_dossier_maintenant(self):
        """Lance le tri du dossier a trier dans un thread separe."""
        if self.tri_worker is not None and self.tri_worker.isRunning():
            self.log_message("\u26a0\ufe0f Un tri est d\u00e9j\u00e0 en cours.")
            return
        files = self.collect_files_to_sort(self.config.DOSSIER_A_TRIER)
        if not files:
            self.log_message("Aucun fichier \u00e0 trier dans le dossier de d\u00e9p\u00f4t.")
            self.sort_status_label.setText("Rien \u00e0 trier")
            return
        self.sort_progress.setVisible(True)
        self.cancel_sort_btn.setVisible(True)
        self.sort_progress.setMaximum(len(files))
        self.sort_progress.setValue(0)
        self.sort_status_label.setText(f"Tri en cours : 0 / {len(files)} photos")
        self.log_message(f"\ud83d\udcc1 Tri de {len(files)} fichier(s)...")
        self.tri_worker = TriWorker(self, files)
        self.tri_worker.progress.connect(self.on_sort_progress)
        self.tri_worker.finished_tri.connect(self.on_sort_finished)
        self.tri_worker.error.connect(lambda m: self.log_message(f"\u26a0\ufe0f {m}"))
        self.tri_worker.start()

    def collect_files_to_sort(self, dossier):
        """Liste les fichiers a trier (tout sauf metadonnees), RAW d'abord."""
        files = []
        for root, _, filenames in os.walk(dossier):
            for filename in filenames:
                filepath = Path(root) / filename
                if filepath.suffix.lower() in (self.config.IMAGE_EXTENSIONS | self.config.RAW_EXTENSIONS):
                    files.append(filepath)
        raw_files = [f for f in files if f.suffix.lower() in self.config.RAW_EXTENSIONS]
        other_files = [f for f in files if f not in raw_files]
        return raw_files + other_files

    def cancel_sort(self):
        """Demande l'annulation du tri en cours."""
        if self.tri_worker is not None and self.tri_worker.isRunning():
            self.tri_worker.cancel()
            self.sort_status_label.setText("Annulation en cours...")
            self.log_message("\u26a0\ufe0f Annulation du tri demand\u00e9e")

    def on_sort_progress(self, index, total, name):
        self.sort_progress.setMaximum(total)
        self.sort_progress.setValue(index)
        if name:
            self.sort_status_label.setText(f"Tri en cours : {index} / {total} photos")

    def on_sort_finished(self, done, errors):
        self.sort_progress.setVisible(False)
        self.cancel_sort_btn.setVisible(False)
        msg = f"Tri termin\u00e9 : {done} tri\u00e9(s)"
        if errors:
            msg += f", {errors} erreur(s)"
        self.sort_status_label.setText(msg)
        self.log_message(f"\u2705 {msg}")
        self.refresh_drop_zone()
        self.refresh_categories_list()

    def eventFilter(self, obj, event):
        """Filtre les événements : dépôt glisser-déposer sur la zone de dépôt."""
        # Depot sur toute la zone de dépôt (cadre compris)
        if obj == getattr(self, 'drop_group', None):
            if event.type() == QEvent.DragEnter or event.type() == QEvent.DragMove:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Drop:
                paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
                if paths:
                    self.on_files_dropped(paths)
                event.acceptProposedAction()
                return True
        # Bouton plein ecran : suivre le coin bas-droit de la photo affichee
        viewport = getattr(self, 'preview_viewport', None)
        if viewport is not None:
            if obj == viewport and event.type() == QEvent.Resize:
                self._position_fullscreen_btn()
            elif obj == viewport.view.viewport() and event.type() == QEvent.Paint:
                self._position_fullscreen_btn()
        return super().eventFilter(obj, event)

    # ====================== DUMP UI (DEBUG) ======================
    def _widget_text(self, widget):
        """Retourne le texte utile d'un widget pour le dump de debug"""
        try:
            if isinstance(widget, (QLabel, QPushButton, QCheckBox, QRadioButton, QGroupBox)):
                return widget.text()
            if isinstance(widget, QLineEdit):
                return widget.text()
            if isinstance(widget, QTextEdit):
                return widget.toPlainText()
            if isinstance(widget, QTabWidget):
                return f"{widget.count()} onglet(s), courant: {widget.tabText(widget.currentIndex())}"
            if isinstance(widget, QListWidget):
                sel = widget.selectedItems()
                sel_txt = sel[0].text() if sel else ""
                return f"{widget.count()} item(s), selection: '{sel_txt}'"
        except Exception:
            pass
        return ""

    def dump_ui_state(self):
        """Enregistre un dump complet de l'interface dans debug_ui/ (raccourci F12)"""
        try:
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            dump_dir = Path(__file__).parent / "debug_ui"
            dump_dir.mkdir(exist_ok=True)

            lines = []
            lines.append(f"=== DUMP UI {timestamp} ===")
            lines.append(f"Fenetre: {self.windowTitle()} pos=({self.x()}, {self.y()}) taille=({self.width()}x{self.height()})")
            lines.append(f"Etat fenetre: {'maximisee' if self.isMaximized() else 'normale'}")

            lines.append("")
            lines.append("--- ARBORESCENCE DES WIDGETS (pos relative a la fenetre) ---")
            for widget in [self] + self.findChildren(QWidget):
                top_left = widget.mapTo(self, QPoint(0, 0))
                info = f"[{widget.metaObject().className()}]"
                if widget.objectName():
                    info += f" nom='{widget.objectName()}'"
                info += f" pos=({top_left.x()}, {top_left.y()}) taille=({widget.width()}x{widget.height()})"
                if not widget.isVisible():
                    info += " INVISIBLE"
                text = self._widget_text(widget)
                if text:
                    text_short = " ".join(text.split())
                    if len(text_short) > 120:
                        text_short = text_short[:117] + "..."
                    info += f" texte='{text_short}'"
                lines.append(info)

            lines.append("")
            lines.append("--- ETAT INTERNE ---")
            lines.append(f"Image courante: {self.current_image_path}")
            lines.append(f"Dossier affiche (panneau tri): {self.current_category}")
            if getattr(self, 'preview_viewport', None) and self.preview_viewport.pixmap_item:
                r = self.preview_viewport.pixmap_item.boundingRect()
                lines.append(f"Pixmap (viewport): taille=({r.width():.0f}x{r.height():.0f})")
            else:
                lines.append("Pixmap: aucun")
            lines.append(f"BASE_DIR: {self.config.BASE_DIR}")
            lines.append(f"Dossier a trier: {self.config.DOSSIER_A_TRIER}")
            lines.append(f"Dossier de sortie: {self.config.DOSSIER_TRIES}")
            lines.append(f"Surveillance active: {self.watchdog_worker is not None}")
            lines.append(f"Modele CLIP charge: {self.model_clip is not None}")
            lines.append(f"Categories: {list(self.config.CATEGORIES.keys())}")

            lines.append("")
            lines.append("--- JOURNAL (30 dernieres lignes) ---")
            if self.log_text:
                lines.extend(self.log_text.toPlainText().splitlines()[-30:])

            dump_file = dump_dir / f"ui_dump_{timestamp}.txt"
            dump_file.write_text("\n".join(lines), encoding="utf-8")

            # Capture PNG de la fenetre (analyse pixels possible avec PIL)
            capture_file = dump_dir / f"capture_{timestamp}.png"
            self.grab().save(str(capture_file))

            msg = f"Dump UI enregistre: {dump_file}"
            logging.info(msg)
            self.log_message(f"[DEBUG] {msg} + capture {capture_file.name}")
        except Exception as e:
            logging.error(f"Erreur lors du dump UI: {e}")
            try:
                self.log_message(f"[DEBUG] Erreur de dump UI: {e}")
            except Exception:
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoSorterApp()
    window.show()
    sys.exit(app.exec_())