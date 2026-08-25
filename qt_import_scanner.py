#!/usr/bin/env python3
"""
Qt Import Scanner
=================

Un script pour analyser un fichier Python et ajouter automatiquement les imports Qt manquants.

Utilisation:
    python qt_import_scanner.py fichier.py

Fonctionnalités:
    - Détecte les classes Qt utilisées dans le code
    - Ajoute les imports manquants dans la section PyQt5
    - Conserve l'ordre et le style existant
"""

import re
import sys
from pathlib import Path


# Mapping des classes Qt à leurs modules respectifs
QT_CLASSES = {
    # QtWidgets
    'QApplication': 'PyQt5.QtWidgets',
    'QMainWindow': 'PyQt5.QtWidgets',
    'QWidget': 'PyQt5.QtWidgets',
    'QVBoxLayout': 'PyQt5.QtWidgets',
    'QHBoxLayout': 'PyQt5.QtWidgets',
    'QLabel': 'PyQt5.QtWidgets',
    'QLineEdit': 'PyQt5.QtWidgets',
    'QPushButton': 'PyQt5.QtWidgets',
    'QListWidget': 'PyQt5.QtWidgets',
    'QTextEdit': 'PyQt5.QtWidgets',
    'QTabWidget': 'PyQt5.QtWidgets',
    'QFrame': 'PyQt5.QtWidgets',
    'QFileDialog': 'PyQt5.QtWidgets',
    'QMessageBox': 'PyQt5.QtWidgets',
    'QCheckBox': 'PyQt5.QtWidgets',
    'QGroupBox': 'PyQt5.QtWidgets',
    'QRadioButton': 'PyQt5.QtWidgets',
    'QInputDialog': 'PyQt5.QtWidgets',
    'QListWidgetItem': 'PyQt5.QtWidgets',
    'QSizePolicy': 'PyQt5.QtWidgets',
    'QGraphicsView': 'PyQt5.QtWidgets',
    'QGraphicsScene': 'PyQt5.QtWidgets',
    'QGraphicsPixmapItem': 'PyQt5.QtWidgets',
    'QGraphicsItem': 'PyQt5.QtWidgets',
    'QScrollBar': 'PyQt5.QtWidgets',
    
    # QtGui
    'QPixmap': 'PyQt5.QtGui',
    'QIcon': 'PyQt5.QtGui',
    'QImage': 'PyQt5.QtGui',
    'QTransform': 'PyQt5.QtGui',
    'QPainter': 'PyQt5.QtGui',
    'QImageReader': 'PyQt5.QtGui',
    'QBrush': 'PyQt5.QtGui',
    'QPen': 'PyQt5.QtGui',
    'QFont': 'PyQt5.QtGui',
    'QColor': 'PyQt5.QtGui',
    'QPalette': 'PyQt5.QtGui',
    'QCursor': 'PyQt5.QtGui',
    'QMouseEvent': 'PyQt5.QtGui',
    'QWheelEvent': 'PyQt5.QtGui',
    'QKeyEvent': 'PyQt5.QtGui',
    'QPaintEvent': 'PyQt5.QtGui',
    
    # QtCore
    'QPoint': 'PyQt5.QtCore',
    'QPointF': 'PyQt5.QtCore',
    'QSize': 'PyQt5.QtCore',
    'QSizeF': 'PyQt5.QtCore',
    'QRect': 'PyQt5.QtCore',
    'QRectF': 'PyQt5.QtCore',
    'Qt': 'PyQt5.QtCore',
    'QThread': 'PyQt5.QtCore',
    'pyqtSignal': 'PyQt5.QtCore',
    'QObject': 'PyQt5.QtCore',
    'QEvent': 'PyQt5.QtCore',
    'QTimer': 'PyQt5.QtCore',
    'QMutex': 'PyQt5.QtCore',
    'QMutexLocker': 'PyQt5.QtCore',
    
    # Autres modules PyQt5
    'QFileSystemWatcher': 'PyQt5.QtCore',
    'QSettings': 'PyQt5.QtCore',
}

# Classes Qt déjà importées (à ne pas ajouter)
QT_ALREADY_IMPORTED = {
    'QApplication', 'QMainWindow', 'QWidget', 'QVBoxLayout', 'QHBoxLayout',
    'QLabel', 'QLineEdit', 'QPushButton', 'QListWidget', 'QTextEdit',
    'QTabWidget', 'QFrame', 'QFileDialog', 'QMessageBox', 'QCheckBox',
    'QGroupBox', 'QRadioButton', 'QInputDialog', 'QListWidgetItem', 'QSizePolicy',
    'QGraphicsView', 'QGraphicsScene',
    'QPixmap', 'QIcon', 'QImage', 'QTransform',
    'QPointF',
    'QPoint', 'QSize', 'Qt', 'QThread', 'pyqtSignal'
}


def find_qt_imports(file_path):
    """
    Trouve tous les imports Qt dans un fichier.
    Retourne un dictionnaire {module: [classes]}
    """
    imports = {
        'PyQt5.QtWidgets': [],
        'PyQt5.QtGui': [],
        'PyQt5.QtCore': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouve les imports existants
    for module in imports.keys():
        pattern = rf'from {module} import \(?([^)]+)\)?'
        match = re.search(pattern, content)
        if match:
            imported = match.group(1).replace('\n', ' ').replace(' ', '')
            imports[module] = imported.split(',')
    
    return imports


def find_used_qt_classes(file_path):
    """
    Trouve toutes les classes Qt utilisées dans le fichier.
    Retourne un ensemble de noms de classes.
    """
    used_classes = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouve toutes les utilisations de classes Qt
    for class_name in QT_CLASSES:
        # Cherche les utilisations qui ne sont pas dans des strings ou des commentaires
        pattern = rf'(?<!#|\"|\')\b{class_name}\b(?!\w)'
        if re.search(pattern, content):
            used_classes.add(class_name)
    
    return used_classes


def add_missing_imports(file_path):
    """
    Ajoute les imports Qt manquants au fichier.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouve les imports existants
    existing_imports = find_qt_imports(file_path)
    
    # Trouve les classes utilisées
    used_classes = find_used_qt_classes(file_path)
    
    # Détermine les classes manquantes pour chaque module
    missing_imports = {
        'PyQt5.QtWidgets': [],
        'PyQt5.QtGui': [],
        'PyQt5.QtCore': []
    }
    
    for class_name in used_classes:
        if class_name not in QT_ALREADY_IMPORTED:
            module = QT_CLASSES.get(class_name)
            if module:
                if class_name not in existing_imports[module]:
                    missing_imports[module].append(class_name)
    
    # Si rien à ajouter, retourne
    if not any(missing_imports.values()):
        print("✅ Aucun import Qt manquant trouvé.")
        return False
    
    # Ajoute les imports manquants
    new_content = content
    
    for module, classes in missing_imports.items():
        if classes:
            # Trouve l'import existant pour ce module
            import_pattern = rf'(from {module} import \(?[^)]*\)?\n)'
            match = re.search(import_pattern, new_content)
            
            if match:
                # L'import existe déjà, on ajoute les classes manquantes
                import_start = match.start()
                import_end = match.end()
                
                # Vérifie si l'import est sur une seule ligne ou multi-lignes
                if '(' in new_content[import_start:import_end]:
                    # Import multi-lignes
                    # Trouve la fin de l'import
                    paren_count = 0
                    end_pos = import_start
                    for i in range(import_start, len(new_content)):
                        if new_content[i] == '(':
                            paren_count += 1
                        elif new_content[i] == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                end_pos = i + 1
                                break
                    
                    # Ajoute les nouvelles classes avant la parenthèse fermante
                    insert_pos = end_pos - 1
                    existing_classes = re.findall(r'\b\w+\b', new_content[import_start:end_pos])
                    existing_classes = [c for c in existing_classes if c in QT_CLASSES or c in ['Qt', 'pyqtSignal']]
                    
                    # Ajoute les classes manquantes
                    new_classes_str = ', '.join(classes)
                    if existing_classes:
                        new_content = new_content[:insert_pos] + ', ' + new_classes_str + new_content[insert_pos:]
                    else:
                        new_content = new_content[:insert_pos] + new_classes_str + new_content[insert_pos:]
                else:
                    # Import sur une seule ligne
                    # Ajoute les classes manquantes
                    existing_line = new_content[import_start:import_end]
                    new_classes_str = ', '.join(classes)
                    new_line = existing_line.rstrip() + ', ' + new_classes_str + '\n'
                    new_content = new_content[:import_start] + new_line + new_content[import_end:]
            else:
                # L'import n'existe pas, on l'ajoute
                import_line = f"from {module} import {', '.join(classes)}\n"
                
                # Trouve où insérer l'import (après les autres imports PyQt5)
                last_pyqt5_import = -1
                for pyqt5_module in ['PyQt5.QtWidgets', 'PyQt5.QtGui', 'PyQt5.QtCore']:
                    pos = new_content.rfind(f'from {pyqt5_module} import')
                    if pos > last_pyqt5_import:
                        last_pyqt5_import = pos
                
                if last_pyqt5_import > 0:
                    # Insère après le dernier import PyQt5
                    insert_pos = new_content.find('\n', last_pyqt5_import) + 1
                    new_content = new_content[:insert_pos] + import_line + new_content[insert_pos:]
                else:
                    # Insère après les autres imports
                    import_end_pos = new_content.rfind('from sentence_transformers')
                    if import_end_pos > 0:
                        insert_pos = new_content.find('\n', import_end_pos) + 1
                        new_content = new_content[:insert_pos] + '\n' + import_line + new_content[insert_pos:]
    
    # Écrit le fichier mis à jour
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Imports Qt manquants ajoutés avec succès.")
    return True


def main():
    if len(sys.argv) < 2:
        print("Utilisation: python qt_import_scanner.py fichier.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ Fichier non trouvé: {file_path}")
        sys.exit(1)
    
    # Ajoute les imports manquants
    modified = add_missing_imports(file_path)
    
    if modified:
        print(f"Fichier mis à jour: {file_path}")
    else:
        print(f"Aucune modification nécessaire pour: {file_path}")


if __name__ == '__main__':
    main()
