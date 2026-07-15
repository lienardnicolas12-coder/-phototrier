# PhotoTrier - Logiciel de tri photo automatique

Un outil puissant pour trier automatiquement vos photos (y compris les fichiers RAW) en utilisant l'IA.

## 🚀 Fonctionnalités

- **Tri automatique** des photos par catégories (paysage, portrait, macro, nocturne, ciel, architecture, urbain, animal, autre)
- **Support des fichiers RAW** (.raw, .cr2, .nef, .arw, .dng, .orf, .rw2, .pef, .raf, .x3f)
- **Support des images classiques** (.jpg, .jpeg, .png, .bmp, .tiff, .webp)
- **Gestion des métadonnées** (déplacement automatique des fichiers .thm, .xmp, .json)
- **Interface graphique** avec PyQt5
- **Surveillance en temps réel** des dossiers avec Watchdog
- **Système de tagging** pour organiser vos photos
- **Support de deux modèles** : CLIP (recommandé) et TensorFlow personnalisé

## 📁 Structure du projet

```
-phototrier/
├── script_tri_RAW.py          # Script CLI pour le tri automatique
├── tri_photos_automatique_gui.py  # Interface graphique
├── requirements.txt           # Dépendances Python
├── README.md                  # Documentation
└── .gitignore                 # Fichiers à ignorer
```

## 🛠️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/lienardnicolas12-coder/-phototrier.git
cd -phototrier
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Sur Windows
python -m venv venv
venv\Scripts\activate

# Sur macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

⚠️ **Note importante** : L'installation peut prendre du temps et nécessiter beaucoup d'espace disque (environ 2-3 Go) à cause des modèles d'IA.

## 📂 Configuration

Par défaut, le logiciel utilise les dossiers suivants dans votre répertoire `Documents`:
- `~/Documents/photos/tri_photo_RAW/a_trier/` - Dossier source (placez vos photos ici)
- `~/Documents/photos/tri_photo_RAW/tries/` - Dossier de destination (photos triées)
- `~/Documents/photos/tri_photo_RAW/models/` - Dossier pour les modèles TensorFlow
- `~/Documents/photos/tri_photo_RAW/tags.json` - Fichier de tags

Vous pouvez modifier ces chemins dans le code ou via l'interface graphique.

## 🎯 Utilisation

### Avec l'interface graphique

```bash
python tri_photos_automatique_gui.py
```

1. **Onglet "Tri Automatique"** :
   - Sélectionnez le dossier à trier
   - Sélectionnez le dossier de sortie
   - Choisissez le modèle (CLIP recommandé)
   - Activez la surveillance pour un tri automatique
   - Cliquez sur "Trier maintenant" pour trier manuellement

2. **Onglet "Gestion des Tags"** :
   - Ajoutez des tags à vos images
   - Recherchez des images par tag
   - Visualisez vos images

3. **Onglet "Modèle TensorFlow"** :
   - Chargez un modèle TensorFlow personnalisé

### Avec le script CLI

```bash
python script_tri_RAW.py
```

Le script va :
1. Créer les dossiers nécessaires
2. Charger le modèle CLIP
3. Trier les photos déjà présentes
4. Surveiller le dossier pour les nouvelles photos

Appuyez sur `Ctrl+C` pour arrêter.

## 🔧 Personnalisation

### Modifier les catégories

Dans les deux scripts, vous pouvez modifier les catégories et leurs descriptions :

```python
CATEGORIES = {
    "paysage": "une photo de paysage avec montagnes, vue large, horizon, nature",
    "portrait": "un portrait de personne, visage, premier plan, sourire",
    # ... ajoutez vos propres catégories
}
```

### Utiliser un modèle différent

Pour CLIP, vous pouvez essayer d'autres modèles :
- `'clip-ViT-B-32'` (par défaut, ~500 Mo)
- `'all-MiniLM-L6-v2'` (plus léger, ~80 Mo)
- `'clip-ViT-L-14'` (plus précis, ~2 Go)

Modifiez dans le code :
```python
model = SentenceTransformer('nom-du-modele', device=device)
```

### Utiliser un modèle TensorFlow

1. Entraînez votre modèle avec Keras
2. Sauvegardez-le au format `.h5` ou `.keras`
3. Dans l'interface graphique, sélectionnez le fichier du modèle
4. Cliquez sur "Charger le modèle TensorFlow"

## ⚠️ Problèmes courants et solutions

### 1. Erreur de mémoire (CUDA out of memory)

**Solution** : Utilisez un modèle plus léger ou réduisez la taille des images.

```python
# Dans le code, modifiez :
if max(image.size) > 512:  # au lieu de 768
    image = image.resize((512, 512))
```

### 2. Erreur de chargement du modèle CLIP

**Solutions** :
- Vérifiez votre connexion Internet
- Libérez de l'espace disque (au moins 2-3 Go)
- Essayez un modèle plus léger : `'all-MiniLM-L6-v2'`
- Installez PyTorch avec CUDA si vous avez une carte graphique NVIDIA

### 3. Les fichiers RAW ne s'ouvrent pas

**Solutions** :
- Installez `rawpy` avec les dépendances système :
  ```bash
  # Sur Ubuntu/Debian
  sudo apt-get install libraw-dev
  
  # Sur macOS
  brew install libraw
  
  # Puis réinstallez rawpy
  pip uninstall rawpy
  pip install rawpy
  ```

### 4. Problème avec les chemins Windows

**Solution** : Utilisez des chemins relatifs ou modifiez les chemins dans la configuration :

```python
# Dans Config class
BASE_DIR = Path.home() / "Documents" / "photos" / "tri_photo_RAW"
```

## 📊 Performances

- **CLIP-ViT-B-32** : ~500 Mo, bon équilibre précision/vitesse
- **all-MiniLM-L6-v2** : ~80 Mo, plus rapide mais moins précis
- Temps de traitement : ~1-2 secondes par image (selon la taille et le matériel)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Ajouter de nouvelles fonctionnalités

## 📄 Licence

Ce projet est sous licence MIT.

---

**Créé par Nicolas Lienard**

Pour toute question ou problème, ouvrez une issue sur GitHub.
