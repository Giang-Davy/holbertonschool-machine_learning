Ce projet implémente un agent Deep Q-Learning (DQN) qui joue à Breakout sur Atari à l'aide de Keras-RL2 et Gymnasium.

1. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```
2. **Installation des ROMs Atari**
```bash
AutoROM --accept-license
```
3. **Modification du fichier Callback.py pour la compatibilité**
Dans une machine local window:
```bash
# Modifier "Python310" selon sa version de Python  
AppData\Roaming\Python\Python310\site-packages\rl\callbacks.py
```
Dans un environnement virtuel linux:
```bash
find ~/.pyenv/versions/*/lib/python*/site-packages/rl -name "callbacks.py"
```
Chercher au début du fichier callback.py:
```bash
from tensorflow.keras import __version__ as KERAS_VERSION
```
Remplacer cette ligne par :
```bash
try:
    from tensorflow.keras import __version__ as KERAS_VERSION
except ImportError:
    KERAS_VERSION = '2.10.0'
```
4. **Utilisation des fichiers**
Pour entraîner:
```bash
python3 train.py
```
Pour tester l'agent entraîné
```bash
python3 play.py
```
Attention, l'entraînement peut prendre plusieurs heures, l'agent déjà entrainé est mis à disposition pour éviter cela, sous le nom de "policy.h5". Donc exécuter directement le fichier "play.py".