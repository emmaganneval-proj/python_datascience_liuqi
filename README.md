# Temps de trajet en transports en commun et facteurs socio-économiques en Île-de-France

## Objectif du projet

Existe-t-il un lien entre les temps de trajet en transport en commun et le niveau de vie / les catégories socio-progressionnelles en Île-de-France ? Plus généralement, quelles caractéristiques du territoire peut-on déduire en fonction de la position dans le réseau francilien ?

## Installation

- Il faut utiliser Python 3.12 (sinon incompatibilité avec pyproj) ;
- Créer un environnement virtuel et installer les dépendances :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Décommenter les deux "telecharger_zip" pour télécharger les données (attention, les deux fichiers font 3,4 Go une fois décompressés).