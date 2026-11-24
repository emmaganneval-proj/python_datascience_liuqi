# Proches en métro, proches en revenus ?

## Objectif du projet

Notre projet étudie les liens entre les disparités socio-économiques des territoires et l'organisation du réseau ferrré d'Île-de-France. Les principales données utilisées sont les données Filosofi pour le niveau de vie, les données IDFM pour le réseau RATP/SNCF en Île-de-France, et dans une moindre mesure les données MOBPRO à des fins d'illustration et de validation.

Nous explorons deux axes :
1. La relation entre accessibilité et niveau de vie ;
2. Les contrastes, continuités et ruptures de niveau de vie le long des lignes IDFM.

## Exécution

Sur Onyxia, le notebook s'exécute sans difficulté sur une session avec au moins 15Gio de mémoire.

Cloner le repo avec : 

```
git clone https://github.com/emmaganneval-proj/python_datascience_liuqi.git
```

Les données sont stockées dans mon bucket Onyxia personnel si elles ne se chargent pas depuis les URL. Pour les charger en cas de problème (uniquement depuis Onyxia) :

```
cd python_datascience_liuqi
aws s3 sync s3://t[mon_nom_de_famille]/data/ ./data/
```


À noter que les données (plus de 3,5Go) ne sont téléchargées qu'une seule fois.