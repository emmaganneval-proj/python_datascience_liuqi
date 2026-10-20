import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import urllib
import os
import pandas as pd
import polars as pl
import zipfile

URL_ZIP_IDFM = "https://files.mobilitydatabase.org/tdg-80921/tdg-80921-202610140154/tdg-80921-202610140154.zip"
URL_ZIP_FILOSOFI = "https://www.insee.fr/fr/statistiques/fichier/8735162/Filosofi2021_carreaux_200m_shp.zip"


def telecharger_zip(url, nom_dossier):
    """Télécharge un zip et le décompresse dans le dossier /data/{nom_dossier}"""
    with (
        urllib.request.urlopen(url) as response,
        open("data/tmp.zip", "wb") as fichier_sortie,
    ):
        fichier_sortie.write(response.read())
    with zipfile.ZipFile("data/tmp.zip") as zf:
        zf.extractall(f"data/{nom_dossier}")
    os.remove("data/tmp.zip")


def charger_filosofi(path):
    """Chargement des données de Filosofi"""
    liste_departements = ["75", "77", "78", "91", "92", "93", "94", "95"]
    df_filosofi = gpd.read_file(path, columns=["lcog_geo", "ind", "ind_snv"])
    df_filosofi = df_filosofi.loc[
        df_filosofi["lcog_geo"].str[:2].isin(liste_departements)
    ]
    return df_filosofi


def charger_lignes(path):
    """Chargement des lignes IDFM"""

    df_routes = pd.read_csv(f"{path}/routes.txt")

    filtre_metro = df_routes["route_type"] == 1
    filtre_rer = (df_routes["route_type"] == 2) & (df_routes["agency_id"] == "IDFM:71")
    df_routes = df_routes.loc[filtre_metro | filtre_rer][["route_id"]]

    df_trips = pd.read_csv(f"{path}/trips.txt")
    df_trips = df_trips.merge(df_routes, on="route_id", how = "inner")

    df_shapes = pd.read_csv(f"{path}/shapes.txt")
    df_shapes = df_shapes.merge(df_trips, on = "shape_id")

    return df_routes, df_trips, df_shapes


def indicateurs_filosofi(df_filosofi):
    """Calcul de nouveaux indicateurs"""
    df_filosofi["nv"] = df_filosofi["ind_snv"] / df_filosofi["ind"]
    return df_filosofi


def definir_quantiles(df_filosofi, indicateur, nb_quantiles):
    """Calcul du quantile de la valeur d'un indicateur pour chaque carreau"""
    quantiles = np.quantile(
        a=df_filosofi["nv"],
        q=[(k + 1) / nb_quantiles for k in range(nb_quantiles - 1)],
        method="inverted_cdf",
        weights=df_filosofi["ind"],
    )
    df_filosofi[f"{indicateur}_quantile"] = np.digitize(
        df_filosofi[indicateur], quantiles
    )
    return df_filosofi


def afficher_carte_filosofi(df_filosofi, indicateur):
    "Affiche la carte pour l'indicateur demandé"
    ax = df_filosofi.plot(
        column=f"{indicateur}_quantile",
        categorical=True,
        cmap="RdYlGn",
        linewidth=0,
        legend=True,
        figsize=(10, 10),
    )
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # telecharger_zip(URL_ZIP_IDFM, "idfm")
    # telecharger_zip(URL_ZIP_FILOSOFI, "filosofi")
    df_routes = charger_lignes("data/idfm")
    # df_filosofi = charger_filosofi("data/filosofi/france_metro/carreaux_200m_met.shp")
    # df_filosofi = indicateurs_filosofi(df_filosofi)
    # df_filosofi = definir_quantiles(df_filosofi, "nv", 6)
    # afficher_carte_filosofi(df_filosofi, "nv")
