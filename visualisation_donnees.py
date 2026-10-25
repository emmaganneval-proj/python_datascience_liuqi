import os
import urllib
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

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


def charger_routes(path):
    """Chargement des lignes IDFM (métro et RER)"""
    df_routes = pd.read_csv(
        f"{path}/routes.txt",
        usecols=[
            "route_id",
            "route_type",
            "route_short_name",
            "route_color",
            "route_text_color",
            "agency_id",
        ],
    )

    filtre_metro = df_routes["route_type"] == 1
    filtre_rer = (df_routes["route_type"] == 2) & (df_routes["agency_id"] == "IDFM:71")
    df_routes = df_routes.loc[filtre_metro | filtre_rer]
    df_routes["mode"] = np.where(df_routes["route_type"] == 1, "metro", "rer")
    return df_routes


def charger_trips(path, df_routes):
    "Chargement des trains"
    df_trips = pd.read_csv(
        f"{path}/trips.txt", usecols=["route_id", "shape_id", "trip_id"]
    )
    df_trips = df_trips.merge(df_routes, on="route_id", how="inner")
    return df_trips


def charger_parcours(path, df_trips):
    """Chargement du tracé des parcours géographiques des lignes IDFM"""

    df_shapes = pd.read_csv(
        f"{path}/shapes.txt",
        usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    ).drop_duplicates()
    df_shapes = df_shapes.merge(
        df_trips.drop_duplicates("shape_id").drop(columns=["trip_id"]), on="shape_id"
    )

    df_shapes = df_shapes.sort_values(["shape_id", "shape_pt_sequence"])

    return df_shapes


def indicateurs_filosofi(df_filosofi):
    """Calcul de nouveaux indicateurs"""
    df_filosofi["nv"] = df_filosofi["ind_snv"] / df_filosofi["ind"]
    return df_filosofi


def definir_quantiles(df_filosofi, indicateur, nb_quantiles):
    """Calcul du quantile de la valeur d'un indicateur pour chaque carreau"""
    quantiles = np.quantile(
        a=df_filosofi[indicateur],
        q=[(k + 1) / nb_quantiles for k in range(nb_quantiles - 1)],
        method="inverted_cdf",
        weights=df_filosofi["ind"],
    )
    df_filosofi[f"{indicateur}_quantile"] = np.digitize(
        df_filosofi[indicateur], quantiles
    )
    return df_filosofi


def generer_carte_filosofi(df_filosofi, indicateur):
    "Génère la carte pour l'indicateur demandé"
    ax = df_filosofi.plot(
        column=f"{indicateur}_quantile",
        categorical=True,
        cmap="RdYlGn",
        linewidth=0,
        legend=True,
        figsize=(10, 10),
    )
    ax.set_axis_off()
    return ax


def construire_lignes_idfm(df_parcours):
    "Construit le tracé des lignes de métro / RER"
    df_shapes = df_parcours.groupby("shape_id").apply(
        lambda group: LineString(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
    )
    gpd_idfm = gpd.GeoDataFrame(
        df_parcours.drop_duplicates("shape_id")
        .drop(columns=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
        .set_index("shape_id"),
        geometry=df_shapes,
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    return gpd_idfm


def ajouter_lignes_graphique(ax, gpd_idfm):
    "Superposer les lignes IDFM à la carte"
    gpd_idfm.plot(
        color="#" + gpd_idfm["route_color"],
        linewidth=np.where(gpd_idfm["mode"] == "metro", 1, 2),
        ax=ax,
    )
    return ax


def charger_arrets(path, df_trips):
    "Chargement des arrêts des trains"
    df_arrets = pd.read_csv(f"{path}/stop_times.txt", usecols=["trip_id", "stop_id"])
    df_arrets = (
        df_arrets.merge(
            df_trips,
            on="trip_id",
        )
        .drop(columns=["trip_id", "shape_id"])
        .drop_duplicates(["route_id", "stop_id"])
    )
    return df_arrets


def charger_stations(path, df_arrets):
    "Chargement de la position des stations"
    df_stations = pd.read_csv(
        f"{path}/stops.txt",
        usecols=["stop_id", "stop_name", "stop_lon", "stop_lat", "parent_station"],
    )

    df_parents = df_stations[["stop_id", "parent_station"]].merge(
        df_arrets, on="stop_id"
    )

    df_stations = (
        df_stations.drop(columns=["parent_station"])
        .merge(
            df_parents.drop(columns=["stop_id"]),
            left_on="stop_id",
            right_on="parent_station",
        )
        .drop_duplicates(["route_id", "stop_id"])
    )
    return df_stations


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # telecharger_zip(URL_ZIP_IDFM, "idfm")
    # telecharger_zip(URL_ZIP_FILOSOFI, "filosofi")

    # df_filosofi = charger_filosofi("data/filosofi/france_metro/carreaux_200m_met.shp")
    # df_filosofi = indicateurs_filosofi(df_filosofi)
    # df_filosofi = definir_quantiles(df_filosofi, "nv", 6)

    df_routes = charger_routes("data/idfm")
    df_trips = charger_trips("data/idfm", df_routes)
    df_parcours = charger_parcours("data/idfm", df_trips)
    gpd_idfm = construire_lignes_idfm(df_parcours)

    df_arrets = charger_arrets("data/idfm", df_trips)
    df_stations = charger_stations("data/idfm", df_arrets)
    # ax = generer_carte_filosofi(df_filosofi, "nv")
    # ax = ajouter_lignes_graphique(ax, gpd_idfm)
    # plt.show()
