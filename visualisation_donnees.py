import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

XLIM = (630_000, 680_000)
YLIM = (6_830_000, 6_880_000)


def generer_carte_filosofi(df_filosofi, indicateur):
    "Génère la carte pour l'indicateur demandé"
    ax = df_filosofi.plot(
        column=f"{indicateur}_quantile",
        categorical=True,
        cmap="RdYlGn",
        linewidth=0,
        legend=True,
        # legend_kwds=f"{indicateur}_nom_quantile",
        figsize=(10, 10),
    )
    ax.set_axis_off()
    return ax


def ajouter_lignes_graphique(ax, gpd_idfm):
    "Superposer les lignes IDFM à la carte"
    gpd_idfm.plot(
        color="#" + gpd_idfm["route_color"],
        linewidth=np.where(gpd_idfm["mode"] == "metro", 1, 2),
        ax=ax,
        zorder=4,
    )
    gpd_idfm.plot(
        color="white",
        linewidth=np.where(gpd_idfm["mode"] == "metro", 2, 3),
        ax=ax,
        zorder=3,
    )
    return ax


def ajouter_stations_graphique(df_stations, ax):
    """Ajoute les stations sur le graphique"""
    gpd_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["stop_lon"], df_stations["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    gpd_stations.plot(
        color="white",
        edgecolor="black",
        linewidth=0.5,
        markersize=np.where(gpd_stations["mode"] == "metro", 5, 10),
        ax=ax,
        zorder=5,
    )
    return ax


def generer_graphique_tranches(df_tranches):
    "Niveau de vie moyen selon le temps de trajet vers Châtelet ou La Défense"
    _, ax = plt.subplots(figsize=(10, 5))
    df_tranches["nv"].plot.bar(
        color=np.where(df_tranches.index == "Non desservi", "grey", "steelblue"),
        width=0.7,
        rot=0,
        ax=ax,
    )
    ax.set_xlabel("Temps de trajet vers Châtelet ou La Défense (marche + métro/RER)")
    ax.set_ylabel("Niveau de vie moyen (€ par an)")
    return ax


def generer_carte_temps(df_filosofi, gpd_carreaux):
    "Carte du temps de trajet total depuis chaque carreau"
    # gpd_carreaux a gardé l'index de df_filosofi : on y récupère les polygones
    gpd_temps = df_filosofi[["geometry"]].join(gpd_carreaux[["temps_total"]])
    ax = gpd_temps.plot(
        column="temps_total",
        cmap="Blues",
        vmax=60,
        linewidth=0,
        legend=True,
        legend_kwds={"label": "Temps de trajet total (min)", "shrink": 0.6},
        missing_kwds={"color": "lightgrey"},
        figsize=(10, 10),
    )
    ax.set_axis_off()
    return ax
