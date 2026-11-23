import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

XLIM = (628_000, 680_000)
YLIM = (6_838_000, 6_885_000)


def generer_carte_filosofi(df_filosofi, indicateur):
    "Génère la carte pour l'indicateur demandé"
    ax = df_filosofi.plot(
        column=f"{indicateur}_quantile",
        categorical=True,
        cmap="RdYlGn",
        linewidth=0,
        legend=True,
        legend_kwds={"loc": "lower left", "bbox_to_anchor": (1, 0), "frameon": False},
        figsize=(12, 10),
    )
    ax.set(facecolor="gainsboro", xticks=[], yticks=[])
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


def ajouter_pastilles_graphique(ax, gpd_terminus):
    "Ajoute le nom des lignes à leurs terminus"
    # Pastilles à moins de 1,6 km l'une de l'autre : alignées autour de leur centre
    groupes = gpd.GeoSeries(gpd_terminus.buffer(800).union_all()).explode()
    gpd_groupes = gpd.GeoDataFrame(
        geometry=groupes.reset_index(drop=True), crs=gpd_terminus.crs
    )
    gpd_terminus = gpd.sjoin(gpd_terminus, gpd_groupes, predicate="within")
    centres = gpd_groupes.centroid.loc[gpd_terminus["index_right"]]
    rangs = gpd_terminus.groupby("index_right").cumcount()
    tailles = gpd_terminus.groupby("index_right")["stop_id"].transform("size")
    gpd_terminus["x"] = centres.x.values + 1400 * (rangs - (tailles - 1) / 2)
    gpd_terminus["y"] = centres.y.values
    for ligne in gpd_terminus.itertuples():
        forme = "round,pad=0.3" if ligne.mode == "rer" else "circle"
        fond = {"boxstyle": forme, "facecolor": "#" + ligne.route_color}
        texte = {"color": "#" + ligne.route_text_color, "ha": "center", "va": "center"}
        ax.text(ligne.x, ligne.y, ligne.route_short_name, bbox=fond, zorder=6, **texte)
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


def generer_nuage_distance_temps(gpd_carreaux):
    "Temps de trajet selon la distance à vol d'oiseau, coloré par le niveau de vie"
    ax = gpd_carreaux.drop(columns="geometry").plot.scatter(
        x="distance_destination",
        y="temps_total",
        c="nv",
        cmap="RdYlGn",
        vmin=15_000,
        vmax=50_000,
        s=1,
        figsize=(10, 6),
    )
    ax.set_xlabel("Distance à vol d'oiseau à Châtelet ou La Défense (km)")
    ax.set_ylabel("Temps de trajet total (min)")
    ax.figure.axes[-1].set_ylabel("Niveau de vie (€ par an)")
    return ax


def generer_graphique_connexion(df_comparaison):
    "Niveau de vie selon la qualité de la connexion, à distance égale"
    ax = df_comparaison.plot.bar(
        color=["seagreen", "orange", "grey"], rot=0, figsize=(10, 5)
    )
    ax.set_xlabel("Distance à vol d'oiseau à Châtelet ou La Défense")
    ax.set_ylabel("Niveau de vie moyen (€ par an)")
    return ax


def generer_profil_ligne(df_profils, gpd_positions, nom_ligne, nv_idf):
    "Niveau de vie par tronçon le long de la ligne, avec le nom des stations"
    df_ligne = df_profils.loc[df_profils["route_short_name"] == nom_ligne]
    gpd_arrets = gpd_positions.loc[gpd_positions["route_short_name"] == nom_ligne]
    couleur = "#" + gpd_arrets["route_color"].iloc[0]
    ax = df_ligne.plot(
        x="position_km",
        y="nv",
        color=couleur,
        label="Le long du tracé",
        figsize=(12, 5),
    )
    # Dégradé : image verticale découpée selon la surface sous la courbe
    surface = ax.fill_between(df_ligne["position_km"], df_ligne["nv"], color="none")
    degrade = ax.imshow(
        np.linspace(1, 0, 100).reshape(-1, 1),
        cmap=LinearSegmentedColormap.from_list("degrade", ["white", couleur]),
        aspect="auto",
        extent=(0, df_ligne["position_km"].max(), 0, df_ligne["nv"].max()),
    )
    degrade.set_clip_path(surface.get_paths()[0], transform=ax.transData)
    ax.axhline(nv_idf, color="grey", linestyle="--", label="Moyenne Île-de-France")
    ax.set_ylim(0, df_ligne["nv"].max() * 1.05)
    ax.legend()
    ax.set_xticks(gpd_arrets["position_km"], gpd_arrets["stop_name"], rotation=90)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("Niveau de vie moyen le long du tracé (€ par an)")
    ax.set_title(f"Ligne {nom_ligne}")
    return ax


def enregistrer_graphique(ax, nom_fichier):
    "Enregistre la figure dans le dossier sorties/"
    os.makedirs("sorties", exist_ok=True)
    ax.figure.savefig(f"sorties/{nom_fichier}.png", dpi=150, bbox_inches="tight")
