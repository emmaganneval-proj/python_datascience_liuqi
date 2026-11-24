import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as effets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

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
    ax.secondary_xaxis("top").set_xlabel("Distance le long du tracé (km)")
    ax.set_title(f"Ligne {nom_ligne}")
    return ax


def enregistrer_graphique(ax, nom_fichier):
    "Enregistre la figure dans le dossier sorties/"
    os.makedirs("sorties", exist_ok=True)
    ax.figure.savefig(f"sorties/{nom_fichier}.png", dpi=150, bbox_inches="tight")


def definir_norme_montagnes(valeurs_nv, nv_moyen):
    "Échelle de couleur rouge-vert centrée sur la moyenne Île-de-France"
    # Même écart de part et d'autre de la moyenne ; les 2 % extrêmes saturent
    ecart = max(
        nv_moyen - np.percentile(valeurs_nv, 2), np.percentile(valeurs_nv, 98) - nv_moyen
    )
    return mcolors.TwoSlopeNorm(
        vcenter=nv_moyen, vmin=nv_moyen - ecart, vmax=nv_moyen + ecart
    )


def bornes_verticales(series_nv):
    "Axe vertical réduit à la plage utile : du niveau de vie le plus bas au plus haut"
    bas = min(serie.min() for serie in series_nv)
    haut = max(serie.max() for serie in series_nv)
    marge = 0.05 * (haut - bas)
    return bas - marge, haut + marge


def dessiner_montagne(ax, df, df_carreaux, couleur, nv_moyen, norme, nv_communes=None):
    "Dessine la montagne d'une ligne sur ax : carreaux traversés, stations, moyennes"
    palette = plt.get_cmap("RdYlGn")  # même palette que la carte Filosofi
    # Une barre par carreau traversé, sur la longueur de tracé qui le traverse
    # Les barres partent de la moyenne Île-de-France : vers le haut si le carreau
    # est plus riche, vers le bas s'il est plus pauvre
    ax.bar(
        df_carreaux["debut"],
        df_carreaux["nv"] - nv_moyen,
        bottom=nv_moyen,
        width=df_carreaux["fin"] - df_carreaux["debut"],
        align="edge",
        color=palette(norme(df_carreaux["nv"])),
        linewidth=0,
        zorder=1,
    )
    # Stations : niveau de vie des habitants dont c'est la station la plus proche
    # Couleur de la ligne, avec un contour foncé pour ressortir sur les barres
    ax.plot(
        df["position_km"], df["nv"], color=couleur, linewidth=2.5, zorder=3,
        path_effects=[effets.withStroke(linewidth=4, foreground="#222222")],
        label="Autour de la station",
    )
    ax.scatter(
        df["position_km"], df["nv"], color="white", edgecolor="black", s=30, zorder=4
    )
    ax.axhline(
        nv_moyen, color="dimgray", linestyle="--", linewidth=1, zorder=2,
        label="Moyenne Île-de-France",
    )
    ax.set_xticks(df["position_km"])
    # Police plus petite pour les lignes avec beaucoup de stations
    taille = 8 if len(df) <= 30 else 7
    ax.set_xticklabels(df["stop_name"], rotation=60, ha="right", fontsize=taille)
    ax.set_ylabel("Niveau de vie moyen (€ par an)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    if "commune" in df.columns:
        ajouter_communes_graphique(ax, df, nv_communes)
    return ax


def ajouter_legende_montagne(ax):
    "Légende, avec une case rouge-vert pour les carreaux"
    palette = plt.get_cmap("RdYlGn")
    poignees, etiquettes = ax.get_legend_handles_labels()
    carreaux = (Patch(color=palette(0.1)), Patch(color=palette(0.9)))
    ax.legend(
        [carreaux, *poignees],
        ["Carreaux de 200 m traversés : écart à la moyenne Île-de-France",
         *etiquettes],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="upper right",
        fontsize=8,
        frameon=False,
    )
    return ax


def ajouter_pastille_titre(ax, ligne, couleur, pad=70):
    "Titre avec le numéro de la ligne dans une pastille de sa couleur"
    # Texte noir sur couleur claire, blanc sur couleur foncée
    clair = np.dot(mcolors.to_rgb(couleur), [0.299, 0.587, 0.114]) > 0.6
    ax.set_title(
        f" {ligne} ",
        loc="left",
        pad=pad,
        fontweight="bold",
        color="black" if clair else "white",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": couleur, "edgecolor": "none"},
    )
    return ax


def generer_graphique_montagne(
    gpd_ligne, df_nv, df_carreaux, nv_moyen, couleur, ligne, nv_communes=None
):
    "Profil du niveau de vie le long d'une ligne : une « montagne » de carreaux"
    df = gpd_ligne.join(df_nv["nv"], on="stop_id")
    norme = definir_norme_montagnes(df_carreaux["nv"], nv_moyen)
    _, ax = plt.subplots(figsize=(14, 6))
    dessiner_montagne(ax, df, df_carreaux, couleur, nv_moyen, norme, nv_communes)
    ax.set_xlim(-0.3, df["position_km"].max() + 0.3)
    ax.set_ylim(bornes_verticales([df_carreaux["nv"], df["nv"]]))
    ax.set_title(
        f"Ligne {ligne} : niveau de vie le long de la ligne", loc="left", pad=70
    )
    ajouter_legende_montagne(ax)
    ax.figure.tight_layout()
    return ax


def generer_montagnes_empilees(lignes, df_nv, nv_moyen, nv_communes=None):
    "Une montagne par ligne, les unes sous les autres, à la même échelle"
    # lignes : dictionnaire {nom de la ligne: (gpd_ligne, couleur, df_carreaux)}
    tables = {
        nom: gpd_ligne.join(df_nv["nv"], on="stop_id")
        for nom, (gpd_ligne, _, _) in lignes.items()
    }
    # Mêmes bornes pour toutes les lignes : hauteurs, couleurs et km comparables
    tous_carreaux = pd.concat([carreaux for _, _, carreaux in lignes.values()])
    norme = definir_norme_montagnes(tous_carreaux["nv"], nv_moyen)
    bornes = bornes_verticales(
        [tous_carreaux["nv"], *(df["nv"] for df in tables.values())]
    )
    km_max = max(df["position_km"].max() for df in tables.values())

    fig, axes = plt.subplots(len(lignes), 1, figsize=(16, 5 * len(lignes)))
    for ax, (nom, df) in zip(axes, tables.items()):
        _, couleur, df_carreaux = lignes[nom]
        dessiner_montagne(
            ax, df, df_carreaux, couleur, nv_moyen, norme, nv_communes
        )
        ax.set_xlim(-0.3, km_max + 0.3)
        ax.set_ylim(bornes)
        ajouter_pastille_titre(ax, nom, couleur, pad=55)
    ajouter_legende_montagne(axes[0])
    fig.suptitle(
        f"Niveau de vie le long de {len(lignes)} lignes de métro",
        x=0.01, ha="left", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # laisse la place au titre général
    return axes


def ajouter_communes_graphique(ax, df, nv_communes=None):
    "Sépare les communes (ou arrondissements) traversées, avec leur nom et leur moyenne"
    # Un bloc = une suite de stations consécutives dans la même commune
    df = df.assign(bloc=(df["commune"] != df["commune"].shift()).cumsum())
    blocs = df.groupby("bloc").agg(
        commune=("commune", "first"),
        debut=("position_km", "min"),
        fin=("position_km", "max"),
    )
    # Frontière à mi-chemin entre la dernière station d'un bloc et la suivante
    frontieres = (blocs["fin"].iloc[:-1].values + blocs["debut"].iloc[1:].values) / 2
    for x in frontieres:
        ax.axvline(x, color="gray", linestyle=":", linewidth=1, zorder=0)
    blocs["gauche"] = [blocs["debut"].iloc[0], *frontieres]
    blocs["droite"] = [*frontieres, blocs["fin"].iloc[-1]]

    # Niveau de vie moyen de toute la commune : un trait horizontal par bloc
    if nv_communes is not None:
        blocs["nv"] = blocs["commune"].map(nv_communes)
        ax.hlines(
            blocs["nv"], blocs["gauche"], blocs["droite"], color="#03A0D4",
            linewidth=2, zorder=2.5,
            label="Moyenne de la commune ou de l'arrondissement",
        )

    # Noms au-dessus du graphique (y en coordonnées de l'axe : 1 = haut)
    for bloc in blocs.itertuples():
        ax.text(
            (bloc.debut + bloc.fin) / 2,
            1.02,
            bloc.commune,
            transform=ax.get_xaxis_transform(),
            rotation=40,
            rotation_mode="anchor",  # le texte part du centre du bloc
            ha="left",
            va="center",
            fontsize=8,
            color="dimgray",
        )
    return ax
