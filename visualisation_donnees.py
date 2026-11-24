import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as effets
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import LightSource, LinearSegmentedColormap
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


def ajouter_legende_montagne(ax, cible=None, **options):
    "Légende, avec une case rouge-vert pour les carreaux (sur ax, ou sur cible)"
    palette = plt.get_cmap("RdYlGn")
    poignees, etiquettes = ax.get_legend_handles_labels()
    carreaux = (Patch(color=palette(0.1)), Patch(color=palette(0.9)))
    options = {"loc": "upper right", "fontsize": 8, "frameon": False, **options}
    (cible or ax).legend(
        [carreaux, *poignees],
        ["Carreaux de 200 m traversés : écart à la moyenne Île-de-France",
         *etiquettes],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        **options,
    )
    return ax


def dessiner_plan_lignes(ax, lignes, gpd_idfm, gpd_communes):
    "Plan de Paris : réseau de métro en gris, et chaque ligne dans sa couleur"
    # lignes : dictionnaire {nom de la ligne: (gpd_ligne, couleur, ...)}
    # Fond : Paris et les communes à moins de 2,5 km des lignes (la silhouette
    # suit les lignes), Paris un peu plus foncé
    traces_lignes = gpd_idfm[gpd_idfm["route_short_name"].isin(list(lignes))]
    couloir = traces_lignes.union_all().buffer(2500)
    paris = gpd_communes[gpd_communes["nom"].str.startswith("Paris")]
    proches = gpd_communes.index[gpd_communes.intersects(couloir)]
    fond = gpd_communes.loc[proches.union(paris.index)]
    fond.plot(ax=ax, color="#f4f4f4", edgecolor="white", linewidth=0.3)
    paris.plot(ax=ax, color="#e4e4e4", edgecolor="white", linewidth=0.5)
    paris.dissolve().boundary.plot(ax=ax, color="black", linewidth=0.8)
    # Tout le réseau en gris, limité au fond
    gpd_idfm.clip(fond.union_all()).plot(ax=ax, color="#c8c8c8", linewidth=0.5)

    bornes = [fond.total_bounds]
    for nom, (gpd_ligne, couleur, *_) in lignes.items():
        traces = gpd_idfm[gpd_idfm["route_short_name"] == nom]
        traces.plot(ax=ax, color="#222222", linewidth=3.5)  # contour foncé
        traces.plot(ax=ax, color=couleur, linewidth=2.2)
        gpd_ligne.plot(
            ax=ax, color="white", edgecolor="black", markersize=5, linewidth=0.4,
            zorder=5,
        )
        # Pastille au terminus de départ (0 km) : indique aussi le sens de lecture
        depart = gpd_ligne.geometry.iloc[0]
        clair = np.dot(mcolors.to_rgb(couleur), [0.299, 0.587, 0.114]) > 0.6
        ax.text(
            depart.x, depart.y, nom, ha="center", va="center", fontsize=8,
            fontweight="bold", color="black" if clair else "white", zorder=6,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": couleur, "edgecolor": "none"},
        )
        bornes.append(traces.total_bounds)

    # Cadre : le fond et toutes les lignes, avec une marge
    bornes = np.vstack(bornes)
    xmin, ymin = bornes[:, :2].min(axis=0) - 1000
    xmax, ymax = bornes[:, 2:].max(axis=0) + 1000
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    # Pas d'axes, mais un fond blanc opaque : masque ce qu'il y a dessous
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_facecolor("white")
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
    gpd_ligne, df_nv, df_carreaux, nv_moyen, couleur, ligne, nv_communes=None,
    gpd_idfm=None, gpd_communes=None,
):
    "Profil du niveau de vie le long d'une ligne : une « montagne » de carreaux"
    df = gpd_ligne.join(df_nv["nv"], on="stop_id")
    norme = definir_norme_montagnes(df_carreaux["nv"], nv_moyen)
    if gpd_idfm is None or gpd_communes is None:
        _, ax = plt.subplots(figsize=(14, 6))
    else:
        # Petit plan de la ligne dans un panneau à droite, calé en bas
        _, (ax, ax_plan) = plt.subplots(
            1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [5, 1]}
        )
        dessiner_plan_lignes(
            ax_plan, {ligne: (gpd_ligne, couleur)}, gpd_idfm, gpd_communes
        )
        ax_plan.set_anchor("S")
    dessiner_montagne(ax, df, df_carreaux, couleur, nv_moyen, norme, nv_communes)
    ax.set_xlim(-0.3, df["position_km"].max() + 0.3)
    ax.set_ylim(bornes_verticales([df_carreaux["nv"], df["nv"]]))
    ax.set_title(
        f"Ligne {ligne} : niveau de vie le long de la ligne", loc="left", pad=70
    )
    ajouter_legende_montagne(ax)
    ax.figure.tight_layout()
    return ax


def creer_figure_empilee(lignes, tables, gpd_idfm=None, gpd_communes=None):
    "Figure à une montagne par ligne, en gardant la place du plan des lignes"
    km_max = max(df["position_km"].max() for df in tables.values())
    # Le plan va dans le vide à droite des lignes courtes ; s'il n'y en a pas
    # (lignes toutes aussi longues, comme les RER), dans une colonne à droite
    debut_plan = 0.74  # en fraction de la largeur des graphiques
    km_debut_plan = -0.3 + debut_plan * (km_max + 0.6)
    libres = [
        i for i, df in enumerate(tables.values())
        if df["position_km"].max() < km_debut_plan - 0.8  # au moins 800 m d'écart
    ]
    avec_plan = gpd_idfm is not None and gpd_communes is not None

    if avec_plan and not libres:
        fig = plt.figure(figsize=(21, 5 * len(lignes)))
        grille = fig.add_gridspec(len(lignes), 2, width_ratios=[16, 5])
        axes = [fig.add_subplot(grille[i, 0]) for i in range(len(lignes))]
        ax_plan = fig.add_subplot(grille[:, 1])
        dessiner_plan_lignes(ax_plan, lignes, gpd_idfm, gpd_communes)
    else:
        fig, axes = plt.subplots(len(lignes), 1, figsize=(16, 5 * len(lignes)))
        axes = np.atleast_1d(axes)
    # Plan à placer dans le vide une fois la mise en page faite
    plan_dans_le_vide = (debut_plan, libres) if avec_plan and libres else None
    return fig, axes, km_max, plan_dans_le_vide


def terminer_figure_empilee(
    fig, axes, lignes, titre, plan_dans_le_vide, gpd_idfm=None, gpd_communes=None
):
    "Titre, mise en page et plan commun dans le vide à droite des lignes courtes"
    fig.suptitle(titre, x=0.01, y=0.995, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # laisse la place au titre et à la légende
    if plan_dans_le_vide is not None:
        debut_plan, libres = plan_dans_le_vide
        haut = axes[libres[0]].get_position()
        bas = axes[libres[-1]].get_position()
        ax_plan = fig.add_axes(
            [haut.x0 + debut_plan * haut.width, bas.y0,
             (1 - debut_plan) * haut.width, haut.y1 - bas.y0]
        )
        dessiner_plan_lignes(ax_plan, lignes, gpd_idfm, gpd_communes)
    return axes


def lister_lignes(lignes):
    "« 1, 9, 13 et 14 » à partir des noms des lignes"
    noms = list(lignes)
    return noms[0] if len(noms) == 1 else ", ".join(noms[:-1]) + " et " + noms[-1]


def generer_montagnes_empilees(
    lignes, df_nv, nv_moyen, nv_communes=None, gpd_idfm=None, gpd_communes=None
):
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
    fig, axes, km_max, plan = creer_figure_empilee(
        lignes, tables, gpd_idfm, gpd_communes
    )
    for ax, (nom, df) in zip(axes, tables.items()):
        _, couleur, df_carreaux = lignes[nom]
        dessiner_montagne(
            ax, df, df_carreaux, couleur, nv_moyen, norme, nv_communes
        )
        ax.set_xlim(-0.3, km_max + 0.3)
        ax.set_ylim(bornes)
        ajouter_pastille_titre(ax, nom, couleur, pad=55)
    # Légende commune, sous le titre général : libère la place pour les plans
    ajouter_legende_montagne(
        axes[0], cible=fig, loc="upper left", bbox_to_anchor=(0.01, 0.985), ncol=3
    )
    return terminer_figure_empilee(
        fig, axes, lignes, f"Niveau de vie le long des lignes {lister_lignes(lignes)}",
        plan, gpd_idfm, gpd_communes,
    )


def dessiner_montagne_population(
    ax, df, couleur, population_moyenne, noms_communes=True
):
    "Montagne du nombre d'habitants rattachés à chaque station (df : stations ordonnées)"
    x, y = df["position_km"], df["population"]
    ax.fill_between(x, y, color=couleur, alpha=0.35, linewidth=0, zorder=1)
    ax.plot(
        x, y, color=couleur, linewidth=2.5, zorder=3,
        path_effects=[effets.withStroke(linewidth=4, foreground="#222222")],
        label="Habitants rattachés à la station",
    )
    ax.scatter(x, y, color="white", edgecolor="black", s=30, zorder=4)
    ax.axhline(
        population_moyenne, color="dimgray", linestyle="--", linewidth=1, zorder=2,
        label="Moyenne des stations de métro et de RER",
    )
    ax.set_xticks(x)
    taille = 8 if len(df) <= 30 else 7
    ax.set_xticklabels(df["stop_name"], rotation=60, ha="right", fontsize=taille)
    ax.set_ylabel("Habitants rattachés")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    if "commune" in df.columns:
        # Sans moyennes de commune (ce sont des niveaux de vie)
        ajouter_communes_graphique(ax, df, noms=noms_communes)
    return ax


def generer_montagnes_population(lignes, df_nv, gpd_idfm=None, gpd_communes=None):
    "Une montagne par ligne : nombre d'habitants dont c'est la station la plus proche"
    # lignes : dictionnaire {nom de la ligne: (gpd_ligne, couleur, ...)}
    tables = {
        nom: gpd_ligne.join(df_nv["population"], on="stop_id")
        for nom, (gpd_ligne, *_) in lignes.items()
    }
    # Repère : moyenne sur toutes les stations de métro et de RER
    population_moyenne = df_nv["population"].mean()
    # Une population part toujours de 0 ; même hauteur pour toutes les lignes
    haut = max(df["population"].max() for df in tables.values()) * 1.08
    fig, axes, km_max, plan = creer_figure_empilee(
        lignes, tables, gpd_idfm, gpd_communes
    )
    for ax, (nom, df) in zip(axes, tables.items()):
        couleur = lignes[nom][1]
        dessiner_montagne_population(ax, df, couleur, population_moyenne)
        ax.set_xlim(-0.3, km_max + 0.3)
        ax.set_ylim(0, haut)
        ajouter_pastille_titre(ax, nom, couleur, pad=55)
    poignees, etiquettes = axes[0].get_legend_handles_labels()
    fig.legend(
        poignees, etiquettes, loc="upper left", bbox_to_anchor=(0.01, 0.985),
        ncol=3, fontsize=8, frameon=False,
    )
    return terminer_figure_empilee(
        fig, axes, lignes,
        f"Habitants rattachés aux stations des lignes {lister_lignes(lignes)}",
        plan, gpd_idfm, gpd_communes,
    )


def generer_montagnes_par_ligne(
    lignes, df_nv, nv_moyen, nv_communes=None, gpd_idfm=None, gpd_communes=None
):
    "Une figure par ligne : niveau de vie en haut, habitants rattachés juste dessous"
    # lignes : dictionnaire {nom de la ligne: (gpd_ligne, couleur, df_carreaux)}
    tables = {
        nom: gpd_ligne.join(df_nv[["nv", "population"]], on="stop_id")
        for nom, (gpd_ligne, *_) in lignes.items()
    }
    # Mêmes échelles verticales et mêmes couleurs pour toutes les lignes :
    # les figures restent comparables entre elles
    tous_carreaux = pd.concat([carreaux for _, _, carreaux in lignes.values()])
    norme = definir_norme_montagnes(tous_carreaux["nv"], nv_moyen)
    bornes_nv = bornes_verticales(
        [tous_carreaux["nv"], *(df["nv"] for df in tables.values())]
    )
    population_moyenne = df_nv["population"].mean()
    haut_population = max(df["population"].max() for df in tables.values()) * 1.08
    avec_plan = gpd_idfm is not None and gpd_communes is not None

    figures = {}
    for nom, df in tables.items():
        gpd_ligne, couleur, df_carreaux = lignes[nom]
        fig = plt.figure(figsize=(20 if avec_plan else 16, 11))
        if avec_plan:  # colonne de droite pour le plan
            grille = fig.add_gridspec(2, 2, width_ratios=[5, 1])
        else:
            grille = fig.add_gridspec(2, 1)
        ax_nv = fig.add_subplot(grille[0, 0])
        ax_pop = fig.add_subplot(grille[1, 0], sharex=ax_nv)  # mêmes km : stations alignées

        dessiner_montagne(ax_nv, df, df_carreaux, couleur, nv_moyen, norme, nv_communes)
        ax_nv.set_ylim(bornes_nv)
        ax_nv.tick_params(labelbottom=False)  # noms des stations : en bas seulement
        ajouter_pastille_titre(ax_nv, nom, couleur, pad=55)

        dessiner_montagne_population(
            ax_pop, df, couleur, population_moyenne, noms_communes=False
        )
        ax_pop.set_ylim(0, haut_population)  # une population part toujours de 0
        ax_pop.set_xlim(-0.3, df["position_km"].max() + 0.3)
        ax_pop.legend(loc="upper right", fontsize=8, frameon=False)

        if avec_plan:
            ax_plan = fig.add_subplot(grille[:, 1])
            dessiner_plan_lignes(ax_plan, {nom: lignes[nom]}, gpd_idfm, gpd_communes)
        ajouter_legende_montagne(
            ax_nv, cible=fig, loc="upper left", bbox_to_anchor=(0.01, 0.975), ncol=3
        )
        fig.suptitle(
            f"Ligne {nom} : niveau de vie, puis habitants rattachés aux stations",
            x=0.01, y=0.995, ha="left", fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))  # place pour le titre et la légende
        figures[nom] = (ax_nv, ax_pop)
    return figures


def ajouter_communes_graphique(ax, df, nv_communes=None, noms=True):
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
    # Entrée ou sortie de Paris : trait noir plein ; autres frontières : pointillés
    dans_paris = blocs["commune"].str.startswith("Paris").values
    premiere_limite = True
    for x, avant, apres in zip(frontieres, dans_paris[:-1], dans_paris[1:]):
        if avant != apres:
            ax.axvline(
                x, color="black", linewidth=1.5, zorder=2.6,
                label="Limite de Paris" if premiere_limite else None,
            )
            premiere_limite = False
        else:
            ax.axvline(x, color="gray", linestyle=":", linewidth=1, zorder=0)
    blocs["gauche"] = [blocs["debut"].iloc[0], *frontieres]
    blocs["droite"] = [*frontieres, blocs["fin"].iloc[-1]]

    # Niveau de vie moyen de toute la commune : un trait horizontal par bloc
    if nv_communes is not None:
        blocs["nv"] = blocs["commune"].map(nv_communes)
        ax.hlines(
            blocs["nv"], blocs["gauche"], blocs["droite"], color="#140396",
            linewidth=2, zorder=2.5,
            label="Moyenne de la commune ou de l'arrondissement",
        )

    # Noms au-dessus du graphique (y en coordonnées de l'axe : 1 = haut)
    for bloc in blocs.itertuples() if noms else []:
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


GRIS_LIGNE = "#9a9a9a"  # lignes non sélectionnées sur le relief


def ordre_lignes(nom):
    "Clé de tri naturel des lignes : 1, 2, 3, 3bis, 4… 14 (et non 1, 10, 11…)"
    chiffres = "".join(c for c in nom if c.isdigit())
    return (int(chiffres) if chiffres else 99, nom)


def colorer_lignes_relief(traits, selection):
    "Lignes sélectionnées dans leur couleur, au premier plan ; les autres en gris fin"
    # traits : {nom de la ligne: (couleur, [tracés matplotlib])}
    for nom, (couleur, morceaux) in traits.items():
        choisie = nom in selection
        for morceau in morceaux:
            morceau.set_color(couleur if choisie else GRIS_LIGNE)
            morceau.set_linewidth(1.6 if choisie else 0.9)
            morceau.set_zorder(5 if choisie else 4)
            morceau.set_path_effects(
                [effets.withStroke(linewidth=2.8, foreground="#222222")] if choisie else []
            )


def ajouter_selecteur_lignes(ax, traits, selection):
    "Boutons (un par ligne) pour choisir les lignes mises en valeur sur le relief"
    import ipywidgets as widgets  # seulement dans le notebook

    selection = set(selection)
    boutons = {}

    def styler(nom):
        "Bouton plein de la couleur de la ligne si elle est choisie, gris sinon"
        couleur = traits[nom][0]
        choisie = nom in selection
        clair = np.dot(mcolors.to_rgb(couleur), [0.299, 0.587, 0.114]) > 0.6
        boutons[nom].style.button_color = couleur if choisie else "#eeeeee"
        boutons[nom].style.text_color = (
            ("black" if clair else "white") if choisie else "#888888"
        )

    def actualiser():
        for nom in boutons:
            styler(nom)
        colorer_lignes_relief(traits, selection)
        ax.figure.canvas.draw_idle()

    def basculer(bouton):
        selection.symmetric_difference_update({bouton.description})
        actualiser()

    for nom in sorted(traits, key=ordre_lignes):
        boutons[nom] = widgets.Button(
            description=nom, layout=widgets.Layout(width="46px"),
            style={"font_weight": "bold"},
        )
        boutons[nom].on_click(basculer)
        styler(nom)

    def tout(choisir):
        def action(_):
            selection.clear()
            if choisir:
                selection.update(traits)
            actualiser()
        return action

    toutes = widgets.Button(description="Toutes", layout=widgets.Layout(width="70px"))
    aucune = widgets.Button(description="Aucune", layout=widgets.Layout(width="70px"))
    toutes.on_click(tout(True))
    aucune.on_click(tout(False))
    return widgets.HBox(
        [widgets.Label("Lignes :"), *boutons.values(), toutes, aucune],
        layout=widgets.Layout(flex_flow="row wrap"),
    )


def points_caches_par_relief(ax, terrain, px, py, pz, tolerance=300):
    "Pour chaque point (x, y, z), True s'il est caché par une colline dans la vue actuelle"
    # Matplotlib n'a pas de test de profondeur en 3D : il dessine les lignes par-dessus
    # la surface, même quand une colline est devant. On avance donc de chaque point
    # vers la caméra et on regarde si le relief passe au-dessus du rayon.
    # terrain : (z, x0, y0, pas) avec z[j, i] l'altitude au point (x0 + i pas, y0 + j pas)
    z_sol, x0, y0, pas = terrain
    hauteur, largeur = z_sol.shape
    relief = np.nan_to_num(z_sol, nan=-np.inf)  # pas de relief : rien ne cache

    # Position de la caméra en coordonnées des données : le seul point que la projection
    # ne sait pas placer (noyau des lignes x, y et w de la matrice de projection)
    camera = np.linalg.svd(ax.get_proj()[[0, 1, 3]])[2][-1]
    if abs(camera[3]) > 1e-12:  # perspective : rayon vers l'œil
        dx, dy, dz = camera[:3, None] / camera[3] - np.array([px, py, pz])
    else:  # vue orthographique : même direction pour tous les points
        d = camera[:3] * (1 if camera[2] * ax.elev >= 0 else -1)
        dx, dy, dz = (np.full(len(px), v) for v in d)
    dh = np.hypot(dx, dy)
    plat = dh < 1e-9  # vue de dessus : rien n'est caché
    dh = np.where(plat, 1, dh)
    ux, uy, pente = dx / dh, dy / dh, dz / dh  # direction au sol et montée par unité

    # On s'arrête quand le rayon passe au-dessus du plus haut sommet ou sort du relief
    portee = np.where(pente > 0, (np.nanmax(z_sol) - pz) / np.where(pente > 0, pente, 1), np.inf)
    portee = np.where(plat, 0, np.minimum(portee, pas * (largeur + hauteur)))

    cache = np.zeros(len(px), dtype=bool)
    restants = np.flatnonzero(portee > 1.5 * pas)
    distance = 1.5 * pas  # on saute la case du point lui-même
    while len(restants):
        k = restants
        i = np.rint((px[k] + ux[k] * distance - x0) / pas).astype(int)
        j = np.rint((py[k] + uy[k] * distance - y0) / pas).astype(int)
        dedans = (i >= 0) & (i < largeur) & (j >= 0) & (j < hauteur)
        sol = np.full(len(k), -np.inf)
        sol[dedans] = relief[j[dedans], i[dedans]]
        cache[k] = sol > pz[k] + pente[k] * distance + tolerance
        distance += pas / 2
        restants = k[~cache[k] & (portee[k] >= distance)]
    return cache


def cacher_derriere_relief(ax, terrain, traits3d, textes):
    "Cache les morceaux de lignes et les noms qui sont derrière une colline"
    # traits3d : liste de (tracé, x, y, z) ; textes : liste de (texte, x, y, z)
    tailles = [len(px) for _, px, _, _ in traits3d] + [1] * len(textes)
    coordonnees = [np.concatenate([np.atleast_1d(t[n]) for t in traits3d + textes]) for n in (1, 2, 3)]
    cache = np.split(points_caches_par_relief(ax, terrain, *coordonnees), np.cumsum(tailles)[:-1])
    for (trait, px, py, pz), masque in zip(traits3d, cache):
        trait.set_data_3d(px, py, np.where(masque, np.nan, pz))
    for (texte, *_), masque in zip(textes, cache[len(traits3d):]):
        texte.set_visible(not masque[0])


def generer_relief_3d(grille, nv_moyen, gpd_idfm, gpd_arrondissements, selection=None):
    "Relief 3D du niveau de vie (altitude = revenus), avec les lignes de métro posées dessus"
    # grille : résultat de construire_grille_nv ; tout est en EPSG:3035
    # selection : lignes mises en valeur (None : toutes) ; les autres sont en gris
    # Renvoie ax et les tracés des lignes, pour changer la sélection ensuite
    x, y, nv = grille["x"], grille["y"], grille["nv"]
    sol = np.floor((np.nanmin(nv) - 5000) / 5000) * 5000  # socle sous la vallée la plus basse
    # Mêmes couleurs que les montagnes : rouge-vert centré sur la moyenne Île-de-France
    norme = definir_norme_montagnes(nv[np.isfinite(nv)], nv_moyen)
    nv_plein = np.nan_to_num(nv, nan=nv_moyen)
    couleurs = plt.get_cmap("RdYlGn")(norme(nv_plein))
    # Ombrage de relief doux (lumière du nord-ouest), comme une carte topographique
    couleurs[..., :3] = LightSource(azdeg=315, altdeg=40).shade_rgb(
        couleurs[..., :3], nv_plein, blend_mode="soft", vert_exag=0.5, dx=200, dy=200
    )

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    # Les cases sans valeur (NaN : hors zone, bois, parcs) ne sont pas dessinées
    ax.plot_surface(
        x / 1000, y / 1000, nv, facecolors=couleurs, rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False, zorder=1,
    )

    i0, j0 = grille["origine"]
    hauteur, largeur = nv.shape

    def altitude(px, py):
        "Niveau de vie de la case de la grille qui contient chaque point"
        i = np.clip(np.floor(px / 200).astype(int) - i0, 0, largeur - 1)
        j = np.clip(np.floor(py / 200).astype(int) - j0, 0, hauteur - 1)
        return grille["nv_lignes"][j, i]

    def points(ligne, pas=50):
        "Points tous les 50 m le long d'une ligne (ou de chacun de ses morceaux)"
        for morceau in getattr(ligne, "geoms", [ligne]):
            if morceau.length > 0:
                pts = shapely.line_interpolate_point(morceau, np.arange(0, morceau.length, pas))
                yield shapely.get_x(pts), shapely.get_y(pts)

    traits3d = []  # tous les tracés, pour cacher ce qui passe derrière le relief

    # Limite de la zone au sol : le socle du relief
    zone = gpd_arrondissements.union_all()
    for anneau in getattr(zone.boundary, "geoms", [zone.boundary]):
        if anneau.length > 2000:  # ignore les petits morceaux parasites de l'union
            for px, py in points(anneau):
                pz = np.full(len(px), sol)
                (trait,) = ax.plot(px / 1000, py / 1000, pz, color="black",
                                   linewidth=1.2, zorder=0.5)
                traits3d.append((trait, px / 1000, py / 1000, pz))

    # Lignes de métro posées sur le relief ; on garde leurs tracés pour la sélection
    metro = gpd_idfm[gpd_idfm["mode"] == "metro"].to_crs(3035)
    traits = {}
    for (nom, couleur), traces in metro.groupby(["route_short_name", "route_color"]):
        morceaux = []
        for trace in traces.geometry:
            for px, py in points(trace.intersection(zone)):
                pz = altitude(px, py) + 600
                morceaux += ax.plot(px / 1000, py / 1000, pz)
                traits3d.append((morceaux[-1], px / 1000, py / 1000, pz))
        traits[nom] = ("#" + couleur, morceaux)
    colorer_lignes_relief(traits, traits if selection is None else selection)

    # Repères : numéro de chaque arrondissement, posé au-dessus du relief
    textes = []
    for arr in gpd_arrondissements.itertuples():
        point = arr.geometry.representative_point()
        tx, ty = point.x / 1000, point.y / 1000
        tz = altitude(np.array([point.x]), np.array([point.y]))[0] + 3000
        texte = ax.text(
            tx, ty, tz, arr.nom.replace("Paris ", ""), fontsize=9, fontweight="bold",
            ha="center", color="#333333", zorder=6,
            path_effects=[effets.withStroke(linewidth=2.5, foreground="white")],
        )
        textes.append((texte, tx, ty, tz))

    # Avant chaque dessin (rotation, zoom, enregistrement), on cache les morceaux de
    # lignes et les noms qui passent derrière une colline
    terrain = (nv, (i0 + 0.5) * 0.2, (j0 + 0.5) * 0.2, 0.2)
    dessin_3d = ax.draw

    def dessiner(renderer):
        cacher_derriere_relief(ax, terrain, traits3d, textes)
        dessin_3d(renderer)

    ax.draw = dessiner

    ax.set_box_aspect((largeur, hauteur, 0.35 * max(largeur, hauteur)))
    ax.view_init(elev=40, azim=-70)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zlim(sol, None)
    ax.set_zlabel("Niveau de vie moyen (€ par an)", labelpad=12)
    ax.zaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax.set_title(
        "Le relief du niveau de vie à Paris (moyenne sur un carré de 1 km de côté centré sur chaque point)",
        loc="left",
    )
    return ax, traits


def nom_ligne(nom, mode):
    "Nom affiché d'une ligne : « M 13 » ou « RER A »"
    return f"RER {nom}" if mode == "rer" else f"M {nom}"


def couleur_texte(couleur):
    "Texte noir sur une couleur claire, blanc sur une couleur foncée"
    return "black" if np.dot(mcolors.to_rgb(couleur), [0.299, 0.587, 0.114]) > 0.6 else "white"


def generer_halteres_paris_banlieue(df_heterogeneite, nv_paris):
    "Niveau de vie le long de chaque ligne, dans Paris et en banlieue (petite couronne)"
    # Une rangée par ligne : point plein = tronçons dans Paris, point creux = en banlieue.
    # À droite : la part des écarts due à la limite de Paris et la part des habitants
    # de banlieue de la ligne qui vivent dans une commune plus riche que Paris.
    df = df_heterogeneite.dropna(subset=["nv_paris", "nv_banlieue"])
    df = df.assign(ecart=df["nv_paris"] - df["nv_banlieue"]).sort_values("ecart")
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(df) + 1.8))
    for y, (nom, ligne) in enumerate(df.iterrows()):
        ax.plot([ligne["nv_banlieue"], ligne["nv_paris"]], [y, y], color="#222222",
                linewidth=5, solid_capstyle="round", zorder=1)
        ax.plot([ligne["nv_banlieue"], ligne["nv_paris"]], [y, y], color=ligne["couleur"],
                linewidth=3.5, solid_capstyle="round", zorder=2)
        ax.scatter(ligne["nv_paris"], y, s=70, color="#222222", zorder=3)
        ax.scatter(ligne["nv_banlieue"], y, s=70, color="white", edgecolor="#222222",
                   linewidth=1.5, zorder=3)
        # Nom de la ligne dans une pastille de sa couleur
        ax.text(-0.01, y, nom_ligne(nom, ligne["mode"]), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=couleur_texte(ligne["couleur"]),
                bbox={"boxstyle": "round,pad=0.3", "facecolor": ligne["couleur"],
                      "edgecolor": "none"})
        # Chiffres à droite
        for x, valeur in [(1.03, ligne["effet_paris"]), (1.17, ligne["part_banlieue_riche"])]:
            ax.text(x, y, f"{valeur:.0%}", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9)
    for x, titre in [(1.03, "Effet\nParis"), (1.17, "Banlieue\nriche")]:
        ax.text(x, len(df) - 0.3, titre, transform=ax.get_yaxis_transform(), ha="center",
                va="bottom", fontsize=9, fontweight="bold")

    # Encadré de lecture de l'effet Paris, avec les deux cas extrêmes de la figure
    def euros(v):
        return f"{v:,.0f} €".replace(",", " ")

    def exemple(nom):
        ligne = df.loc[nom]
        return (
            f"{nom_ligne(nom, ligne['mode'])} ({ligne['effet_paris']:.0%}) : "
            f"{euros(ligne['nv_paris'])} dans Paris, {euros(ligne['nv_banlieue'])} en banlieue"
        )

    # Exemples parmi les lignes bien mesurées des deux côtés (au moins 5 tronçons)
    solides = df[df[["troncons_paris", "troncons_banlieue"]].min(axis=1) >= 5]["effet_paris"]

    ax.text(
        0.01, 0.015,
        "Comment lire l'effet Paris ?\n"
        "Part des écarts de niveau de vie le long de la ligne qui s'explique\n"
        "par le seul fait d'être dans Paris ou en banlieue.\n"
        "• Proche de 100 % : tout se joue à la limite de Paris (Paris d'un\n"
        "   côté, banlieue de l'autre, chacun assez homogène).\n"
        f"   Ex. {exemple(solides.idxmax())}\n"
        "• Proche de 0 % : sortir de Paris ne change presque rien ; les écarts\n"
        "   viennent de quartiers différents à l'intérieur de Paris ou de la banlieue.\n"
        f"   Ex. {exemple(solides.idxmin())}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8, linespacing=1.4,
        color="#333333", zorder=5,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )

    ax.axvline(nv_paris, color="#888888", linestyle="--", linewidth=1, zorder=0)
    ax.text(nv_paris, -0.9, " Moyenne de Paris", color="#666666", fontsize=8, va="center")
    ax.set_yticks([])
    ax.set_ylim(-1.2, len(df) - 0.3)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f} €".replace(",", " "))
    ax.set_xlabel("Niveau de vie moyen des habitants à moins de 400 m de la ligne (€ par an)")
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="#222222", linestyle="", markersize=8,
                       label="Dans Paris"),
            plt.Line2D([], [], marker="o", color="#222222", markerfacecolor="white",
                       linestyle="", markersize=8, label="En banlieue (petite couronne)"),
        ],
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
    )
    ax.set_title(
        "Niveau de vie le long des lignes : dans Paris et en banlieue",
        loc="left", pad=30, fontweight="bold",
    )
    fig.text(
        0.01, -0.02,
        "Banlieue riche : part des habitants de banlieue de la ligne "
        "qui vivent dans une commune plus riche que Paris en moyenne.\n"
        "Lignes 2, 6, 3B et 7B absentes : elles restent dans Paris.",
        fontsize=8, color="#555555", va="top",
    )
    return ax


def generer_carte_communes_riches(gpd_communes, gpd_zones, nv_paris, gpd_idfm):
    "Communes de petite couronne plus riches que Paris, et lignes de métro et RER"
    sommes = gpd_zones.groupby("nom")[["ind_snv", "ind"]].sum()
    nv_communes = sommes["ind_snv"] / sommes["ind"]
    communes = gpd_communes[
        gpd_communes["code_commune"].isin(gpd_zones["code_commune"].unique())
    ].copy()
    communes["nv"] = communes["nom"].map(nv_communes)
    paris = communes["code_commune"].str.startswith("75")
    riches = communes[~paris & (communes["nv"] > nv_paris)]

    fig, ax = plt.subplots(figsize=(11, 10))
    communes[~paris].plot(ax=ax, color="#ececec", edgecolor="white", linewidth=0.5)
    communes[paris].plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=0.5)
    norme = mcolors.Normalize(vmin=nv_paris, vmax=communes["nv"].max())
    # Vert dès le seuil : la commune à peine plus riche que Paris se distingue du gris
    verts = mcolors.ListedColormap(plt.cm.Greens(np.linspace(0.3, 1, 256)))
    riches.plot(ax=ax, column="nv", cmap=verts, norm=norme, edgecolor="white", linewidth=0.5)
    communes[paris].dissolve().boundary.plot(ax=ax, color="black", linewidth=1)

    # Lignes coupées à la petite couronne, dans leur couleur avec un contour foncé
    zone = communes.union_all()
    lignes = gpd_idfm.clip(zone)
    epaisseur = np.where(lignes["mode"] == "rer", 1.6, 0.9)
    lignes.plot(ax=ax, color="#222222", linewidth=epaisseur + 0.8)
    lignes.plot(ax=ax, color="#" + lignes["route_color"], linewidth=epaisseur)

    for commune in riches.itertuples():
        point = commune.geometry.representative_point()
        ax.text(point.x, point.y, commune.nom.replace("-", "-\n", 1) if len(commune.nom) > 14
                else commune.nom, fontsize=7, ha="center", va="center", color="#1a1a1a",
                path_effects=[effets.withStroke(linewidth=2, foreground="white")], zorder=6)

    barre = fig.colorbar(plt.cm.ScalarMappable(norm=norme, cmap=verts), ax=ax,
                         shrink=0.5, pad=0.01)
    barre.set_label("Niveau de vie moyen de la commune (€ par an)")
    barre.ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax.legend(
        handles=[
            Patch(facecolor=verts(0.4), label=f"Plus riche que Paris "
                  f"(> {nv_paris:,.0f} €)".replace(",", " ")),
            Patch(facecolor="#ececec", label="Autre commune de petite couronne"),
            Patch(facecolor="#d9d9d9", edgecolor="black", label="Paris"),
        ],
        loc="lower left", frameon=False, fontsize=9,
    )
    ax.set_axis_off()
    ax.set_title("Les communes de petite couronne plus riches que Paris, et les lignes qui y vont",
                 loc="left", fontweight="bold")
    return ax


def generer_graphique_ecart_distance(df_distance, bornes=(0, 100, 200, 300, 400, 600, 1000, 5000)):
    "Écart moyen à la moyenne Île-de-France selon la distance à la station la plus proche"
    # Un panneau par mode ; carreaux au-dessus de la moyenne (vert) et en dessous (rouge)
    # séparés ; barres d'erreur : intervalle de confiance à 95 % de la moyenne
    df = df_distance.assign(tranche=pd.cut(df_distance["distance"], list(bornes)))
    etiquettes = [f"{int(a)}–{int(b)}" if b < bornes[-1] else f"> {int(a)}"
                  for a, b in zip(bornes[:-1], bornes[1:])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (mode, titre) in zip(axes, [("metro", "Métro"), ("rer", "RER")]):
        for sous, couleur, nom in [(False, "#1a9850", "Carreaux au-dessus de la moyenne"),
                                   (True, "#d73027", "Carreaux en dessous de la moyenne")]:
            groupe = df[(df["mode"] == mode) & (df["sous_moyenne"] == sous)]
            stats = groupe.groupby("tranche", observed=False)["ecart_abs"].agg(["mean", "std", "size"])
            stats = stats.reset_index(drop=True)
            stats = stats[stats["size"] >= 20]  # tranches trop petites : moyenne fragile
            marge = 1.96 * stats["std"] / np.sqrt(stats["size"])
            ax.errorbar(stats.index, stats["mean"], yerr=marge, color=couleur, marker="o",
                        capsize=3, linewidth=1.5, label=nom)
        ax.set_xticks(range(len(etiquettes)), etiquettes, rotation=30)
        ax.set_xlabel("Distance à la station la plus proche, le long de la ligne (m)")
        ax.set_title(titre, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#eeeeee")
        for cote in ("top", "right"):
            ax.spines[cote].set_visible(False)
    axes[0].set_ylabel("Écart moyen à la moyenne Île-de-France (€ par an)")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:,.0f} €".replace(",", " "))
    axes[0].set_ylim(0, None)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Les carreaux proches des stations s'écartent-ils plus de la moyenne ?",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return axes
