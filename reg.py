"""Régressions : temps de trajet, distance à Châtelet et niveau de vie"""
import matplotlib.pyplot as plt
import geopandas as gpd   
import numpy as np
import statsmodels.formula.api as smf

from analyse_donnees import calculer_temps_trajet, joindre_carreaux_stations
from pretraitement_donnees import construire_centres_carreaux


def preparer_stations_chatelet(gpd_stations, graphe_reseau):
    "Temps de trajet et distance à vol d'oiseau de chaque station jusqu'à Châtelet"
    destinations = gpd_stations.loc[
        gpd_stations["stop_name"].isin(["Châtelet - Les Halles", "Châtelet"]), "stop_id"
    ]
    df_temps = calculer_temps_trajet(graphe_reseau, destinations)

    chatelet = gpd_stations.loc[
        gpd_stations["stop_name"] == "Châtelet - Les Halles", "geometry"
    ].iloc[0]
    df_reg = gpd_stations.merge(df_temps, on="stop_id")
    df_reg["distance_km"] = df_reg.distance(chatelet) / 1000
    df_reg = df_reg[df_reg["temps_trajet"] > 0]  # on retire Châtelet lui-même
    return df_reg, df_temps, chatelet


def trouver_seuil(x, y):
    "Seuil (en km) qui sépare le mieux deux régimes linéaires"

    def erreur_totale(seuil):
        erreur = 0
        for masque in [x < seuil, x >= seuil]:
            p, c = np.polyfit(x[masque], y[masque], 1)
            erreur += ((y[masque] - (c + p * x[masque])) ** 2).sum()
        return erreur

    seuils = np.arange(5, 40.5, 0.5)
    return seuils[np.argmin([erreur_totale(s) for s in seuils])]


def graphique_temps_distance(df_reg):
    "Temps de trajet selon la distance : régression simple et en deux morceaux"
    x = df_reg["distance_km"].values
    y = df_reg["temps_trajet"].values

    pente, constante = np.polyfit(x, y, 1)
    seuil = trouver_seuil(x, y)
    pres, loin = x < seuil, x >= seuil
    pente1, const1 = np.polyfit(x[pres], y[pres], 1)
    pente2, const2 = np.polyfit(x[loin], y[loin], 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Gauche : une seule droite
    ax1.scatter(x, y, s=12, alpha=0.5, color="steelblue")
    x_ligne = np.linspace(0, x.max(), 100)
    ax1.plot(x_ligne, constante + pente * x_ligne, color="red", linewidth=2,
             label=f"{pente:.2f} min/km (≈ {60 / pente:.0f} km/h)")
    ax1.set_title("Régression simple")

    # Droite : deux régimes
    ax2.scatter(x[pres], y[pres], s=12, alpha=0.5, color="steelblue")
    ax2.scatter(x[loin], y[loin], s=12, alpha=0.5, color="darkorange")
    x1, x2 = np.linspace(0, seuil, 50), np.linspace(seuil, x.max(), 50)
    ax2.plot(x1, const1 + pente1 * x1, color="navy", linewidth=2,
             label=f"Près du centre : {pente1:.2f} min/km (≈ {60 / pente1:.0f} km/h)")
    ax2.plot(x2, const2 + pente2 * x2, color="red", linewidth=2,
             label=f"Loin du centre : {pente2:.2f} min/km (≈ {60 / pente2:.0f} km/h)")
    ax2.axvline(seuil, color="gray", linestyle="--", linewidth=1)
    ax2.text(seuil + 1, y.max() * 0.95, f"seuil : {seuil:.1f} km", color="gray")
    ax2.set_title("Régression en deux morceaux")

    for ax in (ax1, ax2):
        ax.set_xlabel("Distance à vol d'oiseau jusqu'à Châtelet (km)")
        ax.legend(loc="upper left")
    ax1.set_ylabel("Temps de trajet en transports en commun (min)")
    fig.suptitle("Temps de trajet vers Châtelet selon la distance")
    plt.show()
    return seuil


def preparer_carreaux(df_filosofi, gpd_stations, df_temps, chatelet):
    "Distance et temps de trajet (marche + transports) vers Châtelet pour chaque carreau"
    centres = construire_centres_carreaux(df_filosofi)
    carreaux = joindre_carreaux_stations(
        centres, gpd_stations, df_temps, distance_max=1500, vitesse_marche=75
    )
    carreaux["distance_km"] = carreaux.distance(chatelet) / 1000
    return carreaux


def graphique_niveau_vie(carreaux):
    "Niveau de vie selon la distance et selon le temps (carreaux desservis, pondérés)"
    t = carreaux.dropna(subset=["temps_total", "distance_km", "nv"])
    m_dist = smf.wls("nv ~ distance_km", data=t, weights=t["ind"]).fit()
    m_temps = smf.wls("nv ~ temps_total", data=t, weights=t["ind"]).fit()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Gauche : niveau de vie selon la distance
    ax1.scatter(t["distance_km"], t["nv"], s=2, alpha=0.1, color="steelblue")
    x_ligne = np.linspace(0, t["distance_km"].max(), 100)
    ax1.plot(x_ligne, m_dist.params["Intercept"] + m_dist.params["distance_km"] * x_ligne,
             color="red", linewidth=2, label=f"{m_dist.params['distance_km']:+.0f} € par km")
    ax1.set_xlabel("Distance à vol d'oiseau jusqu'à Châtelet (km)")
    ax1.set_title("Niveau de vie selon la distance")

    # Droite : niveau de vie selon le temps
    ax2.scatter(t["temps_total"], t["nv"], s=2, alpha=0.1, color="darkorange")
    x_ligne = np.linspace(0, t["temps_total"].max(), 100)
    ax2.plot(x_ligne, m_temps.params["Intercept"] + m_temps.params["temps_total"] * x_ligne,
             color="red", linewidth=2, label=f"{m_temps.params['temps_total']:+.0f} € par minute")
    ax2.set_xlabel("Temps de trajet vers Châtelet (min, marche + transports)")
    ax2.set_title("Niveau de vie selon le temps de trajet")

    for ax in (ax1, ax2):
        ax.legend(loc="upper right")
    ax1.set_ylabel("Niveau de vie moyen du carreau (€ par an)")
    ax1.set_ylim(0, 80_000)
    fig.suptitle("Niveau de vie et éloignement de Châtelet (carreaux desservis, pondérés par la population)")
    plt.show()

    # Tableau des résultats (constante, pente, écart-type, p-value, R²)
    for nom, m in [("Niveau de vie ~ distance", m_dist), ("Niveau de vie ~ temps", m_temps)]:
        print(nom)
        print(m.summary2().tables[1][["Coef.", "Std.Err.", "P>|t|"]].round(3))
        print(f"R² = {m.rsquared:.3f}\n")
    return m_dist, m_temps

    # ===================== CARTES =====================

# Épaisseur du trait selon le type de ligne
EPAISSEUR = {"metro": 1, "tram": 0.8, "transilien": 1.5, "rer": 2.5}


def part_par_zone(df_mob, indicateur):
    """Part (en %) d'un indicateur parmi les actifs de chaque commune
    (ou de chaque arrondissement pour Paris).
    indicateur : une CSP ("Ouvriers", "Cadres"...) ou "Transports en commun"."""
    if indicateur == "Transports en commun":
        cible = df_mob["tc"]
    else:
        cible = df_mob["CSP"] == indicateur

    # Part pondérée : poids des personnes concernées / poids de tous les actifs
    somme_cible = (df_mob["IPONDI"] * cible).groupby(df_mob["code_geo"]).sum()
    somme_total = df_mob.groupby("code_geo")["IPONDI"].sum()
    return (somme_cible / somme_total * 100).round(1).rename("part")


def carte(indicateur, df_mob, communes, arr, gpd_idfm, gpd_stations_lignes,
          zone="paris", ligne=None, reseau=False, cmap="Reds"):
    """Carte colorée selon un indicateur (en % des actifs).
    zone   : "paris" (20 arrondissements) ou "idf" (toutes les communes)
    ligne  : numéro d'une ligne à superposer, par exemple "B" (facultatif)
    reseau : True pour tout le réseau, ou une liste de types, ex. ["rer", "transilien"]"""

    # 1. Le fond de carte selon la zone choisie
    if zone == "paris":
        fond = arr
    elif zone == "idf":
        fond = communes
    else:
        raise ValueError('zone doit valoir "paris" ou "idf"')

    part = part_par_zone(df_mob, indicateur)
    fond = fond.merge(part, left_on="INSEE_COM", right_index=True, how="left")

    ax = fond.plot(
        column="part", cmap=cmap, legend=True,
        edgecolor="white",
        linewidth=0.8 if zone == "paris" else 0.1,
        figsize=(12, 10),
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": f"{indicateur} parmi les actifs (%)", "shrink": 0.6},
    )

    # 2a. Paris : nom et valeur de chaque arrondissement
    if zone == "paris":
        for _, r in fond.iterrows():
            p = r.geometry.representative_point()
            ax.annotate(f"{r['nom']}\n{r['part']} %", (p.x, p.y),
                        ha="center", va="center", fontsize=7)

    # 2b. IDF : contours des départements en noir + leur numéro
    if zone == "idf":
        departements = (
            communes.assign(dep=communes["INSEE_COM"].str[:2])
            .dissolve(by="dep").reset_index()
        )
        departements.boundary.plot(ax=ax, color="black", linewidth=1, zorder=2)
        for _, d in departements.iterrows():
            p = d.geometry.representative_point()
            ax.annotate(d["dep"], (p.x, p.y), ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black")

    # 3a. Le réseau (facultatif), sans les arrêts
    if reseau:
        lignes = gpd_idfm if reseau is True else gpd_idfm[gpd_idfm["mode"].isin(reseau)]
        epaisseur = lignes["mode"].map(EPAISSEUR).values
        lignes.plot(ax=ax, color="white", linewidth=epaisseur + 1.5, zorder=3)
        lignes.plot(ax=ax, color="#" + lignes["route_color"], linewidth=epaisseur, zorder=4)

    # 3b. Une seule ligne (facultative), avec un liseré blanc, et ses stations
    if ligne is not None:
        trace = gpd_idfm[gpd_idfm["route_short_name"] == ligne]
        trace.plot(ax=ax, color="white", linewidth=7, zorder=3)
        trace.plot(ax=ax, color="#" + trace["route_color"].iloc[0], linewidth=4, zorder=4)
        gpd_stations_lignes[gpd_stations_lignes["route_short_name"] == ligne].plot(
            ax=ax, color="white", edgecolor="black", markersize=30, zorder=5
        )

    # 4. Cadrage sur la zone (marge de 500 m)
    xmin, ymin, xmax, ymax = fond.total_bounds
    ax.set_xlim(xmin - 500, xmax + 500)
    ax.set_ylim(ymin - 500, ymax + 500)

    nom_zone = "Paris" if zone == "paris" else "Île-de-France"
    titre = f"Part {indicateur.lower()} parmi les actifs ({nom_zone})"
    if ligne is not None:
        titre += f" - ligne {ligne}"
    if reseau:
        titre += " - réseau"
    ax.set_title(titre)
    ax.set_axis_off()
    plt.show()

def resume_ligne(ligne, indicateur, df_mob, communes, gpd_stations_lignes,
                 zones=None, n=5, **autres):
    """Part d'un indicateur autour des gares d'une ligne, résumée par zone.
    ligne      : numéro de la ligne, par exemple "B"
    indicateur : une CSP ("Ouvriers", "Cadres"...) ou "Transports en commun"
    zones      : {nom de la zone: [départements]}, par défaut Paris contre banlieue
    n          : nombre de gares extrêmes à afficher"""
    if zones is None:
        zones = {"Paris": ["75"]}

    # Gares de la ligne, avec la commune (ou l'arrondissement) où elles se trouvent
    gares = gpd_stations_lignes[gpd_stations_lignes["route_short_name"] == ligne]
    gares = gpd.sjoin(gares, communes, predicate="within").drop_duplicates("stop_id")

    # Part de l'indicateur dans la commune de chaque gare
    gares = gares.merge(part_par_zone(df_mob, indicateur), left_on="INSEE_COM", right_index=True)

    # Zone de chaque gare selon son département ("Banlieue" si non précisé)
    dep = gares["INSEE_COM"].str[:2]
    gares["zone"] = "Banlieue"
    for nom_zone, departements in zones.items():
        gares.loc[dep.isin(departements), "zone"] = nom_zone

    # Résumé par zone : nombre de gares, moyenne, minimum et maximum
    resume = gares.groupby("zone")["part"].agg(["count", "mean", "min", "max"]).round(1)
    resume.columns = ["Gares", "Moyenne (%)", "Min (%)", "Max (%)"]
    print(f"Ligne {ligne} : part {indicateur.lower()} parmi les actifs\n")
    print(resume)

    # Les gares extrêmes
    colonnes = ["stop_name", "nom", "zone", "part"]
    print(f"\nPart la plus forte :")
    print(gares.sort_values("part", ascending=False)[colonnes].head(n).to_string(index=False))
    print(f"\nPart la plus faible :")
    print(gares.sort_values("part")[colonnes].head(n).to_string(index=False))
