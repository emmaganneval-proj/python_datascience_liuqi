"""Régressions : temps de trajet, distance à Châtelet et niveau de vie"""
import matplotlib.pyplot as plt
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