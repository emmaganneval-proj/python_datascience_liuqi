import matplotlib.pyplot as plt

from analyse_donnees import (
    calculer_nv_par_tranche_temps,
    calculer_temps_trajet,
    joindre_carreaux_stations,
)
from pretraitement_donnees import (
    URL_ZIP_FILOSOFI,
    URL_ZIP_IDFM,
    charger_arrets,
    charger_filosofi,
    charger_horaires,
    charger_parcours,
    charger_routes,
    charger_stations,
    charger_trips,
    construire_aretes,
    construire_centres_carreaux,
    construire_graphe_reseau,
    construire_lignes_idfm,
    construire_stations,
    definir_quantiles,
    indicateurs_filosofi,
    telecharger_zip,
)
from visualisation_donnees import (
    XLIM,
    YLIM,
    ajouter_lignes_graphique,
    ajouter_stations_graphique,
    generer_carte_filosofi,
    generer_carte_temps,
    generer_graphique_tranches,
)

if __name__ == "__main__":
    # telecharger_zip(URL_ZIP_IDFM, "idfm")
    # telecharger_zip(URL_ZIP_FILOSOFI, "filosofi")

    df_filosofi = charger_filosofi("data/filosofi/france_metro/carreaux_200m_met.shp")
    df_filosofi = indicateurs_filosofi(df_filosofi)
    df_filosofi = definir_quantiles(df_filosofi, "nv", 6)

    df_routes = charger_routes("data/idfm")
    df_trips = charger_trips("data/idfm", df_routes)
    df_parcours = charger_parcours("data/idfm", df_trips)
    gpd_idfm = construire_lignes_idfm(df_parcours)

    df_arrets = charger_arrets("data/idfm", df_trips)
    df_stations = charger_stations("data/idfm", df_arrets)
    gpd_stations = construire_stations(df_stations)

    df_horaires = charger_horaires("data/idfm", df_trips)
    df_aretes = construire_aretes(df_horaires)
    graphe_reseau = construire_graphe_reseau(df_aretes)
    destinations = gpd_stations.loc[
        gpd_stations["stop_name"].isin(
            ["Châtelet - Les Halles", "Châtelet", "La Défense"]
        ),
        "stop_id",
    ]
    df_temps = calculer_temps_trajet(graphe_reseau, destinations)

    gpd_centres = construire_centres_carreaux(df_filosofi)
    gpd_carreaux = joindre_carreaux_stations(
        gpd_centres, gpd_stations, df_temps, distance_max=1500, vitesse_marche=75
    )

    ax = generer_carte_filosofi(df_filosofi, "nv")
    ax = ajouter_lignes_graphique(ax, gpd_idfm)
    ax = ajouter_stations_graphique(df_stations, ax)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    df_tranches = calculer_nv_par_tranche_temps(
        gpd_carreaux, bornes=[0, 10, 20, 30, 40, 50]
    )
    generer_graphique_tranches(df_tranches)

    ax = generer_carte_temps(df_filosofi, gpd_carreaux)
    ax = ajouter_lignes_graphique(ax, gpd_idfm)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.show()
