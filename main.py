import matplotlib.pyplot as plt

from analyse_donnees import (
    ajouter_communes_stations,
    calculer_carreaux_traverses,
    calculer_nv_par_commune,
    calculer_nv_par_station,
    calculer_distance_destinations,
    calculer_nv_par_tranche_temps,
    calculer_profils_lignes,
    calculer_temps_trajet,
    comparer_heterogeneite_lignes,
    comparer_nv_a_distance_egale,
    joindre_carreaux_stations,
    positionner_stations_ligne,
)
from pretraitement_donnees import (
    URL_ZIP_FILOSOFI,
    URL_ZIP_IDFM,
    charger_arrets,
    charger_communes,
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
    construire_positions_stations,
    construire_stations,
    construire_terminus,
    construire_traces_lignes,
    definir_quantiles,
    indicateurs_filosofi,
    telecharger_zip,
)
from visualisation_donnees import (
    XLIM,
    YLIM,
    ajouter_lignes_graphique,
    ajouter_pastilles_graphique,
    ajouter_stations_graphique,
    enregistrer_graphique,
    generer_carte_filosofi,
    generer_carte_temps,
    generer_graphique_montagne,
    generer_montagnes_empilees,
    generer_graphique_connexion,
    generer_graphique_tranches,
    generer_nuage_distance_temps,
    generer_profil_ligne,
)

if __name__ == "__main__":
    telecharger_zip(URL_ZIP_IDFM, "idfm")
    telecharger_zip(URL_ZIP_FILOSOFI, "filosofi")

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

    gpd_terminus = construire_terminus(df_horaires, gpd_stations, df_routes)

    ax = generer_carte_filosofi(df_filosofi, "nv")
    ax = ajouter_lignes_graphique(ax, gpd_idfm)
    ax = ajouter_stations_graphique(df_stations, ax)
    ax = ajouter_pastilles_graphique(
        ax, gpd_terminus.cx[XLIM[0] : XLIM[1], YLIM[0] : YLIM[1]]
    )
    ax.set_title("Niveau de vie et réseau ferré en Île-de-France (2021)")
    ax.get_legend().set_title(
        "Niveau de vie moyen par habitant\n(sextiles de population)"
    )
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    enregistrer_graphique(ax, "carte_niveau_vie")

    df_tranches = calculer_nv_par_tranche_temps(
        gpd_carreaux, bornes=[0, 10, 20, 30, 40, 50]
    )
    ax = generer_graphique_tranches(df_tranches)
    enregistrer_graphique(ax, "niveau_vie_par_tranche_temps")

    ax = generer_carte_temps(df_filosofi, gpd_carreaux)
    ax = ajouter_lignes_graphique(ax, gpd_idfm)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    enregistrer_graphique(ax, "carte_temps_trajet")

    gpd_carreaux = calculer_distance_destinations(
        gpd_carreaux, gpd_stations, destinations
    )
    ax = generer_nuage_distance_temps(gpd_carreaux)
    enregistrer_graphique(ax, "distance_temps_trajet")

    df_comparaison = comparer_nv_a_distance_egale(
        gpd_carreaux, bornes=[0, 5, 10, 15, 20, 30]
    )
    ax = generer_graphique_connexion(df_comparaison)
    enregistrer_graphique(ax, "niveau_vie_connexion_distance_egale")

    gpd_traces = construire_traces_lignes(gpd_idfm)
    gpd_positions = construire_positions_stations(df_stations, gpd_traces)
    nv_idf = df_filosofi["ind_snv"].sum() / df_filosofi["ind"].sum()
    df_profils = calculer_profils_lignes(
        gpd_traces, gpd_centres, distance_max=400, longueur_troncon=500
    )
    for nom_ligne in ["2", "4"]:
        ax = generer_profil_ligne(df_profils, gpd_positions, nom_ligne, nv_idf)
        enregistrer_graphique(ax, f"profil_ligne_{nom_ligne}")
    df_heterogeneite = comparer_heterogeneite_lignes(df_profils)

    # Montagnes : niveau de vie le long des lignes (stations et carreaux traversés)
    df_nv = calculer_nv_par_station(gpd_carreaux)
    nv_moyen = gpd_carreaux["ind_snv"].sum() / gpd_carreaux["ind"].sum()
    gpd_communes = charger_communes()
    nv_communes = calculer_nv_par_commune(gpd_centres, gpd_communes)
    lignes = {}
    for nom in ["1", "9", "13", "14"]:
        gpd_ligne = positionner_stations_ligne(gpd_stations, df_stations, gpd_idfm, nom)
        gpd_ligne = ajouter_communes_stations(gpd_ligne, gpd_communes)
        df_carreaux = calculer_carreaux_traverses(df_filosofi, gpd_ligne, gpd_idfm, nom)
        filtre = df_routes["route_short_name"] == nom
        couleur = "#" + df_routes.loc[filtre, "route_color"].iloc[0]
        lignes[nom] = (gpd_ligne, couleur, df_carreaux)

    axes = generer_montagnes_empilees(
        lignes, df_nv, nv_moyen, nv_communes, gpd_idfm, gpd_communes
    )
    enregistrer_graphique(axes[0], "Metros 1, 9, 13 et 14 : montagnes_empilees")
    plt.show()

    # RER
    lignes = {}
    for nom in ["A", "B"]:
        gpd_ligne = positionner_stations_ligne(gpd_stations, df_stations, gpd_idfm, nom)
        gpd_ligne = ajouter_communes_stations(gpd_ligne, gpd_communes)
        df_carreaux = calculer_carreaux_traverses(df_filosofi, gpd_ligne, gpd_idfm, nom)
        filtre = df_routes["route_short_name"] == nom
        couleur = "#" + df_routes.loc[filtre, "route_color"].iloc[0]
        lignes[nom] = (gpd_ligne, couleur, df_carreaux)

    axes = generer_montagnes_empilees(
        lignes, df_nv, nv_moyen, nv_communes, gpd_idfm, gpd_communes
    )
    enregistrer_graphique(axes[0], "RER A et B : montagnes_empilees")
    plt.show()
