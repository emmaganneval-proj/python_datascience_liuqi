import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def calculer_temps_trajet(graphe, destinations):
    "Temps de trajet minimal (en minutes) vers la destination la plus proche"
    # Graphe inversé : les temps depuis les destinations sont ceux vers elles
    temps, chemins = nx.multi_source_dijkstra(
        graphe.reverse(), set(destinations), weight="temps"
    )
    df_temps = pd.DataFrame(
        {
            "stop_id": list(temps),
            "temps_trajet": list(temps.values()),
            "station_destination": [chemins[station][0] for station in temps],
        }
    )
    return df_temps


def joindre_carreaux_stations(
    gpd_centres, gpd_stations, df_temps, distance_max, vitesse_marche
):
    "Associe chaque carreau à sa station la plus proche et calcule le temps total"
    # Au-delà de distance_max, le carreau est conservé mais sans station (non desservi)
    gpd_carreaux = gpd.sjoin_nearest(
        gpd_centres,
        gpd_stations[["stop_id", "stop_name", "geometry"]],
        how="left",
        max_distance=distance_max,
        distance_col="distance_station",
    ).drop(columns=["index_right"])
    # En cas d'égalité de distance, sjoin_nearest renvoie plusieurs stations
    gpd_carreaux = gpd_carreaux[~gpd_carreaux.index.duplicated()]

    gpd_carreaux = gpd_carreaux.join(df_temps.set_index("stop_id"), on="stop_id")
    gpd_carreaux["temps_marche"] = gpd_carreaux["distance_station"] / vitesse_marche
    gpd_carreaux["temps_total"] = (
        gpd_carreaux["temps_marche"] + gpd_carreaux["temps_trajet"]
    )
    return gpd_carreaux


def calculer_nv_par_tranche_temps(gpd_carreaux, bornes):
    "Niveau de vie moyen et part de la population par tranche de temps total"
    noms_tranches = [
        f"{bornes[i]}-{bornes[i + 1]} min" for i in range(len(bornes) - 1)
    ] + [f"≥ {bornes[-1]} min"]
    gpd_carreaux["tranche_temps"] = (
        pd.cut(
            gpd_carreaux["temps_total"],
            bins=bornes + [np.inf],
            labels=noms_tranches,
            right=False,
        )
        .cat.add_categories("Non desservi")
        .fillna("Non desservi")
    )
    df_tranches = gpd_carreaux.groupby("tranche_temps", observed=True).agg(
        population=("ind", "sum"), revenus=("ind_snv", "sum")
    )
    # Moyenne pondérée par la population, et non moyenne des nv des carreaux
    df_tranches["nv"] = df_tranches["revenus"] / df_tranches["population"]
    df_tranches["part_population"] = (
        df_tranches["population"] / df_tranches["population"].sum()
    )
    return df_tranches


def calculer_distance_destinations(gpd_carreaux, gpd_stations, destinations):
    "Distance à vol d'oiseau (en km) de chaque carreau à la destination la plus proche"
    points = gpd_stations.loc[gpd_stations["stop_id"].isin(destinations)].union_all()
    gpd_carreaux["distance_destination"] = gpd_carreaux.distance(points) / 1000
    return gpd_carreaux


def comparer_nv_a_distance_egale(gpd_carreaux, bornes):
    "Niveau de vie des carreaux mieux ou moins bien connectés, par tranche de distance"
    gpd_carreaux["tranche_distance"] = pd.cut(
        gpd_carreaux["distance_destination"],
        bins=bornes,
        labels=[f"{bornes[i]}-{bornes[i + 1]} km" for i in range(len(bornes) - 1)],
    )
    # Un carreau est mieux connecté si son temps est sous la médiane, à distance égale
    medianes = gpd_carreaux.groupby("tranche_distance", observed=True)[
        "temps_total"
    ].transform("median")
    gpd_carreaux["connexion"] = np.where(
        gpd_carreaux["temps_total"].isna(),
        "non desservi",
        np.where(
            gpd_carreaux["temps_total"] <= medianes,
            "mieux connecté",
            "moins bien connecté",
        ),
    )
    df_comparaison = gpd_carreaux.groupby(
        ["tranche_distance", "connexion"], observed=True
    ).agg(population=("ind", "sum"), revenus=("ind_snv", "sum"))
    df_comparaison["nv"] = df_comparaison["revenus"] / df_comparaison["population"]
    return df_comparaison["nv"].unstack()


def calculer_profils_lignes(gpd_traces, gpd_centres, distance_max, longueur_troncon):
    "Niveau de vie moyen par tronçon de chaque ligne, sur les carreaux proches du tracé"
    liste_profils = []
    for ligne in gpd_traces.itertuples():
        gpd_proches = gpd_centres.loc[
            gpd_centres.distance(ligne.geometry) < distance_max
        ]
        # Abscisse curviligne : distance le long du tracé jusqu'au projeté du carreau
        abscisses = gpd_proches.geometry.apply(ligne.geometry.project)
        df_profil = (
            gpd_proches.assign(troncon=abscisses // longueur_troncon)
            .groupby("troncon", as_index=False)
            .agg(population=("ind", "sum"), revenus=("ind_snv", "sum"))
        )
        df_profil["route_short_name"] = ligne.route_short_name
        df_profil["position_km"] = (
            (df_profil["troncon"] + 0.5) * longueur_troncon / 1000
        )
        liste_profils.append(df_profil)

    df_profils = pd.concat(liste_profils, ignore_index=True)
    df_profils["nv"] = df_profils["revenus"] / df_profils["population"]
    df_profils["ecart_nv"] = df_profils.groupby("route_short_name")["nv"].diff().abs()
    return df_profils


def comparer_heterogeneite_lignes(df_profils):
    "Écart absolu moyen de niveau de vie entre stations successives, par ligne"
    df_heterogeneite = (
        df_profils.groupby("route_short_name")
        .agg(ecart_moyen=("ecart_nv", "mean"), nv_moyen=("nv", "mean"))
        .sort_values("ecart_moyen", ascending=False)
    )
    return df_heterogeneite
