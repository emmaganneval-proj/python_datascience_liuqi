import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point


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

    df_profils = pd.concat(liste_profils).sort_values(["route_short_name", "troncon"])
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


def calculer_nv_par_station(gpd_carreaux):
    "Niveau de vie moyen des habitants rattachés à chaque station"
    df_nv = (
        gpd_carreaux.dropna(subset=["stop_id"])
        .groupby("stop_id")
        .agg(population=("ind", "sum"), revenus=("ind_snv", "sum"))
    )
    # Moyenne pondérée par la population, et non moyenne des nv des carreaux
    df_nv["nv"] = df_nv["revenus"] / df_nv["population"]
    return df_nv


def trace_ligne(gpd_idfm, ligne):
    "Tracé le plus long de la ligne : il dessert toutes les stations (sauf branches)"
    traces = gpd_idfm[gpd_idfm["route_short_name"] == ligne]
    return traces.geometry.loc[traces.length.idxmax()]


def positionner_stations_ligne(gpd_stations, df_stations, gpd_idfm, ligne):
    "Position (en km depuis le terminus ouest ou nord) des stations, le long du tracé"
    trace = trace_ligne(gpd_idfm, ligne)

    ids_ligne = df_stations.loc[df_stations["route_short_name"] == ligne, "stop_id"]
    gpd_ligne = gpd_stations[gpd_stations["stop_id"].isin(ids_ligne)].copy()
    # Stations à plus de 300 m du tracé : sur une autre branche, on les écarte
    gpd_ligne = gpd_ligne[gpd_ligne.distance(trace) < 300]

    gpd_ligne["position_km"] = gpd_ligne.geometry.apply(trace.project) / 1000
    gpd_ligne = gpd_ligne.sort_values("position_km")
    # Lecture d'ouest en est (ou du nord au sud pour une ligne plutôt verticale) :
    # on retourne l'axe si le tracé part dans l'autre sens
    debut, fin = gpd_ligne.geometry.iloc[0], gpd_ligne.geometry.iloc[-1]
    if abs(fin.x - debut.x) > abs(fin.y - debut.y):
        a_retourner = debut.x > fin.x
    else:
        a_retourner = debut.y < fin.y
    if a_retourner:
        gpd_ligne["position_km"] = trace.length / 1000 - gpd_ligne["position_km"]
        gpd_ligne = gpd_ligne.sort_values("position_km")
    # Le premier terminus est à 0 km
    gpd_ligne["position_km"] -= gpd_ligne["position_km"].min()
    return gpd_ligne


def ajouter_communes_stations(gpd_ligne, gpd_communes):
    "Ajoute à chaque station le nom de sa commune (ou de son arrondissement à Paris)"
    gpd_ligne = gpd.sjoin(
        gpd_ligne, gpd_communes[["nom", "geometry"]], how="left", predicate="within"
    ).drop(columns=["index_right"])
    return gpd_ligne.rename(columns={"nom": "commune"})


def calculer_nv_par_commune(gpd_centres, gpd_communes):
    "Niveau de vie moyen de chaque commune (ou arrondissement), pondéré par la population"
    gpd_centres = gpd.sjoin(
        gpd_centres,
        gpd_communes[["nom", "geometry"]].to_crs(gpd_centres.crs),
        predicate="within",
    )
    sommes = gpd_centres.groupby("nom")[["ind_snv", "ind"]].sum()
    return sommes["ind_snv"] / sommes["ind"]


def calculer_carreaux_traverses(df_filosofi, gpd_ligne, gpd_idfm, ligne):
    "Carreaux Filosofi traversés par le tracé, avec leur début et fin (en km) sur la ligne"
    trace = trace_ligne(gpd_idfm, ligne)
    # Même repère que les stations : on retrouve le sens et l'origine à partir d'elles
    km_trace = gpd_ligne.geometry.apply(trace.project) / 1000
    retourne = km_trace.iloc[0] > km_trace.iloc[-1]

    def vers_km_ligne(km):
        return km_trace.max() - km if retourne else km - km_trace.min()

    # Carreaux qui touchent le tracé (index spatial : rapide), puis morceaux de tracé
    proches = df_filosofi.iloc[df_filosofi.sindex.query(trace, predicate="intersects")]
    morceaux = proches.geometry.intersection(trace).explode()
    morceaux = morceaux[morceaux.length > 0]  # on ignore les carreaux juste effleurés

    km_a = morceaux.apply(lambda m: trace.project(Point(m.coords[0])) / 1000)
    km_b = morceaux.apply(lambda m: trace.project(Point(m.coords[-1])) / 1000)
    df_carreaux = pd.DataFrame(
        {
            "debut": np.minimum(vers_km_ligne(km_a), vers_km_ligne(km_b)),
            "fin": np.maximum(vers_km_ligne(km_a), vers_km_ligne(km_b)),
            "nv": proches.loc[morceaux.index, "nv"],
        }
    )
    # On garde la partie entre les deux terminus
    km_max = gpd_ligne["position_km"].max()
    df_carreaux = df_carreaux[(df_carreaux["fin"] > 0) & (df_carreaux["debut"] < km_max)]
    df_carreaux["debut"] = df_carreaux["debut"].clip(lower=0)
    df_carreaux["fin"] = df_carreaux["fin"].clip(upper=km_max)
    return df_carreaux.sort_values("debut").reset_index(drop=True)
