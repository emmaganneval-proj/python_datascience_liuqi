import geopandas as gpd
import networkx as nx
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
