import networkx as nx
import pandas as pd


def calculer_temps_trajet(graphe, destination):
    "Temps de trajet minimal (en minutes) de chaque station vers la destination"
    # Graphe inversé : les temps depuis la destination sont ceux vers elle
    temps = nx.single_source_dijkstra_path_length(
        graphe.reverse(), destination, weight="temps"
    )
    df_temps = pd.DataFrame(list(temps.items()), columns=["stop_id", "temps_trajet"])
    return df_temps
