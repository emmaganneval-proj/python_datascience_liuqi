import gzip
import http.client
import io
import os
import urllib.request
import zipfile

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

URL_ZIP_IDFM = "https://files.mobilitydatabase.org/tdg-80921/tdg-80921-202610140154/tdg-80921-202610140154.zip"
URL_ZIP_FILOSOFI = "https://www.insee.fr/fr/statistiques/fichier/8735162/Filosofi2021_carreaux_200m_shp.zip"
URL_MOBPRO = (
    "https://www.insee.fr/fr/statistiques/fichier/9004795/RP2023_mobpro.parquet"
)
URL_COMMUNES = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions/ile-de-france/communes-ile-de-france.geojson"
URL_ARRONDISSEMENTS = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/arrondissements/exports/geojson"

NOMS_CSP = {
    "1": "Agriculteurs",
    "2": "Artisans, commerçants, chefs d'entreprise",
    "3": "Cadres",
    "4": "Professions intermédiaires",
    "5": "Employés",
    "6": "Ouvriers",
}
NOMS_MODES = {
    "1": "Pas de transport",
    "2": "Marche",
    "3": "Vélo",
    "4": "Deux-roues motorisé",
    "5": "Voiture",
    "6": "Transports en commun",
}


def telecharger_zip(url, nom_dossier, nb_essais=5):
    """Télécharge un zip et le décompresse dans le dossier /data/{nom_dossier}"""
    if os.path.exists(f"data/{nom_dossier}"):
        return
    os.makedirs("data", exist_ok=True)
    for _ in range(nb_essais):
        # Le serveur de l'Insee coupe parfois le transfert : on recommence
        try:
            with urllib.request.urlopen(url) as response:
                contenu = response.read()
            with zipfile.ZipFile(io.BytesIO(contenu)) as zf:
                zf.extractall(f"data/{nom_dossier}")
            return
        except (http.client.IncompleteRead, zipfile.BadZipFile):
            continue
    raise OSError(f"Téléchargement incomplet de {url} après {nb_essais} essais")


def telecharger_fichier(url, nom_fichier, nb_essais=5):
    """Télécharge un fichier dans le dossier /data"""
    if os.path.exists(f"data/{nom_fichier}"):
        return
    os.makedirs("data", exist_ok=True)
    for _ in range(nb_essais):
        try:
            with urllib.request.urlopen(url) as response:
                contenu = response.read()
        except http.client.IncompleteRead:
            continue
        # Le serveur de la Ville de Paris renvoie parfois le fichier compressé
        # sans le signaler : un contenu gzip commence toujours par 1f 8b
        if contenu.startswith(b"\x1f\x8b"):
            contenu = gzip.decompress(contenu)
        # Le serveur de l'Insee coupe parfois le transfert sans erreur :
        # un fichier parquet complet se termine toujours par "PAR1"
        if not nom_fichier.endswith(".parquet") or contenu.endswith(b"PAR1"):
            with open(f"data/{nom_fichier}", "wb") as fichier_sortie:
                fichier_sortie.write(contenu)
            return
    raise OSError(f"Téléchargement incomplet de {url} après {nb_essais} essais")


def charger_filosofi(path):
    """Chargement des données de Filosofi"""
    liste_departements = ["75", "77", "78", "91", "92", "93", "94", "95"]
    df_filosofi = gpd.read_file(path, columns=["lcog_geo", "ind", "ind_snv"])
    df_filosofi = df_filosofi.loc[
        df_filosofi["lcog_geo"].str[:2].isin(liste_departements)
    ]
    return df_filosofi


def charger_mobpro(path):
    """Chargement des trajets domicile-travail des actifs habitant en Île-de-France"""
    df_mob = pd.read_parquet(
        path,
        columns=["COMMUNE", "ARM", "GS", "TRANS", "IPONDI"],
        filters=[("REGION", "==", "11")],
    )
    return df_mob


def charger_routes(path, modes=("metro", "rer", "transilien", "tram")):
    """Chargement des lignes IDFM (métro, RER, Transilien et tram)"""
    df_routes = pd.read_csv(
        f"{path}/routes.txt",
        usecols=[
            "route_id",
            "route_type",
            "route_short_name",
            "route_color",
            "route_text_color",
            "agency_id",
        ],
    )

    df_routes["mode"] = np.select(
        [
            df_routes["route_type"] == 1,
            (df_routes["route_type"] == 2) & (df_routes["agency_id"] == "IDFM:71"),
            (df_routes["route_type"] == 2) & (df_routes["agency_id"] == "IDFM:1046"),
            df_routes["route_type"] == 0,
        ],
        ["metro", "rer", "transilien", "tram"],
        default="autre",
    )
    # Le CDG VAL est une navette interne à l'aéroport de Roissy
    df_routes = df_routes.loc[
        df_routes["mode"].isin(modes) & (df_routes["route_short_name"] != "CDG VAL")
    ]
    return df_routes


def charger_trips(path, df_routes):
    "Chargement des trains"
    df_trips = pd.read_csv(
        f"{path}/trips.txt", usecols=["route_id", "shape_id", "trip_id"]
    )
    df_trips = df_trips.merge(df_routes, on="route_id", how="inner")
    return df_trips


def charger_parcours(path, df_trips):
    """Chargement du tracé des parcours géographiques des lignes IDFM"""

    df_shapes = pd.read_csv(
        f"{path}/shapes.txt",
        usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    ).drop_duplicates()
    df_shapes = df_shapes.merge(
        df_trips.drop_duplicates("shape_id").drop(columns=["trip_id"]), on="shape_id"
    )

    df_shapes = df_shapes.sort_values(["shape_id", "shape_pt_sequence"])

    return df_shapes


def charger_arrets(path, df_trips):
    "Chargement des arrêts des trains"
    df_arrets = pd.read_csv(f"{path}/stop_times.txt", usecols=["trip_id", "stop_id"])
    df_arrets = (
        df_arrets.merge(
            df_trips,
            on="trip_id",
        )
        .drop(columns=["trip_id", "shape_id"])
        .drop_duplicates(["route_id", "stop_id"])
    )
    return df_arrets


def charger_stations(path, df_arrets):
    "Chargement de la position des stations"
    df_stations = pd.read_csv(
        f"{path}/stops.txt",
        usecols=["stop_id", "stop_name", "stop_lon", "stop_lat", "parent_station"],
    )

    df_parents = df_stations[["stop_id", "parent_station"]].merge(
        df_arrets, on="stop_id"
    )

    df_stations = (
        df_stations.drop(columns=["parent_station"])
        .merge(
            df_parents.drop(columns=["stop_id"]),
            left_on="stop_id",
            right_on="parent_station",
        )
        .drop_duplicates(["route_id", "stop_id"])
    )
    return df_stations


def charger_horaires(path, df_trips):
    "Chargement des horaires de passage des trains à chaque station"
    df_horaires = pd.read_csv(
        f"{path}/stop_times.txt",
        usecols=[
            "trip_id",
            "stop_id",
            "stop_sequence",
            "arrival_time",
            "departure_time",
        ],
    )
    df_parents = pd.read_csv(f"{path}/stops.txt", usecols=["stop_id", "parent_station"])
    df_horaires = (
        df_horaires.merge(df_trips[["trip_id", "route_id"]], on="trip_id")
        .merge(df_parents, on="stop_id")
        .drop(columns=["stop_id"])
    )
    return df_horaires


def indicateurs_filosofi(df_filosofi):
    """Calcul de nouveaux indicateurs"""
    df_filosofi["nv"] = df_filosofi["ind_snv"] / df_filosofi["ind"]
    return df_filosofi


def indicateurs_mobpro(df_mob):
    """Calcul de nouveaux indicateurs"""
    df_mob["CSP"] = df_mob["GS"].map(NOMS_CSP)
    df_mob["mode"] = df_mob["TRANS"].map(NOMS_MODES)
    df_mob["tc"] = df_mob["mode"] == "Transports en commun"
    # Lieu de résidence : l'arrondissement à Paris, la commune ailleurs
    df_mob["code_geo"] = np.where(
        df_mob["COMMUNE"] == "75056", df_mob["ARM"], df_mob["COMMUNE"]
    )
    return df_mob


def definir_quantiles(df_filosofi, indicateur, nb_quantiles):
    """Calcul du quantile de la valeur d'un indicateur pour chaque carreau"""
    quantiles = np.quantile(
        a=df_filosofi[indicateur],
        q=[(k + 1) / nb_quantiles for k in range(nb_quantiles - 1)],
        method="inverted_cdf",
        weights=df_filosofi["ind"],
    )
    noms_bornes = [
        f"{quantile / 1000:.1f} k€".replace(".", ",") for quantile in quantiles
    ]
    noms_quantiles = (
        [f"moins de {noms_bornes[0]}"]
        + [
            f"{noms_bornes[i]} – {noms_bornes[i + 1]}"
            for i in range(len(quantiles) - 1)
        ]
        + [f"{noms_bornes[-1]} et plus"]
    )
    df_filosofi[f"{indicateur}_quantile"] = pd.cut(
        x=df_filosofi[f"{indicateur}"],
        bins=np.concatenate(([-np.inf], quantiles, [np.inf])),
        labels=noms_quantiles,
        right=False,
    )
    return df_filosofi


def construire_lignes_idfm(df_parcours):
    "Construit le tracé des lignes ferrées"
    df_shapes = df_parcours.groupby("shape_id").apply(
        lambda group: LineString(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
    )
    gpd_idfm = gpd.GeoDataFrame(
        df_parcours.drop_duplicates("shape_id")
        .drop(columns=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
        .set_index("shape_id"),
        geometry=df_shapes,
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    return gpd_idfm


def construire_aretes(df_horaires):
    "Construit les arêtes du réseau : temps de parcours médian entre deux stations"
    df_aretes = df_horaires.sort_values(["trip_id", "stop_sequence"])
    # Les heures peuvent dépasser 24:00 dans le GTFS : to_timedelta les accepte
    df_aretes["depart"] = pd.to_timedelta(df_aretes["departure_time"])

    # Écart entre deux départs successifs : inclut le temps d'arrêt en station
    df_suivant = df_aretes.groupby("trip_id")[["parent_station", "depart"]].shift(-1)
    df_aretes["station_arrivee"] = df_suivant["parent_station"]
    df_aretes["temps"] = (
        df_suivant["depart"] - df_aretes["depart"]
    ).dt.total_seconds() / 60

    df_aretes = (
        df_aretes.dropna(subset=["station_arrivee"])
        .rename(columns={"parent_station": "station_depart"})
        .groupby(["route_id", "station_depart", "station_arrivee"], as_index=False)
        .agg(temps=("temps", "median"))
    )
    return df_aretes


def construire_graphe_reseau(df_aretes):
    "Construit le graphe orienté du réseau, pondéré par les temps de parcours"
    # Un même tronçon peut être desservi par plusieurs lignes : on garde la plus rapide
    df_aretes = df_aretes.groupby(
        ["station_depart", "station_arrivee"], as_index=False
    ).agg(temps=("temps", "min"))
    graphe_reseau = nx.from_pandas_edgelist(
        df_aretes,
        source="station_depart",
        target="station_arrivee",
        edge_attr="temps",
        create_using=nx.DiGraph,
    )
    return graphe_reseau


def construire_stations(df_stations):
    "Construit la position des stations (une ligne par station)"
    df_stations = df_stations.drop_duplicates("stop_id")[
        ["stop_id", "stop_name", "stop_lon", "stop_lat"]
    ]
    gpd_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["stop_lon"], df_stations["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    return gpd_stations


def construire_centres_carreaux(df_filosofi):
    "Remplace la géométrie des carreaux par leur centre"
    gpd_centres = df_filosofi.set_geometry(df_filosofi.centroid)
    return gpd_centres


def construire_terminus(df_horaires, gpd_stations, df_routes):
    "Construit la position des terminus : stations jamais intermédiaires sur la ligne"
    sequences = df_horaires.groupby("trip_id")["stop_sequence"]
    df_terminus = (
        df_horaires.assign(
            extremite=(df_horaires["stop_sequence"] == sequences.transform("min"))
            | (df_horaires["stop_sequence"] == sequences.transform("max"))
        )
        .groupby(["route_id", "parent_station"], as_index=False)
        .agg(terminus=("extremite", "all"))
        .query("terminus")
    )
    gpd_terminus = gpd_stations.merge(
        df_terminus, left_on="stop_id", right_on="parent_station"
    ).merge(df_routes, on="route_id")
    return gpd_terminus


def construire_traces_lignes(gpd_idfm):
    "Construit le plus long tracé de chaque ligne"
    gpd_traces = (
        gpd_idfm.assign(longueur=gpd_idfm.length)
        .sort_values("longueur", ascending=False)
        .drop_duplicates("route_id")
        .reset_index(drop=True)
    )
    return gpd_traces


def construire_positions_stations(df_stations, gpd_traces):
    "Construit la position (en km) des stations le long du tracé de leur ligne"
    gpd_positions = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["stop_lon"], df_stations["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    traces = gpd.GeoSeries(
        gpd_positions["route_id"].map(gpd_traces.set_index("route_id").geometry),
        crs="EPSG:2154",
    )
    gpd_positions["position_km"] = traces.project(gpd_positions.geometry) / 1000
    # Une station à plus de 600 m du tracé appartient à une autre branche
    gpd_positions = gpd_positions.loc[gpd_positions.distance(traces) < 600]
    return gpd_positions


def charger_communes(path="data/communes.gpkg"):
    "Contours des communes d'Île-de-France, avec Paris découpé en arrondissements"
    if os.path.exists(path):
        return gpd.read_file(path)

    telecharger_fichier(URL_COMMUNES, "communes.geojson")
    telecharger_fichier(URL_ARRONDISSEMENTS, "arrondissements.geojson")

    gpd_communes = gpd.read_file("data/communes.geojson").rename(
        columns={"code": "code_commune"}
    )
    gpd_communes = gpd_communes[gpd_communes["code_commune"] != "75056"]

    gpd_arr = gpd.read_file("data/arrondissements.geojson")
    gpd_arr["code_commune"] = gpd_arr["c_arinsee"].astype(int).astype(str)
    gpd_arr["nom"] = (
        gpd_arr["c_ar"]
        .astype(int)
        .map(lambda n: "Paris 1er" if n == 1 else f"Paris {n}e")
    )

    gpd_communes = pd.concat(
        [gpd_communes, gpd_arr.to_crs(gpd_communes.crs)], ignore_index=True
    )[["code_commune", "nom", "geometry"]].to_crs("EPSG:2154")
    os.makedirs("data", exist_ok=True)
    gpd_communes.to_file(path)
    return gpd_communes


# ===== FONCTION : CHARGER LES LIGNES (MÉTRO, RER, TRANSILIEN, TRAM) =====
def charger_routes_etendu(path, modes=("metro", "rer", "transilien", "tram")):
    """Comme charger_routes (pretraitement_donnees.py), avec le Transilien et le tram en plus.
    modes : les types de lignes à garder"""
    df_routes = pd.read_csv(
        f"{path}/routes.txt",
        usecols=[
            "route_id",
            "route_type",
            "route_short_name",
            "route_color",
            "route_text_color",
            "agency_id",
        ],
    )

    # On ajoute le nom de l'opérateur de chaque ligne (RATP, RER, Transilien, TER...)
    agences = pd.read_csv(f"{path}/agency.txt", usecols=["agency_id", "agency_name"])
    df_routes = df_routes.merge(agences, on="agency_id")

    # On donne un type à chaque ligne
    conditions = [
        df_routes["route_type"] == 1,  # métro
        (df_routes["route_type"] == 2) & (df_routes["agency_name"] == "RER"),  # RER
        (df_routes["route_type"] == 2)
        & (df_routes["agency_name"] == "Transilien"),  # Transilien
        df_routes["route_type"] == 0,  # tram
    ]
    df_routes["mode"] = np.select(
        conditions, ["metro", "rer", "transilien", "tram"], default="autre"
    )
    # -> les TER tombent dans "autre" et sont donc écartés

    # On retire le CDG VAL (navette interne de l'aéroport de Roissy)
    df_routes = df_routes[df_routes["route_short_name"] != "CDG VAL"]

    # Couleur grise pour les lignes qui n'en ont pas
    df_routes["route_color"] = df_routes["route_color"].fillna("888888")

    return df_routes[df_routes["mode"].isin(modes)]
