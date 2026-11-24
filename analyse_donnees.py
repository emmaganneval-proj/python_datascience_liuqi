import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import statsmodels.formula.api as smf
from shapely.geometry import Point

from pretraitement_donnees import construire_centres_carreaux


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


DEPARTEMENTS_PETITE_COURONNE = ("75", "92", "93", "94")


def classer_carreaux_zones(gpd_centres, gpd_communes):
    "Commune et zone (Paris ou banlieue) des carreaux de Paris et de la petite couronne"
    # Une commune de banlieue est « riche » si son niveau de vie moyen dépasse celui de Paris
    # Renvoie les carreaux (avec nom, paris, commune_riche) et le niveau de vie de Paris
    communes = gpd_communes[
        gpd_communes["code_commune"].str[:2].isin(DEPARTEMENTS_PETITE_COURONNE)
    ]
    gpd_zones = gpd.sjoin(
        gpd_centres[["ind", "ind_snv", "geometry"]],
        communes[["nom", "code_commune", "geometry"]].to_crs(gpd_centres.crs),
        predicate="within",
    ).drop(columns="index_right")
    gpd_zones["paris"] = gpd_zones["code_commune"].str.startswith("75")

    sommes = gpd_zones.groupby("nom")[["ind_snv", "ind"]].sum()
    nv_communes = sommes["ind_snv"] / sommes["ind"]
    paris = gpd_zones["paris"]
    nv_paris = gpd_zones.loc[paris, "ind_snv"].sum() / gpd_zones.loc[paris, "ind"].sum()
    gpd_zones["commune_riche"] = ~paris & (gpd_zones["nom"].map(nv_communes) > nv_paris)
    return gpd_zones, nv_paris


def calculer_profils_zones(
    gpd_idfm, gpd_zones, distance_max=400, longueur_troncon=500, population_min=100
):
    "Niveau de vie par tronçon de chaque ligne, en Paris et petite couronne, avec sa zone"
    # Même principe que calculer_profils_lignes : les carreaux à moins de distance_max
    # du tracé sont regroupés par tronçon de longueur_troncon mètres le long de la ligne.
    # On ne garde que les carreaux de Paris et de la petite couronne (gpd_zones) :
    # les RER sont ainsi comparables aux métros.
    # Un tronçon est « Paris » si la majorité de ses habitants vit dans Paris.
    traces = (
        gpd_idfm.assign(longueur=gpd_idfm.length)
        .sort_values("longueur", ascending=False)
        .drop_duplicates("route_short_name")  # tracé le plus long de chaque ligne
    )
    liste_profils = []
    for ligne in traces.itertuples():
        proches = gpd_zones.iloc[
            gpd_zones.sindex.query(ligne.geometry, predicate="dwithin", distance=distance_max)
        ]
        # Abscisse curviligne : distance le long du tracé jusqu'au projeté du carreau
        abscisses = shapely.line_locate_point(ligne.geometry, np.asarray(proches.geometry))
        df_profil = (
            proches.assign(
                troncon=abscisses // longueur_troncon,
                population_paris=proches["ind"].where(proches["paris"], 0),
                population_riche=proches["ind"].where(proches["commune_riche"], 0),
            )
            .groupby("troncon", as_index=False)
            .agg(
                population=("ind", "sum"),
                revenus=("ind_snv", "sum"),
                population_paris=("population_paris", "sum"),
                population_riche=("population_riche", "sum"),
            )
        )
        # Tronçons presque vides (Seine, parcs, zones d'activité) : moyenne trop fragile
        df_profil = df_profil[df_profil["population"] >= population_min]
        df_profil["route_short_name"] = ligne.route_short_name
        df_profil["mode"] = ligne.mode
        df_profil["couleur"] = "#" + ligne.route_color
        df_profil["position_km"] = (df_profil["troncon"] + 0.5) * longueur_troncon / 1000
        liste_profils.append(df_profil)

    df_profils = pd.concat(liste_profils).sort_values(["route_short_name", "troncon"])
    df_profils["nv"] = df_profils["revenus"] / df_profils["population"]
    df_profils["zone"] = np.where(
        df_profils["population_paris"] > df_profils["population"] / 2, "Paris", "Banlieue"
    )
    return df_profils.reset_index(drop=True)


def mesurer_heterogeneite_lignes(df_profils, troncons_min=2):
    "Volatilité, dispersion et effet Paris du niveau de vie le long de chaque ligne"
    # Tout est calculé sur le log du niveau de vie : les écarts se lisent en %
    # - volatilite : écart-type des variations d'un tronçon au suivant (comme la
    #   volatilité d'un cours de bourse, avec « par 500 m » au lieu de « par jour »)
    # - dispersion : écart-type sur la ligne, pondéré par la population
    # - effet_paris : part de la variance due au seul fait d'être dans Paris ou non
    #   (variance entre les deux zones / variance totale) ; vide si la ligne n'a pas
    #   au moins troncons_min tronçons de chaque côté
    resultats = []
    for nom, df in df_profils.groupby("route_short_name"):
        df = df.sort_values("troncon")
        log_nv = np.log(df["nv"])
        poids = df["population"]
        moyenne = np.average(log_nv, weights=poids)
        variance = np.average((log_nv - moyenne) ** 2, weights=poids)

        zones = df.groupby("zone")
        variance_entre = sum(
            poids[groupe.index].sum() * (np.average(log_nv[groupe.index], weights=poids[groupe.index]) - moyenne) ** 2
            for _, groupe in zones
        ) / poids.sum()
        deux_zones = zones.size().reindex(["Paris", "Banlieue"], fill_value=0).min() >= troncons_min

        paris = df[df["zone"] == "Paris"]
        banlieue = df[df["zone"] == "Banlieue"]
        population_banlieue = (banlieue["population"] - banlieue["population_paris"]).sum()
        resultats.append(
            {
                "route_short_name": nom,
                "mode": df["mode"].iloc[0],
                "couleur": df["couleur"].iloc[0],
                "troncons": len(df),
                "troncons_paris": len(paris),
                "troncons_banlieue": len(banlieue),
                "volatilite": log_nv.diff().std(),
                "dispersion": np.sqrt(variance),
                "effet_paris": variance_entre / variance if deux_zones else np.nan,
                "nv_paris": paris["revenus"].sum() / paris["population"].sum()
                if len(paris) >= troncons_min else np.nan,
                "nv_banlieue": banlieue["revenus"].sum() / banlieue["population"].sum()
                if len(banlieue) >= troncons_min else np.nan,
                "part_banlieue_riche": banlieue["population_riche"].sum() / population_banlieue
                if len(banlieue) >= troncons_min and population_banlieue > 0 else np.nan,
            }
        )
    return (
        pd.DataFrame(resultats)
        .set_index("route_short_name")
        .sort_values("dispersion", ascending=False)
    )


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
            "habitants": proches.loc[morceaux.index, "ind"],
        }
    )
    # On garde la partie entre les deux terminus
    km_max = gpd_ligne["position_km"].max()
    df_carreaux = df_carreaux[(df_carreaux["fin"] > 0) & (df_carreaux["debut"] < km_max)]
    df_carreaux["debut"] = df_carreaux["debut"].clip(lower=0)
    df_carreaux["fin"] = df_carreaux["fin"].clip(upper=km_max)
    return df_carreaux.sort_values("debut").reset_index(drop=True)


def calculer_distance_stations(lignes, modes, nv_moyen):
    "Carreaux traversés par chaque ligne, avec leur distance à la station la plus proche"
    # lignes : {nom: (gpd_ligne, couleur, df_carreaux)}, comme pour les montagnes
    # modes : mode de chaque ligne (metro ou rer), indexé par son nom
    # La distance se mesure le long de la ligne, du milieu du carreau à la station
    # la plus proche (en mètres) : c'est l'écart horizontal sur les montagnes
    liste = []
    for nom, (gpd_ligne, _, df_carreaux) in lignes.items():
        stations = gpd_ligne["position_km"].values
        milieux = ((df_carreaux["debut"] + df_carreaux["fin"]) / 2).values
        ecarts = np.abs(milieux[:, None] - stations[None, :])
        proche = ecarts.argmin(axis=1)
        liste.append(
            pd.DataFrame(
                {
                    "ligne": nom,
                    "mode": modes[nom],
                    "nv": df_carreaux["nv"].values,
                    "habitants": df_carreaux["habitants"].values,
                    "distance": ecarts.min(axis=1) * 1000,
                    "station": gpd_ligne["stop_id"].values[proche],
                    "paris": gpd_ligne["commune"].str.startswith("Paris").values[proche],
                }
            )
        )
    df_distance = pd.concat(liste, ignore_index=True)
    df_distance["distance_100m"] = df_distance["distance"] / 100
    df_distance["ecart"] = df_distance["nv"] - nv_moyen  # signé : négatif = plus pauvre
    df_distance["ecart_abs"] = df_distance["ecart"].abs()  # hauteur de la barre
    df_distance["sous_moyenne"] = df_distance["ecart"] < 0
    return df_distance


def regresser_ecart_distance(df_distance):
    "Effet de 100 m d'éloignement de la station sur l'écart au niveau de vie moyen"
    # Une régression par mode (métro, RER) et par spécification, toutes avec un effet
    # fixe par ligne : on compare les carreaux d'une même ligne entre eux.
    # Écarts-types regroupés par station : les carreaux voisins se ressemblent.
    specifications = {
        "Écart absolu": "ecart_abs ~ distance_100m + C(ligne)",
        "Écart absolu, Paris contrôlé": "ecart_abs ~ distance_100m + C(ligne) + paris",
        "Écart signé, Paris contrôlé": "ecart ~ distance_100m + C(ligne) + paris",
    }
    resultats = []
    for mode, df in df_distance.groupby("mode"):
        groupes = pd.factorize(df["station"])[0]
        for nom, formule in specifications.items():
            modele = smf.ols(formule, df).fit(
                cov_type="cluster", cov_kwds={"groups": groupes}
            )
            bas, haut = modele.conf_int().loc["distance_100m"]
            resultats.append(
                {
                    "mode": mode,
                    "modele": nom,
                    "effet_100m": modele.params["distance_100m"],
                    "ic_bas": bas,
                    "ic_haut": haut,
                    "p_valeur": modele.pvalues["distance_100m"],
                    "carreaux": int(modele.nobs),
                }
            )
    return pd.DataFrame(resultats).set_index(["mode", "modele"])


def sommer_voisinage(grille, rayon):
    "Somme de chaque case et de ses voisines dans un carré de (2 rayon + 1) cases"
    bord = np.pad(grille, rayon)
    hauteur, largeur = grille.shape
    return sum(
        bord[rayon + dy : rayon + dy + hauteur, rayon + dx : rayon + dx + largeur]
        for dy in range(-rayon, rayon + 1)
        for dx in range(-rayon, rayon + 1)
    )


def construire_grille_nv(df_filosofi, zone, rayon=2):
    "Grille de 200 m du niveau de vie lissé sur la zone (polygone en EPSG:3035)"
    # Les carreaux Filosofi forment une grille régulière en EPSG:3035
    centres = df_filosofi.geometry.centroid.to_crs(3035)
    xmin, ymin, xmax, ymax = zone.bounds
    i0, j0 = int(xmin // 200) - 5, int(ymin // 200) - 5  # marge de 1 km
    ni, nj = int(xmax // 200) - i0 + 6, int(ymax // 200) - j0 + 6
    i = np.floor(centres.x.values / 200).astype(int) - i0
    j = np.floor(centres.y.values / 200).astype(int) - j0
    garder = (i >= 0) & (i < ni) & (j >= 0) & (j < nj)
    revenus, habitants = np.zeros((nj, ni)), np.zeros((nj, ni))
    np.add.at(revenus, (j[garder], i[garder]), df_filosofi["ind_snv"].values[garder])
    np.add.at(habitants, (j[garder], i[garder]), df_filosofi["ind"].values[garder])

    # Lissage : moyenne pondérée sur un carré de (2 rayon + 1) x 200 m (1 km si rayon=2)
    somme_revenus = sommer_voisinage(revenus, rayon)
    somme_habitants = sommer_voisinage(habitants, rayon)
    nv = somme_revenus / np.maximum(somme_habitants, 1)
    x, y = np.meshgrid((np.arange(ni) + i0 + 0.5) * 200, (np.arange(nj) + j0 + 0.5) * 200)
    # Valeur seulement dans la zone, avec au moins 100 habitants autour et un
    # carreau habité tout près : les bois et les grands parcs restent vides
    habite_pres = sommer_voisinage((habitants > 0).astype(float), 1) > 0
    valide = shapely.contains_xy(zone, x, y) & (somme_habitants >= 100) & habite_pres

    # Pour poser les lignes : au-dessus des zones vides (Seine, parcs), moyenne
    # sur un voisinage trois fois plus large, pour qu'elles survolent sans plonger
    large_revenus = sommer_voisinage(revenus, 3 * rayon)
    large_habitants = sommer_voisinage(habitants, 3 * rayon)
    nv_large = np.where(large_habitants > 0, large_revenus / np.maximum(large_habitants, 1), np.nan)
    return {
        "x": x,
        "y": y,
        "nv": np.where(valide, nv, np.nan),
        "nv_lignes": np.where(valide, nv, nv_large),
        "origine": (i0, j0),
    }
def preparer_stations_poles(gpd_stations, graphe_reseau, noms_poles):
    "Temps de trajet et distance à vol d'oiseau de chaque station jusqu'au pôle le plus proche"
    destinations = gpd_stations.loc[gpd_stations["stop_name"].isin(noms_poles), "stop_id"]
    df_temps = calculer_temps_trajet(graphe_reseau, destinations)

    df_reg = gpd_stations.merge(df_temps, on="stop_id")
    df_reg = calculer_distance_destinations(df_reg, gpd_stations, destinations).rename(
        columns={"distance_destination": "distance_km"}
    )
    df_reg = df_reg[df_reg["temps_trajet"] > 0]  # on retire les pôles eux-mêmes
    return df_reg, df_temps, destinations


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


def preparer_carreaux(df_filosofi, gpd_stations, df_temps, destinations):
    "Distance et temps de trajet (marche + transports) vers le pôle le plus proche pour chaque carreau"
    centres = construire_centres_carreaux(df_filosofi)
    carreaux = joindre_carreaux_stations(
        centres, gpd_stations, df_temps, distance_max=1500, vitesse_marche=75
    )
    carreaux = calculer_distance_destinations(carreaux, gpd_stations, destinations).rename(
        columns={"distance_destination": "distance_km"}
    )
    return carreaux


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
