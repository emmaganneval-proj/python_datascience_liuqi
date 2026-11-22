from pathlib import Path
import urllib.request
import zipfile
import geopandas as gpd

URL = "https://www.insee.fr/fr/statistiques/fichier/8735162/Filosofi2021_carreaux_200m_gpkg.zip"
DATA = Path("data")
DATA.mkdir(exist_ok=True)
zip_path = DATA / "filosofi2021_200m_gpkg.zip"

# 1. Télécharger une seule fois (~300 Mo)
#if not zip_path.exists():
#    urllib.request.urlretrieve(URL, zip_path)

# 2. Décompresser
#with zipfile.ZipFile(zip_path) as z:
#    print(z.namelist())
#    z.extractall(DATA / "filosofi")

# 4. chargement que de l'Île-de-France 
deps_idf = ["75", "77", "78", "91", "92", "93", "94", "95"]
filtre = " OR ".join(f"lcog_geo LIKE '{d}%'" for d in deps_idf)
carreaux = gpd.read_file(gpkg, where=filtre)
carreaux.columns = carreaux.columns.str.lower()

carreaux["niveau_vie_moyen"] = carreaux["ind_snv"] / carreaux["ind"]
print(carreaux.shape, carreaux.crs)
print(carreaux.head())