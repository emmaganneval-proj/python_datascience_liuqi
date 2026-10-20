import pandas as pd


url = ("https://www.insee.fr/fr/statistiques/fichier/8582949/"
       "base-flux-mobilite-domicile-lieu-travail-2022_csv.zip")

df = pd.read_csv(url, sep=";", dtype=str,
                 storage_options={"User-Agent": "Mozilla/5.0"})
print(df.head())
print(df.columns)