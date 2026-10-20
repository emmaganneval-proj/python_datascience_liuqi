import pandas as pd


url = ("https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/"
       "datasets/referentiel-des-lignes/exports/csv?delimiter=%3B")

df = pd.read_csv(url, sep=";")
print(df.head())
print(df.columns)
print(df["transportmode"].unique())  # pour voir les valeurs exactes
df_ferre = df[df["transportmode"].isin(["metro", "rail"])]

#ligne14
l14 = df[(df["transportmode"] == "metro") & (df["name_line"] == "14")]
print(l14.T)  # affiche toutes les colonnes de la ligne

