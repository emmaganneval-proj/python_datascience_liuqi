import pandas as pd
# %% [PARTIE 2] Téléchargement du fichier détail 2023 (à ne faire qu'une fois)
# Le fichier est déjà dans le dossier, téléchargé avec curl.
# Ne décommente que si tu dois le retélécharger (risque de fichier coupé).
#
# import urllib.request
# url = "https://www.insee.fr/fr/statistiques/fichier/9004795/RP2023_mobpro.parquet"
# req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
# with urllib.request.urlopen(req) as r, open("RP2023_mobpro.parquet", "wb") as f:
#     f.write(r.read())

# %% Lecture du fichier détail MOBPRO 2023
cols = ["COMMUNE", "ARM", "DCLT", "TRANS", "GS", "DIPL", "VOIT",
        "REGION", "REGLT", "IPONDI"]
df = pd.read_parquet("RP2023_mobpro.parquet", columns=cols)
print(df.shape)
print(df.dtypes)
print(df.head())

# %% Préparation : poids, Île-de-France, lieu de résidence
# Le poids doit être un nombre
if df["IPONDI"].dtype == "object" or str(df["IPONDI"].dtype) == "str":
    df["IPONDI"] = df["IPONDI"].str.replace(",", ".").astype(float)

# Garder ceux qui habitent ET travaillent en Île-de-France
idf = df[(df["REGION"] == "11") & (df["REGLT"] == "11")].copy()

# Lieu de résidence : l'arrondissement pour Paris, sinon la commune
idf["RESID"] = idf["ARM"].where(idf["ARM"] != "ZZZZZ", idf["COMMUNE"])

print(idf.shape)
print(idf["IPONDI"].sum())   # nombre d'actifs franciliens

# %% Part des transports en commun selon le groupe social
idf["TC"] = (idf["TRANS"] == "6")
tc = idf[idf["TC"]].groupby("GS")["IPONDI"].sum()
total = idf.groupby("GS")["IPONDI"].sum()
print((tc / total).round(3))

total_actifs = idf["IPONDI"].sum()
print(f"En Île-de-France, {total_actifs:,.0f} actifs habitent et travaillent dans la région.".replace(",", " "))
print(f"Le tableau contient {idf.shape[0]:,} lignes (une ligne = une personne).".replace(",", " "))

# %% Heatmap : part des transports en commun, par groupe social et département
import matplotlib.pyplot as plt

noms_gs = {"3": "Cadres", "4": "Prof. intermédiaires",
           "5": "Employés", "6": "Ouvriers"}
noms_dep = {"75": "Paris", "77": "Seine-et-Marne", "78": "Yvelines",
            "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
            "94": "Val-de-Marne", "95": "Val-d'Oise"}

# Département où la personne habite (2 premiers chiffres du code)
idf["DEP"] = idf["RESID"].str[:2]

# On garde les 4 groupes sociaux salariés
idf_gs = idf[idf["GS"].isin(noms_gs)].copy()

# Parmi les gens de chaque case, quelle part prend les transports en commun ?
tc_pers = idf_gs[idf_gs["TC"]].groupby(["GS", "DEP"])["IPONDI"].sum()
tot_pers = idf_gs.groupby(["GS", "DEP"])["IPONDI"].sum()
part = (tc_pers / tot_pers).unstack("DEP").fillna(0) * 100
part = part.rename(index=noms_gs, columns=noms_dep)

print("Part des actifs qui prennent les transports en commun (en %) :")
print(part.round(1))

# Dessin
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(part.values, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(part.shape[1]))
ax.set_xticklabels(part.columns, rotation=30, ha="right")
ax.set_yticks(range(part.shape[0]))
ax.set_yticklabels(part.index)

# Écrire le pourcentage dans chaque case
for i in range(part.shape[0]):
    for j in range(part.shape[1]):
        ax.text(j, i, f"{part.values[i, j]:.0f}%", ha="center", va="center")

fig.colorbar(im, label="% qui prennent les transports en commun")
ax.set_title("Qui prend les transports en commun en Île-de-France ? (2023)")
plt.tight_layout()
plt.savefig("heatmap_tc.png", dpi=150)   # enregistre l'image dans ton dossier
plt.show()
