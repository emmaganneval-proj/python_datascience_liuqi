# %% Imports
import matplotlib.pyplot as plt
import pandas as pd

# %% Constantes
NOMS_GS = {"3": "Cadres", "4": "Prof. intermédiaires",
           "5": "Employés", "6": "Ouvriers"}
ORDRE_DEP = ["75", "92", "93", "94", "78", "91", "95", "77"]
NOMS_DEP = {"75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-St-Denis",
            "94": "Val-de-Marne", "78": "Yvelines", "91": "Essonne",
            "95": "Val-d'Oise", "77": "Seine-et-Marne"}


def nom_arr(code):
    """75108 -> '8e' (et 75101 -> '1er')."""
    n = int(code[-2:])
    return "1er" if n == 1 else f"{n}e"


# %% Téléchargement du fichier détail 2023 (à ne faire qu'une fois)
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
idf["DEP"] = idf["RESID"].str[:2]
idf["TC"] = idf["TRANS"] == "6"          # True = transports en commun

total_actifs = f"{idf['IPONDI'].sum():,.0f}".replace(",", " ")
nb_lignes = f"{idf.shape[0]:,}".replace(",", " ")
print(f"En Île-de-France, {total_actifs} actifs habitent et travaillent dans la région.")
print(f"Le tableau contient {nb_lignes} lignes (une ligne = une personne).")

# %% Part des transports en commun selon le groupe social
tc = idf[idf["TC"]].groupby("GS")["IPONDI"].sum()
total = idf.groupby("GS")["IPONDI"].sum()
for gs, nom in NOMS_GS.items():
    print(f"{nom} : {tc[gs] / total[gs]:.0%} prennent les transports en commun.")

# %% Heatmap 1 : part des transports en commun, par groupe social et département
idf_gs = idf[idf["GS"].isin(NOMS_GS)]
tc_pers = idf_gs[idf_gs["TC"]].groupby(["GS", "DEP"])["IPONDI"].sum()
tot_pers = idf_gs.groupby(["GS", "DEP"])["IPONDI"].sum()
part = (tc_pers / tot_pers).unstack("DEP").reindex(columns=ORDRE_DEP).fillna(0) * 100
part = part.rename(index=NOMS_GS, columns=NOMS_DEP)

print("Part des actifs qui prennent les transports en commun (en %) :")
print(part.round(1))

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(part.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(part.shape[1]))
ax.set_xticklabels(part.columns, rotation=30, ha="right")
ax.set_yticks(range(part.shape[0]))
ax.set_yticklabels(part.index)
for i in range(part.shape[0]):
    for j in range(part.shape[1]):
        ax.text(j, i, f"{part.values[i, j]:.0f}%", ha="center", va="center")
fig.colorbar(im, label="% qui prennent les transports en commun")
ax.set_title("Qui prend les transports en commun en Île-de-France ? (2023)")
plt.tight_layout()
plt.savefig("heatmap_tc.png", dpi=150)
plt.show()

# %% Voyageurs en transports en commun, par groupe social et par trajet
flux = (idf[idf["TC"] & idf["GS"].isin(NOMS_GS)]
        .groupby(["GS", "RESID", "DCLT"], as_index=False)["IPONDI"].sum()
        .rename(columns={"IPONDI": "poids"}))

# On enlève ceux qui travaillent dans leur propre commune (trajet quasi nul)
flux = flux[flux["RESID"] != flux["DCLT"]].copy()
print(f"Nombre de trajets différents : {len(flux):,}".replace(",", " "))
print(flux.head())

# %% Heatmap 2 : où vont travailler les gens (département -> département)
flux["DEP_A"] = flux["RESID"].str[:2]
flux["DEP_B"] = flux["DCLT"].str[:2]
par_dep = flux.groupby(["GS", "DEP_A", "DEP_B"], as_index=False)["poids"].sum()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, gs in zip(axes.flat, NOMS_GS):
    tab = (par_dep[par_dep["GS"] == gs]
           .pivot(index="DEP_A", columns="DEP_B", values="poids")
           .reindex(index=ORDRE_DEP, columns=ORDRE_DEP)
           .fillna(0))
    # En % de chaque ligne : sur 100 trajets au départ, combien vont où ?
    pct = tab.div(tab.sum(axis=1), axis=0).fillna(0) * 100

    im = ax.imshow(pct.values, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(ORDRE_DEP)))
    ax.set_xticklabels([NOMS_DEP[d] for d in ORDRE_DEP], rotation=40, ha="right")
    ax.set_yticks(range(len(ORDRE_DEP)))
    ax.set_yticklabels([NOMS_DEP[d] for d in ORDRE_DEP])
    ax.set_xlabel("Où ils travaillent")
    ax.set_ylabel("Où ils habitent")
    ax.set_title(NOMS_GS[gs])
    for i in range(len(ORDRE_DEP)):
        for j in range(len(ORDRE_DEP)):
            v = pct.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 60 else "black")

fig.colorbar(im, ax=axes, label="% des trajets partis du département", shrink=0.7)
fig.suptitle("Où vont travailler ceux qui prennent les transports en commun ?")
plt.savefig("heatmap_flux.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Heatmap 3 : arrondissements de Paris, tous modes contre transports en commun
def part_par_arr(d):
    """Pour chaque groupe social : % qui travaille dans chaque arrondissement."""
    tab = d.groupby(["DCLT", "GS"])["IPONDI"].sum().unstack("GS").fillna(0)
    tab = tab[list(NOMS_GS)]
    pct = tab.div(tab.sum(axis=0), axis=1) * 100
    pct.columns = [NOMS_GS[c] for c in pct.columns]
    pct.index = [nom_arr(c) for c in pct.index]
    return pct


# Tous ceux qui travaillent à Paris, puis seulement ceux en transports en commun
base = idf[idf["DCLT"].str.startswith("751") & idf["GS"].isin(NOMS_GS)]
pct_tout = part_par_arr(base)
pct_tc = part_par_arr(base[base["TC"]])

# Phrases : les 3 arrondissements principaux de chaque groupe
for titre, pct in [("Tous modes", pct_tout), ("Transports en commun", pct_tc)]:
    for nom in pct.columns:
        top = pct[nom].sort_values(ascending=False).head(3)
        liste = ", ".join(f"le {a} ({v:.0f} %)" for a, v in top.items())
        print(f"[{titre}] {nom} : surtout dans {liste}.")

# Dessin : deux graphiques côte à côte, même échelle de couleurs
vmax = max(pct_tout.values.max(), pct_tc.values.max())
fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
for ax, pct, titre in zip(axes, [pct_tout, pct_tc],
                          ["Tous les modes de transport",
                           "Transports en commun seulement"]):
    im = ax.imshow(pct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(pct.shape[1]))
    ax.set_xticklabels(pct.columns, rotation=25, ha="right")
    ax.set_yticks(range(pct.shape[0]))
    ax.set_yticklabels(pct.index)
    ax.set_title(titre)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct.values[i, j]:.0f}", ha="center", va="center",
                    fontsize=8)

axes[0].set_ylabel("Arrondissement du lieu de travail")
fig.colorbar(im, ax=axes, label="% du groupe qui travaille à Paris", shrink=0.6)
fig.suptitle("Dans quel arrondissement travaillent ceux qui vont à Paris ?")
plt.savefig("heatmap_paris_comparaison.png", dpi=150, bbox_inches="tight")
plt.show()