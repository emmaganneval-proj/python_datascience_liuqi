import pandas as pd
import matplotlib.pyplot as plt

# %% Constantes
# Même population que la heatmap « Où travaillent à Paris… » :
# actifs qui habitent ET travaillent en Île-de-France, travaillent à Paris
# et y vont en transports en commun (TRANS == "6").
cols = ["COMMUNE", "ARM", "DCLT", "TRANS", "GS", "REGION", "REGLT", "IPONDI"]

noms_gs = {"3": "Cadres", "4": "Prof. intermédiaires",
           "5": "Employés", "6": "Ouvriers"}

noms_dep = {"75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
            "94": "Val-de-Marne", "77": "Seine-et-Marne", "78": "Yvelines",
            "91": "Essonne", "95": "Val-d'Oise"}


# %% Fonctions
def heatmap(ax, tableau, actifs, titre, legende):
    im = ax.imshow(tableau.values, cmap="YlOrRd", aspect="auto")
    # Colonne de droite : population active locale, en milliers
    x = tableau.shape[1] - 0.5 + 0.12
    ax.text(x, -0.62, "Actifs\nrésidents\n(milliers)", ha="left", va="bottom",
            fontsize=9, color="dimgray")
    for i, n in enumerate(actifs.values):
        ax.text(x, i, f"{n / 1000:,.0f}".replace(",", " "),
                ha="left", va="center", color="dimgray")
    vmax = tableau.values.max()
    for i in range(tableau.shape[0]):
        for j in range(tableau.shape[1]):
            v = tableau.values[i, j]
            texte = "<1" if round(v) < 1 else f"{v:.0f}"
            couleur = "white" if v > 0.6 * vmax else "black"   # lisible sur le rouge foncé
            ax.text(j, i, texte, ha="center", va="center", color=couleur)
    ax.set_xticks(range(tableau.shape[1]))
    ax.set_xticklabels(tableau.columns)
    ax.xaxis.tick_top()                                    # groupes en haut de chaque bloc
    ax.set_yticks(range(tableau.shape[0]))
    ax.set_yticklabels(tableau.index)
    ax.set_title(titre, loc="left", fontsize=11, pad=28)
    plt.colorbar(im, ax=ax, label=legende, pad=0.14)


# %% Script
if __name__ == "__main__":
    # Lecture et population étudiée
    df = pd.read_parquet("RP2023_mobpro.parquet", columns=cols)

    idf = df[(df["REGION"] == "11") & (df["REGLT"] == "11")].copy()
    idf["RESID"] = idf["ARM"].where(idf["ARM"] != "ZZZZZ", idf["COMMUNE"])

    pop = idf[(idf["TRANS"] == "6")
              & idf["GS"].isin(noms_gs)
              & idf["DCLT"].str.startswith("751")].copy()

    # Tableau 1 : département de résidence (Paris compté comme un tout)
    pop["DEP"] = pop["RESID"].str[:2]
    par_dep = pop.groupby(["DEP", "GS"])["IPONDI"].sum().unstack("GS")
    par_dep = par_dep / par_dep.sum() * 100           # % de chaque groupe
    par_dep = par_dep.loc[list(noms_dep)]             # Paris, petite puis grande couronne
    par_dep = par_dep.rename(index=noms_dep, columns=noms_gs)

    # Tableau 2 : zoom sur ceux qui habitent Paris, par arrondissement
    parisiens = pop[pop["RESID"].str.startswith("751")]
    par_arr = parisiens.groupby(["RESID", "GS"])["IPONDI"].sum().unstack("GS")
    par_arr = par_arr / par_arr.sum() * 100           # % des Parisiens de chaque groupe
    par_arr = par_arr.sort_index()
    par_arr.index = ["1er"] + [f"{i}e" for i in range(2, 21)]
    par_arr = par_arr.rename(columns=noms_gs)

    print(par_dep.round(1))
    print(par_arr.round(1))

    # Population active locale : actifs en emploi qui habitent chaque zone
    # (tous groupes sociaux, tous modes de transport, quel que soit le lieu de travail)
    res = df[df["REGION"] == "11"].copy()
    res["RESID"] = res["ARM"].where(res["ARM"] != "ZZZZZ", res["COMMUNE"])

    actifs_dep = res.groupby(res["RESID"].str[:2])["IPONDI"].sum()
    actifs_dep = actifs_dep.loc[list(noms_dep)].rename(index=noms_dep)

    actifs_arr = (res[res["RESID"].str.startswith("751")]
                  .groupby("RESID")["IPONDI"].sum().sort_index())
    actifs_arr.index = par_arr.index

    print(actifs_dep.round(-3))
    print(actifs_arr.round(-3))

    # Dessin : deux heatmaps l'une sous l'autre
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 13), gridspec_kw={"height_ratios": [8, 20]})

    heatmap(ax1, par_dep, actifs_dep,
            "Département de résidence",
            "% du groupe")
    heatmap(ax2, par_arr, actifs_arr,
            "Zoom sur ceux qui habitent Paris : arrondissement de résidence",
            "% des Parisiens du groupe")

    fig.suptitle("Où habitent ceux qui travaillent à Paris\n"
                 "et y vont en transports en commun ?", fontsize=13)
    fig.text(0.01, 0.005,
             "Source : Insee, RP 2023, fichier détail MOBPRO. "
             "Pourcentages : actifs habitant et travaillant en Île-de-France.\n"
             "Actifs résidents : tous les actifs en emploi de la zone, "
             "quel que soit leur lieu de travail.",
             fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.025, 1, 1))
    plt.savefig("heatmap_residence_tc.png", dpi=150)
    plt.show()