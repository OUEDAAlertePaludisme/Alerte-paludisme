# =============================================================================
# SCRIPT AUTONOME — ANALYSE PALUDISME BURKINA FASO
# Script indépendant — ne nécessite pas pipeline.py
#
# Modèles : KNN | Random Forest | XGBoost | SARIMA
# Découpage : train <= 2022 | test >= 2023
#
# Sorties :
#   resultats/arima_diagnostics/   → figures et tableau ADF/résidus
#   resultats/figures_modeles/     → courbes et métriques 4 modèles
#   resultats/figures_xgboost/     → analyse approfondie meilleur modèle (dynamique)
#   resultats/resultats_complets.xlsx
#
# Lancement dans Spyder :
#   %runfile 'analyse_complete.py' --wdir
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats as sc_stats
from scipy.stats import shapiro, jarque_bera

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "sant.xlsx"   # adapter si nécessaire

OUT_SARIMA  = "resultats/sarima_diagnostics"
OUT_MODELES = "resultats/figures_modeles"
OUT_XGB     = "resultats/figures_xgboost"
OUT_EXCEL   = "resultats/resultats_complets.xlsx"

TARGET   = 'Cas_palu'
# DS_code supprimé  : doublon exact de DS_encoded (corrélation = 1.0)
# mois supprimé     : redondant avec mois_sin + mois_cos (encodage cyclique)
# saison supprimé   : redondant avec mois_sin + mois_cos (discrétisation grossière)
# annee remplacé    : annee_norm = annee - 2010 évite extrapolation linéaire hors train
# Variables météo du mois t incluses : température, pluies, humidité
# sont disponibles en temps réel via les stations météo nationales
# (ANAM/DGMN Burkina Faso) — ne nécessitent pas d'attendre la fin
# du mois contrairement aux données épidémiologiques.
# Référence : OMS/CDS/RBM/2001.32 — Malaria Early Warning Systems.
FEATURES = ['DS_encoded', 'annee_norm',
            'humid_c', 'pluie_mm', 'temp_c']

COULEURS = {
    'SARIMA'      : '#A23B72',
    'KNN'         : '#2E86AB',
    'RandomForest': '#3BB273',
    'XGBoost'     : '#F18F01',
}
DPI = 180

for d in [OUT_SARIMA, OUT_MODELES, OUT_XGB, "resultats"]:
    Path(d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# FONCTION — FIGURE TEST DE STATIONNARITÉ (ADF) — résumé statistique style article
# Deux panneaux colorés (série originale + différenciée), chacun collé à son
# propre résumé ADF (tableau à trois traits). Sortie nommée par son titre.
# =============================================================================
NOIR      = '#1A1A1A'
ORANGE    = '#F18F01'   # série originale (palette d'origine)
BLEU      = '#2E86AB'   # série différenciée


def _nom_fichier(titre):
    interdits = '\\/:*?"<>|'
    return ''.join('-' if c in interdits else c for c in titre).strip()


def _num(x, dec=4):
    return f"{x:.{dec}f}".replace('-', '\u2212')


def _etoiles(p):
    if   p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''


def _table_adf(ax, adf):
    """Trace un tableau ADF à trois traits dans l'axe `ax` (style article)."""
    ax.axis('off')
    stat_t, pval = float(adf[0]), float(adf[1])
    nlags, nobs  = int(adf[2]), int(adf[3])
    cv           = adf[4]
    stationnaire = pval < 0.05
    concl = "Série stationnaire" if stationnaire else "Série non stationnaire"

    et = _etoiles(pval)
    val_stat = _num(stat_t) + (f"  {et}" if et else "")
    lignes = [
        ["Indicateur", "Valeur"],
        ["Statistique ADF ($t$)", val_stat],
        ["$p$-value (MacKinnon)", _num(pval)],
        ["V. critiques (1/5/10 %)",
         f"{_num(cv['1%'],3)} / {_num(cv['5%'],3)} / {_num(cv['10%'],3)}"],
        ["Conclusion ($\\alpha$ = 5 %)", concl],
    ]
    nrows = len(lignes)
    x0, w = 0.04, 0.92
    y0, h = 0.18, 0.74
    tbl = ax.table(cellText=lignes, cellLoc='left', loc='center',
                   colWidths=[0.58, 0.42], bbox=[x0, y0, w, h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('white')
        cell.set_edgecolor('none')
        cell.set_linewidth(0)
        cell.PAD = 0.03
        if r == 0 or r == nrows - 1:
            cell.get_text().set_fontweight('bold')

    rowh = h / nrows
    for yy, lw in [(y0 + h, 1.5), (y0 + h - rowh, 0.7),
                   (y0 + rowh, 0.7), (y0, 1.5)]:
        ax.plot([x0, x0 + w], [yy, yy], color=NOIR, lw=lw,
                clip_on=False, zorder=5, transform=ax.transAxes)

    ax.text(x0, y0 - 0.10,
            "Note. H\u2080 : racine unitaire (non stationnarité). "
            "*** $p$<0,01 ; ** $p$<0,05 ; * $p$<0,10.",
            transform=ax.transAxes, fontsize=8, style='italic', va='top')


def figure_stationnarite(serie, adf_res, adf_stat, adf_pval, stat, d, ds,
                         out_dir, dpi=180, titres_panneaux=True):
    has_diff = d > 0
    ncol = 2 if has_diff else 1

    fig = plt.figure(figsize=(7.0 * ncol, 8.6))
    # 2 lignes : graphiques (haut) puis résumés collés en dessous
    gs = gridspec.GridSpec(2, ncol, height_ratios=[2.7, 2.3],
                           hspace=0.16, wspace=0.18)

    # --- Colonne 1 : série originale + résumé ADF (niveau) -------------------
    ax_g0 = fig.add_subplot(gs[0, 0])
    ax_g0.plot(serie.index, serie.values, color=ORANGE, lw=1.3)
    ax_g0.set_ylabel("Cas de paludisme"); ax_g0.set_xlabel("Année")
    ax_g0.grid(alpha=0.18, lw=0.6)
    ax_g0.spines[['top', 'right']].set_visible(False)
    ax_g0.tick_params(labelsize=9)
    if titres_panneaux:
        prefixe_a = "(a) " if has_diff else ""
        ax_g0.set_title(f"{prefixe_a}Série originale", fontsize=11, loc='left')

    ax_t0 = fig.add_subplot(gs[1, 0])
    _table_adf(ax_t0, adf_res)   # ADF sur la série en niveau

    # --- Colonne 2 : série différenciée + résumé ADF (différence) -----------
    if has_diff:
        sd = serie.diff(d).dropna()
        adf_sd = adfuller(sd, autolag='AIC')

        ax_g1 = fig.add_subplot(gs[0, 1])
        ax_g1.plot(sd.index, sd.values, color=BLEU, lw=1.1)
        ax_g1.axhline(0, color='#555555', lw=0.7, ls='--')
        ax_g1.set_ylabel("Différence première"); ax_g1.set_xlabel("Année")
        ax_g1.grid(alpha=0.18, lw=0.6)
        ax_g1.spines[['top', 'right']].set_visible(False)
        ax_g1.tick_params(labelsize=9)
        if titres_panneaux:
            ax_g1.set_title(f"(b) Série différenciée (d = {d})",
                            fontsize=11, loc='left')

        ax_t1 = fig.add_subplot(gs[1, 1])
        _table_adf(ax_t1, adf_sd)   # ADF sur la série différenciée

    titre = f"Test de stationnarité ADF - {ds}"
    fichier = _nom_fichier(titre) + ".png"
    fig.savefig(f"{out_dir}/{fichier}", dpi=dpi,
                bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)
    return fichier


def _table_residus(ax, sw_stat, sw_pval, jb_stat, jb_pval, lb_stat, lb_pval):
    """Tableau de validation des résidus (style article, trois traits)."""
    ax.axis('off')
    cn = lambda p: "Résidus normaux" if p > 0.05 else "Résidus non normaux"
    cac = "Absence d'autocorrélation" if lb_pval > 0.05 else "Autocorrélation présente"

    def vp(p):
        e = _etoiles(p)
        return _num(p) + (f"  {e}" if e else "")

    lignes = [
        ["Test (hypothèse H\u2080)", "Statistique", "$p$-value", "Décision (\u03b1 = 5 %)"],
        ["Shapiro\u2013Wilk (normalité)",       f"W = {_num(sw_stat)}", vp(sw_pval), cn(sw_pval)],
        ["Jarque\u2013Bera (normalité)",        _num(jb_stat),          vp(jb_pval), cn(jb_pval)],
        ["Ljung\u2013Box, 12 retards (indép.)", f"Q = {_num(lb_stat)}", vp(lb_pval), cac],
    ]
    nrows = len(lignes)
    x0, w = 0.04, 0.92
    y0, h = 0.34, 0.56
    tbl = ax.table(cellText=lignes, cellLoc='left', loc='center',
                   colWidths=[0.34, 0.18, 0.16, 0.32], bbox=[x0, y0, w, h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('white'); cell.set_edgecolor('none'); cell.set_linewidth(0)
        cell.PAD = 0.03
        if r == 0:
            cell.get_text().set_fontweight('bold')

    rowh = h / nrows
    for yy, lw in [(y0 + h, 1.5), (y0 + h - rowh, 0.7), (y0, 1.5)]:
        ax.plot([x0, x0 + w], [yy, yy], color=NOIR, lw=lw,
                clip_on=False, zorder=5, transform=ax.transAxes)

    ax.text(x0, y0 - 0.07,
            "Note. H\u2080 : normalité (Shapiro\u2013Wilk, Jarque\u2013Bera) ; "
            "absence d'autocorrélation (Ljung\u2013Box). "
            "Un $p$ > 0,05 indique le non-rejet de H\u2080 (résultat recherché). "
            "*** $p$<0,01 ; ** $p$<0,05 ; * $p$<0,10.",
            transform=ax.transAxes, fontsize=8, style='italic', va='top')


def figure_residus(resid, sw_stat, sw_pval, jb_stat, jb_pval, lb_stat, lb_pval,
                   ds, out_dir, dpi=180, titres_panneaux=True):
    fig = plt.figure(figsize=(17, 8.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.6, 1.5], hspace=0.30, wspace=0.24)

    # (a) Résidus dans le temps
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(resid.index, resid.values, color='#E84855', linewidth=1)
    ax0.axhline(0, color='black', lw=1, ls='--')
    ax0.fill_between(resid.index, resid.values, 0, where=resid.values > 0,
                     alpha=0.2, color='#E84855')
    ax0.fill_between(resid.index, resid.values, 0, where=resid.values < 0,
                     alpha=0.2, color='#2E86AB')
    ax0.set_ylabel("Résidu"); ax0.set_xlabel("Année"); ax0.grid(alpha=0.2)
    ax0.spines[['top', 'right']].set_visible(False)
    if titres_panneaux:
        ax0.set_title("(a) Résidus dans le temps", fontsize=11, loc='left')

    # (b) QQ-plot
    ax1 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = sc_stats.probplot(resid, dist='norm')
    ax1.scatter(osm, osr, color='#A23B72', s=15, alpha=0.6)
    ax1.plot(osm, slope * np.array(osm) + intercept, 'k--', lw=1.5)
    ax1.set_xlabel("Quantiles théoriques"); ax1.set_ylabel("Quantiles observés")
    ax1.grid(alpha=0.2); ax1.spines[['top', 'right']].set_visible(False)
    if titres_panneaux:
        ax1.set_title("(b) QQ-plot des résidus", fontsize=11, loc='left')

    # (c) Histogramme
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(resid.values, bins=30, color='#A23B72', alpha=0.7,
             edgecolor='white', density=True)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax2.plot(xs, sc_stats.norm.pdf(xs, resid.mean(), resid.std()),
             color='#2C3E50', lw=2, label='Loi normale théorique')
    ax2.set_xlabel("Résidu"); ax2.set_ylabel("Densité")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
    ax2.spines[['top', 'right']].set_visible(False)
    if titres_panneaux:
        ax2.set_title("(c) Distribution des résidus", fontsize=11, loc='left')

    # Tableau récapitulatif (toute la largeur)
    ax_tab = fig.add_subplot(gs[1, :])
    _table_residus(ax_tab, sw_stat, sw_pval, jb_stat, jb_pval, lb_stat, lb_pval)

    titre = f"Validation des résidus - {ds}"
    fichier = _nom_fichier(titre) + ".png"
    fig.savefig(f"{out_dir}/{fichier}", dpi=dpi, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)
    return fichier


# =============================================================================
# FONCTIONS — TESTS STATISTIQUES DE COMPARAISON DES MODÈLES
# Test de Friedman (omnibus) + post-hoc de Nemenyi (Demšar, 2006).
# Produit dans OUT_MODELES : D1 résumé, D2 Friedman, D3 rangs moyens,
# D4 matrice Nemenyi, D5 heatmap, D6 diagramme de différence critique,
# + classeur Excel (repli CSV si Excel indisponible).
# =============================================================================
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def _note_bas(fig, tbl, texte, fontsize=8.5):
    """Place une note italique juste sous un tableau matplotlib (style article)."""
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = tbl.get_window_extent(r).transformed(fig.transFigure.inverted())
        y = bb.y0 - 0.03
    except Exception:
        y = 0.04
    fig.text(0.5, y, texte, ha='center', va='top', fontsize=fontsize, style='italic')

def etoiles(p):
    """Convention d'astérisques de significativité (standard publications)."""
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _rmse(yt, yp):
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))


def _mae(yt, yp):
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 1:
        return np.nan
    return float(np.mean(np.abs(yt[m] - yp[m])))


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE
# -----------------------------------------------------------------------------
def analyse_friedman_nemenyi(merged, modeles, target, out_dir,
                             couleurs=None, dpi=180, metrique='rmse',
                             alpha=0.05):
    """
    Paramètres
    ----------
    merged   : DataFrame contenant une ligne par observation district×mois,
               avec la colonne cible `target` et une colonne de prédiction
               par modèle.
    modeles  : dict {nom_affiché : nom_colonne_prédiction}, ex.
               {'SARIMA':'Pred_ARIMA','KNN':'Pred_KNN',
                'RandomForest':'Pred_RF','XGBoost':'Pred_XGB'}
    target   : nom de la colonne des valeurs réelles.
    out_dir  : dossier de sortie des figures.
    metrique : 'rmse' (défaut) ou 'mae' — mesure de performance par bloc.

    Retour
    ------
    dict avec les tables (DataFrames) et les statistiques.
    """
    if couleurs is None:
        couleurs = {}
    fmes = _rmse if metrique.lower() == 'rmse' else _mae
    metr_label = metrique.upper()
    noms = list(modeles.keys())
    k = len(noms)

    # -- 1. Matrice d'erreur par district (blocs × modèles) -------------------
    lignes = []
    for ds, g in merged.groupby('Nom_DS'):
        ligne = {'District': ds}
        ok = True
        for nom, col in modeles.items():
            val = fmes(g[target].values, g[col].values)
            ligne[nom] = val
            if not np.isfinite(val):
                ok = False
        if ok:
            lignes.append(ligne)

    err_df = pd.DataFrame(lignes).set_index('District')
    N = len(err_df)
    if N < 3:
        raise ValueError(f"Pas assez de districts complets (N={N}) pour Friedman.")

    M = err_df[noms].values  # (N districts, k modèles)

    # -- 2. Rangs par district (1 = meilleur = erreur la plus faible) ---------
    rangs = np.apply_along_axis(rankdata, 1, M)         # (N, k)
    rangs_moyens = rangs.mean(axis=0)                    # (k,)
    rangs_sd     = rangs.std(axis=0, ddof=1)
    rang_df = (pd.DataFrame({'Modele': noms,
                             'Rang_moyen': rangs_moyens,
                             'Rang_ecart_type': rangs_sd})
               .sort_values('Rang_moyen').reset_index(drop=True))
    rang_df.insert(0, 'Rang', range(1, k + 1))
    meilleur = rang_df.iloc[0]['Modele']

    # -- 3. Test de Friedman + Iman-Davenport + Kendall W ---------------------
    chi2, p_fried = friedmanchisquare(*[M[:, j] for j in range(k)])
    ddl_fried = k - 1
    # Iman-Davenport (correction F, recommandée par Demšar 2006)
    denom = (N * (k - 1) - chi2)
    F_id = ((N - 1) * chi2 / denom) if denom > 0 else np.inf
    ddl1, ddl2 = k - 1, (k - 1) * (N - 1)
    from scipy.stats import f as f_dist
    p_id = float(f_dist.sf(F_id, ddl1, ddl2)) if np.isfinite(F_id) else 0.0
    kendall_w = chi2 / (N * (k - 1))          # taille d'effet (0-1)

    # -- 4. Post-hoc de Nemenyi : matrice de p-values + CD --------------------
    SE = np.sqrt(k * (k + 1) / (6.0 * N))
    P = np.ones((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            d = abs(rangs_moyens[i] - rangs_moyens[j])
            q = d / SE
            P[i, j] = float(studentized_range.sf(q * np.sqrt(2), k, np.inf))
    pmat_df = pd.DataFrame(P, index=noms, columns=noms)
    # Différence critique (diagramme de Demšar)
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    CD = q_alpha * SE

    # ordre par rang moyen pour l'affichage
    ordre = rang_df['Modele'].tolist()
    rmoy = dict(zip(noms, rangs_moyens))

    # significativité de chaque modèle VS le meilleur (pour les étoiles)
    p_vs_best = {m: (pmat_df.loc[meilleur, m] if m != meilleur else np.nan)
                 for m in noms}

    # =========================================================================
    # AFFICHAGE CONSOLE
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  TESTS STATISTIQUES DE COMPARAISON — métrique de bloc : {metr_label}")
    print(f"  Blocs = {N} districts | Traitements = {k} modèles | α = {alpha}")
    print("=" * 70)

    print("\n  ── Rangs moyens (1 = meilleur) ──")
    for _, r in rang_df.iterrows():
        et = etoiles(p_vs_best[r['Modele']]) if r['Modele'] != meilleur else '(réf.)'
        print(f"    {int(r['Rang'])}. {r['Modele']:<14s} "
              f"rang moyen = {r['Rang_moyen']:.3f} ± {r['Rang_ecart_type']:.3f}   {et}")

    print("\n  ── Test de Friedman (omnibus) ──")
    print(f"    χ²(ddl={ddl_fried}) = {chi2:.4f}   p = {p_fried:.3e} {etoiles(p_fried)}")
    print(f"    Iman-Davenport F({ddl1},{ddl2}) = {F_id:.4f}   "
          f"p = {p_id:.3e} {etoiles(p_id)}")
    print(f"    Kendall's W (taille d'effet) = {kendall_w:.4f}")
    concl = ("Différences significatives entre modèles → post-hoc justifié"
             if p_fried < alpha else
             "Pas de différence significative globale")
    print(f"    Conclusion : {concl}")

    print("\n  ── Post-hoc de Nemenyi (matrice de p-values) ──")
    print(f"    Différence critique CD = {CD:.4f} (rangs)")
    aff = pmat_df.loc[ordre, ordre].copy()
    print(aff.round(4).to_string())

    # =========================================================================
    # FIGURES
    # =========================================================================

    # ---- D1. Tableau résumé statistique par modèle (avec étoiles) -----------
    res_rows = []
    for m in ordre:
        col = err_df[m]
        res_rows.append([
            m,
            f"{col.mean():.3f} ± {col.std(ddof=1):.3f}",
            f"{rmoy[m]:.3f}",
            ('réf.' if m == meilleur
             else f"{p_vs_best[m]:.4f} {etoiles(p_vs_best[m])}")
        ])
    resume_df = pd.DataFrame(
        res_rows,
        columns=[f'Modèle', f'{metr_label} moyen ± σ (inter-districts)',
                 'Rang moyen', 'p (Nemenyi vs meilleur)'])

    fig, ax = plt.subplots(figsize=(13, 0.62 * (k + 1) + 0.7))
    ax.axis('off')
    tbl = ax.table(cellText=resume_df.values, colLabels=resume_df.columns,
                   cellLoc='center', loc='upper center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            m = ordre[r - 1]
            if m == meilleur:
                cell.set_facecolor('#D5F5E3')
            elif c == 3 and p_vs_best[m] < alpha:
                cell.set_facecolor('#FADBD8')
            else:
                cell.set_facecolor('#F8F9F9')
    plt.tight_layout()
    _note_bas(fig, tbl,
        f"Mesure : {metr_label} par district ; N = {N} districts ; "
        f"meilleur modèle (vert) = {meilleur}. "
        "*** p<0,001 ; ** p<0,01 ; * p<0,05 ; ns = non significatif.")
    plt.savefig(f"{out_dir}/D1_resume_statistique_modeles.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("\n  ✓ D1_resume_statistique_modeles.png")

    # ---- D2. Tableau du test de Friedman ------------------------------------
    fried_rows = [
        ['Statistique de Friedman (χ²)', f"{chi2:.4f}",
         f"ddl = {ddl_fried}", f"p = {p_fried:.3e} {etoiles(p_fried)}"],
        ['Iman-Davenport (F corrigé)', f"{F_id:.4f}",
         f"ddl = ({ddl1}, {ddl2})", f"p = {p_id:.3e} {etoiles(p_id)}"],
        ['Kendall W (taille d\'effet)', f"{kendall_w:.4f}",
         _interp_kendall(kendall_w), '—'],
        ['Nombre de blocs (districts)', f"{N}", '—', '—'],
        ['Nombre de modèles comparés', f"{k}", '—', '—'],
        ['Différence critique (Nemenyi)', f"{CD:.4f}",
         f"α = {alpha}", '—'],
    ]
    fried_df = pd.DataFrame(
        fried_rows, columns=['Indicateur', 'Valeur', 'Paramètre', 'Décision'])

    fig, ax = plt.subplots(figsize=(13, 0.62 * len(fried_rows) + 0.7))
    ax.axis('off')
    tbl = ax.table(cellText=fried_df.values, colLabels=fried_df.columns,
                   cellLoc='center', loc='upper center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r in (1, 2):
            cell.set_facecolor('#D5F5E3' if p_fried < alpha else '#FADBD8')
        elif r == 3:
            cell.set_facecolor('#EBF5FB')
        else:
            cell.set_facecolor('#F8F9F9')
    verdict = ("H₀ REJETÉE — au moins deux modèles diffèrent significativement"
               if p_fried < alpha else
               "H₀ non rejetée — aucune différence significative")
    plt.tight_layout()
    _note_bas(fig, tbl,
        "H\u2080 : tous les modèles ont le même rang moyen (performances équivalentes). "
        f"{verdict}.")
    plt.savefig(f"{out_dir}/D2_test_friedman.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  ✓ D2_test_friedman.png")

    # ---- D3. Tableau des rangs moyens ---------------------------------------
    rang_aff = rang_df.copy()
    rang_aff['Rang_moyen'] = rang_aff['Rang_moyen'].round(3)
    rang_aff['Rang_ecart_type'] = rang_aff['Rang_ecart_type'].round(3)
    rang_aff['Signif. vs meilleur'] = rang_aff['Modele'].map(
        lambda m: 'réf.' if m == meilleur
        else f"{etoiles(p_vs_best[m])} (p={p_vs_best[m]:.4f})")
    rang_aff.columns = ['Rang', 'Modèle', 'Rang moyen', 'Écart-type',
                        'Signif. vs meilleur']

    fig, ax = plt.subplots(figsize=(12, 0.62 * (k + 1) + 0.7))
    ax.axis('off')
    tbl = ax.table(cellText=rang_aff.values, colLabels=rang_aff.columns,
                   cellLoc='center', loc='upper center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            m = rang_aff.iloc[r - 1]['Modèle']
            cell.set_facecolor('#D5F5E3' if m == meilleur else '#F8F9F9')
    plt.tight_layout()
    _note_bas(fig, tbl,
        "Rang 1 = meilleur (RMSE la plus faible par district). "
        "Significativité (post-hoc Nemenyi) : *** p<0,001 ; ** p<0,01 ; * p<0,05 ; ns.")
    plt.savefig(f"{out_dir}/D3_rangs_moyens.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  ✓ D3_rangs_moyens.png")

    # ---- D4. Matrice des p-values de Nemenyi (tableau à étoiles) ------------
    aff = pmat_df.loc[ordre, ordre]
    cell_txt = []
    for i, mi in enumerate(ordre):
        row = []
        for j, mj in enumerate(ordre):
            if mi == mj:
                row.append('—')
            else:
                pv = aff.loc[mi, mj]
                row.append(f"{pv:.4f}\n{etoiles(pv)}")
        cell_txt.append(row)

    fig, ax = plt.subplots(figsize=(2.1 * k + 2, 2.1 * k * 0.5 + 0.9))
    ax.axis('off')
    tbl = ax.table(cellText=cell_txt, rowLabels=ordre, colLabels=ordre,
                   cellLoc='center', loc='upper center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        if r == 0 or c == -1:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r - 1 == c:
            cell.set_facecolor('#D5D8DC')
        else:
            mi = ordre[r - 1]; mj = ordre[c]
            pv = aff.loc[mi, mj]
            cell.set_facecolor('#D5F5E3' if pv < alpha else '#FADBD8')
    plt.tight_layout()
    _note_bas(fig, tbl,
        "Vert = différence significative (p<0,05) ; rouge = non significatif. "
        "*** p<0,001 ; ** p<0,01 ; * p<0,05 ; ns = non significatif.")
    plt.savefig(f"{out_dir}/D4_nemenyi_pvalues.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  ✓ D4_nemenyi_pvalues.png")

    # ---- D5. Heatmap de Nemenyi ---------------------------------------------
    fig, ax = plt.subplots(figsize=(1.4 * k + 3, 1.4 * k + 2))
    Pdisp = aff.values.copy()
    im = ax.imshow(Pdisp, cmap='RdYlGn_r', vmin=0, vmax=0.10, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('p-value (Nemenyi)', fontsize=9)
    ax.set_xticks(range(k)); ax.set_xticklabels(ordre, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(k)); ax.set_yticklabels(ordre, fontsize=9)
    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=11)
            else:
                pv = Pdisp[i, j]
                ax.text(j, i, f"{pv:.3f}\n{etoiles(pv)}",
                        ha='center', va='center', fontsize=8.5,
                        color='black', fontweight='bold')
    ax.text(0.5, -0.20,
            "Vert = significatif (p<0,05) ; rouge = non significatif.",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5, style='italic')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/D5_nemenyi_heatmap.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  ✓ D5_nemenyi_heatmap.png")

    # ---- D6. Diagramme de différence critique (Demšar) ----------------------
    _diagramme_cd(ordre, rmoy, CD, k, N, alpha, couleurs, out_dir, dpi)
    print("  ✓ D6_critical_difference.png")

    return {
        'erreurs_par_district': err_df,
        'rangs_moyens': rang_df,
        'friedman': {'chi2': chi2, 'ddl': ddl_fried, 'p': p_fried,
                     'iman_davenport_F': F_id, 'ddl1': ddl1, 'ddl2': ddl2,
                     'p_id': p_id, 'kendall_w': kendall_w},
        'nemenyi_pvalues': pmat_df,
        'CD': CD, 'meilleur': meilleur, 'resume': resume_df,
    }


# -----------------------------------------------------------------------------
# WRAPPER « CLÉ EN MAIN » — à appeler directement depuis le script principal
# -----------------------------------------------------------------------------
def construire_merged(test, pred_knn, pred_rf, pred_xgb,
                      arima_df_test, target):
    """Reconstitue un DataFrame unique aligné sur les mêmes observations
    district×mois pour les 4 modèles (intersection ML ∩ SARIMA)."""
    ml = test[['Nom_DS', 'annee', 'mois', target]].copy()
    ml['Pred_KNN'] = np.asarray(pred_knn)
    ml['Pred_RF']  = np.asarray(pred_rf)
    ml['Pred_XGB'] = np.asarray(pred_xgb)
    merged = ml.merge(
        arima_df_test[['Nom_DS', 'annee', 'mois', 'Pred_ARIMA']],
        on=['Nom_DS', 'annee', 'mois'], how='inner')
    return merged


def executer_tests_comparaison(test, pred_knn, pred_rf, pred_xgb,
                               arima_df_test, target, out_dir,
                               couleurs=None, dpi=180,
                               metrique='rmse', alpha=0.05,
                               excel_path=None):
    """Point d'entrée unique. Construit les données, lance Friedman + Nemenyi,
    génère les 6 figures et, si demandé, exporte les tables dans un Excel."""
    merged = construire_merged(test, pred_knn, pred_rf, pred_xgb,
                               arima_df_test, target)
    modeles = {'SARIMA': 'Pred_ARIMA', 'KNN': 'Pred_KNN',
               'RandomForest': 'Pred_RF', 'XGBoost': 'Pred_XGB'}
    res = analyse_friedman_nemenyi(merged, modeles, target, out_dir,
                                   couleurs=couleurs, dpi=dpi,
                                   metrique=metrique, alpha=alpha)

    if excel_path:
        fr = res['friedman']
        fried_tab = pd.DataFrame({
            'Indicateur': ['Friedman chi2', 'ddl', 'p-value Friedman',
                           'Iman-Davenport F', 'ddl1', 'ddl2',
                           'p-value Iman-Davenport', 'Kendall W',
                           'Différence critique (CD)', 'N districts'],
            'Valeur': [fr['chi2'], fr['ddl'], fr['p'], fr['iman_davenport_F'],
                       fr['ddl1'], fr['ddl2'], fr['p_id'], fr['kendall_w'],
                       res['CD'], len(res['erreurs_par_district'])]})
        tables = {
            'Resume_statistique': res['resume'],
            'Rangs_moyens': res['rangs_moyens'],
            'Test_Friedman': fried_tab,
            'Nemenyi_pvalues': res['nemenyi_pvalues'],
            'RMSE_par_district': res['erreurs_par_district'],
        }
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as xls:
                for nom, tab in tables.items():
                    idx = nom in ('Nemenyi_pvalues', 'RMSE_par_district')
                    tab.to_excel(xls, nom, index=idx)
            print(f"  ✓ Tables exportées : {excel_path}")
        except Exception as e:
            # Repli CSV (robustesse multi-environnements)
            import os
            base = os.path.splitext(excel_path)[0]
            for nom, tab in tables.items():
                idx = nom in ('Nemenyi_pvalues', 'RMSE_par_district')
                tab.to_csv(f"{base}_{nom}.csv", index=idx,
                           sep=';', encoding='utf-8-sig')
            print(f"  ⚠ Excel indisponible ({type(e).__name__}) → export CSV : "
                  f"{base}_*.csv")
    return res


def _interp_kendall(w):
    if w < 0.1:
        return 'effet négligeable'
    if w < 0.3:
        return 'effet faible'
    if w < 0.5:
        return 'effet modéré'
    return 'effet fort'


def _diagramme_cd(ordre, rmoy, CD, k, N, alpha, couleurs, out_dir, dpi):
    """Diagramme de différence critique (Demšar 2006).
    Orientation standard : meilleur rang (1) à GAUCHE, pire à droite.
    Les étiquettes du meilleur moitié partent vers la gauche, l'autre vers la
    droite — ce qui évite tout croisement de lignes."""
    lo, hi = 1, k
    # ordre est déjà trié du meilleur (rang faible) au pire (rang élevé)
    moitie = (k + 1) // 2

    fig, ax = plt.subplots(figsize=(11, 2.8 + 0.30 * k))
    ax.set_xlim(lo - 0.6, hi + 0.6)        # rang 1 à gauche (non inversé)
    ax.set_ylim(0, 1)
    ax.axis('off')

    y_axe = 0.80
    ax.plot([lo, hi], [y_axe, y_axe], 'k-', lw=1.5)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y_axe, y_axe + 0.03], 'k-', lw=1.2)
        ax.text(t, y_axe + 0.065, str(t), ha='center', fontsize=9)
    ax.text((lo + hi) / 2, y_axe + 0.14, "Rang moyen (1 = meilleur)",
            ha='center', fontsize=9.5, fontweight='bold')

    x_gauche = lo - 0.55
    x_droite = hi + 0.55
    for i, m in enumerate(ordre):
        r = rmoy[m]
        col = couleurs.get(m, '#2C3E50')
        if i < moitie:                       # meilleurs → étiquette à gauche
            y_lab = y_axe - 0.13 - i * 0.135
            xlab, ha, dx = x_gauche, 'right', -0.04
        else:                                # pires → étiquette à droite
            j = i - moitie
            y_lab = y_axe - 0.13 - j * 0.135
            xlab, ha, dx = x_droite, 'left', 0.04
        ax.plot([r, r], [y_axe, y_lab], color=col, lw=1.4)
        ax.plot([r, xlab], [y_lab, y_lab], color=col, lw=1.4)
        ax.text(xlab + dx, y_lab, f"{m} ({r:.2f})",
                ha=ha, va='center', fontsize=10.5, color=col, fontweight='bold')

    # regroupement des modèles NON significativement différents (cliques)
    rgs = [rmoy[m] for m in ordre]           # croissant
    y_grp = y_axe - 0.045
    used = [False] * k
    dec = 0
    for i in range(k):
        if used[i]:
            continue
        j = i
        while j + 1 < k and (rgs[j + 1] - rgs[i]) <= CD:
            j += 1
        if j > i:
            yy = y_grp - dec * 0.035
            ax.plot([rgs[i] - 0.05, rgs[j] + 0.05], [yy, yy],
                    color='#7F8C8D', lw=4.5, solid_capstyle='round')
            for t in range(i, j + 1):
                used[t] = True
            dec += 1

    # barre de la différence critique (échelle de référence)
    y_cd = 0.16
    x0 = lo
    ax.plot([x0, x0 + CD], [y_cd, y_cd], 'k-', lw=3)
    ax.plot([x0, x0], [y_cd - 0.025, y_cd + 0.025], 'k-', lw=1.5)
    ax.plot([x0 + CD, x0 + CD], [y_cd - 0.025, y_cd + 0.025], 'k-', lw=1.5)
    ax.text(x0 + CD / 2, y_cd - 0.075, f"CD = {CD:.3f}",
            ha='center', fontsize=9.5, fontweight='bold')

    ax.text(0.5, 0.03,
            "Modèles reliés par une barre grise = différence NON significative. "
            "Test de Friedman sur %d districts × %d modèles (\u03b1=%.2f)." % (N, k, alpha),
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5, style='italic')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/D6_critical_difference.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()



print("=" * 65)
print("  ANALYSE PALUDISME BURKINA FASO — SCRIPT AUTONOME")
print("=" * 65)

# =============================================================================
# ÉTAPE 1 — CHARGEMENT & FEATURE ENGINEERING
# =============================================================================
print("\n[1/6] Chargement des données...")

if not os.path.exists(DATA_PATH):
    print(f"❌ Fichier introuvable : {DATA_PATH}")
    print(f"   Répertoire courant : {os.getcwd()}")
    sys.exit(1)

le = LabelEncoder()
df = pd.read_excel(DATA_PATH, sheet_name="ML_Dataset", engine="openpyxl")

corrections = {
    "DS Batié": "DS Batie", "DS Boussé": "DS Bousse", "DS Réo": "DS Reo",
    "DS Diébougou": "DS Diebougou", "DS Koupéla": "DS Koupela",
    "DS Léo": "DS Leo", "DS Pô": "DS Po", "DS Saponé": "DS Sapone",
    "DS Séguenega": "DS Seguenega", "DS Zabré": "DS Zabre",
    "DS Ziniare": "DS Ziniare"
}
df["Nom_DS"] = df["Nom_DS"].replace(corrections)
df['DS_encoded'] = le.fit_transform(df['Nom_DS'])

# Encodage numérique du district
ds_list = sorted(df['Nom_DS'].unique())
ds_map  = {ds: i for i, ds in enumerate(ds_list)}
df['DS_code'] = df['Nom_DS'].map(ds_map)

df = df.sort_values(['Nom_DS', 'annee', 'mois']).reset_index(drop=True)

# Cas_palu est directement l'incidence mensuelle brute — aucun calcul population nécessaire

# Variables lagguées climatiques et cible
for col in ['humid_c', 'pluie_mm', 'temp_c']:
    df[f'{col}_lag1'] = df.groupby('Nom_DS')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('Nom_DS')[col].shift(2)
for lag in [1, 2]:
    df[f'Cas_palu_lag{lag}'] = df.groupby('Nom_DS')[TARGET].shift(lag)

# Encodage cyclique du mois (remplace mois brut + saison)
df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)

# annee_norm : centré sur 2010 — évite extrapolation linéaire hors plage train
df['annee_norm'] = df['annee'] - 2010

# trend_3m, trend_6m supprimés : redondants avec Cas_palu_lag1/lag2,
# divisent l'importance sans apporter d'information nouvelle.
# delta_lag1 supprimé : combinaison linéaire exacte de lag1-lag2,
# déjà présents — triple comptage du même signal autorégressif.
# lag3_x_saison supprimé : corrélation globale lag3 = -0.067,
# interaction insuffisante pour compenser le bruit ajouté.

cols_obligatoires = ['humid_c_lag1', 'humid_c_lag2',
                     'pluie_mm_lag1', 'pluie_mm_lag2',
                     'temp_c_lag1',   'temp_c_lag2',
                     'Cas_palu_lag1', 'Cas_palu_lag2']
df = df.dropna(subset=cols_obligatoires).reset_index(drop=True)

print(f"  ✓ {df.shape[0]:,} observations | "
      f"{df['Nom_DS'].nunique()} districts | "
      f"{df['annee'].min()}–{df['annee'].max()}")

# Top 5 districts (pour figures détaillées)
top5 = (df.groupby('Nom_DS')[TARGET].mean()
          .nlargest(5).index.tolist())

# =============================================================================
# ÉTAPE 2 — DIAGNOSTIC SARIMA
# =============================================================================
print("\n[2/6] Diagnostic SARIMA (ADF, ACF/PACF, ordres saisonniers, résidus)...")

S = 12  # Période saisonnière mensuelle

def get_series(data, ds):
    # .mean() et non .sum() : Cas_palu est une incidence (0-1),
    # additionner des incidences n'a pas de sens épidémiologique.
    # En pratique une seule ligne par district×mois après dropna,
    # mais .mean() est robuste à d'éventuels doublons résiduels.
    sub = (data[data['Nom_DS'] == ds]
           .groupby(['annee', 'mois'])[TARGET].mean().reset_index())
    sub['date'] = pd.to_datetime(
        sub[['annee', 'mois']].assign(day=1).rename(
            columns={'annee': 'year', 'mois': 'month'}))
    return sub.sort_values('date').set_index('date')[TARGET]



diag_rows = []

for ds in ds_list:
    serie = get_series(df, ds)
    if len(serie) < 24 or serie.std() == 0:
        continue

    # Test ADF
    adf_res  = adfuller(serie.dropna(), autolag='AIC')
    adf_stat = round(adf_res[0], 4)
    adf_pval = round(adf_res[1], 4)
    stat     = adf_pval < 0.05

    # Détermination de d
    if stat:
        d = 0; serie_diff = serie.copy()
    else:
        d = 1; serie_diff = serie.diff().dropna()
        if adfuller(serie_diff.dropna(), autolag='AIC')[1] >= 0.05:
            d = 2; serie_diff = serie_diff.diff().dropna()

    # Identification D saisonnier
    serie_sdiff = serie_diff.copy()
    if len(serie_sdiff) > S:
        adf_s = adfuller(serie_sdiff.dropna(), autolag='AIC')
        D = 1 if adf_s[1] >= 0.05 else 0
        if D == 1:
            serie_sdiff = serie_sdiff.diff(S).dropna()
    else:
        D = 0

    # Identification p, q, P, Q via ACF/PACF
    max_lag   = min(24, len(serie_sdiff) // 3)
    acf_vals  = acf(serie_sdiff,  nlags=max_lag, fft=True)
    pacf_vals = pacf(serie_sdiff, nlags=max_lag)
    seuil_sig = 2 / np.sqrt(len(serie_sdiff))

    p = min(max((k for k in range(1, min(4, len(pacf_vals)))
                 if abs(pacf_vals[k]) > seuil_sig), default=0), 3)
    q = min(max((k for k in range(1, min(4, len(acf_vals)))
                 if abs(acf_vals[k]) > seuil_sig), default=0), 3)
    if p == 0 and q == 0: p, q = 1, 1

    s_lags = [k for k in range(S, max_lag + 1, S)]
    P = 1 if any(abs(pacf_vals[k]) > seuil_sig for k in s_lags if k < len(pacf_vals)) else 0
    Q = 1 if any(abs(acf_vals[k])  > seuil_sig for k in s_lags if k < len(acf_vals))  else 0
    ordre_str = f"SARIMA({p},{d},{q})({P},{D},{Q})[{S}]"

    # Ajustement SARIMA(p,d,q)(P,D,Q)[12]
    try:
        fit   = SARIMAX(serie, order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        resid = fit.resid.dropna()
    except Exception as e:
        diag_rows.append({'District': ds, 'ADF_pval': adf_pval,
                          'Stationnaire': stat, 'd': d, 'D': D,
                          'p': p, 'q': q, 'P': P, 'Q': Q,
                          'Ordre': ordre_str, 'Erreur': str(e)})
        continue

    # Validation résidus
    sw_stat, sw_pval = shapiro(resid.sample(min(len(resid), 5000), random_state=42))
    jb_stat, jb_pval = jarque_bera(resid)
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    lb_pval = round(float(lb['lb_pvalue'].iloc[0]), 4)

    diag_rows.append({
        'District'           : ds,
        'ADF_stat'           : adf_stat,
        'ADF_pval'           : adf_pval,
        'Stationnaire'       : stat,
        'p': p, 'd': d, 'q': q,
        'P': P, 'D': D, 'Q': Q, 'S': S,
        'Ordre'              : ordre_str,
        'AIC'                : round(fit.aic, 2),
        'BIC'                : round(fit.bic, 2),
        'SW_pval'            : round(sw_pval, 4),
        'Residus_Normaux_SW' : sw_pval > 0.05,
        'JB_pval'            : round(jb_pval, 4),
        'Residus_Normaux_JB' : jb_pval > 0.05,
        'LB_pval'            : lb_pval,
        'Residus_NonAutoCorr': lb_pval > 0.05,
    })

    # Figures détaillées pour le top 5 — 3 figures séparées par district
    if ds in top5:
        ds_clean = ds.replace(' ', '_').replace('/', '_')

        # ── FIGURE 1 : Test de stationnarité (ADF) — résumé statistique sous chaque graphe
        figure_stationnarite(serie, adf_res, adf_stat, adf_pval, stat, d, ds,
                             OUT_SARIMA, DPI)

        # ── FIGURE 2 : Identification des ordres (ACF / PACF) ────────────────
        fig, axes_acf = plt.subplots(1, 2, figsize=(14, 5))
        lags_plot = range(max_lag + 1)

        ax = axes_acf[0]
        ax.bar(lags_plot, acf_vals[:max_lag+1], color='#2E86AB', alpha=0.75,
               edgecolor='white')
        ax.axhline(seuil_sig,  color='red', ls='--', lw=1.2,
                   label=f'Seuil ±{round(seuil_sig, 3)}')
        ax.axhline(-seuil_sig, color='red', ls='--', lw=1.2)
        ax.axhline(0, color='black', lw=0.8)
        # Marquer les lags saisonniers
        for sl in [k for k in range(S, max_lag + 1, S)]:
            ax.axvline(sl, color='purple', ls=':', lw=1, alpha=0.6)
        ax.set_title(f"ACF — série différenciée\n→ Ordre MA retenu : q={q}  Q={Q}",
                     fontsize=10)
        ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrélation")
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        ax = axes_acf[1]
        ax.bar(lags_plot, pacf_vals[:max_lag+1], color='#3BB273', alpha=0.75,
               edgecolor='white')
        ax.axhline(seuil_sig,  color='red', ls='--', lw=1.2,
                   label=f'Seuil ±{round(seuil_sig, 3)}')
        ax.axhline(-seuil_sig, color='red', ls='--', lw=1.2)
        ax.axhline(0, color='black', lw=0.8)
        for sl in [k for k in range(S, max_lag + 1, S)]:
            ax.axvline(sl, color='purple', ls=':', lw=1, alpha=0.6,
                       label='Lag saisonnier (×12)' if sl == S else '')
        ax.set_title(f"PACF — série différenciée\n→ Ordre AR retenu : p={p}  P={P}",
                     fontsize=10)
        ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrélation partielle")
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{OUT_SARIMA}/{_nom_fichier('Identification des ordres - ' + ds)}.png",
                    dpi=DPI, bbox_inches='tight')
        plt.close()

        # ── FIGURE 3 : Validation des résidus — résumé statistique sous les graphes
        lb_stat = float(lb['lb_stat'].iloc[0])
        figure_residus(resid, sw_stat, sw_pval, jb_stat, jb_pval, lb_stat,
                       lb_pval, ds, OUT_SARIMA, DPI)

diag_df = pd.DataFrame(diag_rows)
n = len(diag_df)
if 'Residus_Normaux_SW' in diag_df.columns:
    print(f"  ✓ {n} districts analysés")
    print(f"    Résidus normaux (SW)  : {diag_df['Residus_Normaux_SW'].sum()}/{n}")
    print(f"    Non autocorrélés (LB) : {diag_df['Residus_NonAutoCorr'].sum()}/{n}")
    if 'D' in diag_df.columns:
        d1 = (diag_df['d'] == 1).sum(); D1 = (diag_df['D'] == 1).sum()
        print(f"    Différenciation d=1   : {d1}/{n} districts")
        print(f"    Différenciation D=1   : {D1}/{n} districts")

# =============================================================================
# OPTION 3 — TABLEAUX + GRAPHIQUES POUR CORPS ET ANNEXE
# =============================================================================

if not diag_df.empty and 'ADF_stat' in diag_df.columns:

    # ── Sélection des 5 districts représentatifs ─────────────────────────────
    # 1 stationnaire, 1 non stationnaire urbain, 1 non stationnaire rural,
    # 1 faible charge, 1 atypique
    n_stat    = (diag_df['Stationnaire'] == True).sum()
    n_nonstat = (diag_df['Stationnaire'] == False).sum()

    # District non stationnaire à forte charge
    ds_nonstat = (diag_df[diag_df['Stationnaire'] == False]
                  .sort_values('ADF_pval', ascending=False)
                  .iloc[0]['District'])

    # District stationnaire (si existe)
    ds_stat_row = diag_df[diag_df['Stationnaire'] == True]
    ds_stat = ds_stat_row.iloc[0]['District'] if not ds_stat_row.empty else None

    # Top 5 représentatifs fixes
    rep5 = []
    if ds_nonstat: rep5.append(ds_nonstat)
    if ds_stat and ds_stat not in rep5: rep5.append(ds_stat)
    for ds_add in top5:
        if ds_add not in rep5 and len(rep5) < 5:
            rep5.append(ds_add)
    rep5 = rep5[:5]

    # =========================================================================
    # A. STATIONNARITÉ
    # =========================================================================

    # ── A1. Tableau synthétique stationnarité ─────────────────────────────────
    n_d0 = (diag_df['d'] == 0).sum()
    n_d1 = (diag_df['d'] == 1).sum()
    n_d2 = (diag_df['d'] == 2).sum() if 2 in diag_df['d'].values else 0

    synth_stat = pd.DataFrame([
        ['Stationnaire (d=0)', n_d0, f"{round(n_d0/n*100,1)}%",
         'Aucune différenciation nécessaire'],
        ['Non stationnaire (d=1)', n_d1, f"{round(n_d1/n*100,1)}%",
         'Différenciation d\'ordre 1 appliquée'],
        ['Non stationnaire (d=2)', n_d2, f"{round(n_d2/n*100,1)}%",
         'Différenciation d\'ordre 2 appliquée'],
        ['Total', n, '100%', '70 districts sanitaires'],
    ], columns=['Résultat ADF', 'Nb districts', 'Pourcentage', 'Action'])

    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis('off')
    tbl = ax.table(cellText=synth_stat.values,
                   colLabels=synth_stat.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == 1: cell.set_facecolor('#D5F5E3')
        elif r == 2: cell.set_facecolor('#FADBD8')
        elif r == 3: cell.set_facecolor('#FADBD8')
        elif r == 4: cell.set_facecolor('#EBF5FB')
        cell.set_edgecolor('#BDC3C7')
    plt.tight_layout()
    plt.savefig(f"{OUT_SARIMA}/A1_synth_stationnarite.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("  ✓ A1_synth_stationnarite.png")

    # ── A2. Tableau 5 districts représentatifs — stationnarité ────────────────
    rep5_adf = diag_df[diag_df['District'].isin(rep5)][
        ['District', 'ADF_stat', 'ADF_pval', 'Stationnaire', 'd']].copy()
    rep5_adf['Stationnaire'] = rep5_adf['Stationnaire'].map(
        {True: 'Oui ✓', False: 'Non ✗'})
    rep5_adf.columns = ['District', 'Stat. ADF', 'p-valeur',
                        'Stationnaire', 'd']

    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis('off')
    tbl = ax.table(cellText=rep5_adf.values,
                   colLabels=rep5_adf.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if r <= len(rep5_adf):
                stat_val = str(rep5_adf.iloc[r-1]['Stationnaire'])
                cell.set_facecolor('#D5F5E3' if 'Oui' in stat_val else '#FADBD8')
            cell.set_edgecolor('#BDC3C7')
    plt.tight_layout()
    plt.savefig(f"{OUT_SARIMA}/A2_rep5_stationnarite.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("  ✓ A2_rep5_stationnarite.png")

    # ── A3. Graphique stationnarité — 1 non stationnaire + 1 stationnaire ─────
    districts_graph = []
    if ds_nonstat: districts_graph.append((ds_nonstat, False))
    if ds_stat:    districts_graph.append((ds_stat, True))

    for ds_g, est_stat in districts_graph:
        serie_g = get_series(df, ds_g)
        row_g   = diag_df[diag_df['District'] == ds_g].iloc[0]
        d_g     = int(row_g['d'])
        adf_p_g = float(row_g['ADF_pval'])

        fig, axes_g = plt.subplots(1, 2 if d_g > 0 else 1,
                                   figsize=(14 if d_g > 0 else 7, 4.5))
        if d_g == 0:
            axes_g = [axes_g]

        # Série originale
        ax = axes_g[0]
        couleur_g = '#27AE60' if est_stat else '#F18F01'
        ax.plot(serie_g.index, serie_g.values,
                color=couleur_g, linewidth=1.5)
        statut_g = "Stationnaire" if est_stat else "Non stationnaire"
        ax.text(0.02, 0.95,
                f"Test ADF\np-valeur = {adf_p_g}\n→ {statut_g}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=couleur_g, alpha=0.9))
        ax.set_title(f"{ds_g} — Série originale", fontsize=10)
        ax.set_ylabel("Cas de paludisme")
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        # Série différenciée si nécessaire
        if d_g > 0 and len(axes_g) > 1:
            serie_diff_g = serie_g.diff(d_g).dropna()
            adf_diff_g   = adfuller(serie_diff_g.dropna(), autolag='AIC')
            ax2 = axes_g[1]
            ax2.plot(serie_diff_g.index, serie_diff_g.values,
                     color='#2E86AB', linewidth=1.5)
            ax2.axhline(0, color='black', lw=0.8, ls='--')
            ax2.text(0.02, 0.95,
                     f"Après différenciation (d={d_g})\n"
                     f"p-valeur ADF = {round(adf_diff_g[1], 4)}\n"
                     f"→ Stationnaire ✓",
                     transform=ax2.transAxes, fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='white',
                               edgecolor='#27AE60', alpha=0.9))
            ax2.set_title(f"Série différenciée (d={d_g})", fontsize=10)
            ax2.set_ylabel("Cas de paludisme")
            ax2.grid(alpha=0.3)
            ax2.spines[['top', 'right']].set_visible(False)

        statut_fname = "non_stationnaire" if not est_stat else "stationnaire"
        plt.tight_layout()
        plt.savefig(f"{OUT_SARIMA}/A3_graph_statio_{statut_fname}.png",
                    dpi=DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    print("  ✓ A3_graph_statio_non_stationnaire.png + A3_graph_statio_stationnaire.png")

    # ── A4. Tableau complet annexe — stationnarité ────────────────────────────
    adf_annexe = diag_df[['District', 'ADF_stat', 'ADF_pval',
                           'Stationnaire', 'd']].copy()
    adf_annexe['Stationnaire'] = adf_annexe['Stationnaire'].map(
        {True: 'Oui ✓', False: 'Non ✗'})
    adf_annexe.columns = ['District', 'Stat. ADF', 'p-valeur',
                          'Stationnaire', 'd']
    adf_annexe = adf_annexe.sort_values('p-valeur')

    n_ds  = len(adf_annexe)
    mid   = (n_ds + 1) // 2
    left  = adf_annexe.iloc[:mid].reset_index(drop=True)
    right = adf_annexe.iloc[mid:].reset_index(drop=True)

    fig, axes_ann = plt.subplots(1, 2,
                                 figsize=(18, max(6, mid * 0.32 + 1.2)))
    for ax_a, data_a, lbl in zip(axes_ann, [left, right],
                                  [f'Districts 1–{mid}',
                                   f'Districts {mid+1}–{n_ds}']):
        ax_a.axis('off')
        tbl = ax_a.table(cellText=data_a.values,
                         colLabels=data_a.columns,
                         cellLoc='center', loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1A252F')
                cell.set_text_props(color='white', fontweight='bold')
            elif r <= len(data_a):
                s = str(data_a.iloc[r-1]['Stationnaire'])
                cell.set_facecolor('#D5F5E3' if 'Oui' in s else '#FADBD8')
            cell.set_edgecolor('#BDC3C7')
        ax_a.set_title(lbl, fontsize=9, fontweight='bold', pad=6)

    plt.suptitle(
        "Annexe — Test ADF : résultats complets (70 districts)\n"
        "H₀ : Racine unitaire | Seuil α = 0.05 | "
        "Vert = stationnaire | Rouge = non stationnaire",
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_SARIMA}/A4_annexe_ADF_complet.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("  ✓ A4_annexe_ADF_complet.png")

    # =========================================================================
    # B. VALIDATION DES RÉSIDUS
    # =========================================================================

    # ── B1. Tableau synthétique résidus ──────────────────────────────────────
    n_norm_sw  = diag_df['Residus_Normaux_SW'].sum()
    n_nnorm_sw = n - n_norm_sw
    n_lb_ok    = diag_df['Residus_NonAutoCorr'].sum()
    n_lb_nok   = n - n_lb_ok
    n_tout_ok  = ((diag_df['Residus_Normaux_SW']) &
                  (diag_df['Residus_NonAutoCorr'])).sum()
    n_lb_ok_sw_nok = ((~diag_df['Residus_Normaux_SW']) &
                       (diag_df['Residus_NonAutoCorr'])).sum()

    synth_res = pd.DataFrame([
        ['Normaux + Non autocorrélés', n_tout_ok,
         f"{round(n_tout_ok/n*100,1)}%", 'Modèle entièrement validé'],
        ['Non normaux + Non autocorrélés', n_lb_ok_sw_nok,
         f"{round(n_lb_ok_sw_nok/n*100,1)}%",
         'Prévisions valides, IC à interpréter avec prudence'],
        ['Autocorrélation résiduelle', n_lb_nok,
         f"{round(n_lb_nok/n*100,1)}%", 'Structure résiduelle non capturée'],
        ['Total', n, '100%', '70 districts sanitaires'],
    ], columns=['Résultat', 'Nb districts', 'Pourcentage', 'Implication'])

    fig, ax = plt.subplots(figsize=(15, 2.5))
    ax.axis('off')
    tbl = ax.table(cellText=synth_res.values,
                   colLabels=synth_res.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == 1: cell.set_facecolor('#D5F5E3')
        elif r == 2: cell.set_facecolor('#FEF9E7')
        elif r == 3: cell.set_facecolor('#FADBD8')
        elif r == 4: cell.set_facecolor('#EBF5FB')
        cell.set_edgecolor('#BDC3C7')
    plt.tight_layout()
    plt.savefig(f"{OUT_SARIMA}/B1_synth_residus.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("  ✓ B1_synth_residus.png")

    # ── B2. Tableau 5 districts représentatifs — résidus ─────────────────────
    rep5_res = diag_df[diag_df['District'].isin(rep5)][
        ['District', 'Ordre', 'SW_pval', 'Residus_Normaux_SW',
         'JB_pval', 'LB_pval', 'Residus_NonAutoCorr']].copy()
    rep5_res['Residus_Normaux_SW']  = rep5_res['Residus_Normaux_SW'].map(
        {True: 'Oui ✓', False: 'Non ✗'})
    rep5_res['Residus_NonAutoCorr'] = rep5_res['Residus_NonAutoCorr'].map(
        {True: 'Oui ✓', False: 'Non ✗'})
    rep5_res.columns = ['District', 'Ordre', 'SW p-val', 'Normalité',
                        'JB p-val', 'LB p-val', 'Non autocorrélés']

    fig, ax = plt.subplots(figsize=(15, 2.4))
    ax.axis('off')
    tbl = ax.table(cellText=rep5_res.values,
                   colLabels=rep5_res.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r <= len(rep5_res):
            lb_ok = 'Oui' in str(rep5_res.iloc[r-1]['Non autocorrélés'])
            sw_ok = 'Oui' in str(rep5_res.iloc[r-1]['Normalité'])
            if lb_ok and sw_ok:     cell.set_facecolor('#D5F5E3')
            elif lb_ok and not sw_ok: cell.set_facecolor('#FEF9E7')
            else:                   cell.set_facecolor('#FADBD8')
        cell.set_edgecolor('#BDC3C7')
    plt.tight_layout()
    plt.savefig(f"{OUT_SARIMA}/B2_rep5_residus.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("  ✓ B2_rep5_residus.png")

    # ── B3. Graphiques résidus — non normal + normal (si existe) ─────────────
    # District non normal (le plus fréquent)
    ds_non_norm = (diag_df[diag_df['Residus_Normaux_SW'] == False]
                   .sort_values('SW_pval').iloc[0]['District'])

    # District normal (si existe)
    ds_norm_row = diag_df[diag_df['Residus_Normaux_SW'] == True]
    ds_norm = ds_norm_row.iloc[0]['District'] if not ds_norm_row.empty else None

    for ds_r, label_r in [(ds_non_norm, 'non_normal'),
                           (ds_norm, 'normal') if ds_norm else (None, None)]:
        if ds_r is None:
            continue

        serie_r = get_series(df, ds_r)
        row_r   = diag_df[diag_df['District'] == ds_r].iloc[0]
        sw_r    = float(row_r['SW_pval'])
        lb_r    = float(row_r['LB_pval'])
        ordre_r = str(row_r['Ordre'])

        try:
            fit_r = SARIMAX(serie_r, order=(1, 1, 1),
                            seasonal_order=(1, 0, 1, S),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
            resid_r = fit_r.resid.dropna()
        except Exception:
            continue

        fig, axes_r = plt.subplots(1, 2, figsize=(13, 4.5))

        # QQ-plot
        ax = axes_r[0]
        (osm, osr), (slope, intercept, _) = sc_stats.probplot(resid_r,
                                                                dist='norm')
        ax.scatter(osm, osr, color='#A23B72', s=15, alpha=0.6)
        ax.plot(osm, slope * np.array(osm) + intercept, 'k--', lw=1.5)
        norm_txt = "Normaux ✓" if sw_r > 0.05 else "Non normaux ✗"
        ax.set_title(f"QQ-plot — {ds_r}\nSW p = {sw_r} → {norm_txt}",
                     fontsize=10)
        ax.set_xlabel("Quantiles théoriques")
        ax.set_ylabel("Quantiles observés")
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

        # Histogramme
        ax = axes_r[1]
        ax.hist(resid_r.values, bins=30, color='#A23B72',
                alpha=0.7, edgecolor='white', density=True)
        xs = np.linspace(resid_r.min(), resid_r.max(), 200)
        ax.plot(xs, sc_stats.norm.pdf(xs, resid_r.mean(), resid_r.std()),
                color='#2C3E50', lw=2, label='Loi normale théorique')
        lb_txt = "Non autocorrélés ✓" if lb_r > 0.05 else "Autocorrélation ✗"
        ax.set_title(f"Distribution résidus — {ds_r}\nLB p = {lb_r} → {lb_txt}",
                     fontsize=10)
        ax.set_xlabel("Résidu"); ax.set_ylabel("Densité")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{OUT_SARIMA}/B3_graph_residus_{label_r}.png",
                    dpi=DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    print("  ✓ B3_graph_residus_non_normal.png (+ normal si disponible)")

    # ── B4. Tableau complet annexe — résidus ─────────────────────────────────
    res_annexe = diag_df[['District', 'Ordre', 'SW_pval',
                           'Residus_Normaux_SW', 'JB_pval',
                           'LB_pval', 'Residus_NonAutoCorr']].copy()
    res_annexe['Residus_Normaux_SW']  = res_annexe['Residus_Normaux_SW'].map(
        {True: 'Oui ✓', False: 'Non ✗'})
    res_annexe['Residus_NonAutoCorr'] = res_annexe['Residus_NonAutoCorr'].map(
        {True: 'Oui ✓', False: 'Non ✗'})
    res_annexe.columns = ['District', 'Ordre', 'SW p-val', 'Normalité (SW)',
                          'JB p-val', 'LB p-val', 'Non autocorrélés']

    n_r   = len(res_annexe)
    mid_r = (n_r + 1) // 2
    left_r  = res_annexe.iloc[:mid_r].reset_index(drop=True)
    right_r = res_annexe.iloc[mid_r:].reset_index(drop=True)

    fig, axes_br = plt.subplots(1, 2,
                                figsize=(22, max(6, mid_r * 0.32 + 1.5)))
    for ax_b, data_b, lbl_b in zip(axes_br, [left_r, right_r],
                                    [f'Districts 1–{mid_r}',
                                     f'Districts {mid_r+1}–{n_r}']):
        ax_b.axis('off')
        tbl = ax_b.table(cellText=data_b.values,
                         colLabels=data_b.columns,
                         cellLoc='center', loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1A252F')
                cell.set_text_props(color='white', fontweight='bold')
            elif r <= len(data_b):
                lb_ok = 'Oui' in str(data_b.iloc[r-1]['Non autocorrélés'])
                sw_ok = 'Oui' in str(data_b.iloc[r-1]['Normalité (SW)'])
                if lb_ok and sw_ok:       cell.set_facecolor('#D5F5E3')
                elif lb_ok and not sw_ok: cell.set_facecolor('#FEF9E7')
                else:                     cell.set_facecolor('#FADBD8')
            cell.set_edgecolor('#BDC3C7')
        ax_b.set_title(lbl_b, fontsize=9, fontweight='bold', pad=6)

    plt.suptitle(
        "Annexe — Validation des résidus SARIMA (70 districts)\n"
        "SW & JB : H₀ = Normalité | LB : H₀ = Absence autocorrélation | α = 0.05\n"
        "Vert = tout validé | Jaune = non normal mais non autocorrélé | "
        "Rouge = autocorrélation",
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_SARIMA}/B4_annexe_residus_complet.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("  ✓ B4_annexe_residus_complet.png")

    print("\n  Récapitulatif des fichiers générés :")
    print("  CORPS DU TEXTE :")
    print("    A1 — Tableau synthétique stationnarité")
    print("    A2 — Tableau 5 districts représentatifs (stationnarité)")
    print("    A3 — Graphiques série non stationnaire + stationnaire")
    print("    B1 — Tableau synthétique résidus")
    print("    B2 — Tableau 5 districts représentatifs (résidus)")
    print("    B3 — Graphiques résidus non normaux + normaux")
    print("  ANNEXES :")
    print("    A4 — Tableau complet ADF (70 districts)")
    print("    B4 — Tableau complet résidus (70 districts)")

# =============================================================================
# SORTIES STYLE EVIEWS — TOP 5 DISTRICTS
# =============================================================================

# Recalcul des résidus pour le top 5 (nécessaire pour Q-stats lag par lag)
print("  Génération des sorties style EViews pour le top 5...")

for ds in top5:
    ds_row = diag_df[diag_df['District'] == ds]
    if ds_row.empty or 'Erreur' in ds_row.columns:
        continue

    serie_full = get_series(df, ds)
    if len(serie_full) < 24:
        continue

    # Réajuster SARIMA(1,1,1)(1,0,1)[12] pour obtenir les résidus
    try:
        fit_ev = SARIMAX(serie_full, order=(1, 1, 1),
                         seasonal_order=(1, 0, 1, S),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
        resid_ev = fit_ev.resid.dropna()
    except Exception:
        continue

    adf_stat_v = float(ds_row['ADF_stat'].values[0])
    adf_pval_v = float(ds_row['ADF_pval'].values[0])
    stat_v     = bool(ds_row['Stationnaire'].values[0])
    d_v        = int(ds_row['d'].values[0])
    sw_pval_v  = float(ds_row['SW_pval'].values[0])
    jb_pval_v  = float(ds_row['JB_pval'].values[0])
    lb_pval_v  = float(ds_row['LB_pval'].values[0])
    ordre_v    = str(ds_row['Ordre'].values[0])
    aic_v      = float(ds_row['AIC'].values[0])
    bic_v      = float(ds_row['BIC'].values[0])
    T          = len(serie_full)

    # Valeurs critiques ADF standard
    cv_1  = -3.4816
    cv_5  = -2.8839
    cv_10 = -2.5788

    ds_clean = ds.replace(' ', '_').replace('/', '_')

    # ── SORTIE ADF STYLE EVIEWS ───────────────────────────────────────────────
    texte_adf = (
        f"Null Hypothesis: {ds} has a unit root\n"
        f"Exogenous: Constant\n"
        f"Lag Length: Automatic (based on AIC, maxlag=12)\n"
        f"Sample: {int(serie_full.index.year.min())}M01 "
        f"{int(serie_full.index.year.max())}M{int(serie_full.index.month[-1]):02d}\n"
        f"Included observations: {T}\n"
        f"{'─'*58}\n"
        f"{'':40s} {'t-Statistic':>12s}   {'Prob.*':>8s}\n"
        f"{'─'*58}\n"
        f"{'Augmented Dickey-Fuller test statistic':40s} "
        f"{adf_stat_v:>12.4f}   {adf_pval_v:>8.4f}\n"
        f"{'Test critical values:':40s}\n"
        f"{'    1% level':40s} {cv_1:>12.4f}\n"
        f"{'    5% level':40s} {cv_5:>12.4f}\n"
        f"{'   10% level':40s} {cv_10:>12.4f}\n"
        f"{'─'*58}\n"
        f"*MacKinnon (1996) one-sided p-values"
    )

    # Compter les lignes pour ajuster la hauteur exacte
    n_lignes_adf = texte_adf.count('\n') + 1
    hauteur_adf  = max(2.5, n_lignes_adf * 0.28 + 0.4)

    fig, ax = plt.subplots(figsize=(9, hauteur_adf))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    couleur_box = '#D5F5E3' if stat_v else '#FADBD8'
    ax.text(0.02, 0.98, texte_adf,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=couleur_box,
                      edgecolor='#7F8C8D', alpha=0.9))

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.savefig(f"{OUT_SARIMA}/eviews_ADF_{ds_clean}.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # ── SORTIE RÉSIDUS STYLE EVIEWS (Q-STATISTICS) ───────────────────────────
    max_lag_ev = 12
    ac_vals  = acf(resid_ev,  nlags=max_lag_ev, fft=True)[1:]
    pac_vals = pacf(resid_ev, nlags=max_lag_ev)[1:]

    # Calcul Q-stat et p-valeur lag par lag
    q_stats = []
    p_vals  = []
    for lag in range(1, max_lag_ev + 1):
        lb_res = acorr_ljungbox(resid_ev, lags=[lag], return_df=True)
        q_stats.append(float(lb_res['lb_stat'].iloc[0]))
        p_vals.append(float(lb_res['lb_pvalue'].iloc[0]))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    # En-tête
    entete = (
        f"Correlogram of Residuals — {ordre_v}\n"
        f"Sample: {int(serie_full.index.year.min())}M01 "
        f"{int(serie_full.index.year.max())}M{int(serie_full.index.month[-1]):02d}   "
        f"Included observations: {T}\n"
        f"AIC: {aic_v:.2f}   BIC: {bic_v:.2f}\n"
        f"{'─'*62}\n"
        f"{'Lag':>4s}  {'AC':>8s}  {'PAC':>8s}  {'Q-Stat':>10s}  {'Prob':>8s}  {'':6s}\n"
        f"{'─'*62}\n"
    )

    lignes = ""
    for i in range(max_lag_ev):
        sig = "**" if p_vals[i] < 0.01 else ("*" if p_vals[i] < 0.05 else "  ")
        lignes += (f"{i+1:>4d}  {ac_vals[i]:>8.4f}  {pac_vals[i]:>8.4f}  "
                   f"{q_stats[i]:>10.4f}  {p_vals[i]:>8.4f}  {sig}\n")

    pied = (
        f"{'─'*62}\n"
        f"* Significant at 5%   ** Significant at 1%\n\n"
        f"Normality Tests (Residuals)\n"
        f"{'─'*40}\n"
        f"{'Shapiro-Wilk':30s}  p = {sw_pval_v:.4f}  "
        f"{'✓ Normal' if sw_pval_v > 0.05 else '✗ Non normal'}\n"
        f"{'Jarque-Bera':30s}  p = {jb_pval_v:.4f}  "
        f"{'✓ Normal' if jb_pval_v > 0.05 else '✗ Non normal'}\n"
        f"{'─'*40}"
    )

    texte_complet = entete + lignes + pied

    # Ajuster la hauteur exactement au nombre de lignes
    n_lignes_res = texte_complet.count('\n') + 1
    hauteur_res  = max(3.5, n_lignes_res * 0.27 + 0.4)

    fig, ax = plt.subplots(figsize=(9, hauteur_res))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    couleur_res = '#D5F5E3' if lb_pval_v > 0.05 else '#FEF9E7'
    ax.text(0.02, 0.98, texte_complet,
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=couleur_res,
                      edgecolor='#7F8C8D', alpha=0.9))

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.savefig(f"{OUT_SARIMA}/eviews_residus_{ds_clean}.png",
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("  ✓ Sorties EViews générées pour le top 5")

# =============================================================================
# ÉTAPE 3 — ENTRAÎNEMENT DES MODÈLES
# =============================================================================
print("\n[3/6] Entraînement des modèles...")

features_ext = FEATURES + [
    'humid_c_lag1', 'humid_c_lag2',
    'pluie_mm_lag1', 'pluie_mm_lag2',
    'temp_c_lag1',   'temp_c_lag2',
    'Cas_palu_lag1', 'Cas_palu_lag2',
    'mois_sin', 'mois_cos',
]
# 15 features — météo t + lags t-1/t-2 + autorégressif + saisonnalité :
# DS_encoded, annee_norm,
# humid_c, pluie_mm, temp_c (mois t — disponibles via stations météo),
# 6 lags climatiques t-1/t-2, 2 lags autorégressifs, 2 cycliques
print(f"  Features utilisées ({len(features_ext)}) : {features_ext}")

# Découpage temporel : train <= 2022 | test >= 2023
train = df[df['annee'] <= 2022].copy()
test  = df[df['annee'] >= 2023].copy()
print(f"  Découpage temporel :")
print(f"    Train : {len(train):,} obs | 2010–2022")
print(f"    Test  : {len(test):,} obs  | 2023–2025")

X_train, y_train = train[features_ext], train[TARGET]
X_test,  y_test  = test[features_ext],  test[TARGET]

# =============================================================================
# CALIBRATION DES MODÈLES — TimeSeriesSplit (respect de l'ordre chronologique)
# =============================================================================
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, RandomizedSearchCV

# TimeSeriesSplit : entraîne toujours sur le passé, teste sur le futur immédiat
# Contrairement à KFold shuffle=True qui brise l'ordre temporel
tscv = TimeSeriesSplit(n_splits=5)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

# ── KNN — sélection du k optimal par validation croisée temporelle ────────────
print("  KNN — recherche du k optimal par TimeSeriesSplit (5 folds)...")
k_values  = [3, 5, 7, 9, 11, 15, 21]
rmse_k    = []

for k in k_values:
    knn_cv = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    scores = cross_val_score(knn_cv, X_tr_sc, y_train,
                             cv=tscv,
                             scoring='neg_root_mean_squared_error')
    rmse_k.append(-scores.mean())
    print(f"    k={k:2d} → RMSE CV = {-scores.mean():.4f} ± {scores.std():.4f}")

best_k = k_values[int(np.argmin(rmse_k))]
print(f"  ✓ Meilleur k = {best_k} (RMSE CV = {min(rmse_k):.4f})")

# Figure — courbe RMSE vs k
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, rmse_k, color='#2E86AB', linewidth=2,
        marker='o', markersize=7)
ax.axvline(best_k, color='#E84855', ls='--', lw=1.5,
           label=f'k optimal = {best_k}')
ax.scatter([best_k], [min(rmse_k)], color='#E84855', s=100, zorder=5)
ax.set_xlabel("Nombre de voisins k", fontsize=10)
ax.set_ylabel("RMSE (validation croisée 5-fold)", fontsize=10)
ax.set_xticks(k_values)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_MODELES}/fig_KNN_selection_k.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ fig_KNN_selection_k.png")

# Entraînement final avec le k optimal
knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
knn.fit(X_tr_sc, y_train)
pred_knn = knn.predict(X_te_sc)
print(f"  ✓ KNN entraîné (k={best_k})")

# ── Random Forest — calibration par RandomizedSearchCV + TimeSeriesSplit ─────
print("  Random Forest — calibration des hyperparamètres (RandomizedSearchCV)...")
# Grille RF contrainte pour éviter le sur-apprentissage :
# - max_depth limité à 10 max (None retiré — arbres illimités = mémorisation)
# - min_samples_leaf >= 4 (feuilles avec >= 4 obs — meilleure généralisation)
# - n_estimators >= 200 (stabilité des prédictions)
param_rf = {
    'n_estimators'    : [200, 300, 500],
    'max_depth'       : [5, 7, 10],
    'min_samples_leaf': [4, 8, 15],
    'max_features'    : ['sqrt', 'log2', 0.5],
}
search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_rf,
    n_iter=30,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
search_rf.fit(X_train, y_train)
rf = search_rf.best_estimator_
pred_rf = rf.predict(X_test)
print(f"  ✓ Random Forest calibré — meilleurs hyperparamètres :")
for k, v in search_rf.best_params_.items():
    print(f"      {k} = {v}")
print(f"    MAE CV = {-search_rf.best_score_:.4f}")

# ── XGBoost — calibration par RandomizedSearchCV + TimeSeriesSplit ───────────
print("  XGBoost — calibration des hyperparamètres (RandomizedSearchCV)...")
# Grille XGBoost contrainte pour éviter le sur-apprentissage :
# - max_depth limité à 3-5 (profondeur raisonnable pour données épidémio)
# - reg_alpha et reg_lambda renforcés (pénalisation L1+L2 plus forte)
# - subsample et colsample_bytree < 1.0 forcés (stochasticité obligatoire)
# - learning_rate faible + n_estimators élevé (convergence progressive)
param_xgb = {
    'n_estimators'    : [300, 500, 700],
    'max_depth'       : [3, 4, 5],
    'learning_rate'   : [0.01, 0.03, 0.05],
    'subsample'       : [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha'       : [0.1, 0.5, 1.0, 2.0],
    'reg_lambda'      : [1.5, 2.0, 3.0],
}
search_xgb = RandomizedSearchCV(
    XGBRegressor(objective='reg:squarederror',
                 random_state=42, n_jobs=-1, verbosity=0),
    param_xgb,
    n_iter=50,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
search_xgb.fit(X_train, y_train)
xgb = search_xgb.best_estimator_
pred_xgb = xgb.predict(X_test)
print(f"  ✓ XGBoost calibré — meilleurs hyperparamètres :")
for k, v in search_xgb.best_params_.items():
    print(f"      {k} = {v}")
print(f"    MAE CV = {-search_xgb.best_score_:.4f}")

# ── Figure récapitulative des hyperparamètres retenus (thèse Chapitre 3) ─────
hp_rf  = search_rf.best_params_
hp_xgb = search_xgb.best_params_

hp_rows = [
    ['Random Forest', 'n_estimators',     str(hp_rf.get('n_estimators', '—'))],
    ['Random Forest', 'max_depth',         str(hp_rf.get('max_depth', '—'))],
    ['Random Forest', 'min_samples_leaf',  str(hp_rf.get('min_samples_leaf', '—'))],
    ['Random Forest', 'max_features',      str(hp_rf.get('max_features', '—'))],
    ['Random Forest', 'MAE CV (train)',    f"{-search_rf.best_score_:.4f}"],
    ['XGBoost',       'n_estimators',     str(hp_xgb.get('n_estimators', '—'))],
    ['XGBoost',       'max_depth',         str(hp_xgb.get('max_depth', '—'))],
    ['XGBoost',       'learning_rate',     str(hp_xgb.get('learning_rate', '—'))],
    ['XGBoost',       'subsample',         str(hp_xgb.get('subsample', '—'))],
    ['XGBoost',       'colsample_bytree',  str(hp_xgb.get('colsample_bytree', '—'))],
    ['XGBoost',       'reg_alpha',         str(hp_xgb.get('reg_alpha', '—'))],
    ['XGBoost',       'reg_lambda',        str(hp_xgb.get('reg_lambda', '—'))],
    ['XGBoost',       'MAE CV (train)',    f"{-search_xgb.best_score_:.4f}"],
    ['KNN',           'k optimal',         str(best_k)],
    ['KNN',           'weights',           'distance'],
    ['KNN',           'RMSE CV (train)',   f"{min(rmse_k):.4f}"],
]
hp_df = pd.DataFrame(hp_rows, columns=['Modèle', 'Hyperparamètre', 'Valeur retenue'])

fig, ax = plt.subplots(figsize=(11, len(hp_rows) * 0.45 + 1.2))
ax.axis('off')
tbl_hp = ax.table(cellText=hp_df.values,
                  colLabels=hp_df.columns,
                  cellLoc='center', loc='center')
tbl_hp.auto_set_font_size(False); tbl_hp.set_fontsize(9.5); tbl_hp.scale(1, 1.8)
couleurs_modeles = {'Random Forest': '#E8F8F5', 'XGBoost': '#FEF9E7', 'KNN': '#EBF5FB'}
for (r, c), cell in tbl_hp.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A252F')
        cell.set_text_props(color='white', fontweight='bold')
    elif r <= len(hp_rows):
        modele_r = hp_rows[r - 1][0]
        cell.set_facecolor(couleurs_modeles.get(modele_r, 'white'))
    cell.set_edgecolor('#BDC3C7')

plt.suptitle(
    "Hyperparamètres retenus après calibration\n"
    "Méthode : RandomizedSearchCV — Validation : TimeSeriesSplit (5 folds)\n"
    "Critère d'optimisation : MAE (RF, XGB) | RMSE (KNN)",
    fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_MODELES}/fig_hyperparametres_retenus.png",
            dpi=DPI, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("  ✓ fig_hyperparametres_retenus.png")

# ── Tableau méthodologique : grille explorée × valeur retenue (Chapitre 3) ───
# Croise les grilles de recherche (param_rf, param_xgb, k_values) avec les
# valeurs sélectionnées par validation croisée temporelle.

def _fmt_val(v):
    """Formate une valeur de grille en notation française (virgule décimale)."""
    if isinstance(v, float):
        return f"{v:g}".replace('.', ',')
    if v == 'sqrt':
        return 'racine'
    return str(v)


def _fmt_grille(valeurs):
    return ' ; '.join(_fmt_val(v) for v in valeurs)


# Libellés français des hyperparamètres pour la rédaction
_LIB = {
    'n_neighbors'     : "Nombre de voisins",
    'weights'         : "Pondération",
    'n_estimators'    : "Nombre d'arbres",
    'max_depth'       : "Profondeur maximale",
    'min_samples_leaf': "Effectif minimal par feuille",
    'max_features'    : "Variables candidates par nœud",
    'learning_rate'   : "Taux d'apprentissage",
    'subsample'       : "Taux d'échantillonnage",
    'colsample_bytree': "Variables par arbre",
    'reg_alpha'       : "Pénalisation L1",
    'reg_lambda'      : "Pénalisation L2",
}
_LIB_XGB = dict(_LIB, n_estimators="Nombre d'itérations")

meth_rows = []

# KNN
meth_rows.append(['KNN', _LIB['n_neighbors'], _fmt_grille(k_values)])
meth_rows.append(['KNN', _LIB['weights'], 'Inverse de la distance'])

# Random Forest
for p, grille in param_rf.items():
    meth_rows.append(['Forêt aléatoire', _LIB.get(p, p), _fmt_grille(grille)])

# XGBoost
for p, grille in param_xgb.items():
    meth_rows.append(['XGBoost', _LIB_XGB.get(p, p), _fmt_grille(grille)])

meth_df = pd.DataFrame(
    meth_rows,
    columns=['Modèle', 'Hyperparamètre', 'Valeurs explorées'])

# Export CSV (réutilisable dans Word par collage)
meth_df.to_csv(f"{OUT_MODELES}/tab_hyperparametres_methodo.csv",
               index=False, sep=';', encoding='utf-8-sig')

# Export LaTeX booktabs, modèle fusionné en colonne de gauche
_lignes_tex = []
_modele_prec = None
for m, h, g in meth_rows:
    if _modele_prec is not None and m != _modele_prec:
        _lignes_tex.append(r'\midrule')
    cell_m = m if m != _modele_prec else ''
    _lignes_tex.append(f"{cell_m} & {h} & {g} \\\\")
    _modele_prec = m

_note = (
    "Note : les valeurs candidates sont explorées par validation croisée "
    f"temporelle à {tscv.n_splits} blocs sur la seule période d'entraînement. "
    "La recherche est exhaustive pour le KNN et aléatoire pour les modèles "
    f"d'ensemble ({search_rf.n_iter} tirages pour la forêt aléatoire, "
    f"{search_xgb.n_iter} pour XGBoost). Le critère d'optimisation est la RMSE "
    "pour le KNN et la MAE pour les modèles d'ensemble. Seul le KNN est estimé "
    "sur variables standardisées. Graine aléatoire fixée à 42."
)

_tex = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Espaces de recherche des hyperparamètres}\n"
    "\\label{tab:hyperparametres}\n"
    "\\small\n"
    "\\begin{tabular}{lll}\n"
    "\\toprule\n"
    "Modèle & Hyperparamètre & Valeurs explorées \\\\\n"
    "\\midrule\n"
    + "\n".join(_lignes_tex) + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\begin{minipage}{\\textwidth}\\vspace{2mm}\\footnotesize\n"
    "Source : élaboré par nous-mêmes.\\\\\n"
    + _note + "\n"
    "\\end{minipage}\n"
    "\\end{table}\n"
)

with open(f"{OUT_MODELES}/tab_hyperparametres_methodo.tex", 'w',
          encoding='utf-8') as f:
    f.write(_tex)

print("  ✓ tab_hyperparametres_methodo.csv")
print("  ✓ tab_hyperparametres_methodo.tex")

# =============================================================================
# SARIMA — Principe de parcimonie (Box & Jenkins, 1976)
# =============================================================================
# Ordre uniforme SARIMA(1,1,1)(1,0,1)[12] appliqué à tous les districts.
#
# Justification méthodologique :
#   - Principe de parcimonie (Box & Jenkins, 1976) : parmi deux modèles offrant
#     des performances comparables, le plus simple doit être préféré.
#   - Un modèle avec peu de paramètres est plus stable, plus interprétable et
#     généralement plus performant en prévision hors échantillon.
#   - La partie non saisonnière (1,1,1) capte la tendance locale et la
#     dépendance à court terme.
#   - La partie saisonnière (1,0,1)[12] modélise explicitement le cycle
#     annuel du paludisme sans sur-différenciation (D=0 pour stabilité).
#   - d déterminé individuellement par test ADF (d=0 si stationnaire,
#     d=1 sinon) — seule adaptation district par district retenue.
#
# Référence : Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis:
#             Forecasting and Control. Holden-Day.
# =============================================================================

SARIMA_P, SARIMA_Q = 1, 1   # ordre saisonnier uniforme
SARIMA_p, SARIMA_q = 1, 1   # ordre non saisonnier uniforme
# d déterminé par ADF par district (cf. ÉTAPE 2)

print("  SARIMA(1,d,1)(1,0,1)[12] — principe de parcimonie (Box & Jenkins, 1976)...")
print("  d déterminé par test ADF par district | (peut prendre quelques minutes)")

sarima_rows        = []
sarima_best_orders = {}
n_conv, n_err      = 0, 0

for ds in ds_list:
    tr_s = get_series(train, ds)
    te_s = get_series(test,  ds)

    if len(tr_s) < 24 or tr_s.std() == 0 or len(te_s) == 0:
        continue

    # d déterminé individuellement par test ADF
    adf_p = adfuller(tr_s.dropna(), autolag='AIC')[1]
    d_opt = 0 if adf_p < 0.05 else 1

    ordre_str = f"SARIMA({SARIMA_p},{d_opt},{SARIMA_q})({SARIMA_P},0,{SARIMA_Q})[{S}]"

    try:
        fit = SARIMAX(
            tr_s,
            order=(SARIMA_p, d_opt, SARIMA_q),
            seasonal_order=(SARIMA_P, 0, SARIMA_Q, S),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        _full = pd.concat([tr_s, te_s])
        pred_raw = np.asarray(fit.apply(_full).predict(start=len(tr_s), end=len(_full)-1, dynamic=False), float)
        pred_inc  = np.clip(pred_raw, 0, None)
        real_inc  = te_s.values
        n_conv   += 1

        sarima_best_orders[ds] = {
            'Ordre': ordre_str,
            'AIC'  : round(fit.aic, 2),
            'BIC'  : round(fit.bic, 2),
            'p': SARIMA_p, 'd': d_opt, 'q': SARIMA_q,
            'P': SARIMA_P, 'Q': SARIMA_Q,
        }

        for i in range(len(te_s)):
            sarima_rows.append({
                'Nom_DS'     : ds,
                'annee'      : int(te_s.index[i].year),
                'mois'       : int(te_s.index[i].month),
                'Cas_palu'   : float(real_inc[i]) if not np.isnan(real_inc[i]) else np.nan,
                'Pred_SARIMA': float(pred_inc[i]) if not np.isnan(pred_inc[i]) else np.nan,
                'Ordre'      : ordre_str,
            })

    except Exception as e:
        n_err += 1
        # Fallback d=1 si convergence échoue avec d=0
        try:
            fit_fb = SARIMAX(
                tr_s,
                order=(SARIMA_p, 1, SARIMA_q),
                seasonal_order=(SARIMA_P, 0, SARIMA_Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            _full = pd.concat([tr_s, te_s])
            pred_inc = np.clip(np.asarray(fit_fb.apply(_full).predict(start=len(tr_s), end=len(_full)-1, dynamic=False), float), 0, None)
            real_inc = te_s.values
            ordre_fb = f"SARIMA({SARIMA_p},1,{SARIMA_q})({SARIMA_P},0,{SARIMA_Q})[{S}]"
            sarima_best_orders[ds] = {
                'Ordre': ordre_fb, 'AIC': round(fit_fb.aic, 2),
                'BIC': round(fit_fb.bic, 2),
                'p': SARIMA_p, 'd': 1, 'q': SARIMA_q,
                'P': SARIMA_P, 'Q': SARIMA_Q,
            }
            for i in range(len(te_s)):
                sarima_rows.append({
                    'Nom_DS': ds, 'annee': int(te_s.index[i].year),
                    'mois': int(te_s.index[i].month),
                    'Cas_palu': float(real_inc[i]),
                    'Pred_SARIMA': float(pred_inc[i]),
                    'Ordre': ordre_fb,
                })
            n_conv += 1; n_err -= 1
        except Exception:
            continue

arima_df_test = pd.DataFrame(sarima_rows)
arima_df_test = arima_df_test.rename(columns={'Pred_SARIMA': 'Pred_ARIMA'})

orders_df = pd.DataFrame(sarima_best_orders).T.reset_index()
orders_df.columns = ['District'] + list(orders_df.columns[1:])

# Statistique d : combien de districts nécessitent d=1
n_d0 = sum(1 for v in sarima_best_orders.values() if v['d'] == 0)
n_d1 = sum(1 for v in sarima_best_orders.values() if v['d'] == 1)

print(f"  ✓ SARIMA convergé sur {n_conv} districts ({n_err} erreurs)")
print(f"    d=0 (stationnaire)     : {n_d0} districts")
print(f"    d=1 (différencié)      : {n_d1} districts")
print(f"    Ordre uniforme         : SARIMA(1,d,1)(1,0,1)[12]")
print(f"    Protocole              : prevision a un pas glissante (parametres figes)")
print(f"    Nb observations test   : {len(arima_df_test)}")

# =============================================================================
# MÉTRIQUES
# =============================================================================
def met(yt, yp, name):
    yt = np.array(yt, dtype=float)
    yp = np.array(yp, dtype=float)
    # Filtrer les paires valides
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]; yp = yp[mask]
    if len(yt) < 2:
        return {'Modele': name, 'MAE': None, 'RMSE': None,
                'R2': None, 'SMAPE': None}
    # SMAPE — Symmetric Mean Absolute Percentage Error (Makridakis, 1993)
    # SMAPE = (100/n) · Σ  2·|yp − yt| / (|yt| + |yp|)        domaine [0 ; 200] %
    denom = np.abs(yt) + np.abs(yp)
    ratio = np.divide(2.0 * np.abs(yp - yt), denom,
                      out=np.zeros_like(denom), where=denom != 0)
    smape = np.mean(ratio) * 100
    return {
        'Modele': name,
        'MAE'  : round(mean_absolute_error(yt, yp), 2),
        'RMSE' : round(np.sqrt(mean_squared_error(yt, yp)), 2),
        'R2'   : round(r2_score(yt, yp), 4),
        'SMAPE': round(smape, 2)
    }

arima_real = arima_df_test['Cas_palu'].values
arima_pred = arima_df_test['Pred_ARIMA'].values

metrics_df = pd.DataFrame([
    met(y_test, pred_knn,  'KNN'),
    met(y_test, pred_rf,   'RandomForest'),
    met(y_test, pred_xgb,  'XGBoost'),
    met(arima_real, arima_pred, 'SARIMA'),
]).sort_values('R2', ascending=False).reset_index(drop=True)
metrics_df.insert(0, 'Rang', range(1, len(metrics_df) + 1))

best_name = metrics_df[metrics_df['Modele'] != 'SARIMA'].iloc[0]['Modele']
print(f"\n  Métriques calculées — Meilleur modèle : {best_name}")

# =============================================================================
# TESTS STATISTIQUES DE COMPARAISON (Friedman + post-hoc Nemenyi)
# =============================================================================
executer_tests_comparaison(
    test, pred_knn, pred_rf, pred_xgb, arima_df_test, TARGET,
    OUT_MODELES, couleurs=COULEURS, dpi=DPI, metrique='rmse',
    excel_path=f"{OUT_MODELES}/tests_comparaison_modeles.xlsx")

# =============================================================================
# ÉTAPE 4 — FIGURES RÉSULTATS PAR MODÈLE
# =============================================================================
print("\n[4/6] Génération des figures par modèle...")

# Agrégation nationale mensuelle
test_nat = test.copy()
test_nat['Pred_KNN'] = pred_knn
test_nat['Pred_RF']  = pred_rf
test_nat['Pred_XGB'] = pred_xgb

nat = test_nat.groupby(['annee', 'mois']).agg(
    Cas_reel=('Cas_palu', 'mean'),
    Pred_KNN=('Pred_KNN', 'mean'),
    Pred_RF =('Pred_RF',  'mean'),
    Pred_XGB=('Pred_XGB', 'mean'),
).reset_index()
nat['date'] = pd.to_datetime(
    nat[['annee', 'mois']].assign(day=1).rename(
        columns={'annee': 'year', 'mois': 'month'}))
nat = nat.sort_values('date')

arima_nat = arima_df_test.groupby(['annee', 'mois']).agg(
    Cas_reel  =('Cas_palu',   'mean'),   # moyenne des incidences par district
    Pred_ARIMA=('Pred_ARIMA', 'mean'),   # idem pour les prédictions
).reset_index()
arima_nat['date'] = pd.to_datetime(
    arima_nat[['annee', 'mois']].assign(day=1).rename(
        columns={'annee': 'year', 'mois': 'month'}))
arima_nat = arima_nat.sort_values('date')

# Diagnostic
print(f"  SARIMA agrégé national : {len(arima_nat)} mois | "
      f"{arima_nat['date'].min().strftime('%Y-%m')} → "
      f"{arima_nat['date'].max().strftime('%Y-%m')}")
print(f"  Districts couverts : {arima_df_test['Nom_DS'].nunique()}/70")

# ── Figure IV-A : Courbes 4 modèles ─────────────────────────────────────────
configs = [
    ('SARIMA',       'Modèle 1', arima_nat, 'Pred_ARIMA'),
    ('KNN',          'Modèle 2', nat,       'Pred_KNN'),
    ('RandomForest', 'Modèle 3', nat,       'Pred_RF'),
    ('XGBoost',      'Modèle 4', nat,       'Pred_XGB'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (nom, label, data, pcol) in enumerate(configs):
    ax    = axes[i]
    color = COULEURS[nom]
    m     = metrics_df[metrics_df['Modele'] == nom]
    r2    = m['R2'].values[0];   rmse = m['RMSE'].values[0]
    mae   = m['MAE'].values[0];  smape = m['SMAPE'].values[0]

    ax.plot(data['date'], data['Cas_reel'], color='#2C3E50',
            linewidth=2, label='Cas réels', zorder=3)
    ax.plot(data['date'], data[pcol], color=color,
            linewidth=2, linestyle='--', label='Prédictions', zorder=3)
    ax.fill_between(data['date'], data['Cas_reel'], data[pcol],
                    alpha=0.10, color=color)
    ax.set_title(f"{label} : {nom}", fontsize=11,
                 fontweight='bold', color=color)
    ax.set_xlabel("Période de test (2023–2025)", fontsize=9)
    ax.set_ylabel("Cas de paludisme (national)", fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.98, 0.95,
            f"R² = {r2}  |  RMSE = {rmse}\nMAE = {mae}  |  SMAPE = {smape}%",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig(f"{OUT_MODELES}/fig_IV_A_courbes_4modeles.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ fig_IV_A_courbes_4modeles.png")

# ── Figure IV-B : Scatter plots ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

scatter_cfg = [
    ('SARIMA',       'Modèle 1', arima_df_test['Cas_palu'].values,  arima_pred),
    ('KNN',          'Modèle 2', y_test.values,                      pred_knn),
    ('RandomForest', 'Modèle 3', y_test.values,                      pred_rf),
    ('XGBoost',      'Modèle 4', y_test.values,                      pred_xgb),
]

for i, (nom, label, y_r, y_p) in enumerate(scatter_cfg):
    ax    = axes[i]
    color = COULEURS[nom]
    m     = metrics_df[metrics_df['Modele'] == nom]
    r2    = m['R2'].values[0]
    lim   = max(y_r.max(), y_p.max()) * 1.05

    ax.scatter(y_r, y_p, color=color, alpha=0.3, s=12, edgecolors='none')
    ax.plot([0, lim], [0, lim], 'k--', lw=1.2, label='Prédiction parfaite')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Cas réels", fontsize=10)
    ax.set_ylabel("Cas prédits", fontsize=10)
    ax.set_title(f"{label} : {nom} — R² = {r2}",
                 fontsize=11, fontweight='bold', color=color)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_MODELES}/fig_IV_B_scatter_4modeles.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ fig_IV_B_scatter_4modeles.png")

# ── Figure IV-C : Tableau métriques ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3))
ax.axis('off')
tbl = ax.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r == 1:
        cell.set_facecolor('#FFF3CD')
    elif r % 2 == 0:
        cell.set_facecolor('#F2F3F4')
    cell.set_edgecolor('#D5D8DC')
plt.tight_layout()
plt.savefig(f"{OUT_MODELES}/fig_IV_C_tableau_metriques.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ fig_IV_C_tableau_metriques.png")

# ── Figure IV-D : Importance variables RF vs XGBoost (meilleur en premier) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# Meilleur modèle affiché en premier
pairs_imp = sorted([('RandomForest', rf), ('XGBoost', xgb)], key=lambda x: 0 if x[0]==best_name else 1)
for ax, (nom_mod, model) in zip(axes, pairs_imp):
    color = COULEURS[nom_mod]
    imp   = pd.Series(model.feature_importances_,
                      index=features_ext).sort_values(ascending=True).tail(15)
    ax.barh(imp.index, imp.values, color=color, alpha=0.8,
            edgecolor='white', height=0.65)
    for feat, val in zip(imp.index, imp.values):
        ax.text(val + 0.001, feat, f"{val:.3f}",
                va='center', fontsize=8.5, color='#2C3E50')
    ax.set_title(f"Importance des variables\n{nom_mod}",
                 fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel("Importance relative", fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_MODELES}/fig_IV_D_importance_variables.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ fig_IV_D_importance_variables.png")

# =============================================================================
# ÉTAPE 5 — ANALYSE APPROFONDIE MEILLEUR MODÈLE
# =============================================================================
print(f"\n[5/6] Analyse approfondie — meilleur modèle : {best_name}...")

# ── Sélection dynamique du meilleur modèle (hors SARIMA) ─────────────────────
pred_map = {
    'KNN'         : pred_knn,
    'RandomForest': pred_rf,
    'XGBoost'     : pred_xgb,
}
model_map = {
    'KNN'         : knn,
    'RandomForest': rf,
    'XGBoost'     : xgb,
}
couleur_best = COULEURS[best_name]
pred_best    = pred_map[best_name]
model_best   = model_map[best_name]

# Aligner arrays (sécurité)
y_test_arr   = np.array(y_test, dtype=float).flatten()
pred_best_arr= np.array(pred_best, dtype=float).flatten()
n_min        = min(len(y_test_arr), len(pred_best_arr))
y_test_arr   = y_test_arr[:n_min]
pred_best_arr= pred_best_arr[:n_min]
# Conserver aussi pred_xgb_arr pour compatibilité section alertes
pred_xgb_arr = np.array(pred_xgb, dtype=float).flatten()[:n_min]
residus_best = y_test_arr - pred_best_arr

# Second modèle ML pour comparaison dans la figure détection pics
second_name = metrics_df[
    (metrics_df['Modele'] != 'SARIMA') &
    (metrics_df['Modele'] != best_name)
].iloc[0]['Modele']
pred_second_arr = np.array(pred_map[second_name], dtype=float).flatten()[:n_min]

# ── Figure V-A : Courbe nationale — meilleur modèle ─────────────────────────
pred_col_best = f'Pred_{best_name.replace("RandomForest","RF")}'
if pred_col_best not in nat.columns:
    # fallback : chercher la colonne correspondante
    candidates = [c for c in nat.columns if best_name[:3].upper() in c.upper()]
    pred_col_best = candidates[0] if candidates else 'Pred_XGB'

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(nat['date'], nat['Cas_reel'], color='#2C3E50',
        linewidth=2, label='Cas réels', zorder=3)
ax.plot(nat['date'], nat[pred_col_best], color=couleur_best,
        linewidth=2, linestyle='--', label=f'{best_name} — prédictions', zorder=3)
ax.fill_between(nat['date'], nat['Cas_reel'], nat[pred_col_best],
                alpha=0.12, color=couleur_best)
m_best = metrics_df[metrics_df['Modele'] == best_name].iloc[0]
ax.text(0.01, 0.95,
        f"R² = {m_best['R2']}  |  RMSE = {m_best['RMSE']}  |  MAE = {m_best['MAE']}  |  SMAPE = {m_best['SMAPE']}%",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
ax.set_title(f"Meilleur modèle : {best_name} — Courbe nationale",
             fontsize=11, fontweight='bold', color=couleur_best)
ax.set_xlabel("Période de test (2023–2025)", fontsize=10)
ax.set_ylabel("Cas de paludisme (national)", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_XGB}/fig_V_A_best_courbe_nationale.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_V_A_best_courbe_nationale.png ({best_name})")

# ── Figure V-B : Top 5 districts — meilleur modèle ──────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18))
for ax, ds in zip(axes, top5):
    mask = test['Nom_DS'] == ds
    sub  = test[mask].copy()
    if sum(mask) == 0:
        ax.set_title(f"{ds} — données insuffisantes", fontsize=10)
        continue
    sub['Pred_Best'] = pred_best[mask.values]
    sub['date'] = pd.to_datetime(
        sub[['annee', 'mois']].assign(day=1).rename(
            columns={'annee': 'year', 'mois': 'month'}))
    sub = sub.sort_values('date')
    r2_ds = round(r2_score(sub[TARGET], sub['Pred_Best']), 3) if len(sub) >= 2 else 'N/A'
    ax.plot(sub['date'], sub[TARGET], color='#2C3E50',
            linewidth=1.8, label='Cas réels')
    ax.plot(sub['date'], sub['Pred_Best'], color=couleur_best,
            linewidth=1.8, linestyle='--', label=best_name)
    ax.fill_between(sub['date'], sub[TARGET], sub['Pred_Best'],
                    alpha=0.12, color=couleur_best)
    ax.set_title(f"{ds} — R² = {r2_ds}",
                 fontsize=10, fontweight='bold', color=couleur_best)
    ax.set_ylabel("Cas", fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)
plt.suptitle(f"Meilleur modèle : {best_name} — Top 5 districts",
             fontsize=12, fontweight='bold', color=couleur_best)
plt.tight_layout()
plt.savefig(f"{OUT_XGB}/fig_V_B_best_top5_districts.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_V_B_best_top5_districts.png ({best_name})")

# ── Figure V-C : Résidus — meilleur modèle (4 panneaux) ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

ax = axes[0]
ax.scatter(range(len(residus_best)), residus_best,
           color=couleur_best, alpha=0.3, s=8, edgecolors='none')
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_title("Résidus — distribution temporelle", fontsize=10)
ax.set_xlabel("Observations"); ax.set_ylabel("Résidu (réel − prédit)")
ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
ax.hist(residus_best, bins=50, color=couleur_best,
        alpha=0.7, edgecolor='white', density=True)
xs = np.linspace(residus_best.min(), residus_best.max(), 200)
ax.plot(xs, sc_stats.norm.pdf(xs, residus_best.mean(), residus_best.std()),
        color='#2C3E50', lw=2, label='Loi normale')
ax.set_title("Distribution des résidus", fontsize=10)
ax.set_xlabel("Résidu"); ax.set_ylabel("Densité")
ax.legend(fontsize=8); ax.grid(alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[2]
(osm, osr), (slope, intercept, _) = sc_stats.probplot(residus_best, dist='norm')
ax.scatter(osm, osr, color=couleur_best, alpha=0.4, s=12, edgecolors='none')
ax.plot(osm, slope * np.array(osm) + intercept, 'k--', lw=1.5)
ax.set_title("QQ-plot des résidus", fontsize=10)
ax.set_xlabel("Quantiles théoriques"); ax.set_ylabel("Quantiles observés")
ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

ax = axes[3]
ax.scatter(pred_best_arr, residus_best, color=couleur_best,
           alpha=0.3, s=8, edgecolors='none')
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_title("Résidus vs prédictions", fontsize=10)
ax.set_xlabel("Valeurs prédites"); ax.set_ylabel("Résidu")
ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

plt.suptitle(f"Analyse des résidus — {best_name}",
             fontsize=12, fontweight='bold', color=couleur_best)
plt.tight_layout()
plt.savefig(f"{OUT_XGB}/fig_V_C_best_residus.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_V_C_best_residus.png ({best_name})")

# ── Figure V-D : Détection des pics — meilleur modèle vs second ──────────────
seuil_90    = np.percentile(y_test_arr, 90)
masque_pics = y_test_arr >= seuil_90

# Meilleur modèle
best_pics_r  = y_test_arr[masque_pics];   best_pics_p  = pred_best_arr[masque_pics]
best_norm_r  = y_test_arr[~masque_pics];  best_norm_p  = pred_best_arr[~masque_pics]
best_mae_pics = np.mean(np.abs(best_pics_r - best_pics_p))
best_mae_norm = np.mean(np.abs(best_norm_r - best_norm_p))

# Second modèle (comparaison)
sec_pics_p   = pred_second_arr[masque_pics]
sec_norm_p   = pred_second_arr[~masque_pics]
sec_mae_pics = np.mean(np.abs(best_pics_r - sec_pics_p))
sec_mae_norm = np.mean(np.abs(best_norm_r - sec_norm_p))
couleur_second = COULEURS[second_name]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
lim = max(y_test_arr.max(), pred_best_arr.max(), pred_second_arr.max()) * 1.05

# ── Scatter meilleur modèle ───────────────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(best_norm_r, best_norm_p, color='#BDC3C7', alpha=0.3, s=10,
           label=f'Mois normaux (n={sum(~masque_pics)})')
ax.scatter(best_pics_r, best_pics_p, color='#E84855', alpha=0.7, s=25,
           label=f'Mois de pic P90 (n={sum(masque_pics)})', zorder=3)
ax.plot([0, lim], [0, lim], 'k--', lw=1.2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Cas réels", fontsize=10); ax.set_ylabel("Cas prédits", fontsize=10)
ax.set_title(f"{best_name} — Détection des pics (P90)",
             fontsize=11, fontweight='bold', color=couleur_best)
ax.legend(fontsize=8); ax.grid(alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)

# ── Scatter second modèle ─────────────────────────────────────────────────────
ax = axes[0, 1]
ax.scatter(best_norm_r, sec_norm_p, color='#BDC3C7', alpha=0.3, s=10,
           label=f'Mois normaux (n={sum(~masque_pics)})')
ax.scatter(best_pics_r, sec_pics_p, color='#E84855', alpha=0.7, s=25,
           label=f'Mois de pic P90 (n={sum(masque_pics)})', zorder=3)
ax.plot([0, lim], [0, lim], 'k--', lw=1.2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Cas réels", fontsize=10); ax.set_ylabel("Cas prédits", fontsize=10)
ax.set_title(f"{second_name} — Détection des pics (P90)",
             fontsize=11, fontweight='bold', color=couleur_second)
ax.legend(fontsize=8); ax.grid(alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)

# ── Barres MAE meilleur modèle ────────────────────────────────────────────────
ax = axes[1, 0]
categories = ['Mois normaux', 'Mois de pic (P90)']
bars = ax.bar(categories, [best_mae_norm, best_mae_pics],
              color=['#BDC3C7', '#E84855'], alpha=0.8,
              edgecolor='white', width=0.5)
for bar, val in zip(bars, [best_mae_norm, best_mae_pics]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
            f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel("MAE", fontsize=10)
ax.set_title(f"{best_name} — Erreur normaux vs pics",
             fontsize=11, fontweight='bold', color=couleur_best)
ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

# ── Barres MAE second modèle ──────────────────────────────────────────────────
ax = axes[1, 1]
bars = ax.bar(categories, [sec_mae_norm, sec_mae_pics],
              color=['#BDC3C7', '#E84855'], alpha=0.8,
              edgecolor='white', width=0.5)
for bar, val in zip(bars, [sec_mae_norm, sec_mae_pics]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
            f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel("MAE", fontsize=10)
ax.set_title(f"{second_name} — Erreur normaux vs pics",
             fontsize=11, fontweight='bold', color=couleur_second)
ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

fig.text(0.5, 0.01,
         f"{best_name} — MAE pics : {best_mae_pics:.4f} | MAE normaux : {best_mae_norm:.4f}     "
         f"{second_name} — MAE pics : {sec_mae_pics:.4f} | MAE normaux : {sec_mae_norm:.4f}",
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', alpha=0.9))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f"{OUT_XGB}/fig_V_D_detection_pics_{best_name}_vs_{second_name}.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_V_D_detection_pics_{best_name}_vs_{second_name}.png")

# =============================================================================
# ÉTAPE 6 — EXPORT EXCEL
# =============================================================================
print("\n[6/6] Export Excel...")

with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="Métriques_comparatives", index=False)
    diag_df.to_excel(writer, sheet_name="Diagnostic_SARIMA", index=False)
    arima_df_test.to_excel(writer, sheet_name="Prédictions_SARIMA", index=False)

    # Ordres SARIMA retenus par district
    if 'orders_df' in dir() and not orders_df.empty:
        orders_df.to_excel(writer, sheet_name="Ordres_SARIMA_AIC", index=False)

    # Prédictions ML test
    test_export = test[['Nom_DS', 'annee', 'mois', TARGET]].copy()
    test_export['Pred_KNN'] = pred_knn.round(0).astype(int)
    test_export['Pred_RF']  = pred_rf.round(0).astype(int)
    test_export['Pred_XGB'] = pred_xgb.round(0).astype(int)
    test_export.to_excel(writer, sheet_name="Prédictions_ML", index=False)

    # Importance des variables (tous les modèles à base d'arbres)
    imp_data = {'Variable': features_ext}
    for nom_m, mod_m in [('RF', rf), ('XGBoost', xgb)]:
        imp_data[f'Importance_{nom_m}'] = pd.Series(
            mod_m.feature_importances_, index=features_ext).values.round(4)
    imp_df = pd.DataFrame(imp_data).sort_values(
        f'Importance_{best_name.replace("RandomForest","RF")}',
        ascending=False)
    imp_df.to_excel(writer, sheet_name="Importance_variables", index=False)

    # Hyperparamètres retenus
    hp_export = []
    for mod_label, search in [('RandomForest', search_rf), ('XGBoost', search_xgb)]:
        for k, v in search.best_params_.items():
            hp_export.append({'Modele': mod_label, 'Hyperparametre': k, 'Valeur': str(v)})
    hp_export.append({'Modele': 'KNN', 'Hyperparametre': 'k', 'Valeur': str(best_k)})
    pd.DataFrame(hp_export).to_excel(writer, sheet_name="Hyperparametres", index=False)

    # Tableau méthodologique : grille explorée x valeur retenue (Chapitre 3)
    meth_df.to_excel(writer, sheet_name="Hyperparametres_methodo", index=False)

print(f"  ✓ Excel exporté : {OUT_EXCEL}")


# =============================================================================
# ÉTAPE 7 — SECTION 4.5 : APPLICATION DU SYSTÈME D'ALERTE PRÉCOCE
# =============================================================================
print("\n[7/7] Génération des sorties système d'alerte précoce...")

OUT_ALERTE = "resultats/figures_alerte"
Path(OUT_ALERTE).mkdir(parents=True, exist_ok=True)

mois_labels_fr = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                  'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

# =============================================================================
# CALCUL DES SEUILS Q2/Q3 PAR DISTRICT ET PAR MOIS — MÉTHODE OMS 2018
# =============================================================================
# Baseline 2018-2022 (5 ans) — fenêtre fixe post-interventions PNLP.
# Mois épidémiques exclus via règle de Tukey (Cullen, 1985).
# Q2 = seuil alerte | Q3 = seuil épidémique.
# Références : OMS (2018), Cullen (1985), Tukey (1977), PNLP BF (2021).
# =============================================================================
ANNEE_BASELINE_MIN = 2018
ANNEE_BASELINE_MAX = 2022

train_alerte = df[
    (df['annee'] >= ANNEE_BASELINE_MIN) &
    (df['annee'] <= ANNEE_BASELINE_MAX)
].copy()
print(f'  Baseline alertes : {ANNEE_BASELINE_MIN}-{ANNEE_BASELINE_MAX} '
      f'({ANNEE_BASELINE_MAX - ANNEE_BASELINE_MIN + 1} ans)')

seuils_rows = []
for ds in ds_list:
    for mois_val in range(1, 13):
        serie_hist = train_alerte[
            (train_alerte['Nom_DS'] == ds) &
            (train_alerte['mois']   == mois_val)
        ][TARGET]
        if len(serie_hist) < 3:
            serie_hist = df[
                (df['Nom_DS'] == ds) &
                (df['mois']   == mois_val) &
                (df['annee']  <= ANNEE_BASELINE_MAX)
            ][TARGET]
        if len(serie_hist) < 2:
            continue
        # Règle de Tukey : exclusion années épidémiques
        q1  = serie_hist.quantile(0.25)
        q3v = serie_hist.quantile(0.75)
        iqr = q3v - q1
        serie_norm = serie_hist[serie_hist <= q3v + 1.5 * iqr]
        if len(serie_norm) < 2:
            serie_norm = serie_hist
        seuils_rows.append({
            'District'       : ds,
            'Mois'           : mois_val,
            'Mois_label'     : mois_labels_fr[mois_val - 1],
            'Seuil_Alerte'   : round(serie_norm.quantile(0.50), 6),
            'Seuil_Epidemie' : round(serie_norm.quantile(0.75), 6),
            'N_annees_norm'  : len(serie_norm),
            'N_annees_excl'  : len(serie_hist) - len(serie_norm),
        })

seuils_df = pd.DataFrame(seuils_rows)
print(f'  Seuils calcules : {len(seuils_df)} combinaisons district x mois')

# =============================================================================
# CLASSIFICATION DES PRÉDICTIONS — MÉTHODE OMS 2018
# =============================================================================
test_alerte              = test.copy()
test_alerte['Pred_XGB']  = pred_xgb
test_alerte['Pred_Best'] = pred_best
print(f"  Système d'alerte basé sur : {best_name}")

def classifier_alerte(row):
    match = seuils_df[
        (seuils_df['District'] == row['Nom_DS']) &
        (seuils_df['Mois']     == row['mois'])
    ]
    if match.empty:
        return 'INCONNU'
    pred = row['Pred_Best']
    sa   = match['Seuil_Alerte'].values[0]
    se   = match['Seuil_Epidemie'].values[0]
    if pred >= se:    return 'ROUGE'
    elif pred >= sa:  return 'ORANGE'
    else:             return 'VERT'


test_alerte['Niveau_Alerte'] = test_alerte.apply(classifier_alerte, axis=1)
test_alerte['date'] = pd.to_datetime(
    test_alerte[['annee', 'mois']].assign(day=1).rename(
        columns={'annee': 'year', 'mois': 'month'}))

# =============================================================================
# 4.5.1 — SEUILS CALCULÉS PAR DISTRICT
# =============================================================================

# ── C1a. Tableau seuils 12 mois — DS Bogodogo ────────────────────────────────
seuils_bogo = seuils_df[seuils_df['District'] == 'DS Bogodogo'].copy()
seuils_bogo = seuils_bogo.sort_values('Mois').reset_index(drop=True)

# Niveau dominant observé sur la période test
alerte_bogo_mois = test_alerte[test_alerte['Nom_DS'] == 'DS Bogodogo'].groupby('mois')['Niveau_Alerte'].apply(
    lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
).reset_index()
alerte_bogo_mois.columns = ['Mois', 'Niveau_dominant']
seuils_bogo = seuils_bogo.merge(alerte_bogo_mois, on='Mois', how='left')

tab_bogo = seuils_bogo[['Mois_label', 'Seuil_Alerte', 'Seuil_Epidemie',
                         'N_annees_norm', 'N_annees_excl', 'Niveau_dominant']].copy()
tab_bogo.columns = ['Mois', 'Seuil Alerte (Q2)', 'Seuil Épidémie (Q3)',
                    'Années normales', 'Années exclues', 'Niveau dominant']
# Forcer l'affichage à 6 décimales pour les seuils
tab_bogo['Seuil Alerte (Q2)']   = tab_bogo['Seuil Alerte (Q2)'].apply(lambda x: f"{x:.6f}")
tab_bogo['Seuil Épidémie (Q3)'] = tab_bogo['Seuil Épidémie (Q3)'].apply(lambda x: f"{x:.6f}")

couleurs_niv = {'ROUGE': '#FADBD8', 'ORANGE': '#FEF9E7',
                'VERT': '#D5F5E3', 'N/A': '#F2F3F4'}

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.axis('off')
tbl = ax.table(cellText=tab_bogo.values,
               colLabels=tab_bogo.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A252F')
        cell.set_text_props(color='white', fontweight='bold')
    elif r <= len(tab_bogo):
        niv = str(tab_bogo.iloc[r-1]['Niveau dominant'])
        cell.set_facecolor(couleurs_niv.get(niv, '#F2F3F4'))
    cell.set_edgecolor('#BDC3C7')
plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C1a_seuils_12mois_Bogodogo.png",
            dpi=DPI, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("  ✓ C1a_seuils_12mois_Bogodogo.png")

# ── C1b. Tableau seuils 5 districts représentatifs — mois de septembre ───────
rep5_alerte = list(dict.fromkeys(
    top5[:2] +
    [df.groupby('Nom_DS')[TARGET].mean().nsmallest(1).index[0]] +
    top5[2:4]
))[:5]

seuils_sept = seuils_df[
    (seuils_df['District'].isin(rep5_alerte)) &
    (seuils_df['Mois'] == 9)
][['District', 'Seuil_Alerte', 'Seuil_Epidemie',
   'N_annees_norm', 'N_annees_excl']].copy()
seuils_sept.columns = ['District', 'Seuil Alerte Q2',
                        'Seuil Épidémie Q3',
                        'Années normales', 'Années exclues (Tukey)']
# Forcer l'affichage à 6 décimales
seuils_sept['Seuil Alerte Q2']   = seuils_sept['Seuil Alerte Q2'].apply(lambda x: f"{x:.6f}")
seuils_sept['Seuil Épidémie Q3'] = seuils_sept['Seuil Épidémie Q3'].apply(lambda x: f"{x:.6f}")

fig, ax = plt.subplots(figsize=(14, 2.5))
ax.axis('off')
tbl = ax.table(cellText=seuils_sept.values,
               colLabels=seuils_sept.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A252F')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#F2F3F4')
    cell.set_edgecolor('#BDC3C7')
plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C1b_seuils_rep5_septembre.png",
            dpi=DPI, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("  ✓ C1b_seuils_rep5_septembre.png")

# ── C1c. Annexe — seuils tous districts septembre ────────────────────────────
seuils_tous = seuils_df[seuils_df['Mois'] == 9][
    ['District', 'Seuil_Alerte', 'Seuil_Epidemie']].copy()
seuils_tous.columns = ['District', 'Seuil Alerte Q2', 'Seuil Épidémie Q3']
seuils_tous = seuils_tous.sort_values('Seuil Épidémie Q3', ascending=False).reset_index(drop=True)
# Forcer l'affichage à 6 décimales
seuils_tous['Seuil Alerte Q2']   = seuils_tous['Seuil Alerte Q2'].apply(lambda x: f"{x:.6f}")
seuils_tous['Seuil Épidémie Q3'] = seuils_tous['Seuil Épidémie Q3'].apply(lambda x: f"{x:.6f}")

n_s = len(seuils_tous); mid_s = (n_s + 1) // 2
left_s  = seuils_tous.iloc[:mid_s].reset_index(drop=True)
right_s = seuils_tous.iloc[mid_s:].reset_index(drop=True)

fig, axes_s = plt.subplots(1, 2, figsize=(16, max(6, mid_s * 0.32 + 1.2)))
for ax_s, data_s, lbl_s in zip(axes_s, [left_s, right_s],
                                [f'1–{mid_s}', f'{mid_s+1}–{n_s}']):
    ax_s.axis('off')
    tbl = ax_s.table(cellText=data_s.values,
                     colLabels=data_s.columns,
                     cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1A252F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F2F3F4')
        cell.set_edgecolor('#BDC3C7')
    ax_s.set_title(f'Districts {lbl_s}', fontsize=9, fontweight='bold', pad=6)
plt.suptitle(
    "Annexe — Seuils d'alerte (mois de septembre, 70 districts)\n"
    "Méthode OMS 2018 | Tukey | Q2 = Seuil alerte | Q3 = Seuil épidémique",
    fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C1c_annexe_seuils_tous_districts.png",
            dpi=DPI, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("  ✓ C1c_annexe_seuils_tous_districts.png")

# =============================================================================
# 4.5.2 — ALERTES GÉNÉRÉES SUR LA PÉRIODE 2023–2025
# =============================================================================

# ── C2a. Tableau synthétique alertes ─────────────────────────────────────────
alerte_counts = test_alerte['Niveau_Alerte'].value_counts()
n_test = len(test_alerte)

synth_alerte = pd.DataFrame([
    ['VERT  (Normal)',    alerte_counts.get('VERT',   0),
     f"{round(alerte_counts.get('VERT',   0)/n_test*100, 1)}%",
     'Transmission dans les limites habituelles'],
    ['ORANGE (Alerte)',   alerte_counts.get('ORANGE', 0),
     f"{round(alerte_counts.get('ORANGE', 0)/n_test*100, 1)}%",
     'Vigilance accrue — mobilisation préventive'],
    ['ROUGE  (Épidémie)', alerte_counts.get('ROUGE',  0),
     f"{round(alerte_counts.get('ROUGE',  0)/n_test*100, 1)}%",
     'Intervention immédiate requise'],
    ['Total', n_test, '100%',
     f'{test_alerte["Nom_DS"].nunique()} districts, 2023–2025'],
], columns=['Niveau', 'Nb observations', 'Pourcentage', 'Signification'])

fig, ax = plt.subplots(figsize=(15, 2.8))
ax.axis('off')
tbl = ax.table(cellText=synth_alerte.values,
               colLabels=synth_alerte.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A252F')
        cell.set_text_props(color='white', fontweight='bold')
    elif r == 1: cell.set_facecolor('#D5F5E3')
    elif r == 2: cell.set_facecolor('#FEF9E7')
    elif r == 3: cell.set_facecolor('#FADBD8')
    elif r == 4: cell.set_facecolor('#EBF5FB')
    cell.set_edgecolor('#BDC3C7')
plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C2a_synth_alertes.png",
            dpi=DPI, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("  ✓ C2a_synth_alertes.png")

# ── C2b. Carte thermique alertes ROUGE par district et par mois ──────────────
alerte_pivot = test_alerte.groupby(['Nom_DS', 'mois'])['Niveau_Alerte'].apply(
    lambda x: (x == 'ROUGE').sum()
).unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 18))
im = ax.imshow(alerte_pivot.values, aspect='auto',
               cmap='RdYlGn_r', vmin=0,
               vmax=max(1, alerte_pivot.values.max()))
plt.colorbar(im, ax=ax, label='Nb de mois ROUGE (2023–2025)', shrink=0.4)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                    'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'], fontsize=9)
ax.set_yticks(range(len(alerte_pivot.index)))
ax.set_yticklabels(alerte_pivot.index, fontsize=8)
ax.set_xlabel("Mois", fontsize=10)
ax.set_ylabel("District sanitaire", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C2b_heatmap_alertes_rouge.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ C2b_heatmap_alertes_rouge.png")

# ── C2c. Figure temporelle nationale — alertes + cas ─────────────────────────
# .mean() sur incidence : moyenne inter-districts pour le niveau national
# .sum() serait incorrect — additionner des incidences n'a pas de sens
alerte_nat_grp = test_alerte.groupby('date').agg(
    Cas_reel  =('Cas_palu',      'mean'),
    Pred_XGB  =('Pred_XGB',      'mean'),
    Pred_Best =('Pred_Best',     'mean'),
    Nb_ROUGE  =('Niveau_Alerte', lambda x: (x == 'ROUGE').sum()),
    Nb_ORANGE =('Niveau_Alerte', lambda x: (x == 'ORANGE').sum()),
).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(alerte_nat_grp['date'], alerte_nat_grp['Cas_reel'],
         color='#2C3E50', linewidth=2, label='Cas réels', zorder=3)
ax1.plot(alerte_nat_grp['date'], alerte_nat_grp['Pred_XGB'],
         color=COULEURS['XGBoost'], linewidth=2, ls='--',
         label='XGBoost', zorder=3)
for _, row in alerte_nat_grp.iterrows():
    if row['Nb_ROUGE'] > len(ds_list) * 0.25:
        ax1.axvspan(row['date'] - pd.Timedelta(days=15),
                    row['date'] + pd.Timedelta(days=15),
                    alpha=0.15, color='#E74C3C', zorder=1)
    elif row['Nb_ORANGE'] > len(ds_list) * 0.25:
        ax1.axvspan(row['date'] - pd.Timedelta(days=15),
                    row['date'] + pd.Timedelta(days=15),
                    alpha=0.10, color='#F39C12', zorder=1)
ax1.set_ylabel("Cas de paludisme (national)", fontsize=10)
ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
ax1.spines[['top', 'right']].set_visible(False)

ax2.bar(alerte_nat_grp['date'], alerte_nat_grp['Nb_ROUGE'],
        color='#E74C3C', alpha=0.7, width=20, label='Districts ROUGE')
ax2.bar(alerte_nat_grp['date'], alerte_nat_grp['Nb_ORANGE'],
        color='#F39C12', alpha=0.5, width=20,
        bottom=alerte_nat_grp['Nb_ROUGE'], label='Districts ORANGE')
ax2.set_ylabel("Nb districts", fontsize=9)
ax2.set_xlabel("Période de test (2023–2025)", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C2c_alertes_periode_test.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ C2c_alertes_periode_test.png")

# =============================================================================
# 4.5.3 — VALIDATION RÉTROSPECTIVE
# =============================================================================

# Classer les cas réels selon les seuils
def cas_reel_rouge(row):
    match = seuils_df[
        (seuils_df['District'] == row['Nom_DS']) &
        (seuils_df['Mois']     == row['mois'])
    ]
    if match.empty:
        return False
    return row[TARGET] >= match['Seuil_Epidemie'].values[0]

test_alerte['Cas_reel_rouge'] = test_alerte.apply(cas_reel_rouge, axis=1)

vrais_pos = ((test_alerte['Niveau_Alerte'] == 'ROUGE') &
              (test_alerte['Cas_reel_rouge'] == True)).sum()
faux_pos  = ((test_alerte['Niveau_Alerte'] == 'ROUGE') &
              (test_alerte['Cas_reel_rouge'] == False)).sum()
faux_neg  = ((test_alerte['Niveau_Alerte'] != 'ROUGE') &
              (test_alerte['Cas_reel_rouge'] == True)).sum()
vrais_neg = ((test_alerte['Niveau_Alerte'] != 'ROUGE') &
              (test_alerte['Cas_reel_rouge'] == False)).sum()

sensibilite = round(vrais_pos / (vrais_pos + faux_neg) * 100, 1) if (vrais_pos + faux_neg) > 0 else 0
specificite = round(vrais_neg / (vrais_neg + faux_pos) * 100, 1) if (vrais_neg + faux_pos) > 0 else 0
vpp         = round(vrais_pos / (vrais_pos + faux_pos) * 100, 1) if (vrais_pos + faux_pos) > 0 else 0
vpn         = round(vrais_neg / (vrais_neg + faux_neg) * 100, 1) if (vrais_neg + faux_neg) > 0 else 0

# =============================================================================
# AMÉLIORATION 1 — F-BETA (β=2) SUR SEUIL ACTUEL Q3
# Calcul immédiat AVANT la figure — les scores servent aussi dans C3a
# =============================================================================
from sklearn.metrics import fbeta_score, f1_score

y_true        = test_alerte['Cas_reel_rouge'].astype(int).values
y_pred_actuel = (test_alerte['Niveau_Alerte'] == 'ROUGE').astype(int).values
y_score       = test_alerte['Pred_XGB'].values  # valeur continue XGBoost

f1_actuel = f1_score(y_true, y_pred_actuel)
f2_actuel = fbeta_score(y_true, y_pred_actuel, beta=2)

print("\n" + "=" * 65)
print("  AMÉLIORATION 1 — MÉTRIQUES F-BETA (Seuil actuel Q3)")
print("=" * 65)
print(f"  F1-score  : {f1_actuel:.4f}  (Précision = Rappel, équilibre)")
print(f"  F2-score  : {f2_actuel:.4f}  (Rappel prioritaire β=2)")
print(f"  → Référence de base avant optimisation du seuil")
print("=" * 65)

# ── C3a. Matrice de confusion visuelle ───────────────────────────────────────
fig, axes_cm = plt.subplots(1, 2, figsize=(14, 5))

# Matrice 2×2
cm_data = np.array([[vrais_pos, faux_neg],
                     [faux_pos,  vrais_neg]])
cm_labels = [['VP\n(Épidémie\ncorrectement\ndétectée)',
               'FN\n(Épidémie\nnon détectée)'],
             ['FP\n(Fausse\nalerte)',
              'VN\n(Normal\ncorrectement\nclassé)']]
cm_colors = np.array([['#D5F5E3', '#FADBD8'],
                       ['#FEF9E7', '#D5F5E3']])

ax = axes_cm[0]
ax.axis('off')
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                     facecolor=cm_colors[i][j], edgecolor='white', lw=2))
        ax.text(j + 0.5, 1.5 - i,
                f"{cm_labels[i][j]}\n\nn = {cm_data[i][j]}",
                ha='center', va='center', fontsize=9, fontweight='bold')

ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.text(0.5, 2.1, "Prédit ROUGE",     ha='center', fontsize=10, fontweight='bold')
ax.text(1.5, 2.1, "Prédit non ROUGE", ha='center', fontsize=10, fontweight='bold')
ax.text(-0.15, 1.5, "Réel\nROUGE",     ha='right', va='center', fontsize=10, fontweight='bold')
ax.text(-0.15, 0.5, "Réel non\nROUGE", ha='right', va='center', fontsize=10, fontweight='bold')

# Métriques — tableau incluant F1 et F2
ax = axes_cm[1]
ax.axis('off')
metriques_data = [
    ['Sensibilité (rappel)',       f"{sensibilite}%", 'Épidémies correctement détectées'],
    ['Spécificité',                f"{specificite}%", 'Situations normales correctement classées'],
    ['Valeur prédictive positive', f"{vpp}%",         'Précision des alertes ROUGE'],
    ['Valeur prédictive négative', f"{vpn}%",         'Fiabilité des non-alertes'],
    ['F1-score',                   f"{f1_actuel:.4f}", 'Équilibre Précision / Rappel'],
    ['F2-score (β=2)',             f"{f2_actuel:.4f}", 'Rappel prioritaire — référence SAP'],
]
tbl = ax.table(cellText=metriques_data,
               colLabels=['Indicateur', 'Valeur', 'Signification'],
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A252F')
        cell.set_text_props(color='white', fontweight='bold')
    elif r == 1: cell.set_facecolor('#D5F5E3')
    elif r == 2: cell.set_facecolor('#EBF5FB')
    elif r == 3: cell.set_facecolor('#EBF5FB')
    elif r == 4: cell.set_facecolor('#EBF5FB')
    elif r == 5: cell.set_facecolor('#FEF9E7')
    elif r == 6: cell.set_facecolor('#FEF9E7')
    cell.set_edgecolor('#BDC3C7')

plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C3a_validation_retrospective.png",
            dpi=DPI, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("  ✓ C3a_validation_retrospective.png")
print(f"    Sensibilité={sensibilite}% | Spécificité={specificite}% | "
      f"VPP={vpp}% | VPN={vpn}%")
print(f"    F1={f1_actuel:.4f} | F2={f2_actuel:.4f}")

# ── C3b. Figure — Districts non détectés vs fausses alertes ──────────────────
dist_faux_neg = test_alerte[
    (test_alerte['Niveau_Alerte'] != 'ROUGE') &
    (test_alerte['Cas_reel_rouge'] == True)
]['Nom_DS'].value_counts().head(10)

dist_faux_pos = test_alerte[
    (test_alerte['Niveau_Alerte'] == 'ROUGE') &
    (test_alerte['Cas_reel_rouge'] == False)
]['Nom_DS'].value_counts().head(10)

fig, axes_fn = plt.subplots(1, 2, figsize=(14, 6))

ax = axes_fn[0]
if len(dist_faux_neg) > 0:
    ax.barh(dist_faux_neg.index, dist_faux_neg.values,
            color='#FADBD8', edgecolor='white', height=0.65)
    for ds_n, val in zip(dist_faux_neg.index, dist_faux_neg.values):
        ax.text(val + 0.05, ds_n, str(val), va='center', fontsize=9)
ax.set_title("Faux négatifs — Épidémies non détectées\n(top 10 districts)",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Nombre de mois non détectés")
ax.grid(axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes_fn[1]
if len(dist_faux_pos) > 0:
    ax.barh(dist_faux_pos.index, dist_faux_pos.values,
            color='#FEF9E7', edgecolor='white', height=0.65)
    for ds_n, val in zip(dist_faux_pos.index, dist_faux_pos.values):
        ax.text(val + 0.05, ds_n, str(val), va='center', fontsize=9)
ax.set_title("Faux positifs — Fausses alertes\n(top 10 districts)",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Nombre de fausses alertes")
ax.grid(axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/C3b_faux_neg_faux_pos.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ C3b_faux_neg_faux_pos.png")

# =============================================================================
# EXPORT EXCEL — SECTION 4.5
# =============================================================================
with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl", mode='a',
                    if_sheet_exists='replace') as writer:
    seuils_df.to_excel(writer, sheet_name="Seuils_alerte", index=False)
    test_alerte[['Nom_DS', 'annee', 'mois', TARGET,
                 'Pred_XGB', 'Niveau_Alerte']].to_excel(
        writer, sheet_name="Alertes_test", index=False)

print(f"\n  Récapitulatif section 4.5 :")
print(f"  4.5.1 — Seuils calculés :")
print(f"    C1a — Seuils 12 mois DS Bogodogo (colorié par niveau)")
print(f"    C1b — Tableau 5 districts représentatifs (septembre)")
print(f"    C1c — Annexe : seuils 70 districts (septembre)")
print(f"  4.5.2 — Alertes générées :")
print(f"    C2a — Tableau synthétique VERT/ORANGE/ROUGE")
print(f"    C2b — Carte thermique alertes ROUGE (70 districts × 12 mois)")
print(f"    C2c — Figure temporelle nationale + barres districts en alerte")
print(f"  4.5.3 — Validation rétrospective :")
print(f"    C3a — Matrice de confusion + métriques (+ F1/F2)")
print(f"    C3b — Faux négatifs et faux positifs par district")
print(f"  Excel : Seuils_alerte, Alertes_test")

# =============================================================================
# AMÉLIORATION 2 — OPTIMISATION DU SEUIL VIA COURBE ROC + INDICE DE YOUDEN
# =============================================================================
# Principe : la courbe ROC est calculée sur un SCORE NORMALISÉ
# score = Pred_XGB / Seuil_Epidemie(district, mois)
# Ce ratio est comparable entre tous les districts et aligne les prédictions
# XGBoost avec la logique des seuils historiques OMS.
# score > 1.0 → prédiction au-dessus du seuil épidémique → tendance ROUGE
# score < 1.0 → prédiction en dessous → tendance VERT/ORANGE
# Référence : Youden W.J. (1950). Cancer, 3(1), 32-35.
# =============================================================================
from sklearn.metrics import roc_curve, auc

print("\n" + "=" * 65)
print("  AMÉLIORATION 2 — SEUIL OPTIMAL (ROC + YOUDEN)")
print("=" * 65)

# ── Calcul du score normalisé Pred_XGB / Seuil_Epidemie ──────────────────────
def get_score_normalise(row):
    match = seuils_df[
        (seuils_df['District'] == row['Nom_DS']) &
        (seuils_df['Mois']     == row['mois'])
    ]
    if match.empty:
        return np.nan
    se_val = match['Seuil_Epidemie'].values[0]
    if se_val == 0:
        return np.nan
    return row['Pred_XGB'] / se_val

test_alerte['Score_Norm'] = test_alerte.apply(get_score_normalise, axis=1)

# Filtrer les NaN
mask_valid  = test_alerte['Score_Norm'].notna()
y_true_roc  = y_true[mask_valid]
y_score_roc = test_alerte.loc[mask_valid, 'Score_Norm'].values

print(f"  Score normalisé : {mask_valid.sum()} obs valides / {len(y_true)} total")
print(f"  Score médian    : {np.median(y_score_roc):.4f}  "
      f"(1.0 = exactement au seuil Q3)")
print(f"  Score Q3 actuel correspond à ratio = 1.0 par construction")

# ── Courbe ROC sur score normalisé ────────────────────────────────────────────
fpr, tpr, thresholds_roc = roc_curve(y_true_roc, y_score_roc)
roc_auc = auc(fpr, tpr)

# Indice de Youden : J = TPR - FPR
youden       = tpr - fpr
idx_youden   = np.argmax(youden)
seuil_youden = thresholds_roc[idx_youden]   # ratio optimal Pred/Seuil_Q3

# ── Nouveau classement avec seuil Youden ─────────────────────────────────────
y_pred_youden = (y_score_roc >= seuil_youden).astype(int)

vp_y = int(((y_pred_youden == 1) & (y_true_roc == 1)).sum())
fn_y = int(((y_pred_youden == 0) & (y_true_roc == 1)).sum())
fp_y = int(((y_pred_youden == 1) & (y_true_roc == 0)).sum())
vn_y = int(((y_pred_youden == 0) & (y_true_roc == 0)).sum())

se_y  = round(vp_y / (vp_y + fn_y) * 100, 1) if (vp_y + fn_y) > 0 else 0
sp_y  = round(vn_y / (vn_y + fp_y) * 100, 1) if (vn_y + fp_y) > 0 else 0
vpp_y = round(vp_y / (vp_y + fp_y) * 100, 1) if (vp_y + fp_y) > 0 else 0
vpn_y = round(vn_y / (vn_y + fn_y) * 100, 1) if (vn_y + fn_y) > 0 else 0
f1_y  = f1_score(y_true_roc, y_pred_youden)
f2_y  = fbeta_score(y_true_roc, y_pred_youden, beta=2)

print(f"\n  AUC-ROC      : {roc_auc:.4f}")
print(f"  Seuil Youden : {seuil_youden:.4f}  "
      f"(ratio Pred/Seuil_Q3 | J = {youden[idx_youden]:.4f})")
print(f"  Interprétation : déclencher ROUGE si Pred_XGB >= {seuil_youden:.4f} × Seuil_Q3")
print(f"  ──────────────────────────────────────────")
print(f"  Sensibilité  : {se_y}%   "
      f"(était {sensibilite}%  → Δ = {round(se_y - sensibilite, 1):+}%)")
print(f"  Spécificité  : {sp_y}%   "
      f"(était {specificite}%  → Δ = {round(sp_y - specificite, 1):+}%)")
print(f"  VPP          : {vpp_y}%  | VPN : {vpn_y}%")
print(f"  F1-score     : {f1_y:.4f}  (Δ = {f1_y - f1_actuel:+.4f})")
print(f"  F2-score     : {f2_y:.4f}  (Δ = {f2_y - f2_actuel:+.4f})")
print(f"  VP={vp_y} | FN={fn_y} | FP={fp_y} | VN={vn_y}")
print("=" * 65)

# =============================================================================
# YOUDEN SAISONNIER — Option B
# Saison haute (Jul-Oct) : épidémies fréquentes → Youden agressif
# Saison basse  (Nov-Jun) : épidémies rares     → Q3 conservateur
# Facteur Youden calculé séparément sur chaque sous-période
# =============================================================================
MOIS_SAISON_HAUTE = [7, 8, 9, 10]
MOIS_SAISON_BASSE = [1, 2, 3, 4, 5, 6, 11, 12]

# Données filtrées par saison (sur test_alerte avec Score_Norm)
ta_valid = test_alerte[mask_valid].copy()
ta_valid['y_true_loc']  = y_true_roc
ta_valid['score_norm']  = y_score_roc

# ── Facteur Youden saison haute ───────────────────────────────────────────────
mask_sh = ta_valid['mois'].isin(MOIS_SAISON_HAUTE)
if mask_sh.sum() > 10 and ta_valid.loc[mask_sh, 'y_true_loc'].sum() > 5:
    fpr_sh, tpr_sh, thr_sh = roc_curve(
        ta_valid.loc[mask_sh, 'y_true_loc'],
        ta_valid.loc[mask_sh, 'score_norm'])
    idx_sh       = np.argmax(tpr_sh - fpr_sh)
    facteur_sh   = thr_sh[idx_sh]
    auc_sh       = auc(fpr_sh, tpr_sh)
else:
    facteur_sh = seuil_youden
    auc_sh     = roc_auc

# ── Facteur Youden saison basse ───────────────────────────────────────────────
mask_sb = ta_valid['mois'].isin(MOIS_SAISON_BASSE)
if mask_sb.sum() > 10 and ta_valid.loc[mask_sb, 'y_true_loc'].sum() > 5:
    fpr_sb, tpr_sb, thr_sb = roc_curve(
        ta_valid.loc[mask_sb, 'y_true_loc'],
        ta_valid.loc[mask_sb, 'score_norm'])
    idx_sb       = np.argmax(tpr_sb - fpr_sb)
    facteur_sb   = thr_sb[idx_sb]
    auc_sb       = auc(fpr_sb, tpr_sb)
else:
    facteur_sb = 1.0   # fallback Q3
    auc_sb     = roc_auc

print(f"\n  Youden saisonnier :")
print(f"  Saison haute (Jul-Oct) : facteur = {facteur_sh:.4f} | AUC = {auc_sh:.4f}")
print(f"  Saison basse (Nov-Jun) : facteur = {facteur_sb:.4f} | AUC = {auc_sb:.4f}")

# ── Classement avec Youden saisonnier ────────────────────────────────────────
def classifier_youden_saisonnier(row):
    match = seuils_df[
        (seuils_df['District'] == row['Nom_DS']) &
        (seuils_df['Mois']     == row['mois'])
    ]
    if match.empty:
        return 'INCONNU'
    se_val = match['Seuil_Epidemie'].values[0]
    sa_val = match['Seuil_Alerte'].values[0]
    if se_val == 0:
        return 'INCONNU'
    # Facteur selon saison
    facteur = facteur_sh if row['mois'] in MOIS_SAISON_HAUTE else facteur_sb
    seuil_rouge_sais = se_val * facteur
    seuil_orange     = sa_val
    pred = row['Pred_XGB']
    if pred >= seuil_rouge_sais: return 'ROUGE'
    elif pred >= seuil_orange:   return 'ORANGE'
    else:                        return 'VERT'

test_alerte['Alerte_Saisonnier'] = test_alerte.apply(
    classifier_youden_saisonnier, axis=1)

# Métriques Youden saisonnier
vp_s = int(((test_alerte['Alerte_Saisonnier'] == 'ROUGE') &
             (test_alerte['Cas_reel_rouge'] == True)).sum())
fn_s = int(((test_alerte['Alerte_Saisonnier'] != 'ROUGE') &
             (test_alerte['Cas_reel_rouge'] == True)).sum())
fp_s = int(((test_alerte['Alerte_Saisonnier'] == 'ROUGE') &
             (test_alerte['Cas_reel_rouge'] == False)).sum())
vn_s = int(((test_alerte['Alerte_Saisonnier'] != 'ROUGE') &
             (test_alerte['Cas_reel_rouge'] == False)).sum())

se_s  = round(vp_s / (vp_s + fn_s) * 100, 1) if (vp_s + fn_s) > 0 else 0
sp_s  = round(vn_s / (vn_s + fp_s) * 100, 1) if (vn_s + fp_s) > 0 else 0
vpp_s = round(vp_s / (vp_s + fp_s) * 100, 1) if (vp_s + fp_s) > 0 else 0
vpn_s = round(vn_s / (vn_s + fn_s) * 100, 1) if (vn_s + fn_s) > 0 else 0
y_pred_sais = (test_alerte['Alerte_Saisonnier'] == 'ROUGE').astype(int).values
f1_s  = f1_score(y_true, y_pred_sais)
f2_s  = fbeta_score(y_true, y_pred_sais, beta=2)

print(f"\n  YOUDEN SAISONNIER :")
print(f"  Sensibilité  : {se_s}%   (Δ vs Q3 = {round(se_s-sensibilite,1):+}%)")
print(f"  Spécificité  : {sp_s}%   (Δ vs Q3 = {round(sp_s-specificite,1):+}%)")
print(f"  VPP          : {vpp_s}%  | VPN : {vpn_s}%")
print(f"  F2-score     : {f2_s:.4f}  (Δ vs Q3 = {f2_s-f2_actuel:+.4f})")
print(f"  VP={vp_s} | FN={fn_s} | FP={fp_s} | VN={vn_s}")

# ── Figure C3c : Courbe ROC + tableau comparatif 3 seuils ────────────────────
fig, axes_roc = plt.subplots(1, 2, figsize=(18, 7))

# --- Panneau gauche : Courbe ROC globale ---
ax = axes_roc[0]
ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5,
        label=f'Courbe ROC globale (AUC = {roc_auc:.4f})')
ax.fill_between(fpr, tpr, alpha=0.08, color='#2E86AB')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2,
        label='Modèle aléatoire (AUC=0.5)')

# Point Youden global
ax.scatter(fpr[idx_youden], tpr[idx_youden],
           color='#E74C3C', s=180, zorder=5,
           label=f'Youden global\nSe={se_y}% | Sp={sp_y}%')
ax.annotate(f'Youden global\nSe={se_y}%\nSp={sp_y}%',
            xy=(fpr[idx_youden], tpr[idx_youden]),
            xytext=(fpr[idx_youden] + 0.08, tpr[idx_youden] - 0.12),
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8',
                      edgecolor='#E74C3C', alpha=0.9))

# Point Q3 actuel
idx_q3 = np.argmin(np.abs(thresholds_roc - 1.0))
ax.scatter(fpr[idx_q3], tpr[idx_q3],
           color='#F18F01', s=180, zorder=5, marker='s',
           label=f'Seuil Q3 (ratio=1.0)\nSe={sensibilite}% | Sp={specificite}%')

# Points Youden saisonniers sur courbes
if mask_sh.sum() > 10:
    ax.plot(fpr_sh, tpr_sh, color='#3BB273', linewidth=1.5,
            linestyle='-.', alpha=0.7,
            label=f'ROC saison haute (AUC={auc_sh:.3f})')
    ax.scatter(fpr_sh[idx_sh], tpr_sh[idx_sh],
               color='#3BB273', s=140, zorder=5, marker='^')

if mask_sb.sum() > 10:
    ax.plot(fpr_sb, tpr_sb, color='#A23B72', linewidth=1.5,
            linestyle='-.', alpha=0.7,
            label=f'ROC saison basse (AUC={auc_sb:.3f})')
    ax.scatter(fpr_sb[idx_sb], tpr_sb[idx_sb],
               color='#A23B72', s=140, zorder=5, marker='^')

ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)", fontsize=10)
ax.set_ylabel("Taux de Vrais Positifs (Sensibilité)", fontsize=10)
ax.set_title("(a) Courbes ROC (score normalisé, Pred/Seuil_Q3)", fontsize=11, loc='left')
ax.legend(fontsize=7.5, loc='lower right')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

# --- Panneau droit : tableau comparatif des 3 stratégies (style article) ---
ax = axes_roc[1]
ax.axis('off')
_lignes_roc = [
    ["Stratégie de seuil", "Se (%)", "Sp (%)", "VPP (%)", "VPN (%)", "F1", "F2"],
    ["\u2460 Q3 (OMS)",          f"{sensibilite}", f"{specificite}", f"{vpp}",   f"{vpn}",   f"{f1_actuel:.3f}", f"{f2_actuel:.3f}"],
    ["\u2461 Youden global",     f"{se_y}",        f"{sp_y}",        f"{vpp_y}", f"{vpn_y}", f"{f1_y:.3f}",      f"{f2_y:.3f}"],
    ["\u2462 Youden saisonnier", f"{se_s}",        f"{sp_s}",        f"{vpp_s}", f"{vpn_s}", f"{f1_s:.3f}",      f"{f2_s:.3f}"],
]
_nrows = len(_lignes_roc)
_x0, _w, _y0, _h = 0.02, 0.96, 0.34, 0.50
_tbl = ax.table(cellText=_lignes_roc, cellLoc='center', loc='center',
                colWidths=[0.30, 0.115, 0.115, 0.12, 0.12, 0.11, 0.12],
                bbox=[_x0, _y0, _w, _h])
_tbl.auto_set_font_size(False); _tbl.set_fontsize(10.5)
# Ligne de la meilleure stratégie selon F2 (mise en gras)
_f2_vals = {1: f2_actuel, 2: f2_y, 3: f2_s}
_best_row = max(_f2_vals, key=_f2_vals.get)
for (r, c), cell in _tbl.get_celld().items():
    cell.set_facecolor('white'); cell.set_edgecolor('none'); cell.set_linewidth(0)
    cell.PAD = 0.03
    if c == 0:
        cell.get_text().set_ha('left')
    if r == 0 or r == _best_row:
        cell.get_text().set_fontweight('bold')
_rowh = _h / _nrows
for _yy, _lw in [(_y0 + _h, 1.5), (_y0 + _h - _rowh, 0.7), (_y0, 1.5)]:
    ax.plot([_x0, _x0 + _w], [_yy, _yy], color='#1A1A1A', lw=_lw,
            clip_on=False, zorder=5, transform=ax.transAxes)
ax.set_title("(b) Comparaison des 3 stratégies de seuil", fontsize=11, loc='left')
ax.text(_x0, _y0 - 0.07,
        f"Note. Se = sensibilité ; Sp = spécificité ; VPP/VPN = valeurs prédictives.\n"
        f"F2 (\u03b2 = 2) privilégie le rappel (détection des épidémies). "
        f"AUC globale = {roc_auc:.3f}. En gras : meilleure stratégie selon F2.",
        transform=ax.transAxes, fontsize=8, style='italic', va='top')

plt.tight_layout()
plt.savefig(f"{OUT_ALERTE}/{_nom_fichier('Courbes ROC et comparaison des 3 seuils')}.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ Courbes ROC et comparaison des 3 seuils.png")

# =============================================================================
# ÉTAPE 8 — ANNEXES : TABLEAU DES MÉTRIQUES + TABLEAU DES BIBLIOTHÈQUES
# Tableaux « booktabs » à trois traits (style article), nommés par leur titre.
# Figures dans resultats/annexes/ et feuilles ajoutées à resultats_complets.xlsx.
# =============================================================================
print("\n[8/8] Génération des annexes (métriques + environnement logiciel)...")

import platform
import importlib
import importlib.metadata as _ilmd

OUT_ANNEXE = "resultats/annexes"
Path(OUT_ANNEXE).mkdir(parents=True, exist_ok=True)


def _fmt_cell(x, dec):
    """Formate une valeur numérique avec _num ; gère None/NaN proprement."""
    if x is None:
        return "\u2014"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "\u2014"
    except (TypeError, ValueError):
        pass
    try:
        return _num(float(x), dec)
    except (TypeError, ValueError):
        return str(x)


def _table_booktabs(ax, entetes, lignes, col_widths, note=None,
                    cell_loc='left', bold_premiere_col=False, fontsize=10,
                    y0=0.16, h=0.78):
    """Trace un tableau à trois traits (haut, sous-entête, bas) dans `ax`."""
    ax.axis('off')
    data = [entetes] + lignes
    nrows = len(data)
    x0, w = 0.02, 0.96
    tbl = ax.table(cellText=data, cellLoc=cell_loc, loc='center',
                   colWidths=col_widths, bbox=[x0, y0, w, h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('white')
        cell.set_edgecolor('none')
        cell.set_linewidth(0)
        cell.PAD = 0.04
        if r == 0:
            cell.get_text().set_fontweight('bold')
        if bold_premiere_col and c == 0 and r > 0:
            cell.get_text().set_fontweight('bold')

    rowh = h / nrows
    for yy, lw in [(y0 + h, 1.5), (y0 + h - rowh, 0.7), (y0, 1.5)]:
        ax.plot([x0, x0 + w], [yy, yy], color=NOIR, lw=lw,
                clip_on=False, zorder=5, transform=ax.transAxes)

    if note:
        ax.text(x0, y0 - 0.05, note, transform=ax.transAxes,
                fontsize=8, style='italic', va='top')


# -----------------------------------------------------------------------------
# A.1 — Tableau des métriques de performance (annexe)
# -----------------------------------------------------------------------------
# Métriques avec leurs formules mathématiques (et non leurs valeurs).
# yᵢ : incidence observée ; ŷᵢ : incidence prédite ; ȳ : moyenne observée ;
# n : nombre d'observations de la période de test.
metriques_formules = [
    ("MAE", "Erreur absolue moyenne",
     r"$\mathrm{MAE} = (1/n)\,\sum_i \left|\,y_i-\hat{y}_i\,\right|$"),
    ("RMSE", "Racine de l'erreur quadratique moyenne",
     r"$\mathrm{RMSE} = \sqrt{\,(1/n)\,\sum_i (y_i-\hat{y}_i)^{2}\,}$"),
    ("R\u00b2", "Coefficient de détermination",
     r"$R^{2} = 1 - \sum_i (y_i-\hat{y}_i)^{2} \;/\; \sum_i (y_i-\bar{y})^{2}$"),
    ("SMAPE", "Erreur absolue moyenne symétrique (%)",
     r"$\mathrm{SMAPE} = (100/n)\,\sum_i 2\,|\hat{y}_i-y_i| \;/\; (|y_i|+|\hat{y}_i|)$"),
]

entetes_m = ['Métrique', 'Désignation', 'Formule']
lignes_m = [[sigle, desig, formule] for sigle, desig, formule in metriques_formules]

fig, ax = plt.subplots(figsize=(14.5, 0.72 * (len(lignes_m) + 1) + 0.8))
_table_booktabs(ax, entetes_m, lignes_m,
                col_widths=[0.12, 0.42, 0.46],
                note=None, cell_loc='left',
                bold_premiere_col=True, fontsize=11.5,
                y0=0.12, h=0.82)
titre_m = "Annexe - Formules des metriques de performance"
fic_m = _nom_fichier(titre_m) + ".png"
fig.savefig(f"{OUT_ANNEXE}/{fic_m}", dpi=DPI,
            bbox_inches='tight', pad_inches=0.18)
plt.close(fig)
print(f"  \u2713 {fic_m}")

# Version DataFrame (pour Excel) — formules en notation texte
metriques_annexe_df = pd.DataFrame([
    {"Métrique": "MAE", "Désignation": "Erreur absolue moyenne",
     "Formule": "MAE = (1/n) \u00b7 \u03a3 |y\u1d62 \u2212 \u0177\u1d62|"},
    {"Métrique": "RMSE", "Désignation": "Racine de l'erreur quadratique moyenne",
     "Formule": "RMSE = \u221a[ (1/n) \u00b7 \u03a3 (y\u1d62 \u2212 \u0177\u1d62)\u00b2 ]"},
    {"Métrique": "R\u00b2", "Désignation": "Coefficient de détermination",
     "Formule": "R\u00b2 = 1 \u2212 [ \u03a3(y\u1d62 \u2212 \u0177\u1d62)\u00b2 / \u03a3(y\u1d62 \u2212 \u0233)\u00b2 ]"},
    {"Métrique": "SMAPE", "Désignation": "Erreur absolue moyenne symétrique (%)",
     "Formule": "SMAPE = (100/n) \u00b7 \u03a3 [ 2|\u0177\u1d62 \u2212 y\u1d62| / (|y\u1d62| + |\u0177\u1d62|) ]"},
])[['Métrique', 'Désignation', 'Formule']]


# -----------------------------------------------------------------------------
# A.2 — Tableau des bibliothèques et versions (annexe reproductibilité)
# -----------------------------------------------------------------------------
def _version_pkg(dist, module=None):
    """Récupère la version d'un paquet (métadonnées puis __version__)."""
    try:
        return _ilmd.version(dist)
    except Exception:
        pass
    try:
        mod = importlib.import_module(module or dist)
        return getattr(mod, '__version__', 'n.d.')
    except Exception:
        return 'n.d.'


# Version de R utilisée pour l'application interactive (fournie par l'utilisateur).
# Les versions des paquets R ne sont pas détectables depuis Python : renseigner
# « — » par la sortie de packageVersion("nom_du_paquet") dans R si besoin.
R_VERSION = "4.5.2"

# (Bibliothèque, Version, Environnement, Rôle)
pkgs = [
    ('Python',       platform.python_version(), 'Python',
     "Langage et environnement d'exécution"),
    ('pandas',       _version_pkg('pandas'), 'Python',
     "Manipulation des données tabulaires (DataFrame)"),
    ('NumPy',        _version_pkg('numpy'), 'Python',
     "Calcul numérique vectoriel"),
    ('Matplotlib',   _version_pkg('matplotlib'), 'Python',
     "Production des figures et tableaux"),
    ('SciPy',        _version_pkg('scipy'), 'Python',
     "Tests statistiques (Shapiro, Jarque\u2013Bera, Friedman, Nemenyi)"),
    ('scikit-learn', _version_pkg('scikit-learn', 'sklearn'), 'Python',
     "KNN, Random Forest, métriques, validation croisée temporelle"),
    ('XGBoost',      _version_pkg('xgboost'), 'Python',
     "Modèle XGBoost (gradient boosting)"),
    ('statsmodels',  _version_pkg('statsmodels'), 'Python',
     "SARIMA, test ADF, Ljung\u2013Box"),
    ('openpyxl',     _version_pkg('openpyxl'), 'Python',
     "Lecture et écriture des classeurs Excel"),
    ('R',            R_VERSION, 'R / Shiny',
     "Langage de l'application interactive"),
    ('reticulate',   '\u2014', 'R / Shiny',
     "Pont R \u2194 Python (exécution du pipeline d'analyse)"),
    ('shiny',        '\u2014', 'R / Shiny',
     "Cadre de l'application web interactive"),
    ('leaflet',      '\u2014', 'R / Shiny',
     "Cartographie interactive (cartes choroplèthes)"),
    ('sf',           '\u2014', 'R / Shiny',
     "Données spatiales (lecture des shapefiles, reprojection CRS 4326)"),
]

entetes_p = ['Bibliothèque', 'Version', 'Environnement',
             "Rôle dans la chaîne d'analyse"]
lignes_p = [[nom, ver, env, role] for nom, ver, env, role in pkgs]

fig, ax = plt.subplots(figsize=(14, 0.50 * (len(lignes_p) + 1) + 1.0))
_table_booktabs(ax, entetes_p, lignes_p,
                col_widths=[0.15, 0.10, 0.13, 0.62],
                note=None, cell_loc='left',
                bold_premiere_col=True, fontsize=10.5,
                y0=0.08, h=0.86)
titre_p = "Annexe - Bibliotheques et versions logicielles"
fic_p = _nom_fichier(titre_p) + ".png"
fig.savefig(f"{OUT_ANNEXE}/{fic_p}", dpi=DPI,
            bbox_inches='tight', pad_inches=0.18)
plt.close(fig)
print(f"  \u2713 {fic_p}")

packages_df = pd.DataFrame(pkgs,
                           columns=['Bibliothèque', 'Version', 'Environnement',
                                    "Rôle dans la chaîne d'analyse"])

# -----------------------------------------------------------------------------
# A.3 — Ajout des deux annexes au classeur Excel existant
# -----------------------------------------------------------------------------
try:
    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        metriques_annexe_df.to_excel(
            writer, sheet_name="Annexe_Métriques", index=False)
        packages_df.to_excel(
            writer, sheet_name="Annexe_Bibliothèques", index=False)
    print(f"  \u2713 Feuilles « Annexe_Métriques » et "
          f"« Annexe_Bibliothèques » ajoutées à {OUT_EXCEL}")
except Exception as e:
    # Repli CSV si l'ajout au classeur échoue
    metriques_annexe_df.to_csv(
        f"{OUT_ANNEXE}/annexe_metriques.csv", index=False, encoding="utf-8-sig")
    packages_df.to_csv(
        f"{OUT_ANNEXE}/annexe_bibliotheques.csv", index=False, encoding="utf-8-sig")
    print(f"  \u26a0 Ajout Excel impossible ({e}). Annexes exportées en CSV.")



# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  RÉSULTATS FINAUX")
print("=" * 65)
print("\nPerformances comparatives (période test 2023–2025) :\n")
print(metrics_df[['Rang', 'Modele', 'R2', 'RMSE', 'MAE', 'SMAPE']]
      .to_string(index=False))

print(f"\n🏆 Meilleur modèle : {best_name}")

if 'Residus_Normaux_SW' in diag_df.columns:
    n_diag = len(diag_df)
    print(f"\nDiagnostic SARIMA ({n_diag} districts) :")
    print(f"  Résidus normaux (SW)  : {diag_df['Residus_Normaux_SW'].sum()}/{n_diag}")
    print(f"  Non autocorrélés (LB) : {diag_df['Residus_NonAutoCorr'].sum()}/{n_diag}")

print(f"\n{'─' * 65}")
print(f"  MÉTRIQUES DE DÉTECTION — Seuil actuel (Q3)")
print(f"{'─' * 65}")
print(f"  Sensibilité : {sensibilite}%   — Épidémies correctement détectées")
print(f"  Spécificité : {specificite}%   — Situations normales correctement classées")
print(f"  VPP         : {vpp}%   — Précision des alertes ROUGE")
print(f"  VPN         : {vpn}%   — Fiabilité des non-alertes")
print(f"  ──────────────────────────────────────────")
print(f"  F1-score    : {f1_actuel:.4f}  — équilibre Précision/Rappel")
print(f"  F2-score    : {f2_actuel:.4f}  — Rappel prioritaire (β=2) ← référence SAP")
print(f"  ──────────────────────────────────────────")
print(f"  Interprétation :")
if f2_actuel > f1_actuel:
    print(f"  ✓ F2 ({f2_actuel:.4f}) > F1 ({f1_actuel:.4f})")
    print(f"    Le rappel (Se={sensibilite}%) est bien valorisé par β=2")
print(f"  → Marge d'amélioration : viser F2 > 0.65 via optimisation du seuil")

print(f"\n{'─' * 65}")
print(f"  MÉTRIQUES DE DÉTECTION — Seuil Youden (ROC optimisé)")
print(f"{'─' * 65}")
print(f"  Seuil Youden : {seuil_youden:.6f}  (AUC = {roc_auc:.4f})")
print(f"  Sensibilité  : {se_y}%   (Δ = {round(se_y - sensibilite, 1):+}% vs Q3)")
print(f"  Spécificité  : {sp_y}%   (Δ = {round(sp_y - specificite, 1):+}% vs Q3)")
print(f"  VPP          : {vpp_y}%  | VPN : {vpn_y}%")
print(f"  F1-score     : {f1_y:.4f}  (Δ = {f1_y - f1_actuel:+.4f} vs Q3)")
print(f"  F2-score     : {f2_y:.4f}  (Δ = {f2_y - f2_actuel:+.4f} vs Q3)")
if f2_y > f2_actuel:
    print(f"  ✓ Amélioration confirmée : F2 {f2_actuel:.4f} → {f2_y:.4f}")

print(f"\nFichiers produits :")
print(f"  {OUT_SARIMA}/")
print(f"  {OUT_MODELES}/")
print(f"  {OUT_XGB}/")
print(f"  {OUT_ALERTE}/")
print(f"  {OUT_ANNEXE}/")
print(f"  {OUT_EXCEL}")
print("\n" + "=" * 65)
print("  ANALYSE TERMINÉE")
print("=" * 65)