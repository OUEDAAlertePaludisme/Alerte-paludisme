# =============================================================================
# PIPELINE ML — PALUDISME BURKINA FASO
# Adapté pour reticulate (Shiny)
# Modèles : KNN | Random Forest | XGBoost | SARIMA (par district)
# Découpage temporel : train <= 2022 | test >= 2023
# Baseline : médiane robuste par district et par mois (règle de Tukey)
# Réf. OMS 2018, PNLP BF 2021, Cullen 1985, Tukey 1977
# =============================================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Objets globaux (persistants entre appels reticulate)
_models     = {}
_scaler     = None
_seuils_df  = None
_metrics_df = None
_best_name   = None
_second_name = None
_df          = None
_test_alert  = None

# Facteurs Youden (calculés une fois dans train_models, réutilisés à chaque
# changement de stratégie via apply_strategy())
_youden_factors = {
    'global':       1.0,   # ratio Pred/Seuil_Q3 optimal (global)
    'saison_haute': 1.0,   # Jul-Oct
    'saison_basse': 1.0,   # Nov-Jun
}
MOIS_SAISON_HAUTE = [7, 8, 9, 10]

# Jeu de variables IDENTIQUE à analyse_complete_v11 (15 variables au total
# après extension) : DS_encoded remplace DS_code (encodage identique),
# 'saison' retiré (redondant avec mois_sin + mois_cos).
FEATURES = ['DS_encoded', 'annee_norm', 'humid_c', 'pluie_mm', 'temp_c']
TARGET   = 'Cas_palu'

# -----------------------------------------------------------------------------
# 1. CHARGEMENT & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def load_data(path):
    global _df
    df = pd.read_excel(path, sheet_name="ML_Dataset", engine="openpyxl")
    corrections = {
        "DS Batié": "DS Batie", "DS Boussé": "DS Bousse", "DS Réo": "DS Reo",
        "DS Diébougou": "DS Diebougou", "DS Koupéla": "DS Koupela",
        "DS Léo": "DS Leo", "DS Pô": "DS Po", "DS Saponé": "DS Sapone",
        "DS Séguenega": "DS Seguenega", "DS Zabré": "DS Zabre",
        "DS Ziniare": "DS Ziniare"
    }
    df["Nom_DS"] = df["Nom_DS"].replace(corrections)
    # Encodage numérique du district — IDENTIQUE à analyse_complete_v11
    # (LabelEncoder trie les modalités par ordre alphabétique : même mapping
    # que l'ancien DS_code, corrélation = 1.0)
    df['DS_encoded'] = LabelEncoder().fit_transform(df['Nom_DS'])
    # Tri chronologique par district
    df = df.sort_values(['Nom_DS', 'annee', 'mois']).reset_index(drop=True)
    # Lag features climatiques (t-1, t-2) et cible (t-1, t-2, t-3)
    for col in ['humid_c', 'pluie_mm', 'temp_c']:
        df[f'{col}_lag1'] = df.groupby('Nom_DS')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('Nom_DS')[col].shift(2)
    for lag in [1, 2]:
        df[f'Cas_palu_lag{lag}'] = df.groupby('Nom_DS')[TARGET].shift(lag)
    # Encodage cyclique du mois
    df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
    df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)
    # Saison des pluies (mai-octobre = 1)
    df['saison'] = df['mois'].apply(lambda m: 1 if m in range(5, 11) else 0)
    # annee_norm : centré sur 2010 — évite extrapolation hors plage train
    df['annee_norm'] = df['annee'] - 2010
    df = df.dropna(subset=[
        'humid_c_lag1', 'humid_c_lag2',
        'pluie_mm_lag1', 'pluie_mm_lag2',
        'temp_c_lag1', 'temp_c_lag2',
        'Cas_palu_lag1', 'Cas_palu_lag2'
    ]).reset_index(drop=True)
    _df = df
    return df.shape[0]

# -----------------------------------------------------------------------------
# 2. ENTRAINEMENT
# Découpage : train <= 2022 | test >= 2023
# -----------------------------------------------------------------------------
def train_models():
    global _models, _scaler, _seuils_df, _metrics_df, _best_name, _second_name, _test_alert

    df = _df.copy()
    features_ext = FEATURES + [
        'humid_c_lag1', 'humid_c_lag2', 'pluie_mm_lag1', 'pluie_mm_lag2',
        'temp_c_lag1',  'temp_c_lag2',
        'Cas_palu_lag1', 'Cas_palu_lag2',
        'mois_sin', 'mois_cos'
    ]

    # -------------------------------------------------------------------------
    # Découpage temporel
    # -------------------------------------------------------------------------
    train = df[df['annee'] <= 2022].copy()
    test  = df[df['annee'] >= 2023].copy()

    X_train, y_train = train[features_ext], train[TARGET]
    X_test,  y_test  = test[features_ext],  test[TARGET]

    # -------------------------------------------------------------------------
    # KNN — sélection du k optimal par validation croisée (5-fold)
    # -------------------------------------------------------------------------
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit

    _scaler  = StandardScaler()
    X_tr_sc  = _scaler.fit_transform(X_train)
    X_te_sc  = _scaler.transform(X_test)

    k_values = [3, 5, 7, 9, 11, 15, 21]
    rmse_k   = []
    cv       = TimeSeriesSplit(n_splits=5)  # respecte l'ordre temporel

    for k in k_values:
        knn_cv = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
        scores = cross_val_score(knn_cv, X_tr_sc, y_train,
                                 cv=cv,
                                 scoring='neg_root_mean_squared_error')
        rmse_k.append(-scores.mean())

    best_k = k_values[int(np.argmin(rmse_k))]
    print(f"  KNN — k optimal = {best_k} (RMSE CV = {min(rmse_k):.2f})")

    knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
    knn.fit(X_tr_sc, y_train)
    pred_knn = knn.predict(X_te_sc)

    # -------------------------------------------------------------------------
    # Random Forest — RandomizedSearchCV + TimeSeriesSplit
    # -------------------------------------------------------------------------
    from sklearn.model_selection import RandomizedSearchCV

    tscv = TimeSeriesSplit(n_splits=5)

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
    print(f"  RF — meilleurs hyperparamètres : {search_rf.best_params_}")

    # -------------------------------------------------------------------------
    # XGBoost — RandomizedSearchCV + TimeSeriesSplit
    # -------------------------------------------------------------------------
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
    print(f"  XGB — meilleurs hyperparamètres : {search_xgb.best_params_}")

    # -------------------------------------------------------------------------
    # SARIMA par district — IDENTIQUE à analyse_complete_v11
    # Ordre uniforme SARIMA(1,d,1)(1,0,1)[12] (principe de parcimonie,
    # Box & Jenkins 1976). d déterminé district par district par test ADF
    # (d=0 si stationnaire p<0.05, sinon d=1). Aucune lecture de fichier
    # externe : les ordres sont recalculés à l'identique à chaque appel.
    # -------------------------------------------------------------------------
    def to_district_series(data, district):
        sub = (data[data['Nom_DS'] == district]
               .groupby(['annee', 'mois'])[TARGET].mean()  # mean() correct pour incidence
               .reset_index())
        sub['date'] = pd.to_datetime(
            sub[['annee', 'mois']].assign(day=1).rename(
                columns={'annee': 'year', 'mois': 'month'}))
        return sub.sort_values('date').set_index('date')[TARGET]

    S = 12
    SARIMA_P, SARIMA_Q = 1, 1   # ordre saisonnier uniforme
    SARIMA_p, SARIMA_q = 1, 1   # ordre non saisonnier uniforme

    districts        = sorted(df['Nom_DS'].unique())
    sarima_models     = {}
    sarima_all_real   = []
    sarima_all_pred   = []
    sarima_test_rows  = []

    for ds in districts:
        tr_s = to_district_series(train, ds)
        te_s = to_district_series(test,  ds)
        # Seuil identique à analyse_v11 : au moins 24 mois d'historique
        if len(tr_s) < 24 or tr_s.std() == 0 or len(te_s) == 0:
            continue

        # d déterminé individuellement par test ADF
        adf_p = adfuller(tr_s.dropna(), autolag='AIC')[1]
        d_opt = 0 if adf_p < 0.05 else 1

        try:
            fit = SARIMAX(
                tr_s,
                order=(SARIMA_p, d_opt, SARIMA_q),
                seasonal_order=(SARIMA_P, 0, SARIMA_Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            _full = pd.concat([tr_s, te_s])
            pred  = np.clip(np.asarray(fit.apply(_full).predict(start=len(tr_s), end=len(_full)-1, dynamic=False), float), 0, None)
        except Exception:
            # Fallback d=1 si la convergence échoue avec d=0
            try:
                fit = SARIMAX(
                    tr_s,
                    order=(SARIMA_p, 1, SARIMA_q),
                    seasonal_order=(SARIMA_P, 0, SARIMA_Q, S),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                _full = pd.concat([tr_s, te_s])
                pred  = np.clip(np.asarray(fit.apply(_full).predict(start=len(tr_s), end=len(_full)-1, dynamic=False), float), 0, None)
            except Exception:
                continue
        sarima_models[ds] = fit
        sarima_all_real.extend(te_s.values.tolist())
        sarima_all_pred.extend(pred.tolist())
        for i, (rv, pv) in enumerate(zip(te_s.values, pred)):
            sarima_test_rows.append({
                'Nom_DS':     ds,
                'annee':      int(te_s.index[i].year),
                'mois':       int(te_s.index[i].month),
                'Cas_palu':   float(rv),
                'Pred_SARIMA': round(float(pv), 6)
            })

    sarima_real     = np.array(sarima_all_real)
    sarima_pred_agg = np.array(sarima_all_pred)

    # -------------------------------------------------------------------------
    # Métriques
    # -------------------------------------------------------------------------
    def met(yt, yp, name):
        yt = np.array(yt, dtype=float); yp = np.array(yp, dtype=float)
        # Filtrer les paires valides — IDENTIQUE à analyse_v11
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]; yp = yp[mask]
        if len(yt) < 2:
            return {'Modele': name, 'MAE': None, 'RMSE': None,
                    'R2': None, 'SMAPE': None}
        # SMAPE — Symmetric Mean Absolute Percentage Error (Makridakis, 1993)
        # SMAPE = (100/n) · Σ 2·|yp − yt| / (|yt| + |yp|)   domaine [0 ; 200] %
        denom = np.abs(yt) + np.abs(yp)
        ratio = np.divide(2.0 * np.abs(yp - yt), denom,
                          out=np.zeros_like(denom), where=denom != 0)
        smape = np.mean(ratio) * 100
        return {'Modele': name,
                'MAE'  : round(mean_absolute_error(yt, yp), 2),
                'RMSE' : round(np.sqrt(mean_squared_error(yt, yp)), 2),
                'R2'   : round(r2_score(yt, yp), 4),
                'SMAPE': round(smape, 2)}

    rows = [
        met(y_test,     pred_knn,       'KNN'),
        met(y_test,     pred_rf,        'RandomForest'),
        met(y_test,     pred_xgb,       'XGBoost'),
        met(sarima_real, sarima_pred_agg, 'SARIMA')
    ]
    _metrics_df = pd.DataFrame(rows)

    # Classement multicritère — rang moyen sur les 4 métriques (agrégation de
    # rangs), IDENTIQUE à analyse_complete_v11.
    # MAE, RMSE, SMAPE : plus faible = meilleur | R2 : plus élevé = meilleur.
    # Départage : RMSE croissante. Le meilleur modèle ne dépend plus du seul R2.
    _metrics_df['_r_MAE']   = _metrics_df['MAE'].rank(ascending=True)
    _metrics_df['_r_RMSE']  = _metrics_df['RMSE'].rank(ascending=True)
    _metrics_df['_r_SMAPE'] = _metrics_df['SMAPE'].rank(ascending=True)
    _metrics_df['_r_R2']    = _metrics_df['R2'].rank(ascending=False)
    _metrics_df['Rang_moyen'] = _metrics_df[['_r_MAE', '_r_RMSE',
                                             '_r_SMAPE', '_r_R2']].mean(axis=1)
    _metrics_df = (_metrics_df
                   .sort_values(['Rang_moyen', 'RMSE'], ascending=[True, True])
                   .drop(columns=['_r_MAE', '_r_RMSE', '_r_SMAPE', '_r_R2'])
                   .reset_index(drop=True))
    _metrics_df['Rang_moyen'] = _metrics_df['Rang_moyen'].round(2)
    _metrics_df.insert(0, 'Rang', range(1, len(_metrics_df) + 1))

    ml_m         = _metrics_df[_metrics_df['Modele'] != 'SARIMA'].reset_index(drop=True)
    _best_name   = ml_m.iloc[0]['Modele']
    _second_name = ml_m.iloc[1]['Modele'] if len(ml_m) > 1 else None

    all_preds   = {'KNN': pred_knn, 'RandomForest': pred_rf, 'XGBoost': pred_xgb}
    best_preds  = all_preds[_best_name]
    second_preds = all_preds[_second_name] if _second_name else None

    _models = {
        'KNN':             knn,
        'RandomForest':    rf,
        'XGBoost':         xgb,
        'SARIMA':          sarima_models,
        'sarima_test_rows': sarima_test_rows,
        'scaler':          _scaler,
        'features_ext':    features_ext,
        'second_name':     _second_name,
    }

    # -------------------------------------------------------------------------
    # Seuils d'alerte par district ET par mois — Méthode OMS 2018 (quartiles)
    #
    # LOGIQUE IDENTIQUE à analyse_complete_v11 :
    # la baseline est calculée sur une fenêtre FIXE 2018-2022 (5 ans post-
    # interventions PNLP), et NON sur tout le train (2013-2022). C'est ce
    # point qui alignait la sensibilité entre les deux scripts : calculer les
    # seuils sur tout l'historique décale Q3, donc y_true et le classement.
    #
    #   Seuil_Vigilance = Q1  (75e percentile inférieur, niveau VIGILANCE)
    #   Seuil_Alerte    = Q2  (médiane) — franchissement → alerte
    #   Seuil_Epidemie  = Q3  (75e percentile) → intervention requise
    #
    # Années épidémiques exclues par la règle de Tukey (borne haute).
    # Réf. : OMS 2018 (Malaria surveillance manual), Cullen 1985, Tukey 1977,
    #         PNLP BF 2021.
    # -------------------------------------------------------------------------
    ANNEE_BASELINE_MIN = 2018
    ANNEE_BASELINE_MAX = 2022
    train_alerte = train[
        (train['annee'] >= ANNEE_BASELINE_MIN) &
        (train['annee'] <= ANNEE_BASELINE_MAX)
    ]
    ds_list = sorted(df['Nom_DS'].unique())

    rows_seuils = []
    for ds in ds_list:
        for m in range(1, 13):
            serie_hist = train_alerte[
                (train_alerte['Nom_DS'] == ds) &
                (train_alerte['mois']   == m)
            ][TARGET]
            # Repli : si moins de 3 années sur 2018-2022, élargir à tout le
            # train (<= 2022), comme dans analyse_complete_v11.
            if len(serie_hist) < 3:
                serie_hist = train[
                    (train['Nom_DS'] == ds) &
                    (train['mois']   == m)
                ][TARGET]
            if len(serie_hist) < 2:
                continue
            # Règle de Tukey : exclusion des années épidémiques (borne haute)
            q1v = serie_hist.quantile(0.25)
            q3v = serie_hist.quantile(0.75)
            iqr = q3v - q1v
            serie_norm = serie_hist[serie_hist <= q3v + 1.5 * iqr]
            if len(serie_norm) < 2:
                serie_norm = serie_hist
            baseline        = round(float(serie_norm.quantile(0.50)), 6)  # Q2
            seuil_vigilance = round(float(serie_norm.quantile(0.25)), 6)  # Q1
            seuil_epidemie  = round(float(serie_norm.quantile(0.75)), 6)  # Q3
            rows_seuils.append({
                'Nom_DS':          ds,
                'mois':            m,
                'Baseline':        baseline,
                'Seuil_Vigilance': seuil_vigilance,  # Q1 → VIGILANCE
                'Seuil_Alerte':    baseline,          # Q2 = médiane → ALERTE
                'Seuil_Epidemie':  seuil_epidemie     # Q3 → EPIDEMIE
            })

    seuils    = pd.DataFrame(rows_seuils)
    _seuils_df = seuils

    # -------------------------------------------------------------------------
    # Données de test enrichies avec prédictions et niveaux d'alerte
    # Jointure sur ['Nom_DS', 'mois'] car la baseline est mensuelle
    # -------------------------------------------------------------------------
    ta = test.copy()
    ta['Pred']        = best_preds.round(6)
    if second_preds is not None:
        ta['Pred_2']  = second_preds.round(6)
    ta = ta.merge(seuils, on=['Nom_DS', 'mois'], how='left')

    def assign(p, row, facteur_rouge=1.0):
        """
        Classifie une prédiction selon les seuils OMS avec facteur Youden optionnel.
        facteur_rouge : ratio appliqué au Seuil_Epidemie (Q3)
            1.0           → Seuil Q3 standard (OMS)
            < 1.0         → Seuil abaissé (Youden) → plus sensible
        """
        seuil_rouge_eff = row['Seuil_Epidemie'] * facteur_rouge
        if p >= seuil_rouge_eff:         return 'EPIDEMIE'
        elif p >= row['Seuil_Alerte']:   return 'ALERTE'
        elif p >= row['Seuil_Vigilance']: return 'VIGILANCE'
        else:                             return 'NORMAL'

    # ── Classification initiale avec seuil Q3 (stratégie par défaut) ──────────
    ta['Niveau_Alerte'] = ta.apply(lambda r: assign(r['Pred'], r, 1.0), axis=1)

    if second_preds is not None:
        ta['Niveau_Alerte_2'] = ta.apply(
            lambda r: assign(r['Pred_2'], r, 1.0), axis=1)
        ta['Concordance'] = ta['Niveau_Alerte'] == ta['Niveau_Alerte_2']

    # ── Calcul des facteurs Youden (stockés pour apply_strategy) ─────────────
    try:
        from sklearn.metrics import roc_curve
        import numpy as _np

        # Score normalisé : Pred / Seuil_Epidemie
        ta_tmp = ta.merge(seuils[['Nom_DS', 'mois', 'Seuil_Epidemie']],
                          on=['Nom_DS', 'mois'], how='left',
                          suffixes=('', '_ref'))
        se_col = 'Seuil_Epidemie_ref' if 'Seuil_Epidemie_ref' in ta_tmp.columns \
                 else 'Seuil_Epidemie'
        ta_tmp['score_norm'] = _np.where(
            ta_tmp[se_col] > 0,
            ta_tmp['Pred'] / ta_tmp[se_col],
            _np.nan
        )
        # Vrai positif : cas réel >= Seuil_Epidemie
        ta_tmp['y_true'] = (ta_tmp[TARGET] >= ta_tmp[se_col]).astype(int)

        mask = ta_tmp['score_norm'].notna()
        if mask.sum() > 20 and ta_tmp.loc[mask, 'y_true'].sum() > 5:
            y_t   = ta_tmp.loc[mask, 'y_true'].values
            y_s   = ta_tmp.loc[mask, 'score_norm'].values
            fpr_g, tpr_g, thr_g = roc_curve(y_t, y_s)
            idx_g = _np.argmax(tpr_g - fpr_g)
            _youden_factors['global'] = float(thr_g[idx_g])

            # Saisonnier haute
            mask_sh = mask & ta_tmp['mois'].isin(MOIS_SAISON_HAUTE)
            if mask_sh.sum() > 10 and ta_tmp.loc[mask_sh, 'y_true'].sum() > 3:
                fpr_sh, tpr_sh, thr_sh = roc_curve(
                    ta_tmp.loc[mask_sh, 'y_true'].values,
                    ta_tmp.loc[mask_sh, 'score_norm'].values)
                _youden_factors['saison_haute'] = float(
                    thr_sh[_np.argmax(tpr_sh - fpr_sh)])

            # Saisonnier basse
            mask_sb = mask & ~ta_tmp['mois'].isin(MOIS_SAISON_HAUTE)
            if mask_sb.sum() > 10 and ta_tmp.loc[mask_sb, 'y_true'].sum() > 3:
                fpr_sb, tpr_sb, thr_sb = roc_curve(
                    ta_tmp.loc[mask_sb, 'y_true'].values,
                    ta_tmp.loc[mask_sb, 'score_norm'].values)
                _youden_factors['saison_basse'] = float(
                    thr_sb[_np.argmax(tpr_sb - fpr_sb)])

        print(f"  Facteurs Youden calculés :")
        print(f"    Global       : {_youden_factors['global']:.4f}")
        print(f"    Saison haute : {_youden_factors['saison_haute']:.4f}")
        print(f"    Saison basse : {_youden_factors['saison_basse']:.4f}")
    except Exception as e:
        print(f"  ⚠ Calcul Youden échoué : {e} — facteurs = 1.0 (Q3)")

    _test_alert = ta

    return _best_name

# -----------------------------------------------------------------------------
# 3. ACCESSEURS (appelés depuis R via reticulate)
# -----------------------------------------------------------------------------
def get_metrics():
    return _metrics_df.to_dict('records')

def get_strategies_metrics():
    """Métriques de détection par stratégie de seuil, calculées sur
    l'échantillon test. Référence épidémique : Cas_palu >= Seuil_Epidemie (Q3).
    Remplace les valeurs autrefois codées en dur dans l'interface Shiny :
    l'application affiche ainsi toujours les chiffres réellement obtenus.
    Retourne id, nom, couleur de fond, Se, Sp, VPP, VPN et F2 (%) par stratégie.
    """
    if _test_alert is None or 'Seuil_Epidemie' not in _test_alert.columns:
        return []
    ta   = _test_alert
    pred  = ta['Pred'].values.astype(float)
    seuil = ta['Seuil_Epidemie'].values.astype(float)
    mois  = ta['mois'].values
    y_true = (ta[TARGET].values >= seuil).astype(int)

    def _facteur_saison(m):
        return (_youden_factors['saison_haute'] if m in MOIS_SAISON_HAUTE
                else _youden_factors['saison_basse'])

    strategies = [
        ('Q3',                '① Seuil Q3',          '#FEF9E7',
         np.full(len(ta), 1.0)),
        ('YOUDEN_GLOBAL',     '② Youden global',     '#EBF5FB',
         np.full(len(ta), _youden_factors['global'])),
        ('YOUDEN_SAISONNIER', '③ Youden saisonnier', '#EAFAF1',
         np.array([_facteur_saison(m) for m in mois])),
    ]

    def _conf(y_pred):
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        se  = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
        sp  = 100.0 * tn / (tn + fp) if (tn + fp) else 0.0
        vpp = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
        vpn = 100.0 * tn / (tn + fn) if (tn + fn) else 0.0
        # F2 (β=2) : privilégie la sensibilité — F2 = 5·TP / (5·TP + 4·FN + FP)
        f2  = (5.0 * tp / (5.0 * tp + 4.0 * fn + fp)) \
              if (5.0 * tp + 4.0 * fn + fp) else 0.0
        return se, sp, vpp, vpn, f2

    out = []
    for sid, nom, bg, fact in strategies:
        y_pred = (pred >= seuil * fact).astype(int)
        se, sp, vpp, vpn, f2 = _conf(y_pred)
        out.append({
            'id':  sid, 'nom': nom, 'bg': bg,
            'se':  round(se, 1),  'sp':  round(sp, 1),
            'vpp': round(vpp, 1), 'vpn': round(vpn, 1),
            'f2':  round(f2, 3),
        })
    return out

def get_best_name():
    return _best_name

def get_seuils():
    return _seuils_df.to_dict('records')

def get_test_alert():
    cols = ['Nom_DS', 'annee', 'mois', 'Cas_palu', 'Pred',
            'Baseline', 'Seuil_Vigilance', 'Seuil_Alerte', 'Seuil_Epidemie', 'Niveau_Alerte']
    if 'Pred_2' in _test_alert.columns:
        cols += ['Pred_2', 'Niveau_Alerte_2', 'Concordance']
    return _test_alert[cols].to_dict('records')

def get_second_name():
    """Nom du deuxième meilleur modèle ML."""
    return _second_name

def get_sarima_test():
    """Prédictions SARIMA par district sur la période de test."""
    return _models.get('sarima_test_rows', [])

def get_district_series(district):
    sub = _df[_df['Nom_DS'] == district].sort_values(['annee', 'mois'])
    return sub[['annee', 'mois', TARGET]].to_dict('records')

def get_feature_importance():
    model = _models.get(_best_name)
    if model is None or not hasattr(model, 'feature_importances_'):
        return []
    feats = _models['features_ext']
    imp = list(zip(feats, model.feature_importances_.tolist()))
    imp.sort(key=lambda x: x[1], reverse=True)
    return [{'feature': f, 'importance': round(float(v), 4)} for f, v in imp]

def get_all_districts():
    return sorted(_df['Nom_DS'].unique().tolist())

# -----------------------------------------------------------------------------
# NOUVELLES FONCTIONS — Stratégie de seuil
# -----------------------------------------------------------------------------

def get_youden_factors():
    """
    Retourne les facteurs Youden calculés lors de l'entraînement.
    Utilisé par R pour afficher les valeurs dans l'UI.
    """
    return {
        'global':       round(_youden_factors['global'], 4),
        'saison_haute': round(_youden_factors['saison_haute'], 4),
        'saison_basse': round(_youden_factors['saison_basse'], 4),
    }

def apply_strategy(strategy="Q3"):
    """
    Reclassifie _test_alert selon la stratégie choisie par le décideur.

    Stratégies disponibles :
      "Q3"               → Seuil OMS standard (Seuil_Epidemie × 1.0)
      "YOUDEN_GLOBAL"    → Seuil Youden global (Seuil_Epidemie × facteur_global)
      "YOUDEN_SAISONNIER"→ Seuil Youden saisonnier (facteur haute/basse selon mois)

    Retourne le nom de la stratégie appliquée.
    """
    import numpy as np

    if _test_alert is None or _seuils_df is None:
        return "ERREUR : modèles non entraînés"

    def get_facteur(mois_val):
        if strategy == "Q3":
            return 1.0
        elif strategy == "YOUDEN_GLOBAL":
            return _youden_factors['global']
        elif strategy == "YOUDEN_SAISONNIER":
            return (_youden_factors['saison_haute']
                    if mois_val in MOIS_SAISON_HAUTE
                    else _youden_factors['saison_basse'])
        else:
            return 1.0

    def assign_strat(row):
        match = _seuils_df[
            (_seuils_df['Nom_DS'] == row['Nom_DS']) &
            (_seuils_df['mois']   == row['mois'])
        ]
        if match.empty:
            return 'NORMAL'
        s = match.iloc[0]
        facteur           = get_facteur(row['mois'])
        seuil_rouge_eff   = s['Seuil_Epidemie'] * facteur
        pred              = row['Pred']
        if pred >= seuil_rouge_eff:        return 'EPIDEMIE'
        elif pred >= s['Seuil_Alerte']:    return 'ALERTE'
        elif pred >= s['Seuil_Vigilance']: return 'VIGILANCE'
        else:                              return 'NORMAL'

    _test_alert['Niveau_Alerte'] = _test_alert.apply(assign_strat, axis=1)

    if 'Pred_2' in _test_alert.columns:
        def assign_strat2(row):
            match = _seuils_df[
                (_seuils_df['Nom_DS'] == row['Nom_DS']) &
                (_seuils_df['mois']   == row['mois'])
            ]
            if match.empty:
                return 'NORMAL'
            s = match.iloc[0]
            facteur         = get_facteur(row['mois'])
            seuil_rouge_eff = s['Seuil_Epidemie'] * facteur
            pred2           = row['Pred_2']
            if pred2 >= seuil_rouge_eff:       return 'EPIDEMIE'
            elif pred2 >= s['Seuil_Alerte']:   return 'ALERTE'
            elif pred2 >= s['Seuil_Vigilance']: return 'VIGILANCE'
            else:                              return 'NORMAL'

        _test_alert['Niveau_Alerte_2'] = _test_alert.apply(assign_strat2, axis=1)
        _test_alert['Concordance']     = (
            _test_alert['Niveau_Alerte'] == _test_alert['Niveau_Alerte_2'])

    print(f"  Stratégie appliquée : {strategy}")
    counts = _test_alert['Niveau_Alerte'].value_counts().to_dict()
    print(f"  Distribution : {counts}")
    return strategy

def apply_strategy_future(future_results, strategy="Q3"):
    """
    Reclassifie les prédictions futures selon la stratégie choisie.
    future_results : liste de dicts retournée par predict_future()
    Retourne la liste mise à jour.
    """
    if _seuils_df is None:
        return future_results

    def get_facteur(mois_val):
        if strategy == "Q3":
            return 1.0
        elif strategy == "YOUDEN_GLOBAL":
            return _youden_factors['global']
        elif strategy == "YOUDEN_SAISONNIER":
            return (_youden_factors['saison_haute']
                    if mois_val in MOIS_SAISON_HAUTE
                    else _youden_factors['saison_basse'])
        return 1.0

    for r in future_results:
        s_row = _seuils_df[
            (_seuils_df['Nom_DS'] == r['Nom_DS']) &
            (_seuils_df['mois']   == r['mois'])
        ]
        if s_row.empty:
            continue
        s               = s_row.iloc[0]
        facteur         = get_facteur(r['mois'])
        seuil_rouge_eff = s['Seuil_Epidemie'] * facteur
        pred            = r['Pred_ML']
        if pred >= seuil_rouge_eff:        r['Niveau_Alerte'] = 'EPIDEMIE'
        elif pred >= s['Seuil_Alerte']:    r['Niveau_Alerte'] = 'ALERTE'
        elif pred >= s['Seuil_Vigilance']: r['Niveau_Alerte'] = 'VIGILANCE'
        else:                              r['Niveau_Alerte'] = 'NORMAL'

        if 'Pred_ML2' in r:
            pred2 = r['Pred_ML2']
            if pred2 >= seuil_rouge_eff:       r['Niveau_Alerte_2'] = 'EPIDEMIE'
            elif pred2 >= s['Seuil_Alerte']:   r['Niveau_Alerte_2'] = 'ALERTE'
            elif pred2 >= s['Seuil_Vigilance']: r['Niveau_Alerte_2'] = 'VIGILANCE'
            else:                              r['Niveau_Alerte_2'] = 'NORMAL'
            r['Concordance'] = r['Niveau_Alerte'] == r['Niveau_Alerte_2']

    return future_results

# -----------------------------------------------------------------------------
# 4. PRÉDICTION FUTURE
# Prédiction itérative des n_months prochains mois pour tous les districts.
# Les modèles ML et SARIMA prédisent en parallèle.
# Le niveau d'alerte est comparé à la baseline du mois correspondant.
# -----------------------------------------------------------------------------
def predict_future(n_months=6):
    if _df is None or _best_name is None:
        return []

    model        = _models[_best_name]
    second_name  = _models.get('second_name')
    model_second = _models.get(second_name) if second_name else None
    feats        = _models['features_ext']
    results      = []

    last_year  = int(_df['annee'].max())
    last_month = int(_df[_df['annee'] == last_year]['mois'].max())

    for ds in _df['Nom_DS'].unique():
        sub = _df[_df['Nom_DS'] == ds].sort_values(['annee', 'mois']).tail(6).copy()

        # Série historique pour SARIMA (prédiction itérative)
        sarima_model  = _models['SARIMA'].get(ds)
        sarima_history = (
            _df[_df['Nom_DS'] == ds]
               .groupby(['annee', 'mois'])[TARGET].mean()  # mean() correct pour incidence
               .reset_index()
        )
        sarima_history['date'] = pd.to_datetime(
            sarima_history[['annee', 'mois']].assign(day=1).rename(
                columns={'annee': 'year', 'mois': 'month'}))
        sarima_series = sarima_history.sort_values('date').set_index('date')[TARGET].copy()

        for i in range(n_months):
            nm = last_month + i + 1
            ny = last_year + (nm - 1) // 12
            nm = ((nm - 1) % 12) + 1

            # --- Prédiction ML ---
            row = sub.iloc[-1].copy()
            row['annee']      = ny
            row['annee_norm'] = ny - 2010
            row['mois']       = nm
            row['humid_c_lag1']  = sub.iloc[-1]['humid_c']
            row['humid_c_lag2']  = sub.iloc[-2]['humid_c']  if len(sub) >= 2 else sub.iloc[-1]['humid_c']
            row['pluie_mm_lag1'] = sub.iloc[-1]['pluie_mm']
            row['pluie_mm_lag2'] = sub.iloc[-2]['pluie_mm'] if len(sub) >= 2 else sub.iloc[-1]['pluie_mm']
            row['temp_c_lag1']   = sub.iloc[-1]['temp_c']
            row['temp_c_lag2']   = sub.iloc[-2]['temp_c']   if len(sub) >= 2 else sub.iloc[-1]['temp_c']
            row['Cas_palu_lag1'] = sub.iloc[-1][TARGET]
            row['Cas_palu_lag2'] = sub.iloc[-2][TARGET]     if len(sub) >= 2 else sub.iloc[-1][TARGET]
            row['mois_sin'] = np.sin(2 * np.pi * nm / 12)
            row['mois_cos'] = np.cos(2 * np.pi * nm / 12)

            X = pd.DataFrame([row[feats]])
            if _best_name == 'KNN':
                X = _models['scaler'].transform(X)
            pred_ml = max(0, float(model.predict(X)[0]))

            # --- Prédiction 2ème modèle ---
            pred_ml2 = None
            if model_second is not None:
                X2 = pd.DataFrame([row[feats]])
                if second_name == 'KNN':
                    X2 = _models['scaler'].transform(X2)
                pred_ml2 = max(0, float(model_second.predict(X2)[0]))

            # --- Prédiction SARIMA (mise à jour itérative) ---
            pred_sarima = None
            if sarima_model is not None:
                try:
                    updated    = sarima_model.apply(sarima_series)
                    pred_sarima = max(0, float(updated.forecast(steps=1).iloc[0]))
                    new_date   = pd.Timestamp(year=ny, month=nm, day=1)
                    sarima_series = pd.concat([
                        sarima_series,
                        pd.Series([pred_sarima], index=[new_date])
                    ])
                except Exception:
                    pred_sarima = None

            # --- Niveau d'alerte comparé à la baseline du mois nm ---
            seuil_row = _seuils_df[
                (_seuils_df['Nom_DS'] == ds) & (_seuils_df['mois'] == nm)
            ]
            if seuil_row.empty:
                alerte = 'NORMAL'
            else:
                s = seuil_row.iloc[0]
                if pred_ml >= s['Seuil_Epidemie']:    alerte = 'EPIDEMIE'
                elif pred_ml >= s['Seuil_Alerte']:    alerte = 'ALERTE'
                elif pred_ml >= s['Seuil_Vigilance']: alerte = 'VIGILANCE'
                else:                                  alerte = 'NORMAL'

            result = {
                'Nom_DS':        ds,
                'annee':         int(ny),
                'mois':          int(nm),
                'Pred_ML':       round(pred_ml, 6),
                'Niveau_Alerte': alerte
            }
            if pred_ml2 is not None:
                if seuil_row.empty:
                    alerte2 = 'NORMAL'
                else:
                    s = seuil_row.iloc[0]
                    if pred_ml2 >= s['Seuil_Epidemie']:    alerte2 = 'EPIDEMIE'
                    elif pred_ml2 >= s['Seuil_Alerte']:    alerte2 = 'ALERTE'
                    elif pred_ml2 >= s['Seuil_Vigilance']: alerte2 = 'VIGILANCE'
                    else:                                   alerte2 = 'NORMAL'
                result['Pred_ML2']        = round(pred_ml2, 6)
                result['Niveau_Alerte_2'] = alerte2
                result['Concordance']     = alerte == alerte2
            if pred_sarima is not None:
                result['Pred_SARIMA'] = round(pred_sarima, 6)
            results.append(result)

            # La prédiction ML alimente le pas suivant
            new_row = sub.iloc[-1].copy()
            new_row['annee'] = ny
            new_row['mois']  = nm
            new_row[TARGET]  = pred_ml
            sub = pd.concat([sub, pd.DataFrame([new_row])], ignore_index=True)

    return results