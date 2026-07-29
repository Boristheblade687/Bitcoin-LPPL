from datetime import timedelta
import json
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.stats import kurtosis, skew, t
import streamlit as st

# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Bitcoin PowerLaw + LPPL 2 Harmonics Advanced",
    layout="wide",
    page_icon="₿",
)
st.title("₿ Bitcoin PowerLaw + LPPL (2 Harmonics) - Advanced Analytics")

# Initialisation des variables dans le Session State
DEFAULT_PARAMS = {
    "A": -39.18,
    "B": 5.845,
    "C1": 0.62,
    "omega": 8.635,
    "phi1": -2.11,
    "C2": 0.267,
    "phi2": -3.0,
}

for key, val in DEFAULT_PARAMS.items():
  if key not in st.session_state:
    st.session_state[key] = val

# Constante temporelle de référence
GENESIS_DATE = pd.to_datetime("2009-01-03")


# ==============================================================================
# 1. PARAMÈTRES ET INPUTS (SIDEBAR & CONFIGURATION)
# ==============================================================================
st.sidebar.header("⚙️ Paramètres du Modèle")

# Avertissement légal
st.sidebar.warning(
    "⚠️ **Avertissement :** Ce modèle est conçu exclusivement à des fins de"
    " recherche et de modélisation statistique à long terme. Il ne constitue"
    " en aucun cas un conseil en investissement."
)

# --- GESTION DES CONFIGURATIONS JSON (CHARGEMENT & SAUVEGARDE) ---
st.sidebar.subheader("📁 Gestion de Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Charger Config (JSON)",
    type=["json"],
    help=(
        "❓ Restaurez une configuration de paramètres précédemment sauvegardée"
        " au format JSON."
    ),
)
if uploaded_file is not None:
  try:
    loaded_cfg = json.load(uploaded_file)
    for k_cfg, v_cfg in loaded_cfg.items():
      if k_cfg in DEFAULT_PARAMS:
        st.session_state[k_cfg] = float(v_cfg)
    st.sidebar.success("Configuration chargée avec succès !")
  except Exception as e:
    st.sidebar.error(f"Erreur de lecture du JSON : {e}")

config_dict = {k: st.session_state[k] for k in DEFAULT_PARAMS.keys()}
st.sidebar.download_button(
    "💾 Sauvegarder Config (JSON)",
    data=json.dumps(config_dict, indent=2),
    file_name="lppl_params.json",
    mime="application/json",
    help="❓ Exporte vos paramètres actuels sous forme de fichier JSON.",
)

st.sidebar.markdown("---")

horizon_years = st.sidebar.slider(
    "🔮 Horizon de Projection (Années)",
    min_value=1,
    max_value=3,
    value=3,
    step=1,
    help=(
        "❓ Définit le nombre d'années dans le futur sur lesquelles étendre"
        " les courbes de projection LPPL et Power Law."
    ),
)

st.sidebar.markdown("📊 **Bandes Power Law (Écart-type σ)**")
pl_sigma = st.sidebar.slider(
    "Écart-type (σ) Power Law",
    min_value=0.5,
    max_value=4.0,
    value=1.6,
    step=0.1,
    help=(
        "❓ Multiplicateur d'écart-type (σ) appliqué aux résidus pour tracer"
        " les bandes supérieure et inférieure Power Law."
    ),
)
pl_sigma_upper = pl_sigma
pl_sigma_lower = pl_sigma

with st.sidebar.expander("📌 Power Law (Tendance Fondamentale)", expanded=True):
  st.caption(
      "Ajuste la tendance logarithmique fondamentale du prix (Prix = exp(A +"
      " B * ln(t)))."
  )
  A = st.number_input(
      "A (Ordonnée à l'origine)",
      value=st.session_state["A"],
      step=0.01,
      key="input_A",
      help=(
          "❓ L'intercepte logarithmique de la loi de puissance. Fixe le niveau"
          " de départ à t=1."
      ),
  )
  B = st.number_input(
      "B (Pente / Exposant)",
      value=st.session_state["B"],
      step=0.001,
      key="input_B",
      help=(
          "❓ La pente de la loi de puissance (exposant de croissance dans le"
          " temps)."
      ),
  )

with st.sidebar.expander("🎛️ Options du Modèle & Affichage", expanded=True):
  show_trend = st.checkbox(
      "Afficher la Tendance (Power Law)",
      value=True,
      help=(
          "❓ Affiche la ligne de tendance fondamentale Power Law ainsi que son"
          " canal supérieur et inférieur."
      ),
  )
  show_lppl = st.checkbox(
      "Afficher la Courbe LPPL",
      value=True,
      help=(
          "❓ Affiche la courbe modèle LPPL ajustée (avec oscillations) et ses"
          " projections futures."
      ),
  )
  log_time_axis = st.checkbox(
      "Échelle de Temps Logarithmique (ln(t) sur l'Axe X)",
      value=True,
      help=(
          "❓ Permet de basculer l'axe X des graphiques entre le temps linéaire"
          " calendaire (Date) et le logarithme du temps (ln(t))."
      ),
  )

with st.sidebar.expander("🌊 Harmoniques LPPL", expanded=True):
  st.markdown("**Harmonic 1 (Macro Cycle)**")
  C1 = st.number_input(
      "Amplitude H1 (C1)",
      value=st.session_state["C1"],
      step=0.01,
      key="input_C1",
      help="❓ Amplitude de la première oscillation log-périodique majeure.",
  )
  omega = st.number_input(
      "Omega (ω) - Fréquence",
      value=st.session_state["omega"],
      step=0.001,
      key="input_omega",
      help=(
          "❓ Fréquence angulaire log-périodique. Détermine la vitesse à"
          " laquelle les cycles se compressent au fil du temps."
      ),
  )
  phi1 = st.number_input(
      "Phase H1 (φ1)",
      value=st.session_state["phi1"],
      step=0.01,
      key="input_phi1",
      help="❓ Décalage temporel de la première onde harmonique.",
  )

  st.markdown("**Harmonic 2 (Micro Cycle)**")
  C2 = st.number_input(
      "Amplitude H2 (C2)",
      value=st.session_state["C2"],
      step=0.01,
      key="input_C2",
      help="❓ Amplitude des oscillations secondaires (sous-cycles).",
  )
  phi2 = st.number_input(
      "Phase H2 (φ2)",
      value=st.session_state["phi2"],
      step=0.01,
      key="input_phi2",
      help="❓ Décalage temporel de la seconde onde harmonique.",
  )


# ==============================================================================
# 2. CHARGEMENT ET FILTRAGE DES DONNÉES HISTORIQUES
# ==============================================================================
@st.cache_data(ttl=3600)
def load_btc_data():
  df_cm = pd.DataFrame()
  df_yf = pd.DataFrame()

  # 1. Charger CoinMetrics pour l'historique ancien (2010 -> 2014)
  try:
    url_coinmetrics = (
        "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    )
    df_raw_cm = pd.read_csv(url_coinmetrics)
    df_cm["Date"] = pd.to_datetime(df_raw_cm["time"])
    df_cm["Close"] = pd.to_numeric(df_raw_cm["PriceUSD"], errors="coerce")
    df_cm = df_cm.dropna(subset=["Close"])
  except Exception:
    pass

  # 2. Charger yfinance pour l'historique récent jusqu'à aujourd'hui
  try:
    import yfinance as yf

    df_raw_yf = yf.download("BTC-USD", start="2009-01-03", progress=False)
    if not df_raw_yf.empty:
      if isinstance(df_raw_yf.columns, pd.MultiIndex):
        df_raw_yf = df_raw_yf.droplevel(1, axis=1)
      df_yf = df_raw_yf.reset_index()
      df_yf = df_yf.rename(columns={"Date": "Date", "Close": "Close"})
      df_yf = df_yf[["Date", "Close"]].dropna()
  except Exception:
    pass

  # 3. Fusion intelligente des deux sources
  if not df_cm.empty and not df_yf.empty:
    min_yf_date = df_yf["Date"].min()
    df_cm_old = df_cm[df_cm["Date"] < min_yf_date]
    df = pd.concat([df_cm_old, df_yf], ignore_index=True)
  elif not df_yf.empty:
    df = df_yf
  elif not df_cm.empty:
    df = df_cm
  else:
    st.error(
        "Erreur critique : Impossible de charger les données historiques."
    )
    st.stop()

  df = (
      df[df["Close"] > 0]
      .sort_values("Date", ascending=True)
      .reset_index(drop=True)
  )
  df["Days"] = (df["Date"] - GENESIS_DATE).dt.total_seconds() / 86400.0
  df["Days"] = np.maximum(df["Days"], 1.0)
  df["lnT"] = np.log(df["Days"])
  df["actualLog"] = np.log(df["Close"])
  return df


raw_df = load_btc_data()

st.sidebar.subheader("📅 Période d'Entraînement")
min_date = raw_df["Date"].min().to_pydatetime()
max_date = raw_df["Date"].max().to_pydatetime()

selected_dates = st.sidebar.date_input(
    "Plage de dates d'analyse",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help=(
        "❓ Restreint les données sur lesquelles le modèle est calibré et"
        " évalué."
    ),
)

if len(selected_dates) == 2:
  start_filter, end_filter = selected_dates
  df = raw_df[
      (raw_df["Date"] >= pd.to_datetime(start_filter))
      & (raw_df["Date"] <= pd.to_datetime(end_filter))
  ].copy()
else:
  df = raw_df.copy()

if df.empty:
  st.warning("Aucune donnée disponible pour la plage sélectionnée.")
  st.stop()


# ==============================================================================
# 3. FONCTIONS MATHÉMATIQUES DU MODÈLE & CALCULS
# ==============================================================================
def f_trend(lnT, a_val, b_val):
  return a_val + b_val * lnT


def f_log_model(lnT, a_val, b_val, c1_val, omega_val, p1_val, c2_val, p2_val):
  trend_val = f_trend(lnT, a_val, b_val)
  h1 = c1_val * np.cos(omega_val * lnT + p1_val)
  h2 = c2_val * np.cos(4.0 * omega_val * lnT + p2_val)
  return trend_val + h1 + h2


@st.cache_data
def calculate_bubble_hazard_index(df_data, window=180):
  """Calcule un indice composite de risque de bulle (0 à 100%)

  basé sur la combinaison du Z-Score Power Law, de la compression de
  volatilité et du momentum.
  """
  d = df_data.copy()

  # 1. Composante Valuation : Z-Score Power Law normalisé (capé entre 0 et 4σ)
  z_pl = d["z_score_pl"].fillna(0)
  comp_valuation = np.clip(z_pl / 4.0, 0.0, 1.0)

  # 2. Composante Volatilité glissante (compression des résidus = phase critique)
  rolling_vol = d["residuals"].rolling(window=window, min_periods=30).std()
  vol_median = rolling_vol.median()
  if vol_median > 0:
    comp_vol_compression = np.clip(
        vol_median / (rolling_vol.fillna(vol_median)), 0.5, 2.0
    )
    comp_vol_compression = (comp_vol_compression - 0.5) / 1.5
  else:
    comp_vol_compression = pd.Series(0.5, index=d.index)

  # 3. Composante Momentum / Accélération des prix (ROC sur 90 jours)
  price_roc = d["Close"].pct_change(90).fillna(0)
  comp_momentum = np.clip(price_roc / 2.0, 0.0, 1.0)

  # Indice Composite pondéré (score de 0 à 100)
  hazard_index = (
      0.50 * comp_valuation
      + 0.25 * comp_vol_compression.values
      + 0.25 * comp_momentum.values
  ) * 100.0

  d["Bubble_Hazard_Index"] = np.clip(hazard_index, 0.0, 100.0)
  return d


# ==============================================================================
# 4. OPTIMISATION GLOBALE AUTOMATIQUE (DIFFERENTIAL EVOLUTION)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Calibrage Automatique")

if st.sidebar.button(
    "🤖 Ajuster les paramètres au dataset",
    help=(
        "❓ Lance l'optimisation globale (Differential Evolution) pour estimer"
        " les paramètres de manière robuste."
    ),
):
  lnT_vec = df["lnT"].to_numpy()
  act_log_vec = df["actualLog"].to_numpy()


  def loss_func_fast(params):
    p_A, p_B, p_C1, p_omega, p_p1, p_C2, p_p2 = params
    preds = f_log_model(lnT_vec, p_A, p_B, p_C1, p_omega, p_p1, p_C2, p_p2)
    return np.mean((act_log_vec - preds) ** 2)


  bounds = [
      (-45.0, -25.0),
      (4.5, 6.8),
      (0.0, 1.5),
      (4.0, 16.0),
      (-np.pi, np.pi),
      (0.0, 0.8),
      (-np.pi, np.pi),
  ]

  with st.spinner(
      "Optimisation globale en cours (Recherche globale + Polish)..."
  ):
    res = differential_evolution(
        loss_func_fast,
        bounds=bounds,
        strategy="best1bin",
        maxiter=200,
        popsize=15,
        polish=True,
        seed=42,
    )

  if res.success:
    st.session_state["A"] = float(res.x[0])
    st.session_state["B"] = float(res.x[1])
    st.session_state["C1"] = float(res.x[2])
    st.session_state["omega"] = float(res.x[3])
    st.session_state["phi1"] = float(res.x[4])
    st.session_state["C2"] = float(res.x[5])
    st.session_state["phi2"] = float(res.x[6])

    st.session_state["opt_msg"] = (
        f"A: {res.x[0]:.2f} | B: {res.x[1]:.3f}\n"
        f"C1: {res.x[2]:.2f} | ω: {res.x[3]:.3f} | φ1: {res.x[4]:.2f}\n"
        f"C2: {res.x[5]:.2f} | φ2: {res.x[6]:.2f}"
    )
    st.rerun()
  else:
    st.sidebar.error("L'optimisation globale a échoué.")

if "opt_msg" in st.session_state:
  st.sidebar.success("Ajustement réussi !")
  st.sidebar.info(st.session_state["opt_msg"])


# ==============================================================================
# 5. CALCULS GLOBAUX, POWER LAW, RÉSIDUS & INDICE DE RISQUE DE RUPTURE
# ==============================================================================
df["trend"] = f_trend(df["lnT"].values, A, B)
df["trendPrice"] = np.exp(df["trend"])

df["trend_residuals"] = df["actualLog"] - df["trend"]
trend_res_std = np.std(df["trend_residuals"])
df["z_score_pl"] = (
    df["trend_residuals"] / trend_res_std if trend_res_std > 0 else 0.0
)

df["trendUpperPrice"] = np.exp(df["trend"] + pl_sigma_upper * trend_res_std)
df["trendLowerPrice"] = np.exp(df["trend"] - pl_sigma_lower * trend_res_std)

df["logModel"] = f_log_model(
    df["lnT"].values, A, B, C1, omega, phi1, C2, phi2
)
df["modelPrice"] = np.exp(df["logModel"])

df["residuals"] = df["actualLog"] - df["logModel"]
res_std = np.std(df["residuals"])
df["z_score"] = df["residuals"] / res_std if res_std > 0 else 0.0

# Intégration du calcul du Bubble Hazard Rate
df = calculate_bubble_hazard_index(df)
current_hazard = df["Bubble_Hazard_Index"].iloc[-1]


def get_hazard_status(score):
  if score < 25:
    return "Phase Calme / Accumulation", "#008000"
  elif score < 50:
    return "Maturation Saine", "#38BDF8"
  elif score < 75:
    return "Alerte Bulle (Attention)", "#FFA500"
  else:
    return "Zone Critique / Risque Imminent", "#FF0000"


hazard_txt, hazard_color = get_hazard_status(current_hazard)

ss_res = np.sum((df["actualLog"] - df["logModel"]) ** 2)
ss_tot = np.sum((df["actualLog"] - np.mean(df["actualLog"])) ** 2)
r2_global = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

rmse = np.sqrt(ss_res / len(df))
rmse_pct = 100.0 * (np.exp(rmse) - 1.0)
mae = np.mean(np.abs(df["actualLog"] - df["logModel"]))
mae_pct = 100.0 * (np.exp(mae) - 1.0)

df["upperBand"] = df["modelPrice"] * (1 + rmse_pct / 100.0)
df["lowerBand"] = df["modelPrice"] / (1 + rmse_pct / 100.0)

ratio_history = df["Close"] / df["modelPrice"]
ratio_percentile = (ratio_history <= ratio_history.iloc[-1]).mean() * 100.0


def get_market_state(score):
  if score < 10:
    return "Extreme Value", "#00FF00"
  elif score < 25:
    return "Undervalued", "#008000"
  elif score < 75:
    return "Fair Value", "#FFA500"
  elif score < 90:
    return "Overvalued", "#FF0000"
  else:
    return "Extreme Bubble", "#FF00FF"


state_txt, state_color = get_market_state(ratio_percentile)


# ==============================================================================
# 6. FONCTIONS WALK-FORWARD
# ==============================================================================
@st.cache_data
def run_wf_analysis_fast(
    horizon_days,
    days_arr,
    close_arr,
    p_A,
    p_B,
    p_C1,
    p_omega,
    p_p1,
    p_C2,
    p_p2,
):
  if len(days_arr) <= horizon_days:
    return 0, 0, 0, 0, 0, 0

  t_fut = days_arr[:-horizon_days] + horizon_days
  lnT_fut = np.log(t_fut)

  preds = np.exp(
      f_log_model(lnT_fut, p_A, p_B, p_C1, p_omega, p_p1, p_C2, p_p2)
  )
  actuals = close_arr[horizon_days:]
  starts = close_arr[:-horizon_days]

  log_errs = np.log(actuals / preds)
  wf_rmse = 100.0 * (np.exp(np.sqrt(np.mean(log_errs**2))) - 1.0)
  wf_mae = 100.0 * (np.exp(np.mean(np.abs(log_errs))) - 1.0)

  train_actuals = close_arr[:-horizon_days]
  train_mean_log = np.mean(np.log(train_actuals))
  ss_res_oos = np.sum((np.log(actuals) - np.log(preds)) ** 2)
  ss_tot_oos = np.sum((np.log(actuals) - train_mean_log) ** 2)
  r2_oos = 1.0 - (ss_res_oos / ss_tot_oos) if ss_tot_oos > 0 else 0.0

  correct_dir = ((preds > starts) & (actuals > starts)) | (
      (preds < starts) & (actuals < starts)
  )
  dir_acc = 100.0 * np.mean(correct_dir)
  bull_acc = 100.0 * np.mean(actuals > starts)
  edge = dir_acc - bull_acc

  return dir_acc, bull_acc, edge, wf_mae, wf_rmse, r2_oos


@st.cache_data
def run_rolling_walk_forward(
    df_data, window_days=1095, step_days=90, horizon_days=365
):
  days = df_data["Days"].values
  dates = df_data["Date"].values
  close = df_data["Close"].values
  log_close = df_data["actualLog"].values
  lnT_all = df_data["lnT"].values

  results = []
  n_samples = len(days)

  start_idx = 0
  while start_idx < n_samples:
    train_end_day = days[start_idx] + window_days
    train_end_idx = np.searchsorted(days, train_end_day)

    test_end_day = train_end_day + horizon_days
    test_end_idx = np.searchsorted(days, test_end_day)

    if test_end_idx > n_samples:
      break

    n_train = train_end_idx - start_idx
    n_test = test_end_idx - train_end_idx

    if n_train > 200 and n_test > 30:
      train_lnT = lnT_all[start_idx:train_end_idx]
      train_act_log = log_close[start_idx:train_end_idx]


      def loss_func_local(params):
        p_A, p_B, p_C1, p_omega, p_p1, p_C2, p_p2 = params
        preds = f_log_model(
            train_lnT, p_A, p_B, p_C1, p_omega, p_p1, p_C2, p_p2
        )
        return np.mean((train_act_log - preds) ** 2)


      bounds = [
          (-45.0, -25.0),
          (4.5, 6.8),
          (0.0, 1.5),
          (4.0, 16.0),
          (-np.pi, np.pi),
          (0.0, 0.8),
          (-np.pi, np.pi),
      ]

      init_guess = [-39.18, 5.845, 0.62, 8.635, -2.11, 0.267, -3.0]
      res = minimize(
          loss_func_local, init_guess, bounds=bounds, method="L-BFGS-B"
      )

      if res.success:
        opt_A, opt_B, opt_C1, opt_omega, opt_p1, opt_C2, opt_p2 = res.x
      else:
        opt_A, opt_B, opt_C1, opt_omega, opt_p1, opt_C2, opt_p2 = init_guess

      test_days = days[train_end_idx:test_end_idx]
      test_lnT = np.log(test_days)
      test_actuals = close[train_end_idx:test_end_idx]

      test_preds = np.exp(
          f_log_model(
              test_lnT,
              opt_A,
              opt_B,
              opt_C1,
              opt_omega,
              opt_p1,
              opt_C2,
              opt_p2,
          )
      )

      log_errs = np.log(test_actuals / test_preds)
      rmse_wf = 100.0 * (np.exp(np.sqrt(np.mean(log_errs**2))) - 1.0)
      mae_wf = 100.0 * (np.exp(np.mean(np.abs(log_errs))) - 1.0)

      eval_date = pd.to_datetime(dates[train_end_idx - 1])
      results.append({
          "Date Evaluation": eval_date,
          "RMSE Out-Of-Sample (%)": rmse_wf,
          "MAE Out-Of-Sample (%)": mae_wf,
      })

    step_idx = np.searchsorted(days, days[start_idx] + step_days) - start_idx
    start_idx += max(1, step_idx)

  return pd.DataFrame(results)


# ==============================================================================
# 7. PROJECTIONS & GRAPHIQUE PRINCIPAL
# ==============================================================================
last_date = df["Date"].iloc[-1]
last_days = df["Days"].iloc[-1]

future_horizon_days = int(horizon_years * 365)
future_days_arr = np.arange(
    last_days + 1, last_days + future_horizon_days + 1
)
future_dates_arr = [
    last_date + timedelta(days=int(i)) for i in range(1, future_horizon_days + 1)
]
future_lnT_arr = np.log(future_days_arr)

future_lppl = np.exp(
    f_log_model(future_lnT_arr, A, B, C1, omega, phi1, C2, phi2)
)

future_pl_trend_log = f_trend(future_lnT_arr, A, B)
future_pl_trend = np.exp(future_pl_trend_log)
future_pl_upper = np.exp(future_pl_trend_log + pl_sigma_upper * trend_res_std)
future_pl_lower = np.exp(future_pl_trend_log - pl_sigma_lower * trend_res_std)

days_arr_glob = df["Days"].values
close_arr_glob = df["Close"].values

_, _, _, _, wf_rmse_1y, _ = run_wf_analysis_fast(
    365, days_arr_glob, close_arr_glob, A, B, C1, omega, phi1, C2, phi2
)
_, _, _, _, wf_rmse_2y, _ = run_wf_analysis_fast(
    730, days_arr_glob, close_arr_glob, A, B, C1, omega, phi1, C2, phi2
)
_, _, _, _, wf_rmse_3y, _ = run_wf_analysis_fast(
    1095, days_arr_glob, close_arr_glob, A, B, C1, omega, phi1, C2, phi2
)

res_std_log = np.std(df["residuals"])

wf_milestone_days = np.array([0, 365, 730, 1095], dtype=float)
wf_milestone_errs = (
    np.array(
        [
            res_std_log * 100.0,
            wf_rmse_1y,
            wf_rmse_2y if wf_rmse_2y > 0 else wf_rmse_1y * np.sqrt(2),
            wf_rmse_3y if wf_rmse_3y > 0 else wf_rmse_1y * np.sqrt(3),
        ],
        dtype=float,
    )
    / 100.0
)

future_rel_days = np.arange(1, len(future_days_arr) + 1, dtype=float)
dynamic_uncertainty = np.zeros_like(future_rel_days)

for i, d in enumerate(future_rel_days):
  if d <= 1095:
    dynamic_uncertainty[i] = np.interp(d, wf_milestone_days, wf_milestone_errs)
  else:
    base_err_1095 = np.interp(1095.0, wf_milestone_days, wf_milestone_errs)
    dynamic_uncertainty[i] = base_err_1095 * np.sqrt(d / 1095.0)

if log_time_axis:
  x_hist = df["lnT"]
  x_trend = df["lnT"].tolist() + list(future_lnT_arr)
  x_proj = [df["lnT"].iloc[-1]] + list(future_lnT_arr)
  x_lppl_all = df["lnT"].tolist() + list(future_lnT_arr)
  xaxis_title = "Logarithme du Temps (ln(t) depuis le Genesis)"
else:
  x_hist = df["Date"]
  x_trend = df["Date"].tolist() + future_dates_arr
  x_proj = [df["Date"].iloc[-1]] + future_dates_arr
  x_lppl_all = df["Date"].tolist() + future_dates_arr
  xaxis_title = "Date"

all_pl_trend = df["trendPrice"].tolist() + list(future_pl_trend)
all_pl_upper = df["trendUpperPrice"].tolist() + list(future_pl_upper)
all_pl_lower = df["trendLowerPrice"].tolist() + list(future_pl_lower)

proj_lppl = [df["modelPrice"].iloc[-1]] + list(future_lppl)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.72, 0.28],
    subplot_titles=(
        (
            "Prix & Projections Avancées LPPL (Échelle Logarithmique du Temps"
            " ln(t))"
            if log_time_axis
            else "Prix & Projections Avancées LPPL (Temps Linéaire / Date)"
        ),
        "Analyse des Résidus (Z-Scores LPPL & Power Law)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=df["Close"],
        mode="lines",
        name="Prix BTC",
        line=dict(color="#D1D5DB", width=1.2),
    ),
    row=1,
    col=1,
)

if show_lppl:
  sigma_levels = [
      {"mult": 1.0, "opacity": 0.18, "name": "Canal ±1.0σ (68%)"},
      {"mult": 2.0, "opacity": 0.10, "name": "Canal ±2.0σ (95%)"},
  ]

  for band in sigma_levels:
    m = band["mult"]
    hist_upper = df["modelPrice"] * np.exp(m * res_std_log)
    hist_lower = df["modelPrice"] / np.exp(m * res_std_log)
    fut_upper = future_lppl * np.exp(m * dynamic_uncertainty)
    fut_lower = future_lppl / np.exp(m * dynamic_uncertainty)

    comp_upper = list(hist_upper) + list(fut_upper)
    comp_lower = list(hist_lower) + list(fut_lower)

    fig.add_trace(
        go.Scatter(
            x=x_lppl_all + x_lppl_all[::-1],
            y=comp_upper + comp_lower[::-1],
            fill="toself",
            fillcolor=f"rgba(255, 153, 0, {band['opacity']})",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=band["name"],
        ),
        row=1,
        col=1,
    )

  fig.add_trace(
      go.Scatter(
          x=x_hist,
          y=df["modelPrice"],
          mode="lines",
          name="LPPL Model (Fit)",
          line=dict(color="#FF9900", width=2),
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=x_proj,
          y=proj_lppl,
          mode="lines",
          name="LPPL Projection Centrale",
          line=dict(color="#FF9900", width=2.5, dash="dash"),
      ),
      row=1,
      col=1,
  )

if show_trend:
  fig.add_trace(
      go.Scatter(
          x=x_trend,
          y=all_pl_trend,
          mode="lines",
          name="Power Law Trend",
          line=dict(color="#00BFFF", width=1.5),
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=x_trend,
          y=all_pl_upper,
          mode="lines",
          name=f"Power Law Top Band (+{pl_sigma_upper}σ)",
          line=dict(color="#38BDF8", width=1, dash="dash"),
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=x_trend,
          y=all_pl_lower,
          mode="lines",
          name=f"Power Law Floor Band (-{pl_sigma_lower}σ)",
          line=dict(color="#38BDF8", width=1, dash="dash"),
      ),
      row=1,
      col=1,
  )

fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=df["z_score"],
        mode="lines",
        name="Z-Score LPPL",
        line=dict(color="#FF9900", width=1.2),
        legend="legend2",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=df["z_score_pl"],
        mode="lines",
        name="Z-Score Power Law",
        line=dict(color="#00BFFF", width=1.5),
        legend="legend2",
    ),
    row=2,
    col=1,
)

fig.add_hline(
    y=pl_sigma_upper,
    line_dash="dash",
    line_color="#38BDF8",
    row=2,
    col=1,
    annotation_text=f"+{pl_sigma_upper}σ (PL Top)",
    annotation_position="top right",
)
fig.add_hline(
    y=-pl_sigma_lower,
    line_dash="dash",
    line_color="#38BDF8",
    row=2,
    col=1,
    annotation_text=f"-{pl_sigma_lower}σ (PL Floor)",
    annotation_position="bottom right",
)
fig.add_hline(
    y=2.0,
    line_dash="dot",
    line_color="#FF9900",
    row=2,
    col=1,
    annotation_text="+2.0σ (LPPL Top)",
    annotation_position="top left",
)
fig.add_hline(
    y=-2.0,
    line_dash="dot",
    line_color="#FF9900",
    row=2,
    col=1,
    annotation_text="-2.0σ (LPPL Floor)",
    annotation_position="bottom left",
)
fig.add_hline(y=0.0, line_dash="solid", line_color="gray", row=2, col=1)

fig.update_yaxes(type="log", title_text="Prix (USD)", row=1, col=1)
fig.update_yaxes(title_text="Z-Score (σ)", row=2, col=1)

xaxis_config = (
    dict(title_text=xaxis_title, rangeslider=dict(visible=False))
    if log_time_axis
    else dict(
        title_text=xaxis_title,
        rangeselector=dict(
            y=1.12,
            x=0.0,
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=4, label="4y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
        ),
        rangeslider=dict(visible=False),
    )
)

fig.update_xaxes(xaxis_config)

fig.update_layout(
    template="plotly_dark",
    height=1000,
    margin=dict(l=20, r=20, t=60, b=100),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.29,
        xanchor="center",
        x=0.5,
        font=dict(size=9),
        bgcolor="rgba(0,0,0,0.5)",
    ),
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=9),
        bgcolor="rgba(0,0,0,0.5)",
    ),
)


# ==============================================================================
# 8. DASHBOARD & MÉTRIQUES COMPLÈTES
# ==============================================================================
n_obs = len(df)
k_params = 7

r2_adj = (
    1.0 - ((1.0 - r2_global) * (n_obs - 1) / (n_obs - k_params - 1))
    if n_obs > (k_params + 1)
    else 0.0
)
mape = (
    np.mean(np.abs((df["Close"] - df["modelPrice"]) / df["Close"])) * 100.0
)
res_std_dev = np.std(df["residuals"])
res_skew = skew(df["residuals"])
res_kurt = kurtosis(df["residuals"])

_, _, _, _, wf_rmse_1y_val, r2_oos_1y = run_wf_analysis_fast(
    365, days_arr_glob, close_arr_glob, A, B, C1, omega, phi1, C2, phi2
)
gen_ratio = wf_rmse_1y_val / rmse_pct if rmse_pct > 0 else 1.0

col_chart, col_dash = st.columns([3.2, 1])

with col_chart:
  st.plotly_chart(fig, use_container_width=True)
  with st.expander("❓ Guide de Lecture du Graphique Principal"):
    st.markdown("""
        * **Prix BTC (Gris)** : Cours de clôture quotidien du Bitcoin.
        * **LPPL Model (Orange)** : Courbe ajustée du modèle LPPL combinant la tendance et les oscillations. Les pointillés représentent la projection future.
        * **Power Law Fit (Bleu Cyan)** : Tendance fondamentale A + B * ln(t). Les bandes supérieure et inférieure sont calculées dynamiquement à partir des résidus via le paramètre sigma réglable.
        * **Canaux Multi-Sigma (±1σ, ±2σ, ±3σ)** : Entonnoir de probabilité calibré empiriquement via les erreurs Walk-Forward.
        * **Z-Scores (Panneau Inférieur)** : Mesures de l'écart du prix réel par rapport au modèle LPPL (en orange) et à la Power Law fondamentale (en bleu cyan) exprimés en écarts-types.
        """)

with col_dash:
  # --- Widget 1 : Live & Modèle ---
  with st.container(border=True):
    st.subheader("📌 Live & Modèle")
    if not df.empty:
      current_btc_price = df["Close"].iloc[-1]
      current_model_price = df["modelPrice"].iloc[-1]
      current_z_score = df["z_score"].iloc[-1]

      price_delta = (
          (current_btc_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
          if len(df) > 1
          else 0.0
      )

      st.metric(
          "Prix Actuel BTC",
          f"${current_btc_price:,.2f}",
          delta=f"{price_delta:+.2f}% (24h)",
      )
      st.metric("Prix Théorique (LPPL)", f"${current_model_price:,.2f}")
      st.metric("Z-Score LPPL", f"{current_z_score:.2f}σ")

  # --- Widget 2 : Valuation ---
  with st.container(border=True):
    st.subheader("🎯 Valuation")
    st.metric(
        "Percentile Ratio",
        f"{ratio_percentile:.1f}%",
        help=(
            "❓ Position relative de la valorisation actuelle par rapport à"
            " l'historique complet."
        ),
    )
    st.markdown(
        f"Statut : <span"
        f" style='color:{state_color};font-weight:bold;'>{state_txt}</span>",
        unsafe_allow_html=True,
    )

  # --- Widget 3 : Fit Quality & Robustesse ---
  with st.container(border=True):
    st.subheader("📊 Fit Quality & Robustesse")

    c_g1, c_g2 = st.columns(2)
    c_g1.metric(
        "R² Global",
        f"{r2_global:.4f}",
        help=(
            "❓ Coefficient de détermination expliquant la variance observée"
            " sur le dataset."
        ),
    )
    c_g2.metric(
        "R² Ajusté",
        f"{r2_adj:.4f}",
        help=(
            "❓ Coefficient de détermination ajusté selon le nombre de"
            " paramètres du modèle."
        ),
    )

    st.metric(
        "Forward R² (1Y OOS)",
        f"{r2_oos_1y:.4f}",
        help=(
            "❓ Coefficient de détermination hors-échantillon (Out-Of-Sample)"
            " évalué sur un horizon prédictif de 1 an."
        ),
    )

    st.markdown("---")
    st.caption("📐 **Métriques d'Erreur & Généralisation**")

    c_m3, c_m4 = st.columns(2)
    c_m3.metric(
        "RMSE In-Sample",
        f"{rmse_pct:.1f}%",
        help=(
            "❓ Erreur quadratique moyenne en pourcentage sur la période"
            " d'entraînement."
        ),
    )
    c_m4.metric(
        "MAE In-Sample",
        f"{mae_pct:.1f}%",
        help=(
            "❓ Erreur absolue moyenne en pourcentage sur la période"
            " d'entraînement."
        ),
    )

    c_m5, c_m6 = st.columns(2)
    c_m5.metric(
        "OOS RMSE (1Y)",
        f"{wf_rmse_1y_val:.1f}%",
        help=(
            "❓ Erreur de prédiction hors-échantillon (Out-Of-Sample) sur un"
            " horizon de 1 an."
        ),
    )
    c_m6.metric(
        "Ratio Out/In",
        f"{gen_ratio:.2f}x",
        help=(
            "❓ Ratios < 1.5x indiquent une bonne capacité de généralisation"
            " (faible surapprentissage)."
        ),
    )


# ==============================================================================
# SECTION : INDICATEUR DE RISQUE DE RUPTURE (HAZARD RATE / BUBBLE INDEX)
# ==============================================================================
st.markdown("---")
st.subheader(
    "🚨 Indicateur de Risque de Rupture (Bubble Hazard Rate / Criticality"
    " Index)"
)

with st.expander("❓ Guide de Lecture - Indice de Risque de Rupture"):
  st.markdown("""
    * Cet indice synthétise la probabilité d'entrée en régime critique (système instable type tas de sable).
    * **0 - 25% (Vert)** : Marché sain, en phase de fond ou d'accumulation.
    * **25 - 50% (Bleu)** : Croissance organique alignée sur la Power Law.
    * **50 - 75% (Orange)** : Phase spéculative avancée, signaux d'alerte macro.
    * **> 75% (Rouge)** : Zone de criticité maximale, probabilité élevée de rupture ou de retournement de cycle.
    """)

fig_hazard = go.Figure()
fig_hazard.add_trace(
    go.Scatter(
        x=df["Date"] if not log_time_axis else df["lnT"],
        y=df["Bubble_Hazard_Index"],
        mode="lines",
        name="Bubble Hazard Index (%)",
        line=dict(color=hazard_color, width=2),
        fill="tozeroy",
        fillcolor=(
            "rgba(255, 0, 0, 0.1)"
            if current_hazard > 75
            else "rgba(56, 189, 248, 0.1)"
        ),
    )
)

fig_hazard.add_hline(
    y=75,
    line_dash="dash",
    line_color="#FF0000",
    annotation_text="Seuil Critique (75%)",
    annotation_position="top right",
)
fig_hazard.add_hline(
    y=50,
    line_dash="dot",
    line_color="#FFA500",
    annotation_text="Seuil d'Alerte (50%)",
    annotation_position="top right",
)

fig_hazard.update_layout(
    template="plotly_dark",
    height=300,
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis_title="Indice de Risque (%)",
    xaxis_title=xaxis_title,
    yaxis=dict(range=[0, 100]),
)
st.plotly_chart(fig_hazard, use_container_width=True)

col_haz1, col_haz2 = st.columns([1, 2])
with col_haz1:
  st.metric("Indice de Risque Actuel", f"{current_hazard:.1f} / 100")
with col_haz2:
  st.markdown(
      f"**Statut du Régime :** <span"
      f" style='color:{hazard_color};font-weight:bold;font-size:1.2em;'>{hazard_txt}</span>",
      unsafe_allow_html=True,
  )

# ==============================================================================
# SECTION : HORLOGE DE PHASE GRAVITATIONNELLE (PHASE SPACE CLOCK)
# ==============================================================================
st.markdown("---")
st.subheader("🕒 Horloge de Phase Gravitationnelle (Dynamique Power Law)")

with st.expander("❓ Guide de Lecture - L'Horloge de Phase"):
  st.markdown("""
    * **Principe de l'Horloge** : Ce diagramme représente l'état cyclique du Bitcoin en croisant sa **position** (Z-Score Power Law sur l'axe X) et sa **vitesse de déplacement** (Momentum / Dérivée du Z-Score sur l'axe Y).
    * **Le Mouvement Orbital** : Le marché tourne autour du centre $(0,0)$ (la Power Law parfaite) en suivant une trajectoire en spirale divisée en 4 régimes clés :
      1. **Bas-Droite (Accumulation / Reprise)** : Le prix est sous l'attracteur central mais commence à réaccélérer.
      2. **Haut-Droite (Bull Run / Expansion)** : Le prix dépasse l'attracteur avec une forte dynamique haussière.
      3. **Haut-Gauche (Surchauffe / Sommet de Bulle)** : Le prix est très haut mais la vitesse s'essouffle (point critique de retournement).
      4. **Bas-Gauche (Bear Market / Correction)** : Le prix chute lourdement vers le plancher gravitationnel.
    * **Aiguille Actuelle (Diamant Vert)** : Indique la position exacte et la dynamique instantanée du Bitcoin aujourd'hui sur cette horloge.
    """)

# Calcul de la vitesse du Z-score (fenêtre de 30 jours pour lisser le bruit quotidien)
df["z_velocity"] = df["z_score_pl"].diff(30).fillna(0)

fig_clock = go.Figure()

# Trajectoire historique globale (l'orbite de l'horloge au fil du temps)
fig_clock.add_trace(
    go.Scatter(
        x=df["z_score_pl"],
        y=df["z_velocity"],
        mode="lines+markers",
        marker=dict(
            size=4,
            color=df["Days"],
            colorscale="Inferno",
            colorbar=dict(title="Temps (Genesis ➔ Now)", len=0.6),
            opacity=0.7,
        ),
        line=dict(color="rgba(255, 153, 0, 0.3)", width=1),
        name="Trajectoire Orbitale",
        hovertext=df["Date"].dt.strftime("%Y-%m-%d"),
    )
)

# Point actuel (L'aiguille de l'horloge)
latest_row = df.iloc[-1]
fig_clock.add_trace(
    go.Scatter(
        x=[latest_row["z_score_pl"]],
        y=[latest_row["z_velocity"]],
        mode="markers+text",
        marker=dict(size=16, color="#00FF7F", symbol="diamond"),
        text=["📍 POSITION ACTUELLE"],
        textposition="top center",
        textfont=dict(color="#00FF7F", size=12),
        name="Aiguille Actuelle",
    )
)

# Lignes de division des quadrants (les axes de l'horloge)
fig_clock.add_hline(
    y=0,
    line_dash="dash",
    line_color="rgba(255,255,255,0.3)",
    annotation_text="Vitesse Nulle (Inflexion)",
    annotation_position="bottom right",
)
fig_clock.add_vline(
    x=0,
    line_dash="dash",
    line_color="rgba(255,255,255,0.3)",
    annotation_text="Attracteur Power Law (0σ)",
    annotation_position="top left",
)

fig_clock.update_layout(
    template="plotly_dark",
    height=550,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Position : Z-Score Power Law (Sous-évalué ⟷ Surévalué)",
    yaxis_title=(
        "Vitesse / Momentum : Variation du Z-Score (Décélération"
        " ⟷ Accélération)"
    ),
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
)

st.plotly_chart(fig_clock, use_container_width=True)

# ==============================================================================
# SECTION : DISTRIBUTION EMPIRIQUE DES RÉSIDUS VS LOI DE STUDENT
# ==============================================================================
st.markdown("---")
st.subheader(
    "📊 Distribution empirique des résidus vs Loi de Student (Fat Tails Check)"
)

dist_horizon_options = {
    "In-Sample (Modèle Global)": 0,
    "3 mois (90 jours) Out-Of-Sample": 90,
    "6 mois (180 jours) Out-Of-Sample": 180,
    "1 an (365 jours) Out-Of-Sample": 365,
    "2 ans (730 jours) Out-Of-Sample": 730,
    "3 ans (1095 jours) Out-Of-Sample": 1095,
}

selected_dist_label = st.selectbox(
    "Sélectionner la source des résidus (In-Sample ou Horizon Forward OOS)",
    options=list(dist_horizon_options.keys()),
    index=3,
    key="dist_horizon_selectbox",
    help=(
        "❓ Permet de choisir si l'analyse de distribution et des résidus s'effectue"
        " sur l'In-Sample global ou sur les erreurs de prédiction Out-Of-Sample"
        " pour un horizon de forward spécifique."
    ),
)
horizon_oos_eval = dist_horizon_options[selected_dist_label]

if horizon_oos_eval > 0:
  if len(days_arr_glob) > horizon_oos_eval:
    t_fut_oos = days_arr_glob[:-horizon_oos_eval] + horizon_oos_eval
    lnT_fut_oos = np.log(t_fut_oos)
    preds_oos = np.exp(
        f_log_model(lnT_fut_oos, A, B, C1, omega, phi1, C2, phi2)
    )
    actuals_oos = close_arr_glob[horizon_oos_eval:]
    residuals_oos_log = np.log(actuals_oos / preds_oos)
    res_pct_clean = (np.exp(residuals_oos_log) - 1.0) * 100.0
    dist_mode_label = f"Out-Of-Sample ({selected_dist_label.split(' ')[0]})"
  else:
    res_pct_clean = np.array([])
    dist_mode_label = "OOS (Données insuffisantes)"
else:
  residuals_pct = (np.exp(df["residuals"]) - 1.0) * 100.0
  res_pct_clean = residuals_pct.dropna().values
  dist_mode_label = "In-Sample"

if len(res_pct_clean) > 0:
  df_t, loc_t, scale_t = t.fit(res_pct_clean)
  std_pct_resid = np.std(res_pct_clean)
else:
  df_t, loc_t, scale_t, std_pct_resid = 3.0, 0.0, 1.0, 1.0

col_dist1, col_dist2 = st.columns([2, 1])

with col_dist2:
  st.markdown(f"### 📐 Analyse de forme (Student-t) [{dist_mode_label}]")
  st.markdown(f"""
    Ce graphique superpose ({dist_mode_label}) :
    * **L'histogramme réel** de vos erreurs de prédiction en pourcentage.
    * **La loi de Student ajustée** (df = {df_t:.2f}).
    
    *L'utilisation d'une loi de Student (t-distribution) plutôt qu'une gaussienne classique permet de modéliser proprement les **fat tails** caractéristiques du Bitcoin à long terme.*
    """)

  st.metric(
      "Degrés de liberté (Student df)",
      f"{df_t:.2f}",
      help=(
          "Plus df est bas (< 30), plus les queues de distribution sont"
          " épaisses."
      ),
  )
  st.metric(
      "Kurtosis des Résidus (%)",
      f"{kurtosis(res_pct_clean):.2f}" if len(res_pct_clean) > 0 else "N/A",
      help="Aplatissement mesuré sur la série en pourcentage.",
  )

with col_dist1:
  if len(res_pct_clean) > 0:
    x_range = np.linspace(res_pct_clean.min(), res_pct_clean.max(), 500)
    y_student = t.pdf(x_range, df_t, loc=loc_t, scale=scale_t)

    fig_dist = ff.create_distplot(
        [res_pct_clean],
        [f"Résidus du Modèle (%) [{dist_mode_label}]"],
        bin_size=1.0,
        show_hist=True,
        show_curve=False,
        show_rug=False,
    )
    fig_dist.data[0].marker.color = "#1f77b4"
    fig_dist.add_trace(
        go.Scatter(
            x=x_range,
            y=y_student,
            mode="lines",
            name=f"Loi de Student (df={df_t:.2f})",
            line=dict(color="#00BFFF", width=2.5, dash="dash"),
        )
    )
    fig_dist.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        xaxis_title="Erreur de prédiction / Résidu (%)",
        yaxis_title="Densité de probabilité",
    )
    st.plotly_chart(fig_dist, use_container_width=True)
  else:
    st.warning("Données OOS insuffisantes pour afficher la distribution.")


# ==============================================================================
# SECTION : COURBE DE PRÉDICTION OOS (HORIZON PERSONNALISABLE)
# ==============================================================================
st.markdown("---")

oos_chart_options = {
    "3 mois (90 jours)": 90,
    "6 mois (180 jours)": 180,
    "1 an (365 jours)": 365,
    "2 ans (730 jours)": 730,
    "3 ans (1095 jours)": 1095,
}

if "oos_chart_horizon_selectbox" not in st.session_state:
  st.session_state["oos_chart_horizon_selectbox"] = "1 an (365 jours)"

current_label = st.session_state["oos_chart_horizon_selectbox"]

st.subheader(
    f"📈 Comparaison de la Courbe de Prédiction Out-Of-Sample"
    f" ({current_label}) vs Prix Réel"
)

with st.expander("❓ Guide de Lecture - OOS Historique"):
  st.markdown(f"""
    * Ce graphique trace les projections faites par le modèle global avec un horizon fixe de **{current_label}** en amont, comparées directement au prix réel atteint à cette échéance.
    * Il permet d'évaluer visuellement la robustesse et le biais d'anticipation du modèle sur cet horizon à travers tout l'historique du Bitcoin.
    """)

selected_oos_chart_label = st.selectbox(
    "Sélectionner l'horizon OOS pour la comparaison de prédiction",
    options=list(oos_chart_options.keys()),
    key="oos_chart_horizon_selectbox",
    help=(
        "❓ Permet de choisir l'horizon de prédiction Out-Of-Sample affiché dans"
        " le graphique de comparaison."
    ),
)
h_days = oos_chart_options[selected_oos_chart_label]

fig_oos_parallel = go.Figure()
fig_oos_parallel.add_trace(
    go.Scatter(
        x=x_hist,
        y=df["Close"],
        mode="lines",
        name="Prix BTC Réel",
        line=dict(color="#D1D5DB", width=1.5),
    )
)

label = f"OOS ({selected_oos_chart_label})"
color = "#38BDF8"

days_arr_p = df["Days"].values
dates_arr_p = df["Date"].values
lnT_arr_p = df["lnT"].values

if len(days_arr_p) > h_days:
  t_fut_p = days_arr_p[:-h_days] + h_days
  lnT_fut_p = np.log(t_fut_p)
  preds_p = np.exp(
      f_log_model(lnT_fut_p, A, B, C1, omega, phi1, C2, phi2)
  )

  if log_time_axis:
    x_oos_p = lnT_fut_p
  else:
    x_oos_p = dates_arr_p[h_days:]

  fig_oos_parallel.add_trace(
      go.Scatter(
          x=x_oos_p,
          y=preds_p,
          mode="lines",
          name=label,
          line=dict(color=color, width=1.5, dash="dash"),
      )
  )

fig_oos_parallel.update_yaxes(type="log", title_text="Prix (USD, Log)")
fig_oos_parallel.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=20, r=20, t=30, b=30),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0.5)",
    ),
    xaxis_title=xaxis_title,
)
st.plotly_chart(fig_oos_parallel, use_container_width=True)


# ==============================================================================
# 9. ROLLING WALK-FORWARD & STABILITÉ TEMPORELLE
# ==============================================================================
st.markdown("---")
st.subheader("🔄 Stabilité Temporelle (Rolling Walk-Forward Analysis)")

col_rwf_params, col_rwf_chart = st.columns([1, 3])

with col_rwf_params:
  rwf_window = st.number_input(
      "Taille Fenêtre Train (Jours)",
      value=730,
      step=180,
      help=(
          "❓ Nombre de jours de données utilisés pour la période"
          " d'entraînement glissante."
      ),
  )

  horizon_options = {
      "3 mois (90 jours)": 90,
      "6 mois (180 jours)": 180,
      "1 an (365 jours)": 365,
      "2 ans (730 jours)": 730,
      "3 ans (1095 jours)": 1095,
  }
  selected_horizon_label = st.selectbox(
      "Horizon de Test OOS",
      options=list(horizon_options.keys()),
      index=2,
      key="horizon_test_oos_selectbox",
      help=(
          "❓ Choisit l'horizon de prédiction hors-échantillon testé à chaque"
          " étape glissante."
      ),
  )
  rwf_horizon = horizon_options[selected_horizon_label]

  rwf_step = st.number_input(
      "Pas de Glissement (Jours)",
      value=90,
      step=30,
      help="❓ Pas d'avancement temporel entre chaque évaluation Walk-Forward.",
  )

  metric_choice = st.radio(
      "Métrique d'erreur à afficher",
      ["RMSE", "MAE"],
      help=(
          "❓ Choisir d'afficher la RMSE ou la MAE dans le graphique de"
          " stabilité temporelle."
      ),
  )

df_rwf = run_rolling_walk_forward(
    df,
    window_days=int(rwf_window),
    step_days=int(rwf_step),
    horizon_days=int(rwf_horizon),
)

with col_rwf_chart:
  if not df_rwf.empty:
    fig_rwf = go.Figure()
    if metric_choice == "RMSE":
      fig_rwf.add_trace(
          go.Scatter(
              x=df_rwf["Date Evaluation"],
              y=df_rwf["RMSE Out-Of-Sample (%)"],
              mode="lines",
              name="RMSE OOS (%)",
              line=dict(color="#FF4B4B", width=2),
          )
      )
    else:
      fig_rwf.add_trace(
          go.Scatter(
              x=df_rwf["Date Evaluation"],
              y=df_rwf["MAE Out-Of-Sample (%)"],
              mode="lines",
              name="MAE OOS (%)",
              line=dict(color="#FFA500", width=2),
          )
      )

    fig_rwf.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
        yaxis_title=f"{metric_choice} (%)",
        xaxis_title="Date d'Évaluation",
    )
    st.plotly_chart(fig_rwf, use_container_width=True)
  else:
    st.info(
        "Historique insuffisant pour calculer les fenêtres glissantes"
        " sélectionnées."
    )


# ==============================================================================
# SECTION : POURCENTAGE D'ERREUR RMS / OOS PAR NIVEAU DE SIGMA
# ==============================================================================
st.markdown("---")
st.subheader("🎯 Contribution des Erreurs par Niveau de Sigma (σ) et Brackets")

sigma_horizon_options = {
    "In-Sample (Modèle Global)": 0,
    "3 mois (90 jours) Out-Of-Sample": 90,
    "6 mois (180 jours) Out-Of-Sample": 180,
    "1 an (365 jours) Out-Of-Sample": 365,
    "2 ans (730 jours) Out-Of-Sample": 730,
    "3 ans (1095 jours) Out-Of-Sample": 1095,
}

selected_sigma_label = st.selectbox(
    "Sélectionner l'horizon pour l'analyse des brackets Sigma",
    options=list(sigma_horizon_options.keys()),
    index=3,
    key="sigma_horizon_selectbox",
    help=(
        "❓ Permet de choisir l'horizon In-Sample ou Out-Of-Sample"
        " spécifiquement pour l'analyse de la contribution des erreurs par"
        " brackets de sigma."
    ),
)
horizon_sigma_eval = sigma_horizon_options[selected_sigma_label]

use_oos_sigma = horizon_sigma_eval > 0
analysis_mode_label = (
    f"Out-Of-Sample ({selected_sigma_label.split(' ')[0]})"
    if use_oos_sigma
    else "In-Sample"
)

if use_oos_sigma:
  if len(days_arr_glob) > horizon_sigma_eval:
    t_fut_oos = days_arr_glob[:-horizon_sigma_eval] + horizon_sigma_eval
    lnT_fut_oos = np.log(t_fut_oos)
    preds_oos = np.exp(
        f_log_model(lnT_fut_oos, A, B, C1, omega, phi1, C2, phi2)
    )
    actuals_oos = close_arr_glob[horizon_sigma_eval:]
    residuals_arr = np.log(actuals_oos / preds_oos)
  else:
    residuals_arr = np.array([])
else:
  residuals_arr = df["residuals"].dropna().values

sigma_res = np.std(residuals_arr) if len(residuals_arr) > 0 else 0.0

if sigma_res > 0:
  z_scores_abs = np.abs(residuals_arr / sigma_res)

  brackets = [
      ("≤ 1.0σ", 0.0, 1.0),
      ("1.0σ - 2.0σ", 1.0, 2.0),
      ("2.0σ - 3.0σ", 2.0, 3.0),
      ("> 3.0σ", 3.0, np.inf),
  ]

  bracket_names = []
  pct_points = []
  pct_mse = []

  total_sq_err = np.sum(residuals_arr**2)

  for name, low, high in brackets:
    bracket_names.append(name)
    if high == np.inf:
      mask = z_scores_abs > low
    else:
      mask = (z_scores_abs > low) & (z_scores_abs <= high)

    p_pts = (np.sum(mask) / len(residuals_arr)) * 100.0
    pct_points.append(round(p_pts, 2))

    if total_sq_err > 0:
      p_m = (np.sum((residuals_arr[mask]) ** 2) / total_sq_err) * 100.0
    else:
      p_m = 0.0
    pct_mse.append(round(p_m, 2))

  col_sig1, col_sig2 = st.columns([2, 1])

  with col_sig1:
    fig_sigma_contrib = go.Figure()
    fig_sigma_contrib.add_trace(
        go.Bar(
            x=bracket_names,
            y=pct_points,
            name=f"Part des Points (%) [{analysis_mode_label}]",
            marker_color="#38BDF8",
        )
    )
    fig_sigma_contrib.add_trace(
        go.Bar(
            x=bracket_names,
            y=pct_mse,
            name=f"Contribution à l'Erreur RMS/MSE (%) [{analysis_mode_label}]",
            marker_color="#FF9900",
        )
    )
    fig_sigma_contrib.update_layout(
        template="plotly_dark",
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        xaxis_title="Intervalles d'Écart-type (sigma)",
        yaxis_title="Pourcentage (%)",
    )
    st.plotly_chart(fig_sigma_contrib, use_container_width=True)

  with col_sig2:
    st.markdown(f"### 📊 Analyse des Brackets Sigma ({analysis_mode_label})")
    st.markdown("""
        Ce graphique permet d'identifier l'origine de l'erreur globale :
        * **Part des points (Bleu)** : Fréquence brute des écarts dans chaque tranche.
        * **Contribution RMS/MSE (Orange)** : Poids des erreurs au carré, soulignant l'impact des valeurs extrêmes (fat tails).
        """)

    df_bracket_summary = pd.DataFrame({
        "Bracket": bracket_names,
        "Points (%)": pct_points,
        "MSE (%)": pct_mse,
    })
    st.dataframe(df_bracket_summary, hide_index=True, use_container_width=True)
else:
  st.info("Données insuffisantes pour calculer la répartition par sigma.")


# ==============================================================================
# 10. TABLEAUX DE PERFORMANCE FIXE & EXPORT CSV
# ==============================================================================
st.markdown("---")
col_wf, col_proj = st.columns(2)

with col_wf:
  st.subheader("📈 Performance Walk-Forward (Globale)")

  days_arr = df["Days"].values
  close_arr = df["Close"].values

  wf_data = []
  for h in [365, 730, 1095]:
    acc, bull, edge, mae_h, rmse_h, r2_oos = run_wf_analysis_fast(
        h, days_arr, close_arr, A, B, C1, omega, phi1, C2, phi2
    )
    wf_data.append({
        "Horizon": f"{h}d",
        "Forward R²": f"{r2_oos:.3f}",
        "Dir Acc": f"{acc:.0f}%",
        "Edge": f"{edge:.1f}%",
        "MAE OOS": f"{mae_h:.1f}%",
        "RMSE OOS": f"{rmse_h:.1f}%",
    })
  st.dataframe(
      pd.DataFrame(wf_data), hide_index=True, use_container_width=True
  )

with col_proj:
  st.subheader("🎯 Objectifs de Prix & Export")

  proj_data = []
  export_rows = []
  sigma_cone = 1.0

  for yr in range(1, horizon_years + 1):
    idx = (yr * 365) - 1
    if idx < len(future_dates_arr):
      date_target = future_dates_arr[idx]
      proj_price = future_lppl[idx]
      uncert = dynamic_uncertainty[idx]

      cone_lower = proj_price / np.exp(sigma_cone * uncert)
      cone_upper = proj_price * np.exp(sigma_cone * uncert)

      proj_data.append({
          "Horizon": f"{yr}Y",
          "LPPL Target": f"${proj_price:,.0f}",
          "Cône Projection (±1σ)": f"${cone_lower:,.0f} - ${cone_upper:,.0f}",
      })
      export_rows.append({
          "Horizon": f"{yr}Y",
          "Target_Date": date_target.strftime("%Y-%m-%d"),
          "LPPL_Price_USD": round(proj_price, 2),
          "Cone_Lower_USD": round(cone_lower, 2),
          "Cone_Upper_USD": round(cone_upper, 2),
      })

  st.dataframe(
      pd.DataFrame(proj_data), hide_index=True, use_container_width=True
  )

  df_export = pd.DataFrame(export_rows)
  csv_data = df_export.to_csv(index=False).encode("utf-8")

  st.download_button(
      label="📥 Télécharger les projections (CSV)",
      data=csv_data,
      file_name=f"btc_lppl_projections_{last_date.strftime('%Y%m%d')}.csv",
      mime="text/csv",
      help=(
          "❓ Exporte le tableau des objectifs de prix futurs au format CSV."
      ),
  )


# ==============================================================================
# SECTION : SIMULATEUR DE DCA INTELLIGENT (SMART DCA)
# ==============================================================================
st.markdown("---")
st.subheader("🤖 Simulateur de DCA Intelligent (Smart DCA - Long Terme)")

with st.expander("❓ Guide de Lecture - Smart DCA", expanded=False):
  st.markdown("""
    * **DCA Classique** : Investissement d'un montant fixe à intervalle régulier (ex: chaque semaine ou chaque mois).
    * **Smart DCA** : Modulation dynamique du montant en fonction de l'écart de valorisation (Z-score Power Law) :
      * $Z < -1.0$ (Sous-évaluation) : Multiplicateur x2.0
      * $-1.0 \le Z \le 2.0$ (Zone saine / Fair Value) : Multiplicateur x1.0
      * $Z > 2.0$ (Surchauffe / Bulle) : Multiplicateur x0.5 (ou 0 pour suspension totale)
    """)

# Ligne 1 des paramètres DCA
col_dca_opt1, col_dca_opt2, col_dca_opt3 = st.columns(3)
with col_dca_opt1:
  dca_base_amount = st.number_input(
      "Montant de base ($)", value=100.0, step=50.0, key="dca_base_amt"
  )
with col_dca_opt2:
  dca_freq_label = st.selectbox(
      "Fréquence d'achat",
      ["Hebdomadaire (7j)", "Mensuel (30j)"],
      index=1,
      key="dca_freq_sel",
  )
with col_dca_opt3:
  min_dca_date = raw_df["Date"].min().to_pydatetime()
  max_dca_date = raw_df["Date"].max().to_pydatetime()
  dca_start_date = st.date_input(
      "Date de début du DCA",
      value=min_dca_date,
      min_value=min_dca_date,
      max_value=max_dca_date,
      key="dca_start_date_input",
      help=(
          "❓ Permet de choisir à quelle date historique le premier achat DCA"
          " est déclenché."
      ),
  )

# Ligne 2 des paramètres DCA
col_dca_opt4, col_dca_opt5 = st.columns(2)
with col_dca_opt4:
  dca_strategy = st.selectbox(
      "Stratégie DCA",
      ["Smart DCA (Modulation Z-Score)", "DCA Classique (Fixe)"],
      index=0,
      key="dca_strat_sel",
  )
with col_dca_opt5:
  overheat_action = st.selectbox(
      "Action en Zone de Bulle ($Z > 2.0$)",
      ["Réduire de moitié (0.5x)", "Suspendre les achats (0x)"],
      index=1,
  )

# Filtrage du DataFrame à partir de la date de début choisie
df_dca_filtered = df[df["Date"] >= pd.to_datetime(dca_start_date)].copy()

# Fréquence en jours
step_dca_days = 7 if "Hebdomadaire" in dca_freq_label else 30
dca_sim_df = df_dca_filtered.iloc[::step_dca_days].copy()

invested_classical = 0
btc_classical = 0
invested_smart = 0
btc_smart = 0

dca_history = []

for idx, row in dca_sim_df.iterrows():
  price = row["Close"]
  z_pl = row["z_score_pl"]

  # 1. DCA Classique
  invested_classical += dca_base_amount
  btc_classical += dca_base_amount / price

  # 2. Smart DCA
  current_invest = dca_base_amount
  if "Smart" in dca_strategy:
    if z_pl < -1.0:
      current_invest = dca_base_amount * 2.0
    elif z_pl > 2.0:
      if "Suspendre" in overheat_action:
        current_invest = 0.0
      else:
        current_invest = dca_base_amount * 0.5

  invested_smart += current_invest
  btc_smart += current_invest / price if price > 0 else 0

  val_classical = btc_classical * price
  val_smart = btc_smart * price

  dca_history.append({
      "Date": row["Date"],
      "Investi Classique": invested_classical,
      "Portfolio Classique": val_classical,
      "Investi Smart": invested_smart,
      "Portfolio Smart": val_smart,
  })

df_dca_res = pd.DataFrame(dca_history)

if not df_dca_res.empty:
  last_row = df_dca_res.iloc[-1]

  fin_inv_c = last_row["Investi Classique"]
  fin_val_c = last_row["Portfolio Classique"]
  pnl_c = ((fin_val_c - fin_inv_c) / fin_inv_c) * 100 if fin_inv_c > 0 else 0

  fin_inv_s = last_row["Investi Smart"]
  fin_val_s = last_row["Portfolio Smart"]
  pnl_s = ((fin_val_s - fin_inv_s) / fin_inv_s) * 100 if fin_inv_s > 0 else 0

  col_res1, col_res2 = st.columns(2)
  with col_res1:
    st.markdown("### 📊 DCA Classique (Fixe)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investi", f"${fin_inv_c:,.0f}")
    c2.metric("Valeur Portefeuille", f"${fin_val_c:,.0f}")
    c3.metric("Performance", f"{pnl_c:+.1f}%")

  with col_res2:
    st.markdown("### 🧠 Smart DCA (Basé sur Power Law)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investi", f"${fin_inv_s:,.0f}")
    c2.metric("Valeur Portefeuille", f"${fin_val_s:,.0f}")
    c3.metric(
        "Performance", f"{pnl_s:+.1f}%", delta=f"{pnl_s - pnl_c:+.1f}% vs Fixe"
    )

  # Graphique comparatif
  fig_smart_dca = go.Figure()
  fig_smart_dca.add_trace(
      go.Scatter(
          x=df_dca_res["Date"],
          y=df_dca_res["Portfolio Classique"],
          name="Portfolio DCA Classique",
          line=dict(color="#9CA3AF", width=1.5, dash="dash"),
      )
  )
  fig_smart_dca.add_trace(
      go.Scatter(
          x=df_dca_res["Date"],
          y=df_dca_res["Portfolio Smart"],
          name="Portfolio Smart DCA",
          line=dict(color="#10B981", width=2.5),
      )
  )
  fig_smart_dca.add_trace(
      go.Scatter(
          x=df_dca_res["Date"],
          y=df_dca_res["Investi Smart"],
          name="Capital Total Investi (Smart)",
          line=dict(color="#38BDF8", width=1, dash="dot"),
      )
  )

  fig_smart_dca.update_layout(
      template="plotly_dark",
      height=400,
      margin=dict(l=20, r=20, t=30, b=20),
      yaxis_type="log",
      yaxis_title="USD (Échelle Log)",
      xaxis_title="Date",
      legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
  )
  st.plotly_chart(fig_smart_dca, use_container_width=True)
else:
  st.warning(
      "Aucune donnée disponible à partir de la date de début sélectionnée."
  )


# ==============================================================================
# SECTION FINALE : SCHÉMA CONCEPTUEL (TAS DE SABLE)
# ==============================================================================
st.markdown("---")
st.subheader("📚 Schéma Conceptuel : Le Tas de Sable de Bitcoin & LPPL")
st.image(
    "tas_de_sable.png",
    use_container_width=True,
    caption=(
        "Analogie du tas de sable (Self-Organized Criticality) appliquée à"
        " Bitcoin – Inspiré des travaux de Didier Sornette"
    ),
)
