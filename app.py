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

# ==============================================================================
# GESTION DE LA LANGUE / LANGUAGE SELECTION
# ==============================================================================
if "lang" not in st.session_state:
  st.session_state["lang"] = "Français"

lang = st.sidebar.selectbox(
    "🌍 Langue / Language", ["Français", "English"], key="lang"
)


def t(fr, en):
  return en if lang == "English" else fr


st.title(
    t(
        "₿ Bitcoin PowerLaw + LPPL (2 Harmonics) - Analyses Avancées",
        "₿ Bitcoin PowerLaw + LPPL (2 Harmonics) - Advanced Analytics",
    )
)

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
st.sidebar.header(t("⚙️ Paramètres du Modèle", "⚙️ Model Parameters"))

# Avertissement légal
st.sidebar.warning(
    t(
        "⚠️ **Avertissement :** Ce modèle est conçu exclusivement à des fins de"
        " recherche et de modélisation statistique à long terme. Il ne constitue"
        " en aucun cas un conseil en investissement.",
        "⚠️ **Disclaimer:** This model is designed exclusively for long-term"
        " statistical research and modeling purposes. It does not constitute"
        " investment advice.",
    )
)

# --- GESTION DES CONFIGURATIONS JSON (CHARGEMENT & SAUVEGARDE) ---
st.sidebar.subheader(t("📁 Gestion de Configuration", "📁 Configuration Management"))

uploaded_file = st.sidebar.file_uploader(
    t("Charger Config (JSON)", "Load Config (JSON)"),
    type=["json"],
    help=t(
        "❓ Restaurez une configuration de paramètres précédemment sauvegardée"
        " au format JSON.",
        "❓ Restore a previously saved parameter configuration in JSON format.",
    ),
)
if uploaded_file is not None:
  try:
    loaded_cfg = json.load(uploaded_file)
    for k_cfg, v_cfg in loaded_cfg.items():
      if k_cfg in DEFAULT_PARAMS:
        st.session_state[k_cfg] = float(v_cfg)
    st.sidebar.success(
        t(
            "Configuration chargée avec succès !",
            "Configuration loaded successfully!",
        )
    )
  except Exception as e:
    st.sidebar.error(
        f"{t('Erreur de lecture du JSON', 'Error reading JSON')}: {e}"
    )

config_dict = {k: st.session_state[k] for k in DEFAULT_PARAMS.keys()}
st.sidebar.download_button(
    t("💾 Sauvegarder Config (JSON)", "💾 Save Config (JSON)"),
    data=json.dumps(config_dict, indent=2),
    file_name="lppl_params.json",
    mime="application/json",
    help=t(
        "❓ Exporte vos paramètres actuels sous forme de fichier JSON.",
        "❓ Exports your current parameters as a JSON file.",
    ),
)

st.sidebar.markdown("---")

horizon_years = st.sidebar.slider(
    t("🔮 Horizon de Projection (Années)", "🔮 Projection Horizon (Years)"),
    min_value=1,
    max_value=3,
    value=3,
    step=1,
    help=t(
        "❓ Définit le nombre d'années dans le futur sur lesquelles étendre"
        " les courbes de projection LPPL et Power Law.",
        "❓ Defines the number of years in the future to extend LPPL and Power"
        " Law projection curves.",
    ),
)

st.sidebar.markdown(
    t(
        "📊 **Bandes Power Law (Écart-type σ)**",
        "📊 **Power Law Bands (Standard Deviation σ)**",
    )
)
pl_sigma = st.sidebar.slider(
    t("Écart-type (σ) Power Law", "Power Law Standard Deviation (σ)"),
    min_value=0.5,
    max_value=4.0,
    value=1.5,
    step=0.1,
    help=t(
        "❓ Multiplicateur d'écart-type (σ) appliqué aux résidus pour tracer"
        " les bandes supérieure et inférieure Power Law.",
        "❓ Standard deviation multiplier (σ) applied to residuals to plot"
        " Power Law upper and lower bands.",
    ),
)
pl_sigma_upper = pl_sigma
pl_sigma_lower = pl_sigma

with st.sidebar.expander(
    t("📌 Power Law (Tendance Fondamentale)", "📌 Power Law (Fundamental Trend)"),
    expanded=True,
):
  st.caption(
      t(
          "Ajuste la tendance logarithmique fondamentale du prix (Prix = exp(A +"
          " B * ln(t))).",
          "Adjusts the fundamental logarithmic price trend (Price = exp(A + B *"
          " ln(t))).",
      )
  )
  A = st.number_input(
      t("A (Ordonnée à l'origine)", "A (Intercept)"),
      value=st.session_state["A"],
      step=0.01,
      key="input_A",
      help=t(
          "❓ L'intercepte logarithmique de la loi de puissance. Fixe le niveau"
          " de départ à t=1.",
          "❓ The logarithmic intercept of the power law. Sets the starting"
          " level at t=1.",
      ),
  )
  B = st.number_input(
      t("B (Pente / Exposant)", "B (Slope / Exponent)"),
      value=st.session_state["B"],
      step=0.001,
      key="input_B",
      help=t(
          "❓ La pente de la loi de puissance (exposant de croissance dans le"
          " temps).",
          "❓ The slope of the power law (growth exponent over time).",
      ),
  )

with st.sidebar.expander(
    t("🎛️ Options du Modèle & Affichage", "🎛️ Model Options & Display"),
    expanded=True,
):
  show_trend = st.checkbox(
      t("Afficher la Tendance (Power Law)", "Show Trend (Power Law)"),
      value=True,
      help=t(
          "❓ Affiche la ligne de tendance fondamentale Power Law ainsi que son"
          " canal supérieur et inférieur.",
          "❓ Displays the fundamental Power Law trend line along with its upper"
          " and lower channel.",
      ),
  )
  show_lppl = st.checkbox(
      t("Afficher la Courbe LPPL", "Show LPPL Curve"),
      value=True,
      help=t(
          "❓ Affiche la courbe modèle LPPL ajustée (avec oscillations) et ses"
          " projections futures.",
          "❓ Displays the fitted LPPL model curve (with oscillations) and its"
          " future projections.",
      ),
  )
  log_time_axis = st.checkbox(
      t(
          "Échelle de Temps Logarithmique (ln(t) sur l'Axe X)",
          "Logarithmic Time Scale (ln(t) on X-Axis)",
      ),
      value=False,
      help=t(
          "❓ Permet de basculer l'axe X des graphiques entre le temps linéaire"
          " calendaire (Date) et le logarithme du temps (ln(t)).",
          "❓ Switches the X-axis between calendar linear time (Date) and"
          " logarithmic time (ln(t)).",
      ),
  )

with st.sidebar.expander(
    t("🌊 Harmoniques LPPL", "🌊 LPPL Harmonics"), expanded=True
):
  st.markdown(t("**Harmonic 1 (Macro Cycle)**", "**Harmonic 1 (Macro Cycle)**"))
  C1 = st.number_input(
      t("Amplitude H1 (C1)", "H1 Amplitude (C1)"),
      value=st.session_state["C1"],
      step=0.01,
      key="input_C1",
      help=t(
          "❓ Amplitude de la première oscillation log-périodique majeure.",
          "❓ Amplitude of the first major log-periodic oscillation.",
      ),
  )
  omega = st.number_input(
      t("Omega (ω) - Fréquence", "Omega (ω) - Frequency"),
      value=st.session_state["omega"],
      step=0.001,
      key="input_omega",
      help=t(
          "❓ Fréquence angulaire log-périodique. Détermine la vitesse à"
          " laquelle les cycles se compressent au fil du temps.",
          "❓ Log-periodic angular frequency. Determines the rate at which"
          " cycles compress over time.",
      ),
  )
  phi1 = st.number_input(
      t("Phase H1 (φ1)", "H1 Phase (φ1)"),
      value=st.session_state["phi1"],
      step=0.01,
      key="input_phi1",
      help=t(
          "❓ Décalage temporel de la première onde harmonique.",
          "❓ Time shift of the first harmonic wave.",
      ),
  )

  st.markdown(t("**Harmonic 2 (Micro Cycle)**", "**Harmonic 2 (Micro Cycle)**"))
  C2 = st.number_input(
      t("Amplitude H2 (C2)", "H2 Amplitude (C2)"),
      value=st.session_state["C2"],
      step=0.01,
      key="input_C2",
      help=t(
          "❓ Amplitude des oscillations secondaires (sous-cycles).",
          "❓ Amplitude of secondary oscillations (sub-cycles).",
      ),
  )
  phi2 = st.number_input(
      t("Phase H2 (φ2)", "H2 Phase (φ2)"),
      value=st.session_state["phi2"],
      step=0.01,
      key="input_phi2",
      help=t(
          "❓ Décalage temporel de la seconde onde harmonique.",
          "❓ Time shift of the second harmonic wave.",
      ),
  )


# ==============================================================================
# 2. CHARGEMENT ET FILTRAGE DES DONNÉES HISTORIQUES
# ==============================================================================
@st.cache_data(ttl=3600)
def load_btc_data():
  try:
    url_coinmetrics = (
        "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    )
    df_raw = pd.read_csv(url_coinmetrics)
    df = pd.DataFrame()
    df["Date"] = pd.to_datetime(df_raw["time"])
    df["Close"] = pd.to_numeric(df_raw["PriceUSD"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date", ascending=True)
  except Exception:
    try:
      all_klines = []
      start_time = int(pd.to_datetime("2017-08-17").timestamp() * 1000)
      while True:
        url_binance = (
            f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_time}&limit=1000"
        )
        res = pd.read_json(url_binance)
        if res.empty:
          break
        all_klines.append(res)
        start_time = int(res.iloc[-1][0]) + 86400000
        if len(res) < 1000:
          break

      df_binance = pd.concat(all_klines, ignore_index=True)
      df = pd.DataFrame()
      df["Date"] = pd.to_datetime(df_binance[0], unit="ms")
      df["Close"] = pd.to_numeric(df_binance[4], errors="coerce")
    except Exception as e:
      st.error(
          t(
              "Erreur lors du chargement des données historiques",
              "Error loading historical data",
          )
          + f": {e}"
      )
      st.stop()

  df = df[df["Close"] > 0].reset_index(drop=True)
  df["Days"] = (df["Date"] - GENESIS_DATE).dt.total_seconds() / 86400.0
  df["Days"] = np.maximum(df["Days"], 1.0)
  df["lnT"] = np.log(df["Days"])
  df["actualLog"] = np.log(df["Close"])
  return df


raw_df = load_btc_data()

st.sidebar.subheader(t("📅 Période d'Entraînement", "📅 Training Period"))
min_date = raw_df["Date"].min().to_pydatetime()
max_date = raw_df["Date"].max().to_pydatetime()

selected_dates = st.sidebar.date_input(
    t("Plage de dates d'analyse", "Analysis Date Range"),
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help=t(
        "❓ Restreint les données sur lesquelles le modèle est calibré et"
        " évalué.",
        "❓ Restricts the data on which the model is calibrated and evaluated.",
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
  st.warning(
      t(
          "Aucune donnée disponible pour la plage sélectionnée.",
          "No data available for the selected range.",
      )
  )
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


# ==============================================================================
# 4. OPTIMISATION GLOBALE AUTOMATIQUE (DIFFERENTIAL EVOLUTION)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader(t("🎯 Calibrage Automatique", "🎯 Automatic Calibration"))

if st.sidebar.button(
    t("🤖 Ajuster les paramètres au dataset", "🤖 Fit Parameters to Dataset"),
    help=t(
        "❓ Lance l'optimisation globale (Differential Evolution) pour estimer"
        " les paramètres de manière robuste.",
        "❓ Runs global optimization (Differential Evolution) to robustly"
        " estimate parameters.",
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
      t(
          "Optimisation globale en cours (Recherche globale + Polish)...",
          "Global optimization in progress (Global Search + Polish)...",
      )
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
    st.sidebar.error(
        t(
            "L'optimisation globale a échoué.", "Global optimization failed."
        )
    )

if "opt_msg" in st.session_state:
  st.sidebar.success(t("Ajustement réussi !", "Fitting successful!"))
  st.sidebar.info(st.session_state["opt_msg"])


# ==============================================================================
# 5. CALCULS GLOBAUX, POWER LAW & RÉSIDUS (Z-SCORES)
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

ss_res = np.sum((df["actualLog"] - df["logModel"]) ** 2)
ss_tot = np.sum((df["actualLog"] - np.mean(df["actualLog"])) ** 2)
r2_global = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

rmse = np.sqrt(ss_res / len(df))
rmse_pct = 100.0 * (np.exp(rmse) - 1.0)
mae = np.mean(np.abs(df["actualLog"] - df["logModel"]))
mae_pct = 100.0 * (np.exp(mae) - 1.0)

df["upperBand"] = df["modelPrice"] * (1 + rmse_pct / 100.0)
df["lowerBand"] = df["modelPrice"] / (1 + rmse_pct / 100.0)

df["residuals"] = df["actualLog"] - df["logModel"]
res_std = np.std(df["residuals"])
df["z_score"] = df["residuals"] / res_std if res_std > 0 else 0.0

ratio_history = df["Close"] / df["modelPrice"]
ratio_percentile = (ratio_history <= ratio_history.iloc[-1]).mean() * 100.0


def get_market_state(score):
  if score < 10:
    return t("Valeur Extrême", "Extreme Value"), "#00FF00"
  elif score < 25:
    return t("Sous-évalué", "Undervalued"), "#008000"
  elif score < 75:
    return t("Juste Valeur", "Fair Value"), "#FFA500"
  elif score < 90:
    return t("Surévalué", "Overvalued"), "#FF0000"
  else:
    return t("Bulle Extrême", "Extreme Bubble"), "#FF00FF"


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
          t("Date Evaluation", "Evaluation Date"): eval_date,
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
  xaxis_title = t(
      "Logarithme du Temps (ln(t) depuis le Genesis)",
      "Logarithmic Time (ln(t) since Genesis)",
  )
  custom_hover = df["Date"].dt.strftime("%Y-%m-%d").values
  fut_hover = [d.strftime("%Y-%m-%d") for d in future_dates_arr]
  all_hover = df["Date"].dt.strftime("%Y-%m-%d").tolist() + fut_hover
else:
  x_hist = df["Date"]
  x_trend = df["Date"].tolist() + future_dates_arr
  x_proj = [df["Date"].iloc[-1]] + future_dates_arr
  x_lppl_all = df["Date"].tolist() + future_dates_arr
  xaxis_title = t("Date", "Date")

all_pl_trend = df["trendPrice"].tolist() + list(future_pl_trend)
all_pl_upper = df["trendUpperPrice"].tolist() + list(future_pl_upper)
all_pl_lower = df["trendLowerPrice"].tolist() + list(future_pl_lower)

proj_lppl = [df["modelPrice"].iloc[-1]] + list(future_lppl)
all_lppl_prices = df["modelPrice"].tolist() + list(future_lppl)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.72, 0.28],
    subplot_titles=(
        (
            t(
                "Prix & Projections Avancées LPPL (Échelle Logarithmique du"
                " Temps ln(t))",
                "Price & Advanced LPPL Projections (Logarithmic Time Scale"
                " ln(t))",
            )
            if log_time_axis
            else t(
                "Prix & Projections Avancées LPPL (Temps Linéaire / Date)",
                "Price & Advanced LPPL Projections (Linear Time / Date)",
            )
        ),
        t(
            "Analyse des Résidus (Z-Scores LPPL & Power Law)",
            "Residual Analysis (LPPL & Power Law Z-Scores)",
        ),
    ),
)

fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=df["Close"],
        mode="lines",
        name=t("Prix BTC", "BTC Price"),
        line=dict(color="#D1D5DB", width=1.2),
    ),
    row=1,
    col=1,
)

if show_lppl:
  sigma_levels = [
      {
          "mult": 1.0,
          "opacity": 0.18,
          "name": t("Canal ±1.0σ (68%)", "±1.0σ Channel (68%)"),
      },
      {
          "mult": 2.0,
          "opacity": 0.10,
          "name": t("Canal ±2.0σ (95%)", "±2.0σ Channel (95%)"),
      },
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
          name=t("LPPL Model (Fit)", "LPPL Model (Fit)"),
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
          name=t("LPPL Projection Centrale", "Central LPPL Projection"),
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
          name=t("Power Law Trend", "Power Law Trend"),
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
        name=t("Z-Score LPPL", "LPPL Z-Score"),
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
        name=t("Z-Score Power Law", "Power Law Z-Score"),
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

fig.update_yaxes(type="log", title_text=t("Prix (USD)", "Price (USD)"), row=1, col=1)
fig.update_yaxes(title_text=t("Z-Score (σ)", "Z-Score (σ)"), row=2, col=1)

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
  with st.expander(
      t(
          "❓ Guide de Lecture du Graphique Principal",
          "❓ Main Chart Reading Guide",
      )
  ):
    st.markdown(
        t(
            """
        * **Prix BTC (Gris)** : Cours de clôture quotidien du Bitcoin.
        * **LPPL Model (Orange)** : Courbe ajustée du modèle LPPL combinant la tendance et les oscillations. Les pointillés représentent la projection future.
        * **Power Law Fit (Bleu Cyan)** : Tendance fondamentale A + B * ln(t). Les bandes supérieure et inférieure sont calculées dynamiquement à partir des résidus via le paramètre sigma réglable.
        * **Canaux Multi-Sigma (±1σ, ±2σ, ±3σ)** : Entonnoir de probabilité calibré empiriquement via les erreurs Walk-Forward.
        * **Z-Scores (Panneau Inférieur)** : Mesures de l'écart du prix réel par rapport au modèle LPPL (en orange) et à la Power Law fondamentale (en bleu cyan) exprimés en écarts-types.
        """,
            """
        * **BTC Price (Gray)**: Daily Bitcoin closing price.
        * **LPPL Model (Orange)**: Fitted LPPL model curve combining trend and oscillations. Dotted lines represent future projection.
        * **Power Law Fit (Cyan Blue)**: Fundamental trend A + B * ln(t). Upper and lower bands are dynamically calculated from residuals via the adjustable sigma parameter.
        * **Multi-Sigma Channels (±1σ, ±2σ, ±3σ)**: Probability funnel empirically calibrated via Walk-Forward errors.
        * **Z-Scores (Lower Panel)**: Measures of real price deviation from the LPPL model (in orange) and fundamental Power Law (in cyan blue) expressed in standard deviations.
        """,
        )
    )

# --- Distribution empirique des résidus vs Loi de Student ---
st.markdown("---")
st.subheader(
    t(
        "📊 Distribution empirique des résidus vs Loi de Student (Fat Tails"
        " Check)",
        "📊 Empirical Residual Distribution vs Student-t (Fat Tails Check)",
    )
)

dist_horizon_options = {
    t("In-Sample (Modèle Global)", "In-Sample (Global Model)"): 0,
    t("3 mois (90 jours) Out-Of-Sample", "3 months (90 days) Out-Of-Sample"): 90,
    t(
        "6 mois (180 jours) Out-Of-Sample", "6 months (180 days) Out-Of-Sample"
    ): 180,
    t("1 an (365 jours) Out-Of-Sample", "1 year (365 days) Out-Of-Sample"): 365,
    t("2 ans (730 jours) Out-Of-Sample", "2 years (730 days) Out-Of-Sample"): 730,
    t(
        "3 ans (1095 jours) Out-Of-Sample", "3 years (1095 days) Out-Of-Sample"
    ): 1095,
}

selected_dist_label = st.selectbox(
    t(
        "Sélectionner la source des résidus (In-Sample ou Horizon Forward OOS)",
        "Select residual source (In-Sample or Forward OOS Horizon)",
    ),
    options=list(dist_horizon_options.keys()),
    index=3,
    key="dist_horizon_selectbox",
    help=t(
        "❓ Permet de choisir si l'analyse de distribution et des résidus"
        " s'effectue sur l'In-Sample global ou sur les erreurs de prédiction"
        " Out-Of-Sample pour un horizon de forward spécifique.",
        "❓ Choose whether residual distribution analysis runs on global"
        " In-Sample or Out-Of-Sample prediction errors for a specific forward"
        " horizon.",
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
    dist_mode_label = t(
        "OOS (Données insuffisantes)", "OOS (Insufficient Data)"
    )
else:
  residuals_pct = (np.exp(df["residuals"]) - 1.0) * 100.0
  res_pct_clean = residuals_pct.dropna().values
  dist_mode_label = t("In-Sample", "In-Sample")

if len(res_pct_clean) > 0:
  df_t, loc_t, scale_t = t.fit(res_pct_clean)
  std_pct_resid = np.std(res_pct_clean)
else:
  df_t, loc_t, scale_t, std_pct_resid = 3.0, 0.0, 1.0, 1.0

col_dist1, col_dist2 = st.columns([2, 1])

with col_dist2:
  st.markdown(
      f"### 📐 {t('Analyse de forme (Student-t)', 'Shape Analysis (Student-t)')}"
      f" [{dist_mode_label}]"
  )
  st.markdown(
      t(
          f"""
    Ce graphique superpose ({dist_mode_label}) :
    * **L'histogramme réel** de vos erreurs de prédiction en pourcentage.
    * **La loi de Student ajustée** (df = {df_t:.2f}).
    
    *L'utilisation d'une loi de Student (t-distribution) plutôt qu'une gaussienne classique permet de modéliser proprement les **fat tails** caractéristiques du Bitcoin à long terme.*
    """,
          f"""
    This chart overlays ({dist_mode_label}):
    * **The actual histogram** of your percentage prediction errors.
    * **The fitted Student-t distribution** (df = {df_t:.2f}).
    
    *Using a Student-t distribution instead of a standard Gaussian allows proper modeling of **fat tails** characteristic of long-term Bitcoin data.*
    """,
      )
  )

  st.metric(
      t("Degrés de liberté (Student df)", "Degrees of Freedom (Student df)"),
      f"{df_t:.2f}",
      help=t(
          "Plus df est bas (< 30), plus les queues de distribution sont"
          " épaisses.",
          "Lower df (< 30) indicates heavier distribution tails.",
      ),
  )
  st.metric(
      t("Kurtosis des Résidus (%)", "Residual Kurtosis (%)"),
      f"{kurtosis(res_pct_clean):.2f}" if len(res_pct_clean) > 0 else "N/A",
      help=t(
          "Aplatissement mesuré sur la série en pourcentage.",
          "Flattening measured on the percentage series.",
      ),
  )

with col_dist1:
  if len(res_pct_clean) > 0:
    x_range = np.linspace(res_pct_clean.min(), res_pct_clean.max(), 500)
    y_student = t.pdf(x_range, df_t, loc=loc_t, scale=scale_t)

    fig_dist = ff.create_distplot(
        [res_pct_clean],
        [
            f"{t('Résidus du Modèle (%)', 'Model Residuals (%)')} [{dist_mode_label}]"
        ],
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
            name=f"Student-t (df={df_t:.2f})",
            line=dict(color="#00BFFF", width=2.5, dash="dash"),
        )
    )
    fig_dist.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        xaxis_title=t(
            "Erreur de prédiction / Résidu (%)",
            "Prediction Error / Residual (%)",
        ),
        yaxis_title=t("Densité de probabilité", "Probability Density"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
  else:
    st.warning(
        t(
            "Données OOS insuffisantes pour afficher la distribution.",
            "Insufficient OOS data to display distribution.",
        )
    )

with col_dash:
  with st.container(border=True):
    st.subheader(
        t("📊 Fit Quality & Robustesse", "📊 Fit Quality & Robustness")
    )

    c_g1, c_g2 = st.columns(2)
    c_g1.metric(
        "R² Global",
        f"{r2_global:.4f}",
        help=t(
            "❓ Coefficient de détermination expliquant la variance observée"
            " sur le dataset.",
            "❓ Coefficient of determination explaining variance observed on"
            " the dataset.",
        ),
    )
    c_g2.metric(
        "R² Ajusté",
        f"{r2_adj:.4f}",
        help=t(
            "❓ Coefficient de détermination ajusté selon le nombre de"
            " paramètres du modèle.",
            "❓ Coefficient of determination adjusted for model parameter"
            " count.",
        ),
    )

    st.metric(
        "Forward R² (1Y OOS)",
        f"{r2_oos_1y:.4f}",
        help=t(
            "❓ Coefficient de détermination hors-échantillon (Out-Of-Sample)"
            " évalué sur un horizon prédictif de 1 an.",
            "❓ Out-of-sample coefficient of determination evaluated on a 1-year"
            " predictive horizon.",
        ),
    )

    st.markdown("---")
    st.caption(
        t(
            "📐 **Métriques d'Erreur & Généralisation**",
            "📐 **Error Metrics & Generalization**",
        )
    )

    c_m3, c_m4 = st.columns(2)
    c_m3.metric(
        "RMSE In-Sample",
        f"{rmse_pct:.1f}%",
        help=t(
            "❓ Erreur quadratique moyenne en pourcentage sur la période"
            " d'entraînement.",
            "❓ Root mean square error in percentage over the training period.",
        ),
    )
    c_m4.metric(
        "MAE In-Sample",
        f"{mae_pct:.1f}%",
        help=t(
            "❓ Erreur absolue moyenne en pourcentage sur la période"
            " d'entraînement.",
            "❓ Mean absolute error in percentage over the training period.",
        ),
    )

    c_m5, c_m6 = st.columns(2)
    c_m5.metric(
        "OOS RMSE (1Y)",
        f"{wf_rmse_1y_val:.1f}%",
        help=t(
            "❓ Erreur de prédiction hors-échantillon (Out-Of-Sample) sur un"
            " horizon de 1 an.",
            "❓ Out-of-sample prediction error over a 1-year horizon.",
        ),
    )
    c_m6.metric(
        t("Ratio Out/In", "Out/In Ratio"),
        f"{gen_ratio:.2f}x",
        help=t(
            "❓ Ratios < 1.5x indiquent une bonne capacité de généralisation"
            " (faible surapprentissage).",
            "❓ Ratios < 1.5x indicate good generalization capacity (low"
            " overfitting).",
        ),
    )

  with st.container(border=True):
    st.subheader(t("🎯 Valuation", "🎯 Valuation"))
    st.metric(
        t("Percentile Ratio", "Ratio Percentile"),
        f"{ratio_percentile:.1f}%",
        help=t(
            "❓ Position relative de la valorisation actuelle par rapport à"
            " l'historique complet.",
            "❓ Relative position of current valuation compared to full history.",
        ),
    )
    st.markdown(
        f"{t('Statut', 'Status')} : <span"
        f" style='color:{state_color};font-weight:bold;'>{state_txt}</span>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION : COURBE DE PRÉDICTION OOS (HORIZON PERSONNALISABLE)
# ==============================================================================
st.markdown("---")

oos_chart_options = {
    t("3 mois (90 jours)", "3 months (90 days)"): 90,
    t("6 mois (180 jours)", "6 months (180 days)"): 180,
    t("1 an (365 jours)", "1 year (365 days)"): 365,
    t("2 ans (730 jours)", "2 years (730 days)"): 730,
    t("3 ans (1095 jours)", "3 years (1095 days)"): 1095,
}

if "oos_chart_horizon_selectbox" not in st.session_state:
  st.session_state["oos_chart_horizon_selectbox"] = t(
      "1 an (365 jours)", "1 year (365 days)"
  )

current_label = st.session_state["oos_chart_horizon_selectbox"]

st.subheader(
    f"📈 {t('Comparaison de la Courbe de Prédiction Out-Of-Sample', 'Out-Of-Sample Prediction Curve Comparison')}"
    f" ({current_label}) vs {t('Prix Réel', 'Real Price')}"
)

with st.expander(
    t("❓ Guide de Lecture - OOS Historique", "❓ Reading Guide - Historical OOS")
):
  st.markdown(
      t(
          f"""
    * Ce graphique trace les projections faites par le modèle global avec un horizon fixe de **{current_label}** en amont, comparées directement au prix réel atteint à cette échéance.
    * Il permet d'évaluer visuellement la robustesse et le biais d'anticipation du modèle sur cet horizon à travers tout l'historique du Bitcoin.
    """,
          f"""
    * This chart plots projections made by the global model with a fixed horizon of **{current_label}** ahead, compared directly to the actual price reached at that maturity.
    * It visually assesses the robustness and anticipation bias of the model over this horizon across Bitcoin's entire history.
    """,
      )
  )

selected_oos_chart_label = st.selectbox(
    t(
        "Sélectionner l'horizon OOS pour la comparaison de prédiction",
        "Select OOS horizon for prediction comparison",
    ),
    options=list(oos_chart_options.keys()),
    key="oos_chart_horizon_selectbox",
    help=t(
        "❓ Permet de choisir l'horizon de prédiction Out-Of-Sample affiché dans"
        " le graphique de comparaison.",
        "❓ Selects the Out-Of-Sample prediction horizon displayed in the"
        " comparison chart.",
    ),
)
h_days = oos_chart_options[selected_oos_chart_label]

fig_oos_parallel = go.Figure()
fig_oos_parallel.add_trace(
    go.Scatter(
        x=x_hist,
        y=df["Close"],
        mode="lines",
        name=t("Prix BTC Réel", "Actual BTC Price"),
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

fig_oos_parallel.update_yaxes(
    type="log", title_text=t("Prix (USD, Log)", "Price (USD, Log)")
)
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
st.subheader(
    t(
        "🔄 Stabilité Temporelle (Rolling Walk-Forward Analysis)",
        "🔄 Temporal Stability (Rolling Walk-Forward Analysis)",
    )
)

col_rwf_params, col_rwf_chart = st.columns([1, 3])

with col_rwf_params:
  rwf_window = st.number_input(
      t("Taille Fenêtre Train (Jours)", "Train Window Size (Days)"),
      value=730,
      step=180,
      help=t(
          "❓ Nombre de jours de données utilisés pour la période"
          " d'entraînement glissante.",
          "❓ Number of data days used for the rolling training period.",
      ),
  )

  horizon_options = {
      t("3 mois (90 jours)", "3 months (90 days)"): 90,
      t("6 mois (180 jours)", "6 months (180 days)"): 180,
      t("1 an (365 jours)", "1 year (365 days)"): 365,
      t("2 ans (730 jours)", "2 years (730 days)"): 730,
      t("3 ans (1095 jours)", "3 years (1095 days)"): 1095,
  }
  selected_horizon_label = st.selectbox(
      t("Horizon de Test OOS", "OOS Test Horizon"),
      options=list(horizon_options.keys()),
      index=2,
      key="horizon_test_oos_selectbox",
      help=t(
          "❓ Choisit l'horizon de prédiction hors-échantillon testé à chaque"
          " étape glissante.",
          "❓ Selects the out-of-sample prediction horizon tested at each rolling"
          " step.",
      ),
  )
  rwf_horizon = horizon_options[selected_horizon_label]

  rwf_step = st.number_input(
      t("Pas de Glissement (Jours)", "Rolling Step (Days)"),
      value=90,
      step=30,
      help=t(
          "❓ Pas d'avancement temporel entre chaque évaluation Walk-Forward.",
          "❓ Time advancement step between each Walk-Forward evaluation.",
      ),
  )

  metric_choice = st.radio(
      t("Métrique d'erreur à afficher", "Error metric to display"),
      ["RMSE", "MAE"],
      help=t(
          "❓ Choisir d'afficher la RMSE ou la MAE dans le graphique de"
          " stabilité temporelle.",
          "❓ Choose whether to display RMSE or MAE in the temporal stability"
          " chart.",
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
              x=df_rwf[t("Date Evaluation", "Evaluation Date")],
              y=df_rwf["RMSE Out-Of-Sample (%)"],
              mode="lines",
              name="RMSE OOS (%)",
              line=dict(color="#FF4B4B", width=2),
          )
      )
    else:
      fig_rwf.add_trace(
          go.Scatter(
              x=df_rwf[t("Date Evaluation", "Evaluation Date")],
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
        xaxis_title=t("Date d'Évaluation", "Evaluation Date"),
    )
    st.plotly_chart(fig_rwf, use_container_width=True)
  else:
    st.info(
        t(
            "Historique insuffisant pour calculer les fenêtres glissantes"
            " sélectionnées.",
            "Insufficient history to calculate selected rolling windows.",
        )
    )


# ==============================================================================
# SECTION : POURCENTAGE D'ERREUR RMS / OOS PAR NIVEAU DE SIGMA
# ==============================================================================
st.markdown("---")
st.subheader(
    t(
        "🎯 Contribution des Erreurs par Niveau de Sigma (σ) et Brackets",
        "🎯 Error Contribution by Sigma (σ) Level & Brackets",
    )
)

sigma_horizon_options = {
    t("In-Sample (Modèle Global)", "In-Sample (Global Model)"): 0,
    t("3 mois (90 jours) Out-Of-Sample", "3 months (90 days) Out-Of-Sample"): 90,
    t(
        "6 mois (180 jours) Out-Of-Sample", "6 months (180 days) Out-Of-Sample"
    ): 180,
    t("1 an (365 jours) Out-Of-Sample", "1 year (365 days) Out-Of-Sample"): 365,
    t("2 ans (730 jours) Out-Of-Sample", "2 years (730 days) Out-Of-Sample"): 730,
    t(
        "3 ans (1095 jours) Out-Of-Sample", "3 years (1095 days) Out-Of-Sample"
    ): 1095,
}

selected_sigma_label = st.selectbox(
    t(
        "Sélectionner l'horizon pour l'analyse des brackets Sigma",
        "Select horizon for Sigma bracket analysis",
    ),
    options=list(sigma_horizon_options.keys()),
    index=3,
    key="sigma_horizon_selectbox",
    help=t(
        "❓ Permet de choisir l'horizon In-Sample ou Out-Of-Sample"
        " spécifiquement pour l'analyse de la contribution des erreurs par"
        " brackets de sigma.",
        "❓ Selects In-Sample or Out-Of-Sample horizon specifically for sigma"
        " bracket error contribution analysis.",
    ),
)
horizon_sigma_eval = sigma_horizon_options[selected_sigma_label]

use_oos_sigma = horizon_sigma_eval > 0
analysis_mode_label = (
    f"Out-Of-Sample ({selected_sigma_label.split(' ')[0]})"
    if use_oos_sigma
    else t("In-Sample", "In-Sample")
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
            name=(
                f"{t('Part des Points (%)', 'Points Share (%)')} [{analysis_mode_label}]"
            ),
            marker_color="#38BDF8",
        )
    )
    fig_sigma_contrib.add_trace(
        go.Bar(
            x=bracket_names,
            y=pct_mse,
            name=(
                f"{t('Contribution à l’Erreur RMS/MSE (%)', 'RMS/MSE Error Contribution (%)')} [{analysis_mode_label}]"
            ),
            marker_color="#FF9900",
        )
    )
    fig_sigma_contrib.update_layout(
        template="plotly_dark",
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        xaxis_title=t(
            "Intervalles d'Écart-type (sigma)",
            "Standard Deviation Intervals (sigma)",
        ),
        yaxis_title=t("Pourcentage (%)", "Percentage (%)"),
    )
    st.plotly_chart(fig_sigma_contrib, use_container_width=True)

  with col_sig2:
    st.markdown(
        f"### 📊 {t('Analyse des Brackets Sigma', 'Sigma Bracket Analysis')} ({analysis_mode_label})"
    )
    st.markdown(
        t(
            """
        Ce graphique permet d'identifier l'origine de l'erreur globale :
        * **Part des points (Bleu)** : Fréquence brute des écarts dans chaque tranche.
        * **Contribution RMS/MSE (Orange)** : Poids des erreurs au carré, soulignant l'impact des valeurs extrêmes (fat tails).
        """,
            """
        This chart helps identify the origin of global error:
        * **Points Share (Blue)**: Raw frequency of deviations in each bracket.
        * **RMS/MSE Contribution (Orange)**: Weight of squared errors, highlighting the impact of extreme values (fat tails).
        """,
        )
    )

    df_bracket_summary = pd.DataFrame({
        t("Bracket", "Bracket"): bracket_names,
        t("Points (%)", "Points (%)"): pct_points,
        "MSE (%)": pct_mse,
    })
    st.dataframe(df_bracket_summary, hide_index=True, use_container_width=True)
else:
  st.info(
      t(
          "Données insuffisantes pour calculer la répartition par sigma.",
          "Insufficient data to calculate sigma distribution.",
      )
  )


# ==============================================================================
# 10. TABLEAUX DE PERFORMANCE FIXE & EXPORT CSV
# ==============================================================================
st.markdown("---")
col_wf, col_proj = st.columns(2)

with col_wf:
  st.subheader(
      t(
          "📈 Performance Walk-Forward (Globale)",
          "📈 Walk-Forward Performance (Global)",
      )
  )

  days_arr = df["Days"].values
  close_arr = df["Close"].values

  wf_data = []
  for h in [365, 730, 1095]:
    acc, bull, edge, mae_h, rmse_h, r2_oos = run_wf_analysis_fast(
        h, days_arr, close_arr, A, B, C1, omega, phi1, C2, phi2
    )
    wf_data.append({
        t("Horizon", "Horizon"): f"{h}d",
        "Forward R²": f"{r2_oos:.3f}",
        t("Dir Acc", "Dir Acc"): f"{acc:.0f}%",
        t("Edge", "Edge"): f"{edge:.1f}%",
        "MAE OOS": f"{mae_h:.1f}%",
        "RMSE OOS": f"{rmse_h:.1f}%",
    })
  st.dataframe(
      pd.DataFrame(wf_data), hide_index=True, use_container_width=True
  )

with col_proj:
  st.subheader(t("🎯 Objectifs de Prix & Export", "🎯 Price Targets & Export"))

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
          t("Horizon", "Horizon"): f"{yr}Y",
          t("LPPL Target", "LPPL Target"): f"${proj_price:,.0f}",
          t(
              "Cône Projection (±1σ)", "Projection Cone (±1σ)"
          ): f"${cone_lower:,.0f} - ${cone_upper:,.0f}",
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
      label=t(
          "📥 Télécharger les projections (CSV)",
          "📥 Download Projections (CSV)",
      ),
      data=csv_data,
      file_name=f"btc_lppl_projections_{last_date.strftime('%Y%m%d')}.csv",
      mime="text/csv",
      help=t(
          "❓ Exporte le tableau des objectifs de prix futurs au format CSV.",
          "❓ Exports the future price targets table in CSV format.",
      ),
  )
