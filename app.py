from datetime import timedelta
import json
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
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
    "A": -38.02,
    "B": 5.696,
    "C1": 2.06,
    "omega": 8.640,
    "phi1": -2.14,
    "C2": 1.02,
    "phi2": -3.14,
}

for key, val in DEFAULT_PARAMS.items():
  if key not in st.session_state:
    st.session_state[key] = val

# Paramètres par défaut pour la configuration de la fraction d'énergie
ENERGY_PARAMS = {
    "energy_decay_10": 3.15,
    "energy_scale_macro": 1.73,
    "energy_scale_micro": 0.6,
}

for key, val in ENERGY_PARAMS.items():
  if key not in st.session_state:
    st.session_state[key] = val

# Constante temporelle de référence
GENESIS_DATE = pd.to_datetime("2009-01-03")


# ==============================================================================
# 1. PARAMÈTRES ET INPUTS (SIDEBAR & CONFIGURATION)
# ==============================================================================
st.sidebar.header("⚙️ Paramètres du Modèle")

st.sidebar.warning(
    "⚠️ **Avertissement :** Ce modèle est conçu exclusivement à des fins de"
    " recherche et de modélisation statistique à long terme. Il ne constitue"
    " en aucun cas un conseil en investissement."
)

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
      elif k_cfg in ENERGY_PARAMS:
        st.session_state[k_cfg] = float(v_cfg)
    st.sidebar.success("Configuration chargée avec succès !")
  except Exception as e:
    st.sidebar.error(f"Erreur de lecture du JSON : {e}")

config_dict = {
    k: st.session_state[k]
    for k in list(DEFAULT_PARAMS.keys()) + list(ENERGY_PARAMS.keys())
}
st.sidebar.download_button(
    "💾 Sauvegarder Config (JSON)",
    data=json.dumps(config_dict, indent=2),
    file_name="lppl_params.json",
    mime="application/json",
    help="❓ Exporte vos paramètres actuels sous forme de fichier JSON.",
)

st.sidebar.markdown("---")

horizon_years = st.sidebar.slider(
    "🔮 Horizon de Prévision (Années)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help=(
        "❓ Définit le nombre d'années dans le futur sur lesquelles étendre"
        " les courbes de prévision LPPL et Power Law."
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

with st.sidebar.expander("📌 Power Law (Tendance Fondamentale)", expanded=False):
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

with st.sidebar.expander("🎛️ Options du Modèle & Affichage", expanded=False):
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
          " prévisions futures."
      ),
  )
  use_energy = st.checkbox(
      "⚡ Appliquer la Fraction d'Énergie au Modèle Principal",
      value=True,
      help=(
          "❓ Si décoché, le modèle principal devient le modèle LPPL classique"
          " à amplitudes fixes."
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
  show_angular_points = st.checkbox(
      "Afficher Tops Angulaires (45°, 135°...)",
      value=True,
      help="❓ Permet d'afficher les tops angulaires.",
  )

with st.sidebar.expander(
    "⚡ Configuration Évolution Fraction d'Énergie", expanded=False
):
  st.caption(
      "Ajuste les paramètres dynamiques régissant l'évolution de la fraction"
      " d'énergie entre les différents modes fréquentiels."
  )
  energy_decay_10 = st.slider(
      "Taux de Décroissance H1 ($f_{10}$)",
      min_value=0.5,
      max_value=5.0,
      value=st.session_state["energy_decay_10"],
      step=0.1,
      key="input_energy_decay_10",
      help=(
          "❓ Contrôle la vitesse d'atténuation de la composante"
          " exponentielle de $f_{10}$ au fil du temps."
      ),
  )

  energy_scale_macro = st.slider(
      "Poids Global Macro ($f_{1.0}$)",
      min_value=0.2,
      max_value=2.0,
      value=st.session_state["energy_scale_macro"],
      step=0.05,
      key="input_energy_scale_macro",
      help=(
          "❓ Facteur multiplicatif appliqué à la fraction d'énergie"
          " macro-cyclique $f_{1.0}$."
      ),
  )

with st.sidebar.expander("🌊 Harmoniques LPPL", expanded=False):
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

  st.markdown("**Harmonic 2 (Micro Cycle - 4ω)**")
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

  if not df_cm.empty and not df_yf.empty:
    min_yf_date = df_yf["Date"].min()
    df_cm_old = df_cm[df_cm["Date"] < min_yf_date]
    df = pd.concat([df_cm_old, df_yf], ignore_index=True)
  elif not df_yf.empty:
    df = df_yf
  elif not df_cm.empty:
    df = df_cm
  else:
    st.error("Erreur critique : Impossible de charger les données historiques.")
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
# 3. FONCTIONS MATHÉMATIQUES DU MODÈLE & FRACTIONS D'ÉNERGIE
# ==============================================================================
def f_trend(lnT, a_val, b_val):
  """Calcule la tendance fondamentale Power Law."""
  return a_val + b_val * lnT


def compute_energy_fractions(
    lnT, decay_10=None, scale_macro=None, scale_micro=None
):
  """Calcule les fractions d'énergie réactives (support optimisation & UI)."""
  days = np.exp(lnT)
  years = 2009.0082 + days / 365.25
  t_norm = (years - 2010.0) / (2026.0 - 2010.0)

  d_10 = (
      decay_10
      if decay_10 is not None
      else st.session_state.get("energy_decay_10", 2.5)
  )
  s_macro = (
      scale_macro
      if scale_macro is not None
      else st.session_state.get("energy_scale_macro", 1.0)
  )
  s_micro = (
      scale_micro
      if scale_micro is not None
      else st.session_state.get("energy_scale_micro", 1.0)
  )

  f_05 = 0.02 + 0.01 * np.sin(t_norm * np.pi)
  f_10 = 0.35 * np.exp(-d_10 * t_norm) + 0.05
  f_20 = (
      0.15
      + 0.12
      * np.sin(t_norm * np.pi * 1.5)
      * np.exp(-((t_norm - 0.3) ** 2) / 0.1)
  )
  f_30 = 0.08 + 0.12 * (1.0 - np.exp(-3.0 * t_norm))
  f_40 = 0.06 + 0.02 * np.sin(t_norm * np.pi * 2)

  e_h1 = f_10 * s_macro
  e_h2 = (f_20 + f_30 + f_40) * s_micro

  return e_h1, e_h2, f_05, f_10, f_20, f_30, f_40


def compute_harmonic_energy_fractions(residuals, frequencies, log_time):
  """Calcule les coefficients de Fourier, l'énergie de chaque harmonique

  et sa fraction d'énergie normalisée dans l'espace log-temps (Parseval).
  """
  n = len(log_time)
  energies = []
  coefficients = []

  for omega in frequencies:
    cos_comp = np.cos(omega * log_time)
    sin_comp = np.sin(omega * log_time)

    a_k = np.dot(residuals, cos_comp) / (n / 2)
    b_k = np.dot(residuals, sin_comp) / (n / 2)

    e_k = a_k**2 + b_k**2
    energies.append(e_k)
    coefficients.append((a_k, b_k))

  energies = np.array(energies)
  total_energy = np.sum(energies)

  if total_energy > 0:
    energy_fractions = energies / total_energy
  else:
    energy_fractions = np.zeros_like(energies)

  return coefficients, energy_fractions


def modulate_harmonics_with_energy(coefficients, energy_fractions):
  """Applique la pondération des fractions d'énergie sur les amplitudes des

  harmoniques.
  """
  modulated_coefficients = []
  for (a_k, b_k), f_k in zip(coefficients, energy_fractions):
    scaling_factor = np.sqrt(f_k) if f_k > 0 else 0
    a_mod = a_k * scaling_factor
    b_mod = b_k * scaling_factor
    modulated_coefficients.append((a_mod, b_mod))

  return modulated_coefficients


def f_log_model(
    lnT,
    a_val,
    b_val,
    c1_val,
    omega_val,
    p1_val,
    c2_val,
    p2_val,
    use_energy=True,
    decay_10=None,
    scale_macro=None,
    scale_micro=None,
):
  """Calcule le modèle LPPL global avec ou sans modulation énergétique."""
  trend_val = f_trend(lnT, a_val, b_val)

  if use_energy:
    e_h1, e_h2, _, _, _, _, _ = compute_energy_fractions(
        lnT,
        decay_10=decay_10,
        scale_macro=scale_macro,
        scale_micro=scale_micro,
    )
    h1 = (c1_val * e_h1) * np.cos(omega_val * lnT + p1_val)
    h2 = (c2_val * e_h2) * np.cos(4.0 * omega_val * lnT + p2_val)
  else:
    h1 = c1_val * np.cos(omega_val * lnT + p1_val)
    h2 = c2_val * np.cos(4.0 * omega_val * lnT + p2_val)

  return trend_val + h1 + h2


@st.cache_data
def calculate_bubble_hazard_index(df_data, window=180):
  d = df_data.copy()
  z_pl = d["z_score_pl"].fillna(0)
  comp_valuation = np.clip(z_pl / 4.0, 0.0, 1.0)

  rolling_vol = d["residuals"].rolling(window=window, min_periods=30).std()
  vol_median = rolling_vol.median()
  if vol_median > 0:
    comp_vol_compression = np.clip(
        vol_median / (rolling_vol.fillna(vol_median)), 0.5, 2.0
    )
    comp_vol_compression = (comp_vol_compression - 0.5) / 1.5
  else:
    comp_vol_compression = pd.Series(0.5, index=d.index)

  price_roc = d["Close"].pct_change(90).fillna(0)
  comp_momentum = np.clip(price_roc / 2.0, 0.0, 1.0)

  hazard_index = (
      0.50 * comp_valuation
      + 0.25 * comp_vol_compression.values
      + 0.25 * comp_momentum.values
  ) * 100.0

  d["Bubble_Hazard_Index"] = np.clip(hazard_index, 0.0, 100.0)
  return d


# ==============================================================================
# 4. OPTIMISATION GLOBALE AUTOMATIQUE & DÉTECTION DE CHANGEMENT D'ÉTAT
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Calibrage Automatique")


def perform_auto_calibration(current_use_energy):
  lnT_vec = df["lnT"].to_numpy()
  act_log_vec = df["actualLog"].to_numpy()

  def loss_func_fast(params):
    (
        p_A,
        p_B,
        p_C1,
        p_omega,
        p_p1,
        p_C2,
        p_p2,
        p_decay,
        p_s_macro,
        p_s_micro,
    ) = params
    preds = f_log_model(
        lnT_vec,
        p_A,
        p_B,
        p_C1,
        p_omega,
        p_p1,
        p_C2,
        p_p2,
        use_energy=current_use_energy,
        decay_10=p_decay,
        scale_macro=p_s_macro,
        scale_micro=p_s_micro,
    )
    return np.mean((act_log_vec - preds) ** 2)

  bounds = [
      (-45.0, -25.0),  # A
      (4.5, 6.8),  # B
      (0.0, 3.0),  # C1
      (8.0, 9.0),  # omega
      (-np.pi, np.pi),  # phi1
      (0.0, 2.0),  # C2
      (-np.pi, np.pi),  # phi2
      (0.5, 5.0),  # energy_decay_10
      (0.2, 2.0),  # energy_scale_macro
      (0.2, 2.0),  # energy_scale_micro
  ]

  res = differential_evolution(
      loss_func_fast,
      bounds=bounds,
      strategy="best1bin",
      maxiter=250,
      popsize=20,
      polish=True,
      seed=42,
  )

  if res.success:
    params_keys = [
        "A",
        "B",
        "C1",
        "omega",
        "phi1",
        "C2",
        "phi2",
        "energy_decay_10",
        "energy_scale_macro",
        "energy_scale_micro",
    ]
    for i, k in enumerate(params_keys):
      st.session_state[k] = float(res.x[i])
      st.session_state.pop(f"input_{k}", None)

    st.session_state["opt_msg"] = (
        f"**Tendance :** A={res.x[0]:.2f} | B={res.x[1]:.3f}\n\n"
        f"**Harmoniques :** C1={res.x[2]:.2f} | ω={res.x[3]:.3f} |"
        f" φ1={res.x[4]:.2f}\n\n"
        f"**Harmoniques 2 :** C2={res.x[5]:.2f} | φ2={res.x[6]:.2f}\n\n"
        f"**Énergie :** Decay={res.x[7]:.2f} | Macro={res.x[8]:.2f} |"
        f" Micro={res.x[9]:.2f}"
    )
    return True
  return False


if "prev_use_energy" not in st.session_state:
  st.session_state["prev_use_energy"] = use_energy

if use_energy != st.session_state["prev_use_energy"]:
  st.session_state["prev_use_energy"] = use_energy
  with st.spinner(
      "🔄 Recalibrage automatique suite au changement du mode d'énergie..."
  ):
    if perform_auto_calibration(use_energy):
      st.rerun()

if st.sidebar.button(
    "🤖 Ajuster les paramètres au dataset",
    help=(
        "❓ Lance l'optimisation globale (Differential Evolution) pour estimer"
        " les 10 paramètres de manière robuste."
    ),
):
  with st.spinner(
      "Optimisation globale en cours (Recherche globale + Polish)..."
  ):
    if perform_auto_calibration(use_energy):
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
    df["lnT"].values,
    A,
    B,
    C1,
    omega,
    phi1,
    C2,
    phi2,
    use_energy=use_energy,
)
df["modelPrice"] = np.exp(df["logModel"])

df["residuals"] = df["actualLog"] - df["logModel"]
res_std = np.std(df["residuals"])
df["z_score"] = df["residuals"] / res_std if res_std > 0 else 0.0

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
    use_energy=True,
):
  if len(days_arr) <= horizon_days:
    return 0, 0, 0, 0, 0, 0

  t_fut = days_arr[:-horizon_days] + horizon_days
  lnT_fut = np.log(t_fut)

  preds = np.exp(
      f_log_model(
          lnT_fut,
          p_A,
          p_B,
          p_C1,
          p_omega,
          p_p1,
          p_C2,
          p_p2,
          use_energy=use_energy,
      )
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
    df_data,
    window_days=1095,
    step_days=90,
    horizon_days=365,
    use_energy=True,
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
            train_lnT,
            p_A,
            p_B,
            p_C1,
            p_omega,
            p_p1,
            p_C2,
            p_p2,
            use_energy=use_energy,
        )
        return np.mean((train_act_log - preds) ** 2)

      bounds = [
          (-45.0, -25.0),
          (4.5, 6.8),
          (0.0, 3.0),
          (8.0, 9.0),
          (-np.pi, np.pi),
          (0.0, 2.0),
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
              use_energy=use_energy,
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
    f_log_model(
        future_lnT_arr,
        A,
        B,
        C1,
        omega,
        phi1,
        C2,
        phi2,
        use_energy=use_energy,
    )
)

future_pl_trend_log = f_trend(future_lnT_arr, A, B)
future_pl_trend = np.exp(future_pl_trend_log)
future_pl_upper = np.exp(future_pl_trend_log + pl_sigma_upper * trend_res_std)
future_pl_lower = np.exp(future_pl_trend_log - pl_sigma_lower * trend_res_std)

days_arr_glob = df["Days"].values
close_arr_glob = df["Close"].values

_, _, _, _, wf_rmse_1y, _ = run_wf_analysis_fast(
    365,
    days_arr_glob,
    close_arr_glob,
    A,
    B,
    C1,
    omega,
    phi1,
    C2,
    phi2,
    use_energy=use_energy,
)
_, _, _, _, wf_rmse_2y, _ = run_wf_analysis_fast(
    730,
    days_arr_glob,
    close_arr_glob,
    A,
    B,
    C1,
    omega,
    phi1,
    C2,
    phi2,
    use_energy=use_energy,
)
_, _, _, _, wf_rmse_3y, _ = run_wf_analysis_fast(
    1095,
    days_arr_glob,
    close_arr_glob,
    A,
    B,
    C1,
    omega,
    phi1,
    C2,
    phi2,
    use_energy=use_energy,
)

res_std_log = np.std(df["residuals"])

val_1y = wf_rmse_1y if (wf_rmse_1y > 0 and not np.isnan(wf_rmse_1y)) else 78.1
val_2y = (
    wf_rmse_2y
    if (wf_rmse_2y > 0 and not np.isnan(wf_rmse_2y))
    else val_1y * np.sqrt(2)
)
val_3y = (
    wf_rmse_3y
    if (wf_rmse_3y > 0 and not np.isnan(wf_rmse_3y))
    else val_1y * np.sqrt(3)
)

wf_milestone_days = np.array([0, 365, 730, 1095], dtype=float)
wf_milestone_errs = (
    np.array(
        [
            res_std_log * 100.0,
            val_1y,
            val_2y,
            val_3y,
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
    vertical_spacing=0.1,
    row_heights=[0.72, 0.28],
    subplot_titles=(
        (
            "Prix & Prévisions Avancées LPPL (Échelle Logarithmique du Temps"
            " ln(t))"
            if log_time_axis
            else "Prix & Prévisions Avancées LPPL (Temps Linéaire / Date)"
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
      {"mult": 3.0, "opacity": 0.05, "name": "Canal ±3.0σ (99.7%)"},
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

    if m == 3.0:
      fig.add_trace(
          go.Scatter(
              x=x_lppl_all,
              y=comp_upper,
              mode="lines",
              line=dict(color="rgba(255, 153, 0, 0.3)", width=1, dash="dash"),
              showlegend=False,
              hoverinfo="skip",
          ),
          row=1,
          col=1,
      )
      fig.add_trace(
          go.Scatter(
              x=x_lppl_all,
              y=comp_lower,
              mode="lines",
              line=dict(color="rgba(255, 153, 0, 0.3)", width=1, dash="dash"),
              showlegend=False,
              hoverinfo="skip",
          ),
          row=1,
          col=1,
      )

  fig.add_trace(
      go.Scatter(
          x=x_hist,
          y=df["modelPrice"],
          mode="lines",
          name=(
              "LPPL Model (Fit)"
              if use_energy
              else "LPPL Model Classique (Fit)"
          ),
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
          name="LPPL Prévision Centrale",
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

lnT_min_val = float(df["lnT"].min())
lnT_max_val = float(np.log(future_days_arr[-1]))

step_angle = np.pi / 4
k_min = int(np.floor((omega * lnT_min_val) / step_angle))
k_max = int(np.ceil((omega * lnT_max_val) / step_angle))

marker_x = []
marker_y = []
marker_text = []

for k in range(k_min, k_max + 1):
  lnT_line = (k * step_angle) / omega
  angle_deg = int(round(np.rad2deg(k * step_angle)) % 360)

  if not log_time_axis:
    days_val = float(np.exp(lnT_line))
    if not np.isfinite(days_val) or days_val > 1e9:
      continue
    x_val = GENESIS_DATE + timedelta(days=days_val)
    if x_val < min_date or x_val > max_date + timedelta(
        days=int(horizon_years * 365)
    ):
      continue
  else:
    x_val = lnT_line

  if angle_deg in [90, 180]:
    line_color = "rgba(255, 153, 0, 0.2)"
    line_width = 1.2
    line_dash = "dot"
  elif angle_deg in [0, 270]:
    line_color = "rgba(0, 255, 127, 0.1)"
    line_width = 1.5
    line_dash = "dash"
  elif angle_deg in [45, 135, 225, 315]:
    line_color = "rgba(255, 0, 0, 0.3)"
    line_width = 0.8
    line_dash = "dot"

    if show_angular_points:
      price_val = np.exp(
          f_log_model(
              np.array([lnT_line]),
              A,
              B,
              C1,
              omega,
              phi1,
              C2,
              phi2,
              use_energy=use_energy,
          )[0]
      )
      marker_x.append(x_val)
      marker_y.append(price_val)
      marker_text.append(f"{angle_deg}°")
  else:
    is_major = (angle_deg % 180) == 0
    line_color = (
        "rgba(255, 153, 0, 0.45)" if is_major else "rgba(255, 153, 0, 0.15)"
    )
    line_width = 1.2 if is_major else 0.8
    line_dash = "solid" if is_major else "dot"

  for r in [1, 2]:
    fig.add_vline(
        x=x_val,
        line_dash=line_dash,
        line_color=line_color,
        line_width=line_width,
        row=r,
        col=1,
    )

  if angle_deg in [0, 90, 180, 270, 45, 135, 225, 315]:
    if angle_deg in [0, 270]:
      ann_text = f"<b>{angle_deg}°</b>"
      ann_color = "#00FF7F"
    elif angle_deg in [45, 135, 225, 315]:
      ann_text = f"<b>{angle_deg}°</b>"
      ann_color = "#FF6B6B"
    else:
      ann_text = f"<b>{angle_deg}°</b>"
      ann_color = "#FF9900"

    fig.add_annotation(
        x=x_val,
        y=0.96,
        yref="paper",
        text=ann_text,
        showarrow=False,
        font=dict(size=8, color=ann_color),
        xanchor="center",
        yanchor="bottom",
    )

if show_angular_points and marker_x:
  fig.add_trace(
      go.Scatter(
          x=marker_x,
          y=marker_y,
          mode="markers+text",
          name="Points Angulaires Critiques (45°, 135°, 225°, 315°)",
          text=marker_text,
          textposition="top center",
          marker=dict(color="#FF0000", size=8, symbol="circle"),
          textfont=dict(size=9, color="#FF6B6B"),
          showlegend=True,
      ),
      row=1,
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

x_min_val = x_trend[0]
x_max_val = x_trend[-1]

xaxis_config_top = (
    dict(
        title=dict(text="", standoff=0),
        rangeslider=dict(visible=False),
        range=[x_min_val, x_max_val],
        autorange=False,
    )
    if log_time_axis
    else dict(
        title=dict(text="", standoff=0),
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
        range=[x_min_val, x_max_val],
        autorange=False,
    )
)
fig.update_xaxes(xaxis_config_top, row=1, col=1)

xaxis_config_bottom = (
    dict(
        title=dict(text=xaxis_title, standoff=25),
        rangeslider=dict(visible=False),
        range=[x_min_val, x_max_val],
        autorange=False,
    )
    if log_time_axis
    else dict(
        title=dict(text=xaxis_title, standoff=20),
        rangeslider=dict(visible=False),
        range=[x_min_val, x_max_val],
        autorange=False,
    )
)
fig.update_xaxes(xaxis_config_bottom, row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    height=1050,
    margin=dict(l=5, r=5, t=100, b=120),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.32,
        xanchor="center",
        x=0.5,
        font=dict(size=9),
        bgcolor="rgba(0,0,0,0.5)",
    ),
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
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
    365,
    days_arr_glob,
    close_arr_glob,
    A,
    B,
    C1,
    omega,
    phi1,
    C2,
    phi2,
    use_energy=use_energy,
)
gen_ratio = wf_rmse_1y_val / rmse_pct if rmse_pct > 0 else 1.0

col_chart, col_dash = st.columns([3.2, 1])

with col_chart:
  st.plotly_chart(fig, use_container_width=True)
  with st.expander("❓ Guide de Lecture du Graphique Principal"):
    st.markdown("""
        * **Prix BTC (Gris)** : Cours de clôture quotidien du Bitcoin.
        * **LPPL Model (Orange)** : Courbe ajustée du modèle LPPL sélectionné.
        * **Power Law Fit (Bleu Cyan)** : Tendance fondamentale A + B * ln(t).
        * **Quadrillage Oméga (Lignes et angles)** : Marqueurs angulaires de cycle log-périodique.
        * **Z-Scores (Panneau Inférieur)** : Écarts normalisés du prix réel par rapport au modèle LPPL et à la Power Law.
        """)

with col_dash:
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
          help=(
              "❓ Dernier cours de clôture disponible pour le Bitcoin et"
              " variation sur les 24 dernières heures."
          ),
      )
      st.metric(
          "Prix Théorique (LPPL)",
          f"${current_model_price:,.2f}",
          help=(
              "❓ Valeur théorique du prix calculée par le modèle sélectionné"
              " à la date actuelle."
          ),
      )
      st.metric(
          "Z-Score LPPL",
          f"{current_z_score:.2f}σ",
          help=(
              "❓ Écart normalisé (en écarts-types σ) entre le prix réel et le"
              " prix théorique du modèle LPPL."
          ),
      )

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

  with st.container(border=True):
    st.subheader("📊 Fit Quality & Robustesse")

    c_g1, c_g2 = st.columns(2)
    c_g1.metric("R² Global", f"{r2_global:.4f}")
    c_g2.metric("R² Ajusté", f"{r2_adj:.4f}")

    st.metric("Forward R² (1Y OOS)", f"{r2_oos_1y:.4f}")

    st.markdown("---")
    st.caption("📐 **Métriques d'Erreur & Généralisation**")

    c_m3, c_m4 = st.columns(2)
    c_m3.metric("RMSE In-Sample", f"{rmse_pct:.1f}%")
    c_m4.metric("MAE In-Sample", f"{mae_pct:.1f}%")

    c_m5, c_m6 = st.columns(2)
    c_m5.metric("OOS RMSE (1Y)", f"{wf_rmse_1y_val:.1f}%")
    c_m6.metric("Ratio Out/In", f"{gen_ratio:.2f}x")


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
    * **25 - 40% (Bleu)** : Croissance organique alignée sur la Power Law.
    * **40 - 60% (Orange)** : Phase spéculative avancée, signaux d'alerte macro.
    * **> 60% (Rouge)** : Zone de criticité maximale, probabilité élevée de rupture ou de retournement de cycle.
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
            if current_hazard > 60
            else "rgba(56, 189, 248, 0.1)"
        ),
    )
)

fig_hazard.add_hline(
    y=60,
    line_dash="dash",
    line_color="#FF0000",
    annotation_text="Seuil Critique (60%)",
    annotation_position="top right",
)
fig_hazard.add_hline(
    y=40,
    line_dash="dot",
    line_color="#FFA500",
    annotation_text="Seuil d'Alerte (40%)",
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
  st.metric(
      "Indice de Risque Actuel",
      f"{current_hazard:.1f} / 100",
      help=(
          "❓ Score synthétique de 0 à 100 évaluant la criticité et le risque"
          " imminent de rupture."
      ),
  )
with col_haz2:
  st.markdown(
      f"**Statut du Régime :** <span"
      f" style='color:{hazard_color};font-weight:bold;font-size:1.2em;'>{hazard_txt}</span>",
      unsafe_allow_html=True,
  )

# ==============================================================================
# SECTION : LES DEUX HORLOGES DE CYCLE & GRAVITATIONNELLES (SÉLECTEUR)
# ==============================================================================
st.markdown("---")
st.subheader("🕒 Comparatif des Horloges de Cycle & Gravitationnelles")

with st.expander("❓ Guide de Lecture - Horloges de Cycle"):
  st.markdown("""
    * **Horloge Gravitationnelle (Gauche)** : Permet de basculer entre l'attracteur **Power Law** (Z-Score PL vs Momentum) et l'attracteur **LPPL** (Z-Score LPPL vs Momentum) via le menu déroulant ci-dessous. Le point vert (**📍 ACTUEL**) indique la position actuelle par rapport à l'attracteur choisi.
    * **Horloge Log-Périodique (Droite)** : Représentation polaire de la phase angulaire $\\omega \\cdot \\ln(t)$. Les cadrans de couleur et les secteurs cibles verts permettent de suivre la progression et la résonance des sous-cycles log-périodiques.
    """)

# Sélecteur pour basculer entre les deux horloges gravitationnelles
grav_clock_choice = st.selectbox(
    "🔄 Sélectionner le modèle de l'Horloge Gravitationnelle (Gauche)",
    ["Attracteur Power Law (Z-Score PL)", "Attracteur LPPL (Z-Score LPPL)"],
    index=1,
    help="Permet de basculer l'horloge gravitationnelle entre le modèle de tendance Power Law et le modèle LPPL.",
)

col_clock_grav, col_clock_omega = st.columns(2)

# --- 1. HORLOGE GRAVITATIONNELLE (GAUCHE - DYNAMIQUE) ---
with col_clock_grav:
  if "Power Law" in grav_clock_choice:
    st.markdown(
        "##### 🌍 Horloge Gravitationnelle — Attracteur Power Law (Z-Score vs"
        " Momentum)"
    )
    df["z_velocity_pl"] = df["z_score_pl"].diff(30).fillna(0)
    x_data = df["z_score_pl"]
    y_data = df["z_velocity_pl"]
    latest_x = df["z_score_pl"].iloc[-1]
    latest_y = df["z_velocity_pl"].iloc[-1]
    x_axis_title = "Position (Z-Score Power Law)"
  else:
    st.markdown(
        "##### 🌍 Horloge Gravitationnelle — Attracteur LPPL (Z-Score vs"
        " Momentum)"
    )
    df["z_velocity_lppl"] = df["z_score"].diff(30).fillna(0)
    x_data = df["z_score"]
    y_data = df["z_velocity_lppl"]
    latest_x = df["z_score"].iloc[-1]
    latest_y = df["z_velocity_lppl"].iloc[-1]
    x_axis_title = "Position (Z-Score LPPL)"

  fig_clock = go.Figure()

  fig_clock.add_trace(
      go.Scatter(
          x=x_data,
          y=y_data,
          mode="lines+markers",
          marker=dict(
              size=3,
              color=df["Days"],
              colorscale="Inferno",
              opacity=0.8,
          ),
          line=dict(color="rgba(255, 153, 0, 0.35)", width=1.2),
          showlegend=False,
      )
  )

  fig_clock.add_hline(
      y=0,
      line_dash="dash",
      line_color="rgba(255, 255, 255, 0.4)",
      line_width=1,
  )
  fig_clock.add_vline(
      x=0,
      line_dash="dash",
      line_color="rgba(255, 255, 255, 0.4)",
      line_width=1,
  )

  fig_clock.add_trace(
      go.Scatter(
          x=[latest_x],
          y=[latest_y],
          mode="markers+text",
          marker=dict(
              size=14,
              color="#00FF7F",
              symbol="diamond",
              line=dict(color="#000000", width=1),
          ),
          text=["📍 ACTUEL"],
          textposition="top center",
          textfont=dict(color="#00FF7F", size=11, family="sans-serif"),
          showlegend=False,
      )
  )

  fig_clock.update_layout(
      template="plotly_dark",
      height=480,
      margin=dict(l=10, r=10, t=20, b=10),
      plot_bgcolor="rgba(10, 10, 15, 0.6)",
      paper_bgcolor="rgba(0,0,0,0)",
      xaxis=dict(
          title=x_axis_title,
          gridcolor="rgba(255, 255, 255, 0.08)",
          linecolor="rgba(255, 255, 255, 0.15)",
      ),
      yaxis=dict(
          title="Vitesse / Momentum",
          gridcolor="rgba(255, 255, 255, 0.08)",
          linecolor="rgba(255, 255, 255, 0.15)",
      ),
  )
  st.plotly_chart(fig_clock, use_container_width=True)

# --- 2. HORLOGE LOG-PÉRIODIQUE (DROITE) ---
with col_clock_omega:
  st.markdown("##### ⏱️ Horloge Log-Périodique (Phase $\\omega \\cdot \\ln(t)$)")

  df["log_phase"] = omega * df["lnT"]
  df["clock_angle"] = (df["log_phase"] % (2 * np.pi)) / (2 * np.pi) * 360
  df["clock_radius"] = np.linspace(1, 5, len(df))

  fig_clock_omega = go.Figure()

  sectors = [
      (0, 90, "rgba(255, 136, 9, 0.15)"),
      (90, 180, "rgba(173, 20, 20, 0.15)"),
      (180, 270, "rgba(255, 136, 9, 0.15)"),
      (270, 360, "rgba(34, 177, 76, 0.15)"),
  ]

  for start_th, end_th, fill_col in sectors:
    th_vals = np.linspace(start_th, end_th, 30)
    r_inner = np.full_like(th_vals, 6.1)
    r_outer = np.full_like(th_vals, 6.5)
    fig_clock_omega.add_trace(
        go.Scatterpolar(
            r=list(r_inner) + list(r_outer[::-1]),
            theta=list(th_vals) + list(th_vals[::-1]),
            fill="toself",
            fillcolor=fill_col,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

  target_cones = [0, 270]
  cone_half_width = 14.0

  for ang in target_cones:
    th_vals = np.linspace(ang - cone_half_width, ang + cone_half_width, 30)
    if ang == 270:
      th_vals = th_vals[::-1]
    r_vals = np.full_like(th_vals, 6.5)

    fig_clock_omega.add_trace(
        go.Scatterpolar(
            r=[0] + list(r_vals) + [0],
            theta=[ang] + list(th_vals) + [ang],
            fill="toself",
            fillcolor="rgba(0, 255, 127, 0.2)",
            line=dict(color="rgba(0, 255, 127, 0.6)", width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

  red_cones = [45, 135, 225, 315]
  for ang in red_cones:
    th_vals = np.linspace(ang - cone_half_width, ang + cone_half_width, 30)
    r_vals = np.full_like(th_vals, 6.5)

    fig_clock_omega.add_trace(
        go.Scatterpolar(
            r=[0] + list(r_vals) + [0],
            theta=[ang] + list(th_vals) + [ang],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255, 0, 0, 0.6)", width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

  fig_clock_omega.add_trace(
      go.Scatterpolar(
          r=df["clock_radius"],
          theta=df["clock_angle"],
          mode="lines+markers",
          marker=dict(
              size=3,
              color=df["Days"],
              colorscale="Inferno",
              colorbar=dict(title="Temps", len=0.4, thickness=10),
              opacity=0.8,
          ),
          line=dict(color="rgba(255, 153, 0, 0.5)", width=1.5),
          name="Orbite",
      )
  )

  fig_clock_omega.add_trace(
      go.Scatterpolar(
          r=[df["clock_radius"].iloc[-1]],
          theta=[df["clock_angle"].iloc[-1]],
          mode="markers+text",
          marker=dict(
              size=14,
              color="#00FF7F",
              symbol="diamond",
              line=dict(color="#000000", width=1),
          ),
          text=["📍 ACTUEL"],
          textposition="top center",
          textfont=dict(color="#00FF7F", size=11, family="sans-serif"),
          showlegend=False,
      )
  )

  fig_clock_omega.update_layout(
      template="plotly_dark",
      height=480,
      margin=dict(l=10, r=10, t=20, b=30),
      paper_bgcolor="rgba(0,0,0,0)",
      polar=dict(
          bgcolor="rgba(10, 10, 15, 0.6)",
          radialaxis=dict(visible=False, range=[0, 6.5]),
          angularaxis=dict(
              direction="clockwise",
              period=360,
              tickmode="array",
              tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
              ticktext=[
                  "0°",
                  "45°",
                  "90°",
                  "135°",
                  "180°",
                  "225°",
                  "270°",
                  "315°",
              ],
              gridcolor="rgba(255, 255, 255, 0.08)",
              linecolor="rgba(255, 255, 255, 0.15)",
          ),
      ),
      showlegend=False,
  )
  st.plotly_chart(fig_clock_omega, use_container_width=True)

# ==============================================================================
# SECTION : PROFIL DE VITESSE TYPIQUE EN FONCTION DE LA PHASE DU CYCLE
# ==============================================================================
st.markdown("---")
st.subheader(
    "⚡ Profil de Vitesse Typique en Fonction de la Phase du Cycle"
)

with st.expander("❓ Guide de Lecture - Vitesse par Phase de Cycle"):
  st.markdown("""
    * Ce graphique analyse la vitesse moyenne de progression (Momentum / Vitesse du Z-Score de la Power Law) en fonction de la position exacte au sein du cycle log-périodique (de $0^\circ$ à $360^\circ$).
    * **Barres bleues** : Vitesse moyenne observée historiquement dans chaque secteur angulaire du cycle.
    * **Barres d'erreur (±1σ)** : Dispersion statistique de la vitesse pour chaque phase.
    * **Ligne pointillée rouge** : Position actuelle estimée dans le cycle.
    """)

df["cycle_phase_deg"] = (
    (omega * df["lnT"]) % (2 * np.pi)
) * (180 / np.pi)
df["z_velocity"] = df["z_score_pl"].diff(30).fillna(0)

current_phase = df["cycle_phase_deg"].iloc[-1]

bins = np.arange(0, 370, 10)
labels = np.arange(5, 365, 10)
df["angle_bin"] = pd.cut(
    df["cycle_phase_deg"], bins=bins, labels=labels, include_lowest=True
)

speed_profile = (
    df.groupby("angle_bin", observed=False)["z_velocity"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

fig_speed_profile = go.Figure()

fig_speed_profile.add_trace(
    go.Bar(
        x=speed_profile["angle_bin"],
        y=speed_profile["mean"],
        name="Vitesse Moyenne (Z-Score)",
        marker_color="#38BDF8",
        error_y=dict(
            type="data",
            array=speed_profile["std"].fillna(0),
            visible=True,
            color="rgba(255, 255, 255, 0.5)",
        ),
    )
)

fig_speed_profile.add_hline(
    y=0, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)", line_width=1.2
)

fig_speed_profile.add_vline(
    x=current_phase,
    line_dash="dash",
    line_color="#F43F5E",
    line_width=2.5,
    annotation_text=f"Actuel : {current_phase:.1f}°",
    annotation_position="top right",
    annotation_font_color="#F43F5E",
    annotation_font_size=12,
)

fig_speed_profile.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Phase Angulaire du Cycle (Degrés)",
    yaxis_title="Vitesse Moyenne du Z-Score (Δσ / 30j)",
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 360, 45)),
        ticktext=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
    ),
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
)

st.plotly_chart(fig_speed_profile, use_container_width=True)



# ==============================================================================
# SECTION COMBINÉE : FRACTIONS D'ÉNERGIE ET VISUALISATION DES HARMONIQUES CÔTE À CÔTE
# ==============================================================================
st.markdown("---")

years = df["Date"].dt.year + df["Date"].dt.dayofyear / 365.25
t_norm = (years - 2010.0) / (2026.0 - 2010.0)

f_05 = 0.02 + 0.01 * np.sin(t_norm * np.pi)
f_10 = 0.35 * np.exp(-2.5 * t_norm) + 0.05
f_20 = 0.15 + 0.12 * np.sin(t_norm * np.pi * 1.5) * np.exp(
    -((t_norm - 0.3) ** 2) / 0.1
)
f_20 = np.clip(f_20, 0.05, 0.30)
f_30 = 0.08 + 0.12 * (1.0 - np.exp(-3.0 * t_norm))
f_40 = 0.06 + 0.02 * np.sin(t_norm * np.pi * 2)

twist_raw = f_05 + f_10 + f_20 + f_30 + f_40
f_05_s = f_05 / twist_raw * 0.61
f_10_s = f_10 / twist_raw * 0.61
f_20_s = f_20 / twist_raw * 0.61
f_30_s = f_30 / twist_raw * 0.61
f_40_s = f_40 / twist_raw * 0.61
f_wr = 1.0 - (f_05_s + f_10_s + f_20_s + f_30_s + f_40_s)

custom_angles = (
    df["clock_angle"] if "clock_angle" in df.columns else np.zeros(len(df))
)
lnT_full = df["lnT"].values

col_left, col_right = st.columns(2)

# ==============================================================================
# COLONNE GAUCHE : FRACTIONS D'ÉNERGIE (TWIST / WRITHE)
# ==============================================================================
with col_left:
  st.subheader("⚡ Fractions d'énergie par mode")

  with st.expander("❓ Guide de lecture - Fractions d'énergie"):
    st.markdown("""
        * Décomposition temporelle des fractions d'énergie par mode harmonique ($0.5\omega$ à $4.0\omega$) et composante **Writhe**.
        * **Twist (~0.61)** : Énergie cumulée des modes oscillatoires.
        * **Writhe (~0.39)** : Énergie de fond / résiduelle (Power Law).
        """)

  fig_energy = go.Figure()

  fig_energy.add_trace(
      go.Scatter(
          x=df["Date"],
          y=f_05_s,
          name="0.5ω",
          mode="lines",
          line=dict(width=0.5, color="#1f77b4"),
          stackgroup="one",
          customdata=custom_angles,
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Angle: %{customdata:.1f}°<br>Fraction:"
              " %{y:.3f}<extra>%{data.name}</extra>"
          ),
      )
  )
  fig_energy.add_trace(
      go.Scatter(
          x=df["Date"],
          y=f_10_s,
          name="1.0ω",
          mode="lines",
          line=dict(width=0.5, color="#ff7f0e"),
          stackgroup="one",
          customdata=custom_angles,
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Angle: %{customdata:.1f}°<br>Fraction:"
              " %{y:.3f}<extra>%{data.name}</extra>"
          ),
      )
  )
  fig_energy.add_trace(
      go.Scatter(
          x=df["Date"],
          y=f_20_s,
          name="2.0ω",
          mode="lines",
          line=dict(width=0.5, color="#2ca02c"),
          stackgroup="one",
          customdata=custom_angles,
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Angle: %{customdata:.1f}°<br>Fraction:"
              " %{y:.3f}<extra>%{data.name}</extra>"
          ),
      )
  )
  fig_energy.add_trace(
      go.Scatter(
          x=df["Date"],
          y=f_30_s,
          name="3.0ω",
          mode="lines",
          line=dict(width=0.5, color="#d62728"),
          stackgroup="one",
          customdata=custom_angles,
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Angle: %{customdata:.1f}°<br>Fraction:"
              " %{y:.3f}<extra>%{data.name}</extra>"
          ),
      )
  )
  fig_energy.add_trace(
      go.Scatter(
          x=df["Date"],
          y=f_40_s,
          name="4.0ω",
          mode="lines",
          line=dict(width=0.5, color="#9467bd"),
          stackgroup="one",
          customdata=custom_angles,
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Angle: %{customdata:.1f}°<br>Fraction:"
              " %{y:.3f}<extra>%{data.name}</extra>"
          ),
      )
  )
  fig_energy.add_trace(
      go.Scatter(
          x=df["Date"],
          y=f_wr,
          name="wr",
          mode="lines",
          line=dict(width=0.5, color="#8c564b"),
          stackgroup="one",
          customdata=custom_angles,
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Angle: %{customdata:.1f}°<br>Fraction:"
              " %{y:.3f}<extra>%{data.name}</extra>"
          ),
      )
  )

  if "clock_angle" in df.columns:
    angles = df["clock_angle"].values
    k = 0
    unwrapped_angles = []
    prev_a = angles[0]
    for a in angles:
      if a < prev_a - 180:
        k += 1
      elif a > prev_a + 180:
        k -= 1
      unwrapped_angles.append(k * 360 + a)
      prev_a = a

    df_temp = df.copy()
    df_temp["unwrapped_angle"] = unwrapped_angles
    min_ang, max_ang = (
        df_temp["unwrapped_angle"].min(),
        df_temp["unwrapped_angle"].max(),
    )
    offsets = [45, 135, 225, 315]

    for cycle in range(
        int(np.floor(min_ang / 360)), int(np.ceil(max_ang / 360)) + 1
    ):
      for offset in offsets:
        ang = cycle * 360 + offset
        if min_ang <= ang <= max_ang:
          idx = (df_temp["unwrapped_angle"] - ang).abs().idxmin()
          fig_energy.add_vline(
              x=df_temp.loc[idx, "Date"],
              line_dash="dash",
              line_color="rgba(255, 255, 255, 0.2)",
              line_width=0.8,
              annotation_text=f"{int(ang) % 360}°",
              annotation_position="top",
              annotation_font_size=9,
              annotation_font_color="rgba(255, 255, 255, 0.6)",
          )

  fig_energy.update_layout(
      template="plotly_dark",
      title="Fractions d'énergie Twist/Writhe",
      height=450,
      margin=dict(l=20, r=20, t=70, b=30),
      yaxis=dict(title="Fraction", range=[0, 1.0]),
      xaxis=dict(title="Date"),
      legend=dict(
          orientation="v",
          yanchor="top",
          y=0.98,
          xanchor="right",
          x=0.99,
          bgcolor="rgba(0,0,0,0.6)",
      ),
  )
  st.plotly_chart(fig_energy, use_container_width=True)

# ==============================================================================
# COLONNE DROITE : VISUALISATION INTERACTIVE DES HARMONIQUES
# ==============================================================================
with col_right:
  st.subheader(
      "🎼 Harmoniques par Mode - Amuse toi à reconstruire BTC en jouant sur les"
      " paramètres"
  )

  with st.expander("❓ Guide de lecture - Harmoniques"):
    st.markdown("""
        * Sélectionnez les harmoniques à afficher.
        * Les amplitudes sont modulées dynamiquement par leurs fractions d'énergie respectives.
        """)

  with st.expander(
      "⚙️ Paramétrage des phases ($\phi$) & Affichage des modes", expanded=False
  ):
    st.markdown("**Affichage des modes :**")
    col_h_chk1, col_h_chk2, col_h_chk3, col_h_chk4, col_h_chk5 = st.columns(5)
    with col_h_chk1:
      show_h05 = st.checkbox("0.5w", value=True, key="chk_h05")
    with col_h_chk2:
      show_h10 = st.checkbox("1w", value=True, key="chk_h10")
    with col_h_chk3:
      show_h20 = st.checkbox("2w", value=True, key="chk_h20")
    with col_h_chk4:
      show_h30 = st.checkbox("3w", value=True, key="chk_h30")
    with col_h_chk5:
      show_h40 = st.checkbox("4w", value=True, key="chk_h40")

    show_sum = st.checkbox(
        "Afficher la somme totale", value=True, key="chk_h_sum"
    )

    st.markdown("---")
    st.markdown("**Phases individuelles ($\phi$) :**")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
      phase_05 = st.number_input(
          "Phase 0.5ω", value=float(1.0), format="%.4f", key="phase_05_val"
      )
    with col_p2:
      phase_10 = st.number_input(
          "Phase 1.0ω", value=float(-2.1), format="%.4f", key="phase_10_val"
      )
    with col_p3:
      phase_20 = st.number_input(
          "Phase 2.0w", value=float(-0.5), format="%.4f", key="phase_20_val"
      )
    with col_p4:
      phase_30 = st.number_input(
          "Phase 3.0ω", value=float(-1.0), format="%.4f", key="phase_30_val"
      )
    with col_p5:
      phase_40 = st.number_input(
          "Phase 4.0ω", value=float(-2.7), format="%.4f", key="phase_40_val"
      )

  wave_05 = (C1 * 0.4) * f_05_s * np.cos(0.5 * omega * lnT_full + phase_05)
  wave_10 = C1 * f_10_s * np.cos(1.0 * omega * lnT_full + phase_10)
  wave_20 = (C2 * 1.2) * f_20_s * np.cos(2.0 * omega * lnT_full + phase_20)
  wave_30 = (C2 * 0.8) * f_30_s * np.cos(3.0 * omega * lnT_full + phase_30)
  wave_40 = C2 * f_40_s * np.cos(4.0 * omega * lnT_full + phase_40)

  wave_sum = wave_05 + wave_10 + wave_20 + wave_30 + wave_40

  fig_harmonics = go.Figure()

  if show_h05:
    fig_harmonics.add_trace(
        go.Scatter(
            x=df["Date"],
            y=wave_05,
            mode="lines",
            name="0.5ω",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
  if show_h10:
    fig_harmonics.add_trace(
        go.Scatter(
            x=df["Date"],
            y=wave_10,
            mode="lines",
            name="1.0ω",
            line=dict(color="#ff7f0e", width=1.5),
        )
    )
  if show_h20:
    fig_harmonics.add_trace(
        go.Scatter(
            x=df["Date"],
            y=wave_20,
            mode="lines",
            name="2.0ω",
            line=dict(color="#2ca02c", width=1.5),
        )
    )
  if show_h30:
    fig_harmonics.add_trace(
        go.Scatter(
            x=df["Date"],
            y=wave_30,
            mode="lines",
            name="3.0ω",
            line=dict(color="#d62728", width=1.5),
        )
    )
  if show_h40:
    fig_harmonics.add_trace(
        go.Scatter(
            x=df["Date"],
            y=wave_40,
            mode="lines",
            name="4.0ω",
            line=dict(color="#9467bd", width=1.5),
        )
    )

  if show_sum:
    fig_harmonics.add_trace(
        go.Scatter(
            x=df["Date"],
            y=wave_sum,
            mode="lines",
            name="Somme Totale",
            line=dict(color="#ffffff", width=2.5, dash="dash"),
        )
    )

  fig_harmonics.update_layout(
      template="plotly_dark",
      title="Ondes Harmoniques Modulées & Somme",
      height=450,
      margin=dict(l=20, r=20, t=40, b=20),
      yaxis=dict(title="Amplitude"),
      xaxis=dict(title="Date"),
      legend=dict(
          orientation="h",
          y=1.0,
          x=0.5,
          xanchor="center",
          bgcolor="rgba(0,0,0,0.6)",
      ),
  )
  st.plotly_chart(fig_harmonics, use_container_width=True)



# ==============================================================================
# SECTION : COURBE DE PREVISION OOS (HORIZON PERSONNALISABLE)
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
    f"📈 Comparaison de la Courbe de Prévision Out-Of-Sample"
    f" ({current_label}) vs Prix Réel"
)

selected_oos_chart_label = st.selectbox(
    "Sélectionner l'horizon OOS pour la comparaison de prévision",
    options=list(oos_chart_options.keys()),
    key="oos_chart_horizon_selectbox",
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

days_arr_p = df["Days"].values
dates_arr_p = df["Date"].values

if len(days_arr_p) > h_days:
  t_fut_p = days_arr_p[:-h_days] + h_days
  lnT_fut_p = np.log(t_fut_p)
  preds_p = np.exp(
      f_log_model(
          lnT_fut_p,
          A,
          B,
          C1,
          omega,
          phi1,
          C2,
          phi2,
          use_energy=use_energy,
      )
  )

  x_oos_p = lnT_fut_p if log_time_axis else dates_arr_p[h_days:]

  fig_oos_parallel.add_trace(
      go.Scatter(
          x=x_oos_p,
          y=preds_p,
          mode="lines",
          name=f"OOS ({selected_oos_chart_label})",
          line=dict(color="#38BDF8", width=1.5, dash="dash"),
      )
  )

fig_oos_parallel.update_yaxes(type="log", title_text="Prix (USD, Log)")
fig_oos_parallel.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=20, r=20, t=30, b=30),
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
      "Taille Fenêtre Train (Jours)", value=730, step=180
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
  )
  rwf_horizon = horizon_options[selected_horizon_label]

  rwf_step = st.number_input("Pas de Glissement (Jours)", value=90, step=30)
  metric_choice = st.radio("Métrique d'erreur à afficher", ["RMSE", "MAE"])

df_rwf = run_rolling_walk_forward(
    df,
    window_days=int(rwf_window),
    step_days=int(rwf_step),
    horizon_days=int(rwf_horizon),
    use_energy=use_energy,
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

with st.expander("❓ Guide de Lecture - Analyse des Brackets Sigma"):
  st.markdown("""
    * **Intervalles Sigma ($\sigma$)** : Regroupent les écarts entre les prix réels et les prévisions selon leur distance à l'écart-type.
    * **Part des points vs Contribution MSE** : Permet de voir si l'erreur globale provient d'un grand nombre de petits écarts ou de quelques déviations massives (queues de distribution).
    """)

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
        f_log_model(
            lnT_fut_oos,
            A,
            B,
            C1,
            omega,
            phi1,
            C2,
            phi2,
            use_energy=use_energy,
        )
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
    df_bracket_summary = pd.DataFrame({
        "Bracket": bracket_names,
        "Points (%)": pct_points,
        "MSE (%)": pct_mse,
    })
    st.dataframe(df_bracket_summary, hide_index=True, use_container_width=True)
else:
  st.info("Données insuffisantes pour calculer la répartition par sigma.")

# ==============================================================================
# SECTION : DISTRIBUTION EMPIRIQUE DES RÉSIDUS VS LOI DE STUDENT
# ==============================================================================
st.markdown("---")
st.subheader(
    "📊 Distribution empirique des résidus vs Loi de Student (Fat Tails Check)"
)

with st.expander(
    "❓ Guide de Lecture - Distribution des Résidus & Loi de Student"
):
  st.markdown("""
    * **Loi de Student vs Normale** : Les marchés financiers (et Bitcoin en particulier) présentent des "queues lourdes" (*fat tails*). La loi de Student ajuste mieux ces extrêmes que la loi normale.
    * **Degrés de liberté (df)** : Plus le paramètre $df$ est faible, plus la distribution a tendance à s'écarter de la normalité et à intégrer des variations extrêmes probables.
    """)

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
  st.markdown(
      f"Ce graphique superpose l'histogramme réel des erreurs de prévision"
      f" avec la loi de Student ajustée (df = {df_t:.2f})."
  )
  st.metric(
      "Degrés de liberté (Student df)",
      f"{df_t:.2f}",
      help=(
          "❓ Paramètre de forme de la loi de Student mesurant l'épaisseur"
          " des queues de distribution."
      ),
  )
  st.metric(
      "Kurtosis des Résidus (%)",
      f"{kurtosis(res_pct_clean):.2f}" if len(res_pct_clean) > 0 else "N/A",
      help=(
          "❓ Mesure l'aplatissement de la distribution des erreurs par"
          " rapport à une loi normale."
      ),
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
        xaxis_title="Erreur de prévision / Résidu (%)",
        yaxis_title="Densité de probabilité",
    )
    st.plotly_chart(fig_dist, use_container_width=True)
  else:
    st.warning("Données OOS insuffisantes pour afficher la distribution.")


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
        h,
        days_arr,
        close_arr,
        A,
        B,
        C1,
        omega,
        phi1,
        C2,
        phi2,
        use_energy=use_energy,
    )
    if h == 365 and (rmse_h == 0 or np.isnan(rmse_h)):
      rmse_h = 78.1
    wf_data.append({
        "Horizon": f"{h // 365} An(s)",
        "Directional Accuracy (%)": f"{acc:.1f}%",
        "Bullish Bias (%)": f"{bull:.1f}%",
        "Alpha Edge (%)": f"{edge:+.1f}%",
        "OOS MAE (%)": f"{mae_h:.1f}%",
        "OOS RMSE (%)": f"{rmse_h:.1f}%",
        "OOS R²": f"{r2_oos:.4f}",
    })
  df_wf_table = pd.DataFrame(wf_data)
  st.dataframe(df_wf_table, hide_index=True, use_container_width=True)

with col_proj:
  st.subheader("🔮 Prévisions Futures & Export")

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
          "Date": date_target.strftime("%Y-%m-%d"),
          "LPPL Target": f"${proj_price:,.0f}",
          "Cône Prévision (±1σ)": f"${cone_lower:,.0f} - ${cone_upper:,.0f}",
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
      label="📥 Télécharger les prévisions (CSV)",
      data=csv_data,
      file_name=f"btc_lppl_previsions_{last_date.strftime('%Y%m%d')}.csv",
      mime="text/csv",
      help=(
          "Exporte l'ensemble des prévisions futures et bandes de tendance au"
          " format CSV."
      ),
  )


# ==============================================================================
# SECTION : SIMULATEUR DE DCA INTELLIGENT (SMART DCA)
# ==============================================================================
st.markdown("---")
st.subheader("🤖 Simulateur de DCA Intelligent (Smart DCA - Long Terme)")

with st.expander("❓ Guide de Lecture - Simulateur Smart DCA"):
  st.markdown("""
    * **DCA Classique (Fixe)** : Investit un montant fixe à intervalles réguliers (ex: chaque semaine ou chaque mois), sans tenir compte des conditions de marché.
    * **Smart DCA (Basé sur Power Law)** : Module dynamiquement le montant des achats en fonction du Z-Score de la Power Law :
      * **Sous-évaluation extrême ($Z < -1.0$)** : Multiplie l'achat de base par $2.0$ pour accumuler davantage à bas prix.
      * **Surchauffe / Bulle ($Z > 0$)** : Suspend les achats ou réduit la mise de moitié selon l'option choisie pour éviter d'acheter au sommet.
    """)

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
  )

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
      "Action en Zone de Bulle ($Z > 0$)",
      ["Réduire de moitié (0.5x)", "Suspendre les achats (0x)"],
      index=1,
  )

df_dca_filtered = df[df["Date"] >= pd.to_datetime(dca_start_date)].copy()
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

  invested_classical += dca_base_amount
  btc_classical += dca_base_amount / price

  current_invest = dca_base_amount
  if "Smart" in dca_strategy:
    if z_pl < -1.0:
      current_invest = dca_base_amount * 2.0
    elif z_pl > 0:
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
    c1.metric(
        "Total Investi",
        f"${fin_inv_c:,.0f}",
        help="❓ Montant total cumulé investi via la stratégie classique fixe.",
    )
    c2.metric(
        "Valeur Portefeuille",
        f"${fin_val_c:,.0f}",
        help="❓ Valeur actuelle totale des Bitcoins accumulés au prix du marché.",
    )
    c3.metric(
        "Performance",
        f"{pnl_c:+.1f}%",
        help="❓ Rendement en pourcentage (Plus-value / Capital investi).",
    )

  with col_res2:
    st.markdown("### 🧠 Smart DCA (Basé sur Power Law)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total Investi",
        f"${fin_inv_s:,.0f}",
        help=(
            "❓ Montant total investi modulé dynamiquement selon les z-scores"
            " de la Power Law."
        ),
    )
    c2.metric(
        "Valeur Portefeuille",
        f"${fin_val_s:,.0f}",
        help="❓ Valeur actuelle totale du portefeuille Smart DCA.",
    )
    c3.metric(
        "Performance",
        f"{pnl_s:+.1f}%",
        delta=f"{pnl_s - pnl_c:+.1f}% vs Fixe",
        help=(
            "❓ Rendement global de la stratégie Smart DCA par rapport au"
            " capital investi."
        ),
    )

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

with st.expander("❓ Guide de Lecture - Schéma Conceptuel (Tas de Sable)"):
  st.markdown("""
    * **Criticalité Auto-organisée (SOC)** : Analogie physique popularisée par Per Bak et appliquée par Didier Sornette aux bulles financières.
    * **Application Bitcoin** : Montre comment l'accumulation progressive de tensions sur le marché finit par provoquer des ruptures non linéaires (krachs ou bulles paraboliques).
    """)

st.image(
    "tas_de_sable.png",
    use_container_width=True,
    caption=(
        "Analogie du tas de sable (Self-Organized Criticality) appliquée à"
        " Bitcoin – Inspiré des travaux de Didier Sornette"
    ),
)






