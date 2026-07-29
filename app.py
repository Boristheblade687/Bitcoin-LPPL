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
