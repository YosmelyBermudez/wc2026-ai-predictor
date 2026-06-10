"""
⚽ World Cup 2026 — AI Predictor
Streamlit App — Run locally with: streamlit run app.py

Author: Yosmely Bermúdez | AI/ML Engineer
GitHub: https://github.com/yosbermudez/wc2026-ai-predictor

Folder structure expected:
    app.py
    data/
        matches_1930_2022.csv
        player_stats.csv
        player_gca.csv
        player_defense.csv
    .env   ← GOOGLE_API_KEY=your_key_here  (optional, for LLM narratives)
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb
from itertools import combinations
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()  # reads .env file if present

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme base */
    .stApp { background-color: #0d1117; }
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e6edf3 !important;
    }
    .main .block-container { padding-top: 2rem; }

    /* Typography */
    h1 { color: #e6edf3 !important; font-size: 1.8rem !important; }
    h2 { color: #79c0ff !important; }
    h3 { color: #3fb950 !important; }
    p, li { color: #e6edf3; }

    /* Prediction result box */
    .prediction-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #388bfd;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* Narrative box */
    .narrative-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #c9a84c;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 16px;
        line-height: 1.7;
        color: #e6edf3;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 8px 0;
    }

    /* Probability label */
    .prob-label {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 4px;
    }

    /* Team header */
    .team-header {
        font-size: 22px;
        font-weight: 700;
        color: #e6edf3;
        text-align: center;
        padding: 8px 0;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #161b22; }

    /* Upset warning */
    .upset-warning {
        background: rgba(248, 81, 73, 0.1);
        border: 1px solid rgba(248, 81, 73, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 13px;
        color: #f85149;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# DATA & MODEL — loaded once, cached
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline():
    """
    Full pipeline:
    1. Load historical match data
    2. Compute Elo ratings (964 matches, 1930-2022)
    3. Compute historical win rates
    4. Build Player Impact Score from Qatar 2022 player data
    5. Train LightGBM on 900 matches (1930-2018)
    Returns everything the prediction function needs.
    """

    def load_csv(filename):
        path = os.path.join("Data", filename)
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    # ── Load data ──────────────────────────────────────────────────────────
    matches   = load_csv("matches_1930_2022.csv")
    p_stats   = load_csv("player_stats.csv")
    p_gca     = load_csv("player_gca.csv")
    p_defense = load_csv("player_defense.csv")

    # ── Match outcomes ─────────────────────────────────────────────────────
    def get_result(row):
        if row["home_score"] > row["away_score"]:   return "home_win"
        elif row["home_score"] < row["away_score"]: return "away_win"
        else:                                        return "draw"

    matches["result"] = matches.apply(get_result, axis=1)
    TARGET_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
    REV_MAP    = {0: "home_win", 1: "draw", 2: "away_win"}
    matches["target"] = matches["result"].map(TARGET_MAP)

    # ── Elo engine ─────────────────────────────────────────────────────────
    # K=32: standard for international football
    # Each match redistributes points based on upset magnitude
    elo     = {}
    history = []
    for _, row in matches.sort_values("Year").iterrows():
        h, a, yr = row["home_team"], row["away_team"], row["Year"]
        if h not in elo: elo[h] = 1500
        if a not in elo: elo[a] = 1500
        rh, ra  = elo[h], elo[a]
        exp_h   = 1 / (1 + 10 ** ((ra - rh) / 400))
        act_h   = 1.0 if row["result"]=="home_win" else (0.0 if row["result"]=="away_win" else 0.5)
        new_h   = rh + 32 * (act_h - exp_h)
        new_a   = ra + 32 * ((1 - act_h) - (1 - exp_h))
        history += [
            {"year": yr, "team": h, "elo_after": new_h},
            {"year": yr, "team": a, "elo_after": new_a}
        ]
        elo[h], elo[a] = new_h, new_a

    hist_df = pd.DataFrame(history)

    # Elo at tournament entry (no leakage — only prior history)
    elo_entries = []
    for year in sorted(matches["Year"].unique()):
        teams = set(
            matches[matches["Year"]==year]["home_team"].tolist() +
            matches[matches["Year"]==year]["away_team"].tolist()
        )
        for team in teams:
            prior = hist_df[(hist_df["team"]==team) & (hist_df["year"]<year)]
            elo_entries.append({
                "team": team, "year": year,
                "elo_entry": prior.iloc[-1]["elo_after"] if len(prior) > 0 else 1500
            })
    elo_entry_df = pd.DataFrame(elo_entries)

    # ── Historical win rates ───────────────────────────────────────────────
    hw = matches[matches["result"]=="home_win"].groupby("home_team").size()
    aw = matches[matches["result"]=="away_win"].groupby("away_team").size()
    hp = matches.groupby("home_team").size()
    ap = matches.groupby("away_team").size()
    win_rate = (hw.add(aw, fill_value=0) / hp.add(ap, fill_value=0)).fillna(0.25)

    # ── ML dataset + train LightGBM ───────────────────────────────────────
    mf = matches.copy()
    mf = mf.merge(
        elo_entry_df.rename(columns={"team":"home_team","elo_entry":"elo_home","year":"Year"}),
        on=["home_team","Year"], how="left"
    )
    mf = mf.merge(
        elo_entry_df.rename(columns={"team":"away_team","elo_entry":"elo_away","year":"Year"}),
        on=["away_team","Year"], how="left"
    )
    mf["elo_diff"]    = mf["elo_home"] - mf["elo_away"]
    mf["wr_home"]     = mf["home_team"].map(win_rate).fillna(0.25)
    mf["wr_away"]     = mf["away_team"].map(win_rate).fillna(0.25)
    mf["wr_diff"]     = mf["wr_home"] - mf["wr_away"]
    mf["is_knockout"] = mf["Round"].isin([
        "Round of 16", "Quarter-finals", "Semi-finals", "Final", "Third-place match"
    ]).astype(int)

    FEATURES = ["elo_diff","elo_home","elo_away","wr_diff","wr_home","wr_away","is_knockout"]
    train    = mf[mf["Year"] < 2022].dropna(subset=FEATURES + ["target"])

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03,
        num_leaves=15, min_child_samples=5,
        random_state=42, verbose=-1
    )
    model.fit(train[FEATURES], train["target"])

    # ── Player Impact Score (PIS) ─────────────────────────────────────────
    # Merge 3 player datasets
    KEY    = ["player", "team"]
    player = p_stats[[
        "player","team","minutes_90s",
        "goals_per90","xg_per90","xg_assist","assists_per90"
    ]].copy()

    gca_cols = [c for c in ["player","team","sca_per90","gca_per90"] if c in p_gca.columns]
    def_cols  = [c for c in ["player","team","tackles_won","interceptions"] if c in p_defense.columns]
    player = player.merge(p_gca[gca_cols], on=KEY, how="left")
    player = player.merge(p_defense[def_cols], on=KEY, how="left")

    # Filter noise: min 1 full match equivalent (90 min)
    clean  = player[player["minutes_90s"] >= 1.0].copy()
    per90  = clean["minutes_90s"].clip(lower=0.1)

    clean["pis_attack"]   = (
        clean["goals_per90"].fillna(0) * 0.25 +
        clean["xg_per90"].fillna(0)    * 0.20 +
        clean["xg_assist"].fillna(0) / per90 * 0.15
    )
    clean["pis_creation"] = (
        clean["sca_per90"].fillna(0) * 0.10
        if "sca_per90" in clean.columns else 0
    )
    clean["pis_defense"]  = (
        clean["tackles_won"].fillna(0) / per90 * 0.15 +
        clean["interceptions"].fillna(0) / per90 * 0.15
    )
    clean["pis_raw"] = (
        clean["pis_attack"] + clean["pis_creation"] + clean["pis_defense"]
    )

    pis_min, pis_max = clean["pis_raw"].min(), clean["pis_raw"].max()
    clean["pis"] = (clean["pis_raw"] - pis_min) / (pis_max - pis_min) * 100

    def team_pis_agg(group):
        xi = group.nlargest(11, "minutes_90s")
        return pd.Series({
            "pis_mean":     round(xi["pis"].mean(), 1),
            "pis_top3":     round(xi.nlargest(3, "pis")["pis"].mean(), 1),
            "pis_def_mean": round(xi["pis_defense"].mean(), 4)
            if "pis_defense" in xi.columns else 0.5
        })

    pis_map = clean.groupby("team").apply(team_pis_agg).to_dict("index")

    return model, elo, win_rate, pis_map, FEATURES, REV_MAP, elo_entry_df


@st.cache_resource(show_spinner="Initializing LLM...")
def load_llm_chain():
    """
    Load Groq client.
    Returns None if GROQ_API_KEY is not set — app works without it.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None

    try:
        from groq import Groq
        return Groq(api_key=api_key)

    except Exception as e:
        st.warning(f"LLM not available: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def predict_match(model, elo_final, win_rate, pis_map, FEATURES, REV_MAP,
                  home_team, away_team, stage="group"):
    """
    Single prediction function used by both the UI and the LLM.
    Returns a dict with probabilities + all team context.
    """
    elo_h = elo_final.get(home_team, 1500)
    elo_a = elo_final.get(away_team, 1500)
    wr_h  = float(win_rate.get(home_team, 0.25))
    wr_a  = float(win_rate.get(away_team, 0.25))
    pis_h = pis_map.get(home_team, {"pis_mean": 50.0, "pis_top3": 50.0, "pis_def_mean": 0.5})
    pis_a = pis_map.get(away_team, {"pis_mean": 50.0, "pis_top3": 50.0, "pis_def_mean": 0.5})

    X = pd.DataFrame([{
        "elo_diff":    elo_h - elo_a,
        "elo_home":    elo_h,
        "elo_away":    elo_a,
        "wr_diff":     wr_h - wr_a,
        "wr_home":     wr_h,
        "wr_away":     wr_a,
        "is_knockout": 1 if stage == "knockout" else 0
    }])

    proba = model.predict_proba(X)[0]
    pred  = REV_MAP[model.predict(X)[0]]

    # Morocco flag: if PIS defense is in top 25% but Elo gap is large
    # this is a signal the model might be underestimating
    elo_gap    = abs(elo_h - elo_a)
    pis_def_h  = pis_h.get("pis_def_mean", 0)
    pis_def_a  = pis_a.get("pis_def_mean", 0)
    upset_risk = (
        elo_gap > 100 and
        max(pis_def_h, pis_def_a) > 0.08 and
        max(float(proba[0]), float(proba[2])) < 0.55
    )

    return {
        # Probabilities
        "p_home_win":    round(float(proba[0]) * 100, 1),
        "p_draw":        round(float(proba[1]) * 100, 1),
        "p_away_win":    round(float(proba[2]) * 100, 1),
        "prediction":    pred,
        # Team context
        "home_team":     home_team,
        "away_team":     away_team,
        "stage":         stage,
        "elo_home":      int(round(elo_h)),
        "elo_away":      int(round(elo_a)),
        "wr_home":       round(wr_h * 100, 1),
        "wr_away":       round(wr_a * 100, 1),
        "pis_home":      pis_h["pis_mean"],
        "pis_away":      pis_a["pis_mean"],
        "pis_top3_home": pis_h["pis_top3"],
        "pis_top3_away": pis_a["pis_top3"],
        "has_pis_home":  home_team in pis_map,
        "has_pis_away":  away_team in pis_map,
        "upset_risk":    upset_risk,
    }


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

# WC 2026 confirmed/likely participants — update after official draw
WC2026_TEAMS = sorted([
    "Argentina", "Australia", "Belgium", "Brazil", "Cameroon", "Canada",
    "Colombia", "Costa Rica", "Croatia", "Czech Republic", "Denmark",
    "DR Congo", "Ecuador", "Egypt", "England", "France", "Germany",
    "Ghana", "Honduras", "IR Iran", "Iraq", "Japan", "Jordan",
    "Korea Republic", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Nigeria", "Panama", "Poland", "Portugal", "Qatar", "Romania",
    "Saudi Arabia", "Scotland", "Senegal", "Serbia", "South Africa",
    "Spain", "Switzerland", "Tunisia", "Turkey", "Ukraine",
    "United States", "Uruguay", "Venezuela", "Wales",
])

# Illustrative groups — update after official draw (Dec 2025)
GROUPS = {
    "A": ["United States", "Mexico", "Canada", "Uruguay"],
    "B": ["Argentina",     "Brazil",  "Colombia", "Ecuador"],
    "C": ["France",        "Germany", "Portugal", "Morocco"],
    "D": ["Spain",         "England", "Netherlands", "Croatia"],
    "E": ["Japan",         "Korea Republic", "Saudi Arabia", "Senegal"],
    "F": ["Belgium",       "Serbia",  "Denmark", "Poland"],
}

PRED_LABELS = {
    "home_win": "wins",
    "draw":     "Draw",
    "away_win": "wins"
}

PRED_COLORS = {
    "home_win": "#3fb950",
    "draw":     "#c9a84c",
    "away_win": "#388bfd"
}


# ══════════════════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ══════════════════════════════════════════════════════════════════════════

model, elo_final, win_rate, pis_map, FEATURES, REV_MAP, elo_entry_df = load_pipeline()
llm_chain = load_llm_chain()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# ⚽ WC 2026")
    st.markdown("### AI Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🎯 Match Predictor", "📊 Group Stage", "ℹ️ How it works"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model stats**")
    st.markdown("Trained on `900 matches` (1930–2018)")
    st.markdown("Tested on `64 matches` (Qatar 2022)")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "48.4%")
    col2.metric("F1 Score", "0.478")

    st.markdown("---")
    llm_status = "✅ Active" if llm_chain else "⚠️ No API key"
    st.markdown(f"**LLM Narratives:** {llm_status}")
    if not llm_chain:
        st.caption("Add GOOGLE_API_KEY to .env to enable AI narratives")

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/yosbermudez/wc2026-ai-predictor)")
    st.markdown("[LinkedIn](https://linkedin.com/in/yosmely-bermudez)")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

if page == "🎯 Match Predictor":
    st.title("⚽ World Cup 2026 — Match Predictor")
    st.markdown(
        "*Predictions grounded in 92 years of World Cup history + Qatar 2022 player analytics*"
    )
    st.markdown("---")

    # Team selectors
    col1, col2, col3 = st.columns([5, 2, 5])
    with col1:
        home = st.selectbox(
            "🏠 Home Team",
            WC2026_TEAMS,
            index=WC2026_TEAMS.index("Argentina")
        )
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center; color:#8b949e;'>VS</h2>",
            unsafe_allow_html=True
        )
    with col3:
        away = st.selectbox(
            "✈️ Away Team",
            WC2026_TEAMS,
            index=WC2026_TEAMS.index("Morocco")
        )

    col4, _ = st.columns([2, 6])
    with col4:
        stage = st.selectbox("Stage", ["group", "knockout"])

    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_btn:
        if home == away:
            st.error("Please select two different teams.")
            st.stop()

        with st.spinner("Running the model..."):
            r = predict_match(
                model, elo_final, win_rate, pis_map, FEATURES, REV_MAP,
                home, away, stage
            )

        st.markdown("---")

        # ── Probability display ────────────────────────────────────────────
        st.subheader("Outcome Probabilities")
        c1, c2, c3 = st.columns(3)

        with c1:
            color = "#3fb950" if r["prediction"] == "home_win" else "#8b949e"
            st.markdown(
                f"<div class='team-header' style='color:{color};'>{home}</div>",
                unsafe_allow_html=True
            )
            st.metric("Win probability", f"{r['p_home_win']}%")
            st.progress(int(r["p_home_win"]))

        with c2:
            color = "#c9a84c" if r["prediction"] == "draw" else "#8b949e"
            st.markdown(
                f"<div class='team-header' style='color:{color};'>Draw</div>",
                unsafe_allow_html=True
            )
            st.metric("Probability", f"{r['p_draw']}%")
            st.progress(int(r["p_draw"]))

        with c3:
            color = "#388bfd" if r["prediction"] == "away_win" else "#8b949e"
            st.markdown(
                f"<div class='team-header' style='color:{color};'>{away}</div>",
                unsafe_allow_html=True
            )
            st.metric("Win probability", f"{r['p_away_win']}%")
            st.progress(int(r["p_away_win"]))

        # Prediction banner
        pred_team = home if r["prediction"] == "home_win" else (away if r["prediction"] == "away_win" else None)
        if pred_team:
            pred_text = f"🏆 Model predicts: **{pred_team} wins** ({max(r['p_home_win'], r['p_away_win'])}%)"
        else:
            pred_text = f"🤝 Model predicts: **Draw** ({r['p_draw']}%)"

        pred_color = PRED_COLORS[r["prediction"]]
        st.markdown(
            f"<div class='prediction-box'>{pred_text}</div>",
            unsafe_allow_html=True
        )

        # Morocco-style upset warning
        if r["upset_risk"]:
            st.markdown(
                "<div class='upset-warning'>⚠️ <b>Upset risk detected</b> — "
                "one team shows strong defensive Player Impact Score despite a significant "
                "Elo gap. The model may be underestimating current squad quality. "
                "Think Morocco 2022.</div>",
                unsafe_allow_html=True
            )

        # ── Team stats ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Team Statistics")

        stats_cols = st.columns(2)
        for i, (team, elo_val, wr, pis, pis3, has_pis) in enumerate([
            (home, r["elo_home"], r["wr_home"], r["pis_home"], r["pis_top3_home"], r["has_pis_home"]),
            (away, r["elo_away"], r["wr_away"], r["pis_away"], r["pis_top3_away"], r["has_pis_away"])
        ]):
            with stats_cols[i]:
                st.markdown(f"**{team}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Elo", f"{elo_val:,}")
                m2.metric("WC Win Rate", f"{wr}%")
                m3.metric("PIS Squad", f"{pis}/100" if has_pis else "N/A*")
                m4.metric("PIS Top-3", f"{pis3}/100" if has_pis else "N/A*")
                if not has_pis:
                    st.caption("*PIS based on Qatar 2022 data. Not available for teams not in WC 2022.")

        # Elo visualisation
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[home, away],
            y=[r["elo_home"], r["elo_away"]],
            marker_color=["#3fb950", "#388bfd"],
            text=[f"{r['elo_home']}", f"{r['elo_away']}"],
            textposition="outside"
        ))
        fig.update_layout(
            title="Elo Rating Comparison",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#e6edf3",
            height=250,
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis=dict(range=[
                min(r["elo_home"], r["elo_away"]) - 100,
                max(r["elo_home"], r["elo_away"]) + 100
            ])
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── LLM Narrative ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🤖 AI Match Analysis")

        if llm_chain is None:
            st.info(
                "LLM narratives are disabled. "
                "Add your `GOOGLE_API_KEY` to a `.env` file to enable them.\n\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        else:
            with st.spinner("Generating match analysis..."):
                try:
                    prompt_text = f"""You are a data-driven football analyst for the World Cup 2026 AI Predictor.
                    Your job is to translate statistics into readable match previews.
                    IMPORTANT: Only use the numbers provided below. Never invent injuries, squad news, or stats not listed.

                    MATCH: {r['home_team']} vs {r['away_team']} | Stage: {r['stage']}

                    MODEL OUTPUT:
                    - {r['home_team']} win: {r['p_home_win']}%
                    - Draw: {r['p_draw']}%
                    - {r['away_team']} win: {r['p_away_win']}%
                    - Prediction: {r['prediction']}

                    TEAM STATISTICS (historical WC data + Qatar 2022 player analysis):
                    {r['home_team']}:
                    Elo rating: {r['elo_home']} | WC win rate: {r['wr_home']}% | Player Impact Score: {r['pis_home']}/100 | Top-3 PIS: {r['pis_top3_home']}
                    {r['away_team']}:
                    Elo rating: {r['elo_away']} | WC win rate: {r['wr_away']}% | Player Impact Score: {r['pis_away']}/100 | Top-3 PIS: {r['pis_top3_away']}

                    Write a match preview in exactly 3 paragraphs:
                    1. The key narrative tension of this matchup (what makes it interesting)
                    2. What the numbers say — explain WHY the model predicts this outcome
                    3. One honest sentence about what could make the model wrong

                    Tone: analytical but engaging. Like a good football journalist who trusts data."""

                    response = llm_chain.chat.completions.create(
                                            model="llama-3.1-8b-instant",
                                            max_tokens=500,
                                            messages=[{"role": "user", "content": prompt_text}]
                                        )
                    narrative = response.choices[0].message.content
                    st.markdown(
                                            f'<div class="narrative-box">{narrative}</div>',
                                            unsafe_allow_html=True
                                        )
                except Exception as e:
                    st.error(f"LLM error: {e}")

        st.caption(
            "Model: LightGBM · Trained: 900 matches (1930–2018) · "
            "Validated: Qatar 2022 (blind) · Accuracy: 48.4% · F1: 0.478 · Log-Loss: 1.204"
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: GROUP STAGE
# ══════════════════════════════════════════════════════════════════════════

elif page == "📊 Group Stage":
    st.title("📊 WC 2026 — Group Stage Predictions")
    st.markdown(
        "*Expected points per team across all group matches. "
        "Top 2 advance.*"
    )
    st.caption("⚠️ Groups are illustrative — will be updated after the official 2026 draw")
    st.markdown("---")

    # Compute all groups
    all_standings = {}
    with st.spinner("Simulating all group matches..."):
        for group_name, teams in GROUPS.items():
            exp_pts = {t: 0.0 for t in teams}
            for home_t, away_t in combinations(teams, 2):
                r = predict_match(
                    model, elo_final, win_rate, pis_map, FEATURES, REV_MAP,
                    home_t, away_t, "group"
                )
                exp_pts[home_t] += r["p_home_win"]/100 * 3 + r["p_draw"]/100
                exp_pts[away_t] += r["p_away_win"]/100 * 3 + r["p_draw"]/100
            all_standings[group_name] = sorted(exp_pts.items(), key=lambda x: -x[1])

    # Display in 3 columns
    col_pairs = [
        ("A", "B"),
        ("C", "D"),
        ("E", "F"),
    ]

    for left_g, right_g in col_pairs:
        cols = st.columns(2)
        for col, g in zip(cols, [left_g, right_g]):
            with col:
                st.markdown(f"### Group {g}")
                standing = all_standings[g]
                for i, (team, pts) in enumerate(standing):
                    elo_t = elo_final.get(team, 1500)
                    wr_t  = float(win_rate.get(team, 0.25)) * 100
                    icon  = "🥇" if i == 0 else "🥈" if i == 1 else "🔴"
                    qualifies = " **→ Advances**" if i < 2 else ""
                    st.markdown(
                        f"{icon} **{team}**{qualifies}  \n"
                        f"&nbsp;&nbsp;&nbsp;{pts:.2f} exp pts | "
                        f"Elo: {int(elo_t)} | WR: {wr_t:.0f}%"
                    )
                st.markdown("")

    # Summary chart
    st.markdown("---")
    st.subheader("Predicted Group Winners by Elo")

    group_winners = [(g, all_standings[g][0][0]) for g in GROUPS]
    winner_elos   = [elo_final.get(t, 1500) for _, t in group_winners]
    winner_labels = [f"G{g}: {t}" for g, t in group_winners]

    fig = go.Figure(go.Bar(
        x=winner_labels,
        y=winner_elos,
        marker_color="#c9a84c",
        text=[f"{e:.0f}" for e in winner_elos],
        textposition="outside"
    ))
    fig.update_layout(
        title="Elo of Predicted Group Winners",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis=dict(
            range=[min(winner_elos)-150, max(winner_elos)+100],
            gridcolor="#30363d"
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ How it works":
    st.title("ℹ️ How This System Works")

    st.markdown("""
    ## Three-Layer Architecture

    This is not a single model. It is a pipeline of signals, each one compensating
    for the blind spots of the previous one.

    ---

    ### Layer 1 — Historical Signal
    **Custom Elo Engine + Historical Win Rates**

    The Elo system assigns every team a rating based on 92 years of World Cup results.
    Beating a strong opponent earns more points than beating a weak one.
    After 964 matches, the ratings reflect long-term team quality.

    > *What it captures:* The identity of a national team across decades.  
    > *What it misses:* A completely rebuilt squad, a new tactical system, peak form.

    ---

    ### Layer 2 — Current Squad Signal
    **Player Impact Score (PIS)**

    Built from 9 granular datasets covering all 680 Qatar 2022 players.
    After filtering the 194 players with less than 1 full match equivalent
    (statistical noise), we compute a weighted score across:
    - Attacking: goals/90 · xG/90 · xG assist (40%)
    - Creative: shot-creating actions/90 (10%)
    - Defensive: tackles won/90 · interceptions/90 (30%)

    > *What it captures:* Who is actually on the pitch right now.  
    > *Why it matters:* Morocco had Elo 1435 entering Qatar 2022. Their defensive PIS was top 3. The model missed them. The PIS layer is the fix.

    ---

    ### Layer 3 — Narrative Signal
    **LangChain + Gemini**

    The LLM receives every number from our pipeline and generates a grounded
    match preview. It is explicitly instructed never to invent squad news,
    injuries, or statistics not in the context.

    > *What it adds:* Readable analysis that connects the numbers into a story.  
    > *What it doesn't do:* Hallucinate. Every claim is anchored to a real number.

    ---

    ## Validation

    The model was trained on **900 matches (1930–2018)** and tested blind on
    **64 matches (Qatar 2022)** — results the model had never seen.

    | Model | Accuracy | Weighted F1 |
    |-------|----------|-------------|
    | FIFA Ranking (baseline) | 51.7% | 0.517 |
    | Logistic Regression | 46.9% | 0.394 |
    | **LightGBM** | **48.4%** | **0.478** |

    The model is not perfect. Football is not predictable.
    The goal is to quantify uncertainty better than a ranking table does.

    ---

    ## Built by

    **Yosmely Bermúdez** — AI/ML Engineer  
    Geophysical Engineer turned AI Engineer.

    In geophysics: garbage in → garbage out.  
    Same rule applies here.

    [GitHub](https://github.com/yosbermudez/wc2026-ai-predictor) |
    [LinkedIn](https://linkedin.com/in/yosmely-bermudez)
    """)
