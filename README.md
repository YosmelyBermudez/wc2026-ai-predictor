# ⚽ World Cup 2026 — AI Predictor

> *"The model was looking at Morocco's past. The player data was describing Morocco's present."*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-purple?style=flat-square)](https://lightgbm.readthedocs.io)
[![LangChain](https://img.shields.io/badge/LangChain-LLM_layer-green?style=flat-square)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Building_live-orange?style=flat-square)]()
[![WC2026](https://img.shields.io/badge/World_Cup-2026-gold?style=flat-square)]()

A production ML system that predicts World Cup 2026 match outcomes by combining
**92 years of historical data**, a **custom Elo engine**, a **Player Impact Score**
built from 9 granular datasets, and an **LLM narrative layer** that generates
human-readable match analysis.

Built live — one commit per day — predictions dropping before June 11. 🏆

---

## The Problem

Football prediction is genuinely hard:

- **3 outcome classes** — home win / draw / away win (not binary)
- **Class imbalance** — 55% home wins, 22% draws, 22% away wins
- **High variance** — one red card changes everything
- **Small test sets** — only 64 matches per World Cup
- **The draw problem** — our model's draw F1 is 0.29. Honestly? That's the sport.

The goal is not to predict who wins.
The goal is to assign probabilities that are **better calibrated than a FIFA ranking table**.

---

## Architecture

```
Historical data (1930–2022) · 964 matches · 18 CSV files
              │
              ▼
  ┌─────────────────────────┐
  │   Layer 1: Elo Engine   │  ← 92 years of match history
  │   + Historical Win Rates│    captures long-term team identity
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  LightGBM Classifier    │  ← trained on 900 matches (1930–2018)
  │  Match probabilities    │    tested blind on Qatar 2022
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Layer 2: Player Impact │  ← 9 datasets · 680 players · 486 after noise filter
  │  Score (PIS)            │    captures current squad quality Elo can't see
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Layer 3: LLM Narrative │  ← LangChain + Gemini
  │  Match analysis         │    connects signals into human-readable output
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Streamlit App          │  ← deployed on Hugging Face Spaces
  │  Public predictions     │    updated every matchday
  └─────────────────────────┘
```

---

## Results — Qatar 2022 Blind Test

Trained on **900 matches (1930–2018)**, tested on **64 matches (Qatar 2022)** — the model never saw them.

| Model | Accuracy | Weighted F1 | Log-Loss |
|-------|----------|-------------|----------|
| FIFA Ranking (baseline) | 51.7% | 0.517 | — |
| Logistic Regression | 46.9% | 0.394 | — |
| **LightGBM** | **48.4%** | **0.478** | **1.204** |

**Per-class breakdown (LightGBM):**

| Class | Precision | Recall | F1 | Notes |
|-------|-----------|--------|----|-------|
| Home Win | 0.65 | 0.69 | 0.67 | ✅ Solid |
| Away Win | 0.35 | 0.35 | 0.35 | ⚠️ Needs work |
| Draw | 0.31 | 0.27 | 0.29 | ❌ Hardest class in football |

**Feature Importance (LightGBM):**

```
wr_diff      2642  ██████████████████████████████  #1  win rate differential
elo_diff     2478  ████████████████████████████    #2  Elo gap
elo_away     2266  █████████████████████████       #3  away team absolute strength
elo_home     2188  ████████████████████████        #4  home team absolute strength
wr_home      1450  ████████████████                #5  home team historical WR
wr_away      1319  ██████████████                  #6  away team historical WR
is_knockout   257  ██                              #7  stage type

FIFA rank: not in top features — replaced by Elo + historical win rates
```

---

## The Morocco Problem

Morocco beat Spain. Then Portugal. Then nearly France.
The model didn't see any of it coming.

```
Morocco Elo entering Qatar 2022:    1435
Spain Elo entering Qatar 2022:      1605   ← model predicted 77% Spain win
Actual result:                      Draw   ← Morocco advanced on penalties
```

That gap — between historical signal and current squad reality — is exactly why
this system has three layers. Historical Elo looks backward.
Player Impact Score looks at who is **actually on the pitch**.

---

## WC 2026 Group Stage Predictions

*Groups are illustrative — will be updated after the official draw.*
*Expected points = P(win)×3 + P(draw)×1, summed across 3 group matches.*

| Group | 1st | 2nd | Notable |
|-------|-----|-----|---------|
| A | 🇺🇾 Uruguay (5.89) | 🇲🇽 Mexico (4.41) | USA hosts but doesn't qualify |
| B | 🇧🇷 Brazil (6.34) | 🇦🇷 Argentina (4.85) | Group of death |
| C | 🇩🇪 Germany (7.78) | 🇵🇹 Portugal (4.66) | France 3rd — biggest model upset |
| D | 🇳🇱 Netherlands (5.38) | 🇪🇸 Spain (3.91) | England narrowly misses |
| E | 🇸🇳 Senegal (5.95) | 🇯🇵 Japan (3.89) | Africa & Asia surprise |
| F | 🇧🇪 Belgium (TBD) | 🇷🇸 Serbia (TBD) | — |

> **Spicy take from the model:** France lands in 3rd place in Group C behind Germany and Portugal.
> Elo says France (1693) is one of the strongest teams in the world.
> The group draw says the bracket is brutal.
> We'll see.

---

## Repository Structure

```
wc2026-ai-predictor/
│
├── README.md
│
├── notebooks/
│   ├── Day1_EDA_FeatureEngineering.ipynb   ✅ complete
│   └── Day2_Model_Training.ipynb           ✅ complete
│
├── src/                                    🔜 refactoring from notebooks
│   ├── elo_engine.py
│   ├── feature_engineering.py
│   ├── pis_pipeline.py
│   └── model.py
│
├── app/                                    🔜 Day 3
│   └── streamlit_app.py
│
├── outputs/
│   ├── wc2026_group_predictions.csv        ✅ generated
│   └── wc2026_team_strength.csv            ✅ generated
│
└── data/                                   ← not tracked (see .gitignore)
    └── .gitkeep
```

---

## Data Sources

| Dataset | Rows | Key Contents |
|---------|------|--------------|
| `matches_1930_2022.csv` | 964 | All WC matches: goals, xG, stage, teams |
| `world_cup.csv` | 22 | One row per tournament: host, champion |
| `fifa_ranking_2022-10-06.csv` | 211 | FIFA ranking at WC 2022 start |
| `data__1_.csv` | 64 | Qatar 2022 matches: 53 features each |
| `team_data.csv` | 32 | Qatar 2022 teams: 189 features each |
| `player_stats.csv` | 680 | Base stats: goals, assists, xG, xA |
| `player_shooting.csv` | 680 | Shot quality, npxG, distance |
| `player_defense.csv` | 680 | Tackles, interceptions, blocks |
| `player_gca.csv` | 680 | Goal and shot-creating actions |
| `player_keepers.csv` + `keepersadv.csv` | ~50 | Save%, PSxG, distribution |
| *+ 4 more player datasets* | 680 | Passing, possession, misc, playing time |

Raw CSVs are not tracked in this repo. See notebooks for loading instructions.

---

## Stack

```python
# Core
pandas          # data manipulation
numpy           # numerical operations
scikit-learn    # preprocessing, metrics, Logistic Regression
lightgbm        # gradient boosting — main model
plotly          # visualizations

# Coming Day 3
langchain       # LLM orchestration
google-generativeai  # Gemini API
streamlit       # web app interface
```

---

## Roadmap

- [x] **Day 1** — EDA + Feature Engineering
  - [x] Elo rating engine (964 matches, K=32)
  - [x] Historical win rate pipeline
  - [x] Player Impact Score (9 datasets, 486 signal players)
  - [x] Master feature table

- [x] **Day 2** — Model Training + Validation
  - [x] FIFA ranking baseline
  - [x] Logistic Regression with Elo features
  - [x] LightGBM — leave-one-tournament-out validation
  - [x] Error analysis (Morocco, Japan, Croatia upsets)
  - [x] WC 2026 group stage predictions

- [ ] **Day 3** — Streamlit App + LLM Layer
  - [ ] LangChain + Gemini narrative generation
  - [ ] Streamlit interface with match selector
  - [ ] Deploy on Hugging Face Spaces
  - [ ] Live prediction updates per matchday

- [ ] **During the tournament** — Live predictions
  - [ ] Pre-match analysis for every group game
  - [ ] Post-match: model hit rate tracker
  - [ ] Elo updates after each result

---

## The Backstory

I'm a Geophysical Engineer turned AI/ML Engineer.

In geophysics, the first rule is: **garbage in → garbage out**.
Before you touch a model, you separate signal from noise.

That rule is why this project filters 194 of 680 Qatar 2022 players
before computing the Player Impact Score. A striker who played 12 minutes
doesn't have a shooting profile — he has a high-variance anomaly.

The same principle is why we use log-loss instead of accuracy,
leave-one-tournament-out instead of random split,
and feature importance analysis instead of just reporting the final number.

Building this live. Predictions drop before June 11.

---

## Author

**Yosmely Bermúdez** · AI/ML Engineer  
[LinkedIn](https://linkedin.com/in/yosmely-bermudez) · [Portfolio](https://yosmelybermudez.github.io/)

---

*Built with real data, honest metrics, and a lot of football fever. ⚽*
