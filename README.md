# Recruitment Support - FBref Data Pipeline

A Python-based data pipeline for football recruitment analytics using FBref data. This project processes player statistics, computes role-based scores, generates player shortlists, and identifies comparable players for specific tactical roles.

## Overview

This project extracts, transforms, and analyzes player performance data from FBref to support football recruitment decisions. It evaluates players against specific tactical roles (Ball-Playing CB, Deep-Lying Playmaker, Winger Creator) using weighted scoring algorithms and provides actionable insights through:

- **Role Scoring**: Evaluates players against defined tactical roles with weighted metrics
- **Player Shortlists**: Generates ranked lists of top candidates per role
- **Comparables Analysis**: Identifies similar players using machine learning similarity metrics
- **Tableau-Ready Exports**: Produces dimension and fact tables optimized for visualization

## Features

- **Multi-League Data Extraction**: Supports multiple leagues and seasons via FBref
- **Comprehensive Metrics**: Processes standard, passing, defense, possession, shooting, and advanced stats
- **Percentile Ranking**: Contextualizes player performance within league/season/position groups
- **Role-Based Analytics**: Configurable role definitions with must-have criteria and weighted scoring
- **Risk Flagging**: Identifies potential concerns (age, playing time, error rates)
- **Data Mart Architecture**: Organized dimensional modeling for analytics consumption

## Installation

### Requirements

- Python >= 3.12, < 3.13
- pip or compatible package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd recruitment-support-v1
```

2. Install the package and dependencies:
```bash
pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

### Dependencies

- `soccerdata>=1.8.0` - FBref data extraction
- `pandas>=2.2` - Data manipulation
- `numpy>=2.0` - Numerical operations
- `pyarrow>=16.0` - Parquet file support
- `pyyaml>=6.0` - Configuration file parsing
- `scikit-learn>=1.5` - Machine learning utilities for similarity calculations
- `typer>=0.12` - CLI framework
- `rich>=13.0` - Enhanced terminal output

## Configuration

The pipeline is configured via YAML files in the `configs/` directory.

### Main Configuration (`configs/v1.yaml`)

```yaml
project:
  name: recruitment-support-fbref
  version: v1

fbref:
  leagues: ["ENG-Premier League"]
  seasons: ["2425"]
  data_dir: "data/soccerdata_cache/FBref"
  no_cache: false
  no_store: false

filters:
  min_minutes: 900
  league_scope: "ENG-Premier League"
  same_league_only: true

roles:
  role_defs_path: "configs/roles_v1.yaml"
  position_map_path: "configs/position_map.yaml"

exports:
  out_dir: "data/exports/tableau"
```

### Role Definitions (`configs/roles_v1.yaml`)

Defines tactical roles with:
- **Position buckets**: Eligibility criteria (CB, DMCM, WIDE)
- **Must-have filters**: Minimum thresholds (minutes, key metrics)
- **Weighted scoring**: Relative importance of metrics per role
- **Negative metrics**: Metrics where lower values are better

Currently defined roles:
- **BPCB** (Ball-Playing CB): Center-back with strong distribution and progression
- **DLP** (Deep-Lying Playmaker): Midfielder focused on deep-lying creative passing
- **WCR** (Winger Creator): Wide attacker with dribbling and chance creation

## Usage

The pipeline consists of three main scripts that should be run in sequence:

### 1. Run Pipeline (`scripts/run_pipeline.py`)

Extracts data from FBref, processes player statistics, and computes role scores.

```bash
python scripts/run_pipeline.py
```

**What it does:**
- Downloads/caches FBref data for specified leagues and seasons
- Flattens and merges multiple stat tables (standard, passing, defense, etc.)
- Applies data cleaning and filtering (minimum minutes, position buckets)
- Calculates canonical metrics (per-90 rates, percentages)
- Computes percentile rankings within league/season/position groups
- Applies role scoring algorithms based on `configs/roles_v1.yaml`

**Output:**
- `data/intermediate/player_season_base.parquet` - Raw merged data
- `data/intermediate/player_season_clean.parquet` - Cleaned and filtered data
- `data/intermediate/player_season_scored.parquet` - Data with percentiles and role scores

### 2. Build Marts (`scripts/build_marts.py`)

Creates dimensional data marts and exports to Tableau format.

```bash
python scripts/build_marts.py
```

**What it does:**
- Generates unique player and team identifiers
- Builds dimension tables (`dim_player`, `dim_team`)
- Creates fact tables (`fact_player_season`, `fact_role_profile_card`)
- Exports all tables as both Parquet and CSV formats

**Output:**
- `data/marts/dim_player.parquet` & `.csv`
- `data/marts/dim_team.parquet` & `.csv`
- `data/marts/fact_player_season.parquet` & `.csv`
- `data/marts/fact_role_profile_card.parquet` & `.csv`
- CSV files also written to `data/exports/tableau/`

### 3. Build Comparables (`scripts/build_comparables.py`)

Identifies similar players for each role using cosine similarity on percentile features.

```bash
python scripts/build_comparables.py [--top-n 10]
```

**Options:**
- `--top-n`: Number of comparables per player (default: 10)

**What it does:**
- For each eligible player in each role, finds most similar players
- Uses RobustScaler normalization and cosine distance
- Generates reason codes explaining why players are similar
- Filters to same league/season scope (v1)

**Output:**
- `data/marts/fact_comparables.parquet`
- `data/exports/tableau/fact_comparables.csv`

### 4. Build Shortlist (`scripts/build_shortlist.py`)

Generates ranked shortlists of top candidates per role.

```bash
python scripts/build_shortlist.py [--top-n 50]
```

**Options:**
- `--top-n`: Number of players per role shortlist (default: 50)

**What it does:**
- Ranks eligible players by role score
- Computes sub-scores (progression, creation, defending, security, finishing)
- Adds risk flags (low minutes, high errors, age concerns)
- Includes evidence strings (key metrics with percentiles)
- Filters to top N per role

**Output:**
- `data/marts/fact_shortlist.parquet`
- `data/exports/tableau/fact_shortlist.csv`

## Project Structure

```
recruitment-support-v1/
├── configs/                 # Configuration files
│   ├── v1.yaml             # Main pipeline configuration
│   ├── roles_v1.yaml       # Role definitions and scoring weights
│   └── position_map.yaml   # Position mapping rules
├── scripts/                 # Executable pipeline scripts
│   ├── run_pipeline.py     # Data extraction and scoring
│   ├── build_marts.py      # Dimension and fact table creation
│   ├── build_comparables.py # Similarity analysis
│   └── build_shortlist.py  # Shortlist generation
├── src/rsfbref/            # Main package code
│   ├── config.py           # Configuration loading
│   ├── io/                 # Data I/O
│   │   └── fbref_reader.py # FBref data extraction wrapper
│   ├── transform/          # Data transformation
│   │   ├── flatten.py      # Column name flattening
│   │   ├── player_season.py # Stat table merging
│   │   └── clean_player_season.py # Data cleaning and feature engineering
│   ├── features/           # Feature engineering
│   │   └── percentiles.py  # Percentile calculations
│   ├── analytics/          # Analytics algorithms
│   │   ├── roles.py        # Role scoring logic
│   │   ├── comparables.py  # Player similarity analysis
│   │   └── shortlist.py    # Shortlist generation
│   ├── marts/              # Data mart builders
│   │   ├── build_dims.py   # Dimension table creation
│   │   └── build_facts.py  # Fact table creation
│   └── export/             # Export utilities
│       └── tableau.py      # Tableau CSV export
├── data/                   # Data directory (gitignored)
│   ├── intermediate/       # Intermediate processing files
│   ├── marts/              # Final data marts (Parquet)
│   ├── exports/            # Export files (CSV)
│   └── soccerdata_cache/   # FBref data cache
└── pyproject.toml          # Package configuration
```

## Data Pipeline Flow

```
1. FBref Data Extraction
   └─> Multiple stat tables (standard, passing, defense, etc.)

2. Data Transformation
   ├─> Flatten column names (MultiIndex → snake_case)
   ├─> Merge tables on (league, season, team, player)
   └─> Clean and filter (position buckets, minimum minutes)

3. Feature Engineering
   ├─> Calculate per-90 metrics
   ├─> Compute percentages and rates
   └─> Generate percentile rankings (by league/season/position)

4. Role Scoring
   ├─> Apply must-have filters
   ├─> Calculate weighted scores from percentiles
   └─> Generate role eligibility flags

5. Analytics
   ├─> Build comparables (cosine similarity)
   ├─> Generate shortlists (ranked by score)
   └─> Compute risk flags and sub-scores

6. Data Marts
   ├─> Create dimension tables (players, teams)
   ├─> Create fact tables (player seasons, role profiles, etc.)
   └─> Export to Parquet and CSV formats
```

## Metrics Computed

### Passing & Progression
- `pass_cmp_pct`: Pass completion percentage
- `passes_att_p90`: Passes attempted per 90 minutes
- `prog_passes_p90`: Progressive passes per 90
- `passes_final_third_p90`: Passes into final third per 90
- `long_pass_cmp_pct`: Long pass completion percentage

### Chance Creation
- `key_passes_p90`: Key passes per 90
- `xa_p90`: Expected assists per 90
- `sca_p90`: Shot-creating actions per 90
- `crosses_pa_p90`: Crosses into penalty area per 90

### Defending & Duels
- `tkl_int_p90`: Tackles + interceptions per 90
- `clr_p90`: Clearances per 90
- `errors_p90`: Errors leading to shots per 90
- `aerial_win_pct`: Aerial duel win percentage

### Carrying & Dribbling
- `prog_carries_p90`: Progressive carries per 90
- `carries_pa_p90`: Carries into penalty area per 90
- `succ_takeons_p90`: Successful take-ons per 90
- `takeon_succ_pct`: Take-on success percentage

### Risk Metrics
- `mis_dis_p90`: Miscontrols + dispossessions per 90
- `fouls_p90`: Fouls committed per 90

### Finishing
- `Per_90_Minutes_npxG`: Non-penalty expected goals per 90

All metrics are also available as percentile ranks (`pct_*`) within league/season/position buckets.

## Position Buckets

Players are categorized into position buckets for role eligibility:

- **GK**: Goalkeepers (excluded from v1 roles)
- **CB**: Center-backs (for BPCB role)
- **DMCM**: Defensive/Central midfielders (for DLP role)
- **WIDE**: Wide forwards/wingers (for WCR role)
- **OTHER**: Players not fitting above categories

## Output Schema

### Dimension Tables

**dim_player**
- `player_id`: Unique player identifier (SHA1 hash)
- `player_name`: Player name
- `nation`: Nationality
- `age`: Age
- `born`: Birth date
- `position_raw`: Raw FBref position
- `position_bucket`: Categorized position

**dim_team**
- `team_id`: Unique team identifier (SHA1 hash)
- `team_name`: Team name
- `league`: League name
- `season`: Season identifier

### Fact Tables

**fact_player_season**
- All dimension keys (player_id, team_id, league, season)
- Playing time (minutes, nineties)
- All canonical metrics
- All percentile columns (`pct_*`)
- Role scores (`score_BPCB`, `score_DLP`, `score_WCR`)

**fact_role_profile_card**
- Long-format table for visualization
- One row per (player, team, season, role, KPI)
- Contains `kpi_name`, `kpi_value`, `kpi_pct`, `role_score`

**fact_comparables**
- `anchor_player_id`, `comparable_player_id`
- `role_id`, `distance`, `rank`
- `reason_1`, `reason_2`, `reason_3`: Similarity explanations

**fact_shortlist**
- Top N players per role
- `total_score`: Overall role score
- `sub_progression`, `sub_creation`, `sub_defending`, etc.: Sub-scores
- `risk_flags`: Comma-separated risk indicators
- `risk_count`: Number of risk flags
- `evidence_1` through `evidence_5`: Key metric strings

## Customization

### Adding New Roles

1. Edit `configs/roles_v1.yaml` to add role definition:
   - Define `role_id`, `role_name`, `position_bucket`
   - Set `must_have` criteria
   - Configure `weights` for scoring
   - Specify `negative_metrics` if applicable

2. Update `scripts/build_comparables.py` and `scripts/build_shortlist.py` to include new role_id in loops

3. Add role-specific features in `src/rsfbref/analytics/comparables.py` if needed

### Adding New Metrics

1. Add metric calculation in `src/rsfbref/transform/clean_player_season.py`
2. Include metric in `metric_cols` list in `scripts/run_pipeline.py`
3. Add to appropriate role definitions in `configs/roles_v1.yaml`

### Changing Data Sources

Modify `configs/v1.yaml` to:
- Change `leagues` list
- Adjust `seasons` list
- Update `data_dir` for cache location

## Data Caching

FBref data is cached locally in `data/soccerdata_cache/FBref/` to avoid re-downloading. To refresh data:
- Set `no_cache: true` in config (will re-download but not save)
- Delete cache directory and re-run
- Set `no_store: true` to disable caching entirely

## Testing

Run tests with pytest:
```bash
pytest tests/
```
