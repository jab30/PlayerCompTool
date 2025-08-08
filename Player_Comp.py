# ncaa_player_comp_tool.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="NCAA Player Comp Tool",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all NCAA data files"""
    try:
        data_2025 = pd.read_csv('NCAA Cards 25 - 2025_Data.csv')
        data_2024 = pd.read_csv('NCAA Cards 25 - 2024_Data.csv')
        data_2023 = pd.read_csv('NCAA Cards 25 - 2023_Data.csv')
        data_2022 = pd.read_csv('NCAA Cards 25 - 2022_Data.csv')

        # Clean data to avoid Arrow conversion issues
        for df in [data_2025, data_2024, data_2023, data_2022]:
            if 'playerFullName' in df.columns:
                df['playerFullName'] = df['playerFullName'].astype(str)
            for col in ['newestTeamName', 'pos']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        # Add year column to each dataset
        data_2025['Year'] = 2025
        data_2024['Year'] = 2024
        data_2023['Year'] = 2023
        data_2022['Year'] = 2022

        return data_2025, data_2024, data_2023, data_2022
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


def clean_stat_value(value):
    """Clean and normalize statistical values"""
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    if isinstance(value, str):
        if '%' in value:
            try:
                return float(value.replace('%', '')) / 100.0
            except:
                return np.nan
        try:
            return float(value)
        except:
            return np.nan
    return float(value) if not pd.isna(value) else np.nan


def get_comparison_stats():
    """Define stats used for player comparison"""
    # Keep outcomes in the set but we will weight them very low
    return [
        'AVG', 'wOBA', 'xWOBA', '2B', 'HR', 'K%', 'BB%', 'ExitVel', 'Air EV',
        '90thExitVel', 'MinMxExitVel', 'LaunchAng', 'LA10-30%', 'EV95+LA10-30%',
        'Swing%', 'InZoneSwing%', 'Z-Contact%', 'Chase%', 'SwStrk%', 'Miss vs 93+ FB',
        'xSLG vs 93+ FB', 'Miss vs 83+Spin', 'xSLG vs 83+ Spin', 'O-Contact%',
        'HardHit%', 'Contact%'
    ]


def get_percentiles_data():
    """Percentile reference data for radar chart only"""
    percentiles = {
        'K%': {'0th': 2.2, '50th': 17.7, '100th': 40.6},
        'BB%': {'0th': 2.5, '50th': 11.0, '100th': 26.0},
        'Air EV': {'0th': 50.7, '50th': 87.4, '100th': 104.0},
        'LA10-30%': {'0th': 0.0, '50th': 29.2, '100th': 72.7},
        'EV95+LA10-30%': {'0th': 0.0, '50th': 15.4, '100th': 50.0},
        'Swing%': {'0th': 25.9, '50th': 43.0, '100th': 61.7},
        'InZoneSwing%': {'0th': 36.1, '50th': 65.9, '100th': 90.7},
        'Z-Contact%': {'0th': 61.3, '50th': 85.6, '100th': 97.7},
        'Chase%': {'0th': 2.5, '50th': 24.1, '100th': 44.7},
        'SwStrk%': {'0th': 1.4, '50th': 9.2, '100th': 21.1},
        'Miss vs 93+ FB': {'0th': 0.0, '50th': 21.4, '100th': 100.0},
        'xSLG vs 93+ FB': {'0th': 0.0, '50th': 0.2, '100th': 3.4},
        'Miss vs 83+Spin': {'0th': 0.0, '50th': 34.5, '100th': 100.0},
        'xSLG vs 83+ Spin': {'0th': 0.0, '50th': 0.2, '100th': 2.0},
        'O-Contact%': {'0th': 24.2, '50th': 60.7, '100th': 94.4},
        'HardHit%': {'0th': 0.0, '50th': 36.0, '100th': 100.0},
        'Contact%': {'0th': 54.0, '50th': 78.6, '100th': 94.7},
        '90thExitVel': {'0th': 75.1, '50th': 102.0, '100th': 115.9},
        'MinMxExitVel': {'0th': 71.0, '50th': 106.6, '100th': 121.1},
        'LaunchAng': {'0th': -18.7, '50th': 10.1, '100th': 36.9}
    }
    return percentiles


def calculate_percentile_rank(value, stat_name, percentiles_data):
    """Percentile rank for radar chart"""
    if stat_name not in percentiles_data:
        return 50
    min_val = percentiles_data[stat_name]['0th']
    median_val = percentiles_data[stat_name]['50th']
    max_val = percentiles_data[stat_name]['100th']
    if pd.isna(value) or value == '' or value == '-':
        return 50
    clean_val = clean_stat_value(value)
    if pd.isna(clean_val):
        return 50
    if isinstance(value, str) and '%' in str(value):
        clean_val = clean_val * 100
    if max_val == min_val:
        return 50
    if clean_val <= min_val:
        return 0
    elif clean_val <= median_val:
        if median_val == min_val:
            return 50
        return 0 + (50 * (clean_val - min_val) / (median_val - min_val))
    elif clean_val <= max_val:
        if max_val == median_val:
            return 50
        return 50 + (50 * (clean_val - median_val) / (max_val - median_val))
    else:
        return 100


def prepare_comparison_data(*datasets):
    """Prepare historical data for comparison from selected datasets"""
    valid_datasets = [df for df in datasets if df is not None and len(df) > 0]
    if not valid_datasets:
        return pd.DataFrame()
    historical_data = pd.concat(valid_datasets, ignore_index=True)

    column_mappings = {
        'Z-Swing': 'InZoneSwing%',
        'O-Swing%': 'Chase%',
        'Miss vs Spin 83+': 'Miss vs 83+Spin',
    }

    # Map names if needed
    for old_col, new_col in column_mappings.items():
        if old_col in historical_data.columns:
            if new_col not in historical_data.columns:
                historical_data[new_col] = historical_data[old_col]
            else:
                mask = historical_data[new_col].isin(['-', '', np.nan]) | historical_data[new_col].isna()
                historical_data.loc[mask, new_col] = historical_data.loc[mask, old_col]

    # Create placeholders for stats that do not exist in older years
    missing_stats_2024 = ['O-Contact%', 'HardHit%', 'Contact%']
    for stat in missing_stats_2024:
        if stat not in historical_data.columns:
            historical_data[stat] = np.nan

    return historical_data


# -----------------------------
# New: standardization and reliability helpers
# -----------------------------

def fit_standardization(historical_data, comparison_stats):
    """Compute means and stds for each stat on historical data for z-scores"""
    means = {}
    stds = {}
    for s in comparison_stats:
        if s in historical_data.columns:
            col = pd.to_numeric(historical_data[s], errors="coerce")
            means[s] = np.nanmean(col)
            std = np.nanstd(col)
            stds[s] = std if std and std > 1e-8 else 1.0
    return means, stds


def zscore(value, mean, std):
    v = clean_stat_value(value)
    if pd.isna(v):
        return np.nan
    return (v - mean) / std


# You can adapt these to your real column names for attempts or exposures
ATTEMPT_COLUMN_FALLBACKS = {
    "PA": ["PA", "PlateAppr", "PlateAppearances", "PAs"],
    "BBE": ["BattedBalls", "BBE", "Batted Balls"],
    "Seen_93plus_FB": ["Seen_93plus_FB", "FB_93plus_Seen", "FB93Seen"],
    "Seen_83plus_Spin": ["Seen_83plus_Spin", "Spin_83plus_Seen", "Spin83Seen"],
}

def get_first_available(row, keys):
    for k in keys:
        if k in row and not pd.isna(row.get(k)):
            return row.get(k)
    return np.nan


def stat_attempts(row, stat):
    """Best guess attempts driver per stat"""
    if stat in ["K%", "BB%", "Swing%", "InZoneSwing%", "Z-Contact%", "Chase%", "SwStrk%", "O-Contact%", "Contact%"]:
        return get_first_available(row, ATTEMPT_COLUMN_FALLBACKS["PA"])
    if stat in ["ExitVel", "Air EV", "90thExitVel", "EV95+LA10-30%", "HardHit%", "LaunchAng", "LA10-30%"]:
        return get_first_available(row, ATTEMPT_COLUMN_FALLBACKS["BBE"])
    if stat in ["Miss vs 93+ FB", "xSLG vs 93+ FB"]:
        return get_first_available(row, ATTEMPT_COLUMN_FALLBACKS["Seen_93plus_FB"])
    if stat in ["Miss vs 83+Spin", "xSLG vs 83+ Spin"]:
        return get_first_available(row, ATTEMPT_COLUMN_FALLBACKS["Seen_83plus_Spin"])
    if stat in ["AVG", "2B", "HR", "wOBA", "xWOBA"]:
        return get_first_available(row, ATTEMPT_COLUMN_FALLBACKS["PA"])
    return np.nan


K_VALUES = {
    "plate": 150,    # PA-based stats
    "bbe": 80,       # batted-ball based
    "pitch": 75,     # pitch exposure based
    "realized": 300  # realized outcomes
}

def k_for(stat):
    if stat in ["K%","BB%","Swing%","InZoneSwing%","Z-Contact%","Chase%","SwStrk%","O-Contact%","Contact%"]:
        return K_VALUES["plate"]
    if stat in ["ExitVel","Air EV","90thExitVel","EV95+LA10-30%","HardHit%","LaunchAng","LA10-30%"]:
        return K_VALUES["bbe"]
    if stat in ["Miss vs 93+ FB","xSLG vs 93+ FB","Miss vs 83+Spin","xSLG vs 83+ Spin"]:
        return K_VALUES["pitch"]
    if stat in ["AVG","2B","HR","wOBA","xWOBA"]:
        return K_VALUES["realized"]
    return 150


def reliability_w(n, k):
    if pd.isna(n):
        return 0.0
    return float(n) / float(n + k)


# New weights: downweight realized outcomes, emphasize skills
NEW_WEIGHTS = {
    # contact quality
    "Air EV": 2.6, "90thExitVel": 2.6, "EV95+LA10-30%": 2.5, "ExitVel": 2.2, "HardHit%": 2.2, "MinMxExitVel": 2.0,
    # decisions and contact
    "Z-Contact%": 2.6, "Chase%": 2.6, "SwStrk%": 2.4, "InZoneSwing%": 2.2, "Contact%": 2.2, "O-Contact%": 2.0, "BB%": 2.0, "K%": 2.0,
    # pitch recognition
    "Miss vs 93+ FB": 2.3, "Miss vs 83+Spin": 2.3,
    # expected results
    "xWOBA": 2.2, "xSLG vs 93+ FB": 1.9, "xSLG vs 83+ Spin": 1.9, "wOBA": 1.2,
    # batted-ball shape
    "LaunchAng": 1.4, "LA10-30%": 1.8,
    # realized outcomes very low
    "AVG": 0.5, "2B": 0.6, "HR": 0.8,
    # swing level aggregate
    "Swing%": 1.6,
}

def stat_weight(stat):
    return NEW_WEIGHTS.get(stat, 1.0)


def calculate_player_similarity(target_player, historical_data, comparison_stats, stat_means, stat_stds):
    """Similarity using z-scores, reliability weighting, and robust clipping"""
    # Clean target player stats to numeric once
    target_stats = {}
    for stat in comparison_stats:
        if stat in target_player:
            target_stats[stat] = clean_stat_value(target_player[stat])

    similarities = []

    for _, historical_player in historical_data.iterrows():
        if 'playerId' in historical_player and 'playerId' in target_player:
            if historical_player['playerId'] == target_player['playerId']:
                continue

        total_similarity = 0.0
        valid_weight_sum = 0.0
        stat_similarities = {}

        for stat in comparison_stats:
            if stat not in target_stats or stat not in historical_player:
                continue

            target_val = target_stats[stat]
            hist_val = clean_stat_value(historical_player[stat])

            if pd.isna(target_val) or pd.isna(hist_val):
                continue

            mean = stat_means.get(stat)
            std = stat_stds.get(stat)
            if mean is None or std is None:
                continue

            t_z = zscore(target_val, mean, std)
            h_z = zscore(hist_val, mean, std)
            if pd.isna(t_z) or pd.isna(h_z):
                continue

            # robust absolute difference on z-scale
            z_diff = abs(t_z - h_z)
            z_diff = min(z_diff, 2.5)  # clip huge gaps

            # reliability per player-stat
            n_hist = stat_attempts(historical_player, stat)
            n_targ = stat_attempts(target_player, stat)
            rw_hist = reliability_w(n_hist, k_for(stat))
            rw_targ = reliability_w(n_targ, k_for(stat))
            rw = (rw_hist + rw_targ) / 2.0

            w = stat_weight(stat) * rw
            if w <= 0:
                continue

            # convert distance to similarity in [0,1]
            similarity = 1.0 - (z_diff / 2.5)
            similarity = max(0.0, similarity)

            total_similarity += similarity * w
            valid_weight_sum += w
            stat_similarities[stat] = similarity

        # Require 60 percent of total possible weight available
        total_possible_weight = sum(stat_weight(s) for s in comparison_stats)
        coverage = (valid_weight_sum / total_possible_weight) * 100 if total_possible_weight > 0 else 0

        if valid_weight_sum >= (0.60 * total_possible_weight):
            avg_similarity = total_similarity / valid_weight_sum if valid_weight_sum > 0 else 0.0
            similarities.append({
                'player': historical_player.get('playerFullName', 'Unknown'),
                'team': historical_player.get('newestTeamName', 'Unknown'),
                'position': historical_player.get('pos', 'Unknown'),
                'year': historical_player.get('Year', 'Unknown'),
                'similarity_score': avg_similarity,
                'valid_stats': len(stat_similarities),
                'total_possible_stats': len(comparison_stats),
                'coverage_pct': round(coverage, 1),
                'stat_breakdown': stat_similarities,
                'player_data': historical_player
            })

    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities


def create_comparison_chart(target_player, comp_players, stats_to_show):
    """Radar chart comparing target player with comps using percentiles"""
    percentiles_data = get_percentiles_data()
    available_stats = [stat for stat in stats_to_show if stat in percentiles_data]
    if not available_stats:
        st.warning("No percentile data available for selected stats")
        return None

    categories = available_stats + [available_stats[0]]
    fig = go.Figure()

    # Target
    target_percentiles = []
    for stat in available_stats:
        percentile_rank = calculate_percentile_rank(target_player.get(stat), stat, percentiles_data)
        target_percentiles.append(percentile_rank)
    target_percentiles.append(target_percentiles[0])

    fig.add_trace(go.Scatterpolar(
        r=target_percentiles,
        theta=categories,
        fill='toself',
        name=f"{target_player.get('playerFullName', 'Target Player')} (2025)",
        line_color='red',
        fillcolor='rgba(255,0,0,0.15)',
        line_width=3
    ))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, comp in enumerate(comp_players[:5]):
        comp_percentiles = []
        for stat in available_stats:
            percentile_rank = calculate_percentile_rank(comp['player_data'].get(stat), stat, percentiles_data)
            comp_percentiles.append(percentile_rank)
        comp_percentiles.append(comp_percentiles[0])

        # simple rgba from hex
        hexc = colors[i % len(colors)]
        r = int(hexc[1:3], 16)
        g = int(hexc[3:5], 16)
        b = int(hexc[5:7], 16)

        fig.add_trace(go.Scatterpolar(
            r=comp_percentiles,
            theta=categories,
            fill='toself',
            name=f"{comp['player']} ({comp['year']})",
            line_color=hexc,
            fillcolor=f'rgba({r},{g},{b},0.10)',
            line_width=2,
            opacity=0.8
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0th', '25th', '50th', '75th', '100th'],
                tickfont=dict(color='black', size=11),
                gridcolor='lightgray',
                title=dict(text="Percentile Rank", font=dict(color='black', size=12))
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='black'),
                rotation=90
            )
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='black', size=11),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        title={
            'text': "Player Comparison - Percentile Rankings<br><sub>Higher values = better performance relative to 2025 NCAA population</sub>",
            'x': 0.5,
            'font': {'size': 16, 'color': 'black'}
        },
        height=700,
        width=700,
        font=dict(size=11, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig


def main():
    st.title("NCAA Player Comparison Tool")
    st.markdown("Find the most statistically similar players from 2022-2024 to any 2025 player")

    # Load data
    data_2025, data_2024, data_2023, data_2022 = load_data()
    if data_2025 is None:
        st.error("Could not load data files. Please ensure CSV files are in the correct location.")
        return

    # Sidebar controls
    st.sidebar.header("Player Selection")

    player_names = sorted([name for name in data_2025['playerFullName'].dropna().unique() if name])

    selected_player_name = st.sidebar.selectbox(
        "Choose a 2025 player:",
        player_names,
        index=0 if player_names else None
    )

    if not selected_player_name:
        st.warning("No players available for selection")
        return

    target_player = data_2025[data_2025['playerFullName'] == selected_player_name].iloc[0]

    # Comparison controls
    st.sidebar.header("Comparison Settings")
    num_comps = st.sidebar.slider("Number of comparisons to show:", 1, 20, 10)
    min_games = st.sidebar.slider("Minimum games played:", 0, 50, 10)

    year_options = st.sidebar.multiselect(
        "Include players from years:",
        [2024, 2023, 2022],
        default=[2024, 2023, 2022]
    )

    position_filter = st.sidebar.multiselect(
        "Filter by position:",
        ['All'] + sorted(pd.concat([data_2024, data_2023, data_2022])['pos'].dropna().unique().tolist()),
        default=['All']
    )

    # Compute comparisons
    if st.sidebar.button("Find Player Comparisons") or 'comparisons' not in st.session_state:
        if not year_options:
            st.error("Please select at least one year for historical comparisons")
            return

        with st.spinner("Calculating player similarities with z-scores and reliability weighting..."):
            historical_datasets = []
            if 2024 in year_options and data_2024 is not None:
                historical_datasets.append(data_2024)
            if 2023 in year_options and data_2023 is not None:
                historical_datasets.append(data_2023)
            if 2022 in year_options and data_2022 is not None:
                historical_datasets.append(data_2022)

            if not historical_datasets:
                st.error("No valid historical data found for selected years")
                return

            historical_data = prepare_comparison_data(*historical_datasets)

            # Optional filters
            if min_games > 0 and 'G' in historical_data.columns:
                historical_data = historical_data[historical_data['G'] >= min_games]

            if 'All' not in position_filter and position_filter:
                historical_data = historical_data[historical_data['pos'].isin(position_filter)]

            comparison_stats = get_comparison_stats()

            # Fit z-score parameters on the selected historical set
            stat_means, stat_stds = fit_standardization(historical_data, comparison_stats)

            # Calculate similarities
            similarities = calculate_player_similarity(
                target_player,
                historical_data,
                comparison_stats,
                stat_means,
                stat_stds
            )

            st.session_state.comparisons = similarities[:num_comps]
            st.session_state.target_player = target_player
            st.session_state.comparison_stats = comparison_stats
            st.session_state.selected_years = year_options

            st.success(
                f"Found {len(similarities)} potential matches from {len(historical_data)} players across {len(year_options)} years"
            )

            if st.sidebar.checkbox("Show model weights and standardization"):
                st.write("Model: Rule-based weighted similarity on z-scores with reliability weighting")
                st.json({s: stat_weight(s) for s in comparison_stats})
                st.write("Standardization means")
                st.json(stat_means)
                st.write("Standardization stds")
                st.json(stat_stds)

    # Display results
    if 'comparisons' in st.session_state:
        st.header(f"Top {len(st.session_state.comparisons)} Most Similar Players")

        years_searched = ", ".join(map(str, st.session_state.selected_years))
        st.info(f"Emphasizing EV, decisions, contact, and expected power. Searched years: {years_searched}")

        comp_data = []
        for i, comp in enumerate(st.session_state.comparisons):
            comp_data.append({
                'Rank': i + 1,
                'Player': comp['player'],
                'Team': comp['team'],
                'Position': comp['position'],
                'Year': comp['year'],
                'Similarity Score': f"{comp['similarity_score'] * 100:.1f}%",
                'Stats Coverage': f"{comp['coverage_pct']:.1f}%",
                'Valid Stats': f"{comp['valid_stats']}/{comp['total_possible_stats']}"
            })

        comp_df = pd.DataFrame(comp_data).astype(str)
        st.dataframe(comp_df, use_container_width=True)

        # Statistical comparison section
        st.header("Statistical Comparison")

        available_stats = st.session_state.comparison_stats
        percentile_stats = list(get_percentiles_data().keys())
        available_for_chart = [stat for stat in available_stats if stat in percentile_stats]

        selected_stats = st.multiselect(
            "Select stats to compare (percentile rankings available):",
            available_for_chart,
            default=available_for_chart[:8] if len(available_for_chart) >= 8 else available_for_chart
        )

        if selected_stats and len(st.session_state.comparisons) > 0:
            st.subheader("Detailed Statistical Comparison")

            detailed_comp = {'Stat': selected_stats}
            detailed_comp[st.session_state.target_player['playerFullName']] = [
                str(st.session_state.target_player.get(stat, 'N/A')) for stat in selected_stats
            ]

            for comp in st.session_state.comparisons[:5]:
                comp_name = f"{comp['player']} ({comp['year']})"
                comp_values = []
                for stat in selected_stats:
                    value = comp['player_data'].get(stat, 'N/A')
                    if pd.isna(value) or value == '' or value is None:
                        if comp['year'] == 2024 and stat in ['O-Contact%', 'HardHit%', 'Contact%']:
                            value = 'N/A (2024)'
                        else:
                            value = 'N/A'
                    comp_values.append(str(value))
                detailed_comp[comp_name] = comp_values

            detailed_df = pd.DataFrame(detailed_comp)
            for col in detailed_df.columns:
                detailed_df[col] = detailed_df[col].astype(str)
            st.dataframe(detailed_df, use_container_width=True)

            if len(selected_stats) >= 3:
                st.subheader("Player Comparison - Percentile Rankings")
                st.info("Chart shows percentile rankings relative to all 2025 NCAA players. Higher values = better performance.")
                radar_fig = create_comparison_chart(
                    st.session_state.target_player,
                    st.session_state.comparisons,
                    selected_stats
                )
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning("Please select at least 3 stats to display the radar chart")


if __name__ == "__main__":
    main()
