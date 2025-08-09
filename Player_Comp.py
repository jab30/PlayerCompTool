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
        # Load all datasets with correct file paths
        data_2025 = pd.read_csv('NCAA Cards 25 - 2025_Data.csv')
        data_2024 = pd.read_csv('NCAA Cards 25 - 2024_Data.csv')
        data_2023 = pd.read_csv('NCAA Cards 25 - 2023_Data.csv')
        data_2022 = pd.read_csv('NCAA Cards 25 - 2022_Data.csv')

        # Clean data to avoid Arrow conversion issues
        for df in [data_2025, data_2024, data_2023, data_2022]:
            # Convert playerFullName to string to avoid mixed types
            if 'playerFullName' in df.columns:
                df['playerFullName'] = df['playerFullName'].astype(str)
            # Convert other text columns to string
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
            return float(value.replace('%', '')) / 100
        try:
            return float(value)
        except:
            return np.nan

    return float(value) if not pd.isna(value) else np.nan


def get_comparison_stats():
    """Define ALL stats used for player comparison - using every column specified"""
    return [
        'AVG', 'wOBA', 'xWOBA', '2B', 'HR', 'K%', 'BB%', 'ExitVel', 'Air EV',
        '90thExitVel', 'MinMxExitVel', 'LaunchAng', 'LA10-30%', 'EV95+LA10-30%',
        'Swing%', 'InZoneSwing%', 'Z-Contact%', 'Chase%', 'SwStrk%', 'Miss vs 93+ FB',
        'xSLG vs 93+ FB', 'Miss vs 83+Spin', 'xSLG vs 83+ Spin', 'O-Contact%',
        'HardHit%', 'Contact%'
    ]


def get_percentiles_data():
    """Return the percentile reference data from the 2025 dataset analysis"""
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
    """Calculate the percentile rank for a given stat value using 0th, 50th, and 100th percentiles"""
    if stat_name not in percentiles_data:
        return 50  # Default to 50th percentile if stat not found

    # Get the three reference points
    min_val = percentiles_data[stat_name]['0th']  # Minimum value (0th percentile)
    median_val = percentiles_data[stat_name]['50th']  # Median value (50th percentile)
    max_val = percentiles_data[stat_name]['100th']  # Maximum value (100th percentile)

    if pd.isna(value) or value == '' or value == '-':
        return 50  # Default to 50th percentile for missing values

    # Clean the value
    clean_val = clean_stat_value(value)
    if pd.isna(clean_val):
        return 50

    # Convert percentages back to actual values for comparison
    if isinstance(value, str) and '%' in str(value):
        clean_val = clean_val * 100  # Convert back to percentage scale

    # Handle edge cases where min = max (no variance)
    if max_val == min_val:
        return 50

    # Calculate percentile rank using linear interpolation
    if clean_val <= min_val:
        return 0
    elif clean_val <= median_val:
        # Linear interpolation between 0th and 50th percentile
        if median_val == min_val:
            return 50
        return 0 + (50 * (clean_val - min_val) / (median_val - min_val))
    elif clean_val <= max_val:
        # Linear interpolation between 50th and 100th percentile
        if max_val == median_val:
            return 50
        return 50 + (50 * (clean_val - median_val) / (max_val - median_val))
    else:
        # Above 100th percentile - cap at 100th percentile
        return 100


def prepare_comparison_data(*datasets):
    """Prepare historical data for comparison from selected datasets"""
    # Filter out empty datasets and combine
    valid_datasets = [df for df in datasets if df is not None and len(df) > 0]

    if not valid_datasets:
        return pd.DataFrame()

    historical_data = pd.concat(valid_datasets, ignore_index=True)

    # Handle column name differences across years
    # 2024 data has different column names and missing stats
    column_mappings = {
        # 2024 specific mappings
        'Z-Swing': 'InZoneSwing%',  # 2024 uses Z-Swing instead of InZoneSwing%
        'O-Swing%': 'Chase%',  # 2024 uses O-Swing% for Chase%
        'Miss vs Spin 83+': 'Miss vs 83+Spin',  # 2024 column name difference
    }

    # Debug: Print available columns for troubleshooting
    print("DEBUG - Available columns in historical data:")
    print(sorted(historical_data.columns.tolist()))

    for old_col, new_col in column_mappings.items():
        if old_col in historical_data.columns:
            print(f"DEBUG - Mapping {old_col} -> {new_col}")
            # Only map if the target column doesn't already exist or has missing data
            if new_col not in historical_data.columns:
                historical_data[new_col] = historical_data[old_col]
            else:
                # Merge the data - use new_col where it exists, fill with old_col where it's missing
                mask = historical_data[new_col].isin(['-', '', np.nan]) | historical_data[new_col].isna()
                historical_data.loc[mask, new_col] = historical_data.loc[mask, old_col]
                print(f"DEBUG - Merged {old_col} into existing {new_col} column")
        else:
            print(f"DEBUG - Column {old_col} not found in historical data")

    # Debug: Check if Chase% exists after mapping
    if 'Chase%' in historical_data.columns:
        print("DEBUG - Chase% column exists after mapping")
        chase_sample = historical_data['Chase%'].head(10).tolist()
        print(f"DEBUG - Chase% sample values: {chase_sample}")
        # Count non-missing values
        non_missing = historical_data['Chase%'][~historical_data['Chase%'].isin(['-', '', np.nan])].count()
        total = len(historical_data)
        print(f"DEBUG - Chase% non-missing values: {non_missing}/{total}")
    else:
        print("DEBUG - Chase% column still missing after mapping")

    # Handle missing stats in 2024 data by creating placeholder columns
    # This prevents "None" values in comparisons
    missing_stats_2024 = ['O-Contact%', 'HardHit%', 'Contact%']
    for stat in missing_stats_2024:
        if stat not in historical_data.columns:
            historical_data[stat] = np.nan  # Fill with NaN so they're excluded from similarity calc

    return historical_data


def calculate_player_similarity(target_player, historical_data, comparison_stats):
    """Calculate similarity scores between target player and historical players using ALL specified stats"""

    # Clean target player stats
    target_stats = {}
    for stat in comparison_stats:
        if stat in target_player:
            target_stats[stat] = clean_stat_value(target_player[stat])

    similarities = []

    for idx, historical_player in historical_data.iterrows():
        # Skip if same player (by playerId)
        if 'playerId' in historical_player and 'playerId' in target_player:
            if historical_player['playerId'] == target_player['playerId']:
                continue

        # Calculate similarity for this player using ALL stats
        valid_comparisons = 0
        total_similarity = 0
        stat_similarities = {}

        historical_stats = {}
        for stat in comparison_stats:
            if stat in historical_player:
                historical_stats[stat] = clean_stat_value(historical_player[stat])

        # Calculate weighted similarity for each stat
        for stat in comparison_stats:
            if stat in target_stats and stat in historical_stats:
                target_val = target_stats[stat]
                hist_val = historical_stats[stat]

                if not pd.isna(target_val) and not pd.isna(hist_val):
                    # Enhanced normalization with EXIT VELOCITY and PLATE DISCIPLINE as highest weights
                    if stat in ['ExitVel', 'Air EV', '90thExitVel', 'MinMxExitVel']:
                        # EXIT VELOCITY STATS - HIGHEST PRIORITY (typically 70-110 mph range)
                        if stat == 'MinMxExitVel':
                            max_possible_diff = 50.0
                        else:
                            max_possible_diff = 40.0
                        diff = min(abs(target_val - hist_val) / max_possible_diff, 1.0)
                        weight = 2.5  # HIGHEST weight for exit velocity metrics
                    elif stat in ['K%', 'BB%', 'Swing%', 'InZoneSwing%', 'Z-Contact%', 'Chase%',
                                  'SwStrk%', 'O-Contact%', 'Contact%']:
                        # PLATE DISCIPLINE STATS - HIGHEST PRIORITY (0-100% range, stored as 0-1)
                        max_possible_diff = 1.0
                        diff = abs(target_val - hist_val) / max_possible_diff
                        weight = 2.5  # HIGHEST weight for plate discipline/contact stats
                    elif stat in ['HardHit%', 'EV95+LA10-30%']:
                        # QUALITY OF CONTACT - Very high priority (combines exit velo + discipline)
                        max_possible_diff = 1.0
                        diff = abs(target_val - hist_val) / max_possible_diff
                        weight = 2.3  # Very high weight for hard contact metrics
                    elif stat in ['Miss vs 93+ FB', 'Miss vs 83+Spin']:
                        # PITCH RECOGNITION - Very high priority for plate discipline
                        max_possible_diff = 1.0
                        diff = abs(target_val - hist_val) / max_possible_diff
                        weight = 2.2  # Very high weight for pitch recognition
                    elif stat in ['AVG', 'wOBA', 'xWOBA']:
                        # Traditional batting stats - high but secondary to exit velo/discipline
                        max_possible_diff = 1.0
                        diff = abs(target_val - hist_val) / max_possible_diff
                        weight = 2.0  # High weight for key offensive stats
                    elif stat in ['2B', 'HR']:
                        # Power counting stats - high weight
                        max_val = max(abs(target_val), abs(hist_val), 1)
                        diff = abs(target_val - hist_val) / max_val
                        weight = 1.9  # High weight for power numbers
                    elif 'xSLG' in stat:
                        # Expected slugging (0.000-2.000+ range typically)
                        max_possible_diff = 2.0
                        diff = abs(target_val - hist_val) / max_possible_diff
                        weight = 1.8  # Good weight for expected performance
                    elif stat in ['LaunchAng', 'LA10-30%']:
                        # Launch angle metrics - medium-high priority
                        if stat == 'LaunchAng':
                            max_possible_diff = 70.0  # Launch angle range
                        else:
                            max_possible_diff = 1.0  # Percentage
                        diff = min(abs(target_val - hist_val) / max_possible_diff, 1.0)
                        weight = 1.6  # Medium-high weight for launch metrics
                    else:
                        # Default handling for any other stats
                        max_val = max(abs(target_val), abs(hist_val), 0.001)
                        diff = abs(target_val - hist_val) / max_val
                        weight = 1.0

                    # Convert difference to similarity (0-1, where 1 is identical)
                    similarity = max(0, 1 - diff)
                    weighted_similarity = similarity * weight

                    total_similarity += weighted_similarity
                    valid_comparisons += weight  # Use weight in denominator for proper averaging
                    stat_similarities[stat] = similarity

        # Calculate total possible weight dynamically based on available stats for this player
        def get_stat_weight(stat):
            if stat in ['ExitVel', 'Air EV', '90thExitVel', 'MinMxExitVel']:
                return 2.5
            elif stat in ['K%', 'BB%', 'Swing%', 'InZoneSwing%', 'Z-Contact%', 'Chase%', 'SwStrk%', 'O-Contact%',
                          'Contact%']:
                return 2.5
            elif stat in ['HardHit%', 'EV95+LA10-30%']:
                return 2.3
            elif stat in ['Miss vs 93+ FB', 'Miss vs 83+Spin']:
                return 2.2
            elif stat in ['AVG', 'wOBA', 'xWOBA']:
                return 2.0
            elif stat in ['2B', 'HR']:
                return 1.9
            elif 'xSLG' in stat:
                return 1.8
            elif stat in ['LaunchAng', 'LA10-30%']:
                return 1.6
            else:
                return 1.0

        # Calculate total possible weight for ALL stats
        total_possible_weight = sum(get_stat_weight(stat) for stat in comparison_stats)

        # Count actual number of stats compared (not weighted)
        actual_stats_compared = len(stat_similarities)

        # Calculate actual coverage percentage with more precision
        coverage_percentage = round((valid_comparisons / total_possible_weight) * 100, 1)

        # Require a substantial number of valid comparisons (at least 60% of possible stats)
        if valid_comparisons >= (total_possible_weight * 0.6):  # At least 60% of weighted stats available
            avg_similarity = total_similarity / valid_comparisons
            similarities.append({
                'player': historical_player.get('playerFullName', 'Unknown'),
                'team': historical_player.get('newestTeamName', 'Unknown'),
                'position': historical_player.get('pos', 'Unknown'),
                'year': historical_player.get('Year', 'Unknown'),
                'similarity_score': avg_similarity,
                'valid_stats': actual_stats_compared,
                'total_possible_stats': len(comparison_stats),
                'coverage_pct': coverage_percentage,
                'stat_breakdown': stat_similarities,
                'player_data': historical_player
            })

    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities


def create_comparison_chart(target_player, comp_players, stats_to_show):
    """Create radar chart comparing target player with comps using percentiles"""

    # Get percentile reference data
    percentiles_data = get_percentiles_data()

    # Filter stats to only those we have percentile data for
    available_stats = [stat for stat in stats_to_show if stat in percentiles_data]

    if not available_stats:
        st.warning("No percentile data available for selected stats")
        return None

    # Prepare data for radar chart using percentiles
    # Duplicate first stat at end to close the loop
    categories = available_stats + [available_stats[0]]

    fig = go.Figure()

    # Add target player
    target_percentiles = []
    for stat in available_stats:
        percentile_rank = calculate_percentile_rank(target_player.get(stat), stat, percentiles_data)
        target_percentiles.append(percentile_rank)
    # Close the loop by repeating first value
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

    # Add comparison players
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, comp in enumerate(comp_players[:5]):  # Top 5 comps
        comp_percentiles = []
        for stat in available_stats:
            percentile_rank = calculate_percentile_rank(comp['player_data'].get(stat), stat, percentiles_data)
            comp_percentiles.append(percentile_rank)
        # Close the loop by repeating first value
        comp_percentiles.append(comp_percentiles[0])

        fig.add_trace(go.Scatterpolar(
            r=comp_percentiles,
            theta=categories,
            fill='toself',
            name=f"{comp['player']} ({comp['year']})",
            line_color=colors[i % len(colors)],
            fillcolor=f'rgba{tuple(list(int(colors[i % len(colors)][j:j + 2], 16) for j in (1, 3, 5)) + [0.1])}',
            line_width=2,
            opacity=0.8
        ))

    # Update layout for percentiles - FIXED TICK LABELS
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
                rotation=90  # Start at top
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
            'text': "Player Comparison - Percentile Rankings",
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

    # Get list of 2025 players
    player_names = sorted([name for name in data_2025['playerFullName'].dropna().unique() if name])

    selected_player_name = st.sidebar.selectbox(
        "Choose a 2025 player:",
        player_names,
        index=0 if player_names else None
    )

    if not selected_player_name:
        st.warning("No players available for selection")
        return

    # Get selected player data
    target_player = data_2025[data_2025['playerFullName'] == selected_player_name].iloc[0]

    # Comparison controls
    st.sidebar.header("Comparison Settings")
    num_comps = st.sidebar.slider("Number of comparisons to show:", 1, 20, 10)

    min_games = st.sidebar.slider("Minimum games played:", 0, 50, 10)

    # Year selection for historical comparisons
    year_options = st.sidebar.multiselect(
        "Include players from years:",
        [2024, 2023, 2022],
        default=[2024, 2023, 2022]  # All years by default
    )

    position_filter = st.sidebar.multiselect(
        "Filter by position:",
        ['All'] + sorted(pd.concat([data_2024, data_2023, data_2022])['pos'].dropna().unique().tolist()),
        default=['All']
    )

    # Calculate comparisons
    if st.sidebar.button("Find Player Comparisons") or 'comparisons' not in st.session_state:
        if not year_options:
            st.error("Please select at least one year for historical comparisons")
            return

        with st.spinner("Calculating player similarities using exit velocity & plate discipline focus..."):
            # Prepare historical data from selected years only
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

            # Filter by minimum games
            if min_games > 0:
                historical_data = historical_data[historical_data['G'] >= min_games]

            # Filter by position
            if 'All' not in position_filter and position_filter:
                historical_data = historical_data[historical_data['pos'].isin(position_filter)]

            # Get comparison stats
            comparison_stats = get_comparison_stats()

            # Calculate similarities
            similarities = calculate_player_similarity(target_player, historical_data, comparison_stats)

            st.session_state.comparisons = similarities[:num_comps]
            st.session_state.target_player = target_player
            st.session_state.comparison_stats = comparison_stats
            st.session_state.selected_years = year_options

            # Show summary of search
            st.success(
                f"Found {len(similarities)} potential matches from {len(historical_data)} players across {len(year_options)} years")

            # Debug: Show which stats are missing from historical data
            if st.sidebar.checkbox("Show debug info"):
                missing_stats = [stat for stat in comparison_stats if stat not in historical_data.columns]
                if missing_stats:
                    st.sidebar.warning(f"Missing stats in historical data: {', '.join(missing_stats)}")

    # Display results
    if 'comparisons' in st.session_state:
        st.header(f"Top {len(st.session_state.comparisons)} Most Similar Players")

        # Show search parameters
        years_searched = ", ".join(map(str, st.session_state.selected_years))
        st.info(f"Prioritizing Exit Velocity & Plate Discipline | Searched years: {years_searched}")

        # Create comparison table
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

        comp_df = pd.DataFrame(comp_data)
        # Clean the dataframe to avoid Arrow conversion issues
        comp_df = comp_df.astype(str)
        st.dataframe(comp_df, use_container_width=True)

        # Statistical comparison section
        st.header("Statistical Comparison")

        # Select stats to display (filter to only those with percentile data)
        available_stats = st.session_state.comparison_stats
        percentile_stats = list(get_percentiles_data().keys())
        available_for_chart = [stat for stat in available_stats if stat in percentile_stats]

        selected_stats = st.multiselect(
            "Select stats to compare (percentile rankings available):",
            available_for_chart,
            default=available_for_chart[:8] if len(available_for_chart) >= 8 else available_for_chart
        )

        if selected_stats and len(st.session_state.comparisons) > 0:
            # Create detailed comparison table
            st.subheader("Detailed Statistical Comparison")

            detailed_comp = {'Stat': selected_stats}

            # Add target player
            detailed_comp[st.session_state.target_player['playerFullName']] = [
                str(st.session_state.target_player.get(stat, 'N/A')) for stat in selected_stats
            ]

            # Add comparison players
            for i, comp in enumerate(st.session_state.comparisons[:5]):  # Top 5 comps
                comp_name = f"{comp['player']} ({comp['year']})"
                comp_values = []
                for stat in selected_stats:
                    value = comp['player_data'].get(stat, 'N/A')
                    # Handle missing stats better - show year-specific note
                    if pd.isna(value) or value == '' or value is None:
                        if comp['year'] == 2024 and stat in ['O-Contact%', 'HardHit%', 'Contact%']:
                            value = 'N/A (2024)'  # Indicate this stat wasn't collected in 2024
                        else:
                            value = 'N/A'
                    comp_values.append(str(value))
                detailed_comp[comp_name] = comp_values

            detailed_df = pd.DataFrame(detailed_comp)

            # Ensure all columns are strings to prevent Arrow issues
            for col in detailed_df.columns:
                detailed_df[col] = detailed_df[col].astype(str)

            st.dataframe(detailed_df, use_container_width=True)

            # Create radar chart
            if len(selected_stats) >= 3:
                st.subheader("Player Comparison - Percentile Rankings")
                st.info(
                    "Chart shows percentile rankings relative to all 2025 NCAA players.")

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
