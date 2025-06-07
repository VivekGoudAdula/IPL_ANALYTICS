import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# Set page config
st.set_page_config(
    page_title="IPL Match Analysis Dashboard",
    page_icon="🏏",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ipl_dataset.csv')
        # Clean and preprocess data
        df['season'] = df['full_scorecard'].str.extract(r'match/(\d+)/')
        df['season'] = pd.to_numeric(df['season'].str[:4], errors='coerce')
        df['season'] = df['season'].fillna(2018)  # Fill missing values with 2018 as it's a common year in the data
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def plot_wins_by_team(df):
    st.subheader("🏆 Wins by Team")
    wins = df['winner'].value_counts().reset_index()
    wins.columns = ['Team', 'Wins']
    
    fig = px.bar(
        wins.head(15),  # Show top 15 teams by wins
        x='Team',
        y='Wins',
        title='Total Wins by Team (Top 15)',
        color='Team',
        text='Wins',
        labels={'Wins': 'Number of Wins', 'Team': 'Team'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_toss_decision_impact(df):
    st.subheader("🎯 Toss Decision Impact")
    try:
        df['toss_win_match_win'] = df['toss_winner'] == df['winner']
        toss_impact = df['toss_win_match_win'].value_counts(normalize=True).mul(100).round(1)
        
        labels = ['Toss Winner Won Match', 'Toss Loser Won Match']
        colors = px.colors.qualitative.Pastel[:2]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=toss_impact.values,
            marker=dict(colors=colors)
        )])
        
        fig.update_layout(
            title='Match Outcome Based on Toss',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate toss decision impact chart: {e}")

def plot_matches_by_season(df):
    st.subheader("📅 Matches Played by Season")
    matches_by_season = df['season'].value_counts().sort_index()
    
    fig = px.line(
        x=matches_by_season.index,
        y=matches_by_season.values,
        title='Number of Matches Played Each Season',
        labels={'x': 'Season', 'y': 'Number of Matches'},
        markers=True
    )
    fig.update_traces(line_color='#FF4B4B')
    fig.update_layout(xaxis_tickformat='d')
    st.plotly_chart(fig, use_container_width=True)

def plot_head_to_head(df, team1, team2):
    """Generate head-to-head statistics between two teams."""
    matches = df[((df['team1'] == team1) & (df['team2'] == team2)) | 
                 ((df['team1'] == team2) & (df['team2'] == team1))]
    
    if matches.empty:
        st.warning(f"No matches found between {team1} and {team2}")
        return
    
    # Calculate statistics
    total_matches = len(matches)
    team1_wins = len(matches[matches['winner'] == team1])
    team2_wins = len(matches[matches['winner'] == team2])
    no_result = total_matches - (team1_wins + team2_wins)
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{team1} Wins", team1_wins)
    with col2:
        st.metric(f"{team2} Wins", team2_wins)
    with col3:
        st.metric("No Result/Abandoned", no_result)
    
    # Plot pie chart
    if team1_wins + team2_wins > 0:  # Only plot if there are completed matches
        fig = px.pie(
            names=[f"{team1} Wins", f"{team2} Wins", "No Result"],
            values=[team1_wins, team2_wins, no_result],
            title=f"{team1} vs {team2} - Head to Head",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_player_performance(df, player_name):
    """Show performance metrics for a specific player."""
    st.subheader(f"Player Performance: {player_name}")
    
    # Check if player exists in the data
    if 'player_of_match' not in df.columns or player_name not in df['player_of_match'].values:
        st.warning(f"No performance data available for {player_name}")
        return
    
    # Calculate player stats
    player_matches = df[df['player_of_match'] == player_name]
    total_awards = len(player_matches)
    teams_played_for = list(set(player_matches['team1'].unique().tolist() + 
                              player_matches['team2'].unique().tolist()))
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Player of the Match Awards", total_awards)
    with col2:
        st.metric("Teams Played For", ", ".join(teams_played_for) if teams_played_for else "N/A")

def plot_venue_stats(df):
    """Show statistics by venue."""
    # Check which venue column is available
    venue_col = None
    if 'venue' in df.columns:
        venue_col = 'venue'
    elif 'stadium' in df.columns:
        venue_col = 'stadium'
    
    if not venue_col:
        st.warning("No venue data available in the dataset.")
        return
    
    # Get venue stats
    venue_stats = df[venue_col].value_counts().reset_index()
    venue_stats.columns = ['Venue', 'Matches']
    
    if venue_stats.empty:
        st.warning("No venue statistics available.")
        return
    
    # Show top 10 venues by match count
    st.subheader("🏟️ Top Venues")
    top_venues = venue_stats.head(10)
    
    fig1 = px.bar(
        top_venues,
        x='Venue',
        y='Matches',
        title='Top 10 Venues by Number of Matches',
        color='Matches',
        color_continuous_scale='Viridis',
        labels={'Venue': 'Stadium', 'Matches': 'Number of Matches'}
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Show venue win distribution
    if 'winner' in df.columns and venue_col:
        st.subheader("🏆 Win Distribution by Venue")
        venue_wins = df[df['winner'].notna()].groupby([venue_col, 'winner']).size().reset_index(name='wins')
        
        if not venue_wins.empty:
            # Get top 5 venues by number of matches
            top_venue_names = venue_stats.head(5)['Venue'].tolist()
            top_venue_wins = venue_wins[venue_wins[venue_col].isin(top_venue_names)]
            
            if not top_venue_wins.empty:
                fig2 = px.sunburst(
                    top_venue_wins,
                    path=[venue_col, 'winner'],
                    values='wins',
                    title='Win Distribution at Top Venues',
                    color='wins',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)

def plot_player_of_match(df):
    st.subheader("⭐ Top Performers")
    top_players = df['player_of_match'].value_counts().head(10).reset_index()
    top_players.columns = ['Player', 'Awards']
    
    fig = px.bar(
        top_players,
        x='Player',
        y='Awards',
        title='Top 10 Players with Most Player of the Match Awards',
        color='Awards',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_data_overview(df):
    st.header("📊 Data Overview")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(df))
    col2.metric("Number of Teams", len(set(df['team1'].unique().tolist() + df['team2'].unique().tolist())))
    col3.metric("Seasons", df['season'].nunique())
    
    # Show dataset info
    with st.expander("🔍 View Dataset Info"):
        st.write("**Shape:**", df.shape)
        
        # Basic statistics
        st.write("### Basic Statistics")
        st.dataframe(df.describe(include='all').fillna(''))
        
        # Null values
        st.write("### Missing Values")
        null_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
        st.dataframe(null_df[null_df['Missing Values'] > 0])

def main():
    st.title("🏏 IPL Match Analysis Dashboard")
    st.markdown("### Analyzing IPL match data to uncover interesting insights")
    
    # File uploader
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload IPL Dataset (CSV)", type=["csv"])
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Preprocess data
            if 'full_scorecard' in df.columns:
                df['season'] = df['full_scorecard'].str.extract(r'match/(\d+)/')
                df['season'] = pd.to_numeric(df['season'].str[:4], errors='coerce')
                df['season'] = df['season'].fillna(2018)
            else:
                st.warning("Could not extract season from 'full_scorecard' column. Using default season values.")
                df['season'] = 2023  # Default season if column not found
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            df = pd.DataFrame()
    else:
        df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check if the file exists and is in the correct format.")
        return
    
    # Sidebar filters
    st.sidebar.header('Filters')
    
    # Season filter
    if 'season' in df.columns:
        seasons = sorted(df['season'].unique(), reverse=True)
        selected_seasons = st.sidebar.multiselect(
            'Select Seasons',
            options=seasons,
            default=seasons[:2] if len(seasons) > 1 else seasons
        )
        if selected_seasons:
            df = df[df['season'].isin(selected_seasons)]
    
    # Team filter
    if all(col in df.columns for col in ['team1', 'team2']):
        # Convert all team names to strings and remove any NaN/None values
        team1_list = df['team1'].dropna().astype(str).unique().tolist()
        team2_list = df['team2'].dropna().astype(str).unique().tolist()
        all_teams = sorted(list(set(team1_list + team2_list)))
        selected_teams = st.sidebar.multiselect(
            'Select Teams',
            options=all_teams,
            default=all_teams[:2] if len(all_teams) > 1 else all_teams
        )
        if selected_teams:
            df = df[df['team1'].isin(selected_teams) | df['team2'].isin(selected_teams)]
    
    # Show data overview
    show_data_overview(df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Team Performance", 
        "🤜🤛 Head to Head", 
        "👤 Player Stats",
        "📊 Match Analysis",
        "🏟️ Venue Stats"
    ])
    
    with tab1:
        st.header("🏆 Team Performance")
        if 'winner' in df.columns:
            plot_wins_by_team(df)
        
        if all(col in df.columns for col in ['toss_winner', 'winner']):
            plot_toss_decision_impact(df)
    
    with tab2:
        st.header("🤜🤛 Head to Head Comparison")
        if all(col in df.columns for col in ['team1', 'team2', 'winner']):
            # Convert all team names to strings and remove any NaN/None values
            team1_list = df['team1'].dropna().astype(str).unique().tolist()
            team2_list = df['team2'].dropna().astype(str).unique().tolist()
            all_teams = sorted(list(set(team1_list + team2_list)))
            col1, col2 = st.columns(2)
            with col1:
                team1 = st.selectbox('Select Team 1', all_teams, index=0)
            with col2:
                # Ensure team2 is different from team1
                other_teams = [t for t in all_teams if t != team1]
                team2 = st.selectbox('Select Team 2', other_teams, 
                                   index=0 if len(other_teams) > 0 else None)
            
            if team1 and team2 and team1 != team2:
                plot_head_to_head(df, team1, team2)
    
    with tab3:
        st.header("👤 Player Performance")
        if 'player_of_match' in df.columns:
            # Show top players selector
            top_players = df['player_of_match'].value_counts().head(20).index.tolist()
            selected_player = st.selectbox('Select a Player', top_players)
            if selected_player:
                plot_player_performance(df, selected_player)
            
            # Show top performers
            st.subheader("🏆 Top Performers")
            plot_player_of_match(df)
    
    with tab4:
        st.header("📊 Match Analysis")
        if 'season' in df.columns:
            plot_matches_by_season(df)
    
    with tab5:
        st.header("🏟️ Venue Statistics")
        plot_venue_stats(df)
    
    # Add download button for filtered data
    if not df.empty:
        st.sidebar.markdown("---")
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"filtered_ipl_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Venue filter (if venue data exists)
    if 'stadium' in df.columns:
        venues = sorted(df['stadium'].dropna().unique())
        selected_venues = st.sidebar.multiselect(
            'Select Venues',
            options=venues,
            default=venues[:2] if len(venues) > 1 else venues
        )
        if selected_venues:
            df = df[df['stadium'].isin(selected_venues)]
    
    # Add some space at the bottom
    st.markdown("---")
    st.markdown("### Data Source: IPL Match Data")
    st.markdown("Explore the data using the filters in the sidebar to discover more insights!")

if __name__ == "__main__":
    main()
