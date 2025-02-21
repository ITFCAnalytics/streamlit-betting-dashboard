from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap
import seaborn as sns
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup, Comment
import os
from pathlib import Path
import time
from scipy import stats
from statistics import mean
from math import pi
import streamlit as st

root = os.getcwd() + '/'

# This section creates the programs that gather data from FBRef.com... Data is from FBRef and Opta
def _get_table(soup):
    return soup.find_all('table')[0]

def _parse_row(row):
    cols = None
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    return cols

def get_df(path):
    URL = path
    time.sleep(4)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    table = _get_table(soup)
    data = []
    headings=[]
    headtext = soup.find_all("th",scope="col")
    for i in range(len(headtext)):
        heading = headtext[i].get_text()
        headings.append(heading)
    headings=headings[1:len(headings)]
    data.append(headings)
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')

    for row_index in range(len(rows)):
        row = rows[row_index]
        cols = _parse_row(row)
        data.append(cols)

    data = pd.DataFrame(data)
    data = data.rename(columns=data.iloc[0])
    data = data.reindex(data.index.drop(0))
    data = data.replace('',0)
    return data

pl_path = 'https://fbref.com/en/comps/9/Premier-League-Stats#all_results2024-202591'

pl_table = get_df(pl_path)
pl_table = pl_table.iloc[:, 0:1]
#pl_table['Position'] = range(1, len(pl_table) + 1)

pl_table.head()

ch_path = 'https://fbref.com/en/comps/10/Championship-Stats#all_results2024-2025101'

ch_table = get_df(ch_path)
ch_table = ch_table.iloc[:, 0:1]
#ch_table['Position'] = range(1, len(ch_table) + 1)

ch_table.head()

df = pd.read_csv(f'{root}Final FBRef Match Logs for 2024-2025.csv')

df = df[df['Round'].str.contains('Matchweek', case=True)]

#columns = ['Team', 'Opponent', 'Venue', 'Result']
points_conditions = [
    (df['Result'] == 'W'),
    (df['Result'] == 'D'),
    (df['Result'] == 'L')
]
points_results = ['3', '1', '0']

#df_filtered = df[columns]
df['Points'] = np.select(points_conditions, points_results)
df['Gameweek'] = df.groupby('Team').cumcount() + 1
df['Score'] = df['GF'].astype(str) + '-' + df['GA'].astype(str)
df['6 Game xGD avg'] = df.groupby('Team')['xGD'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())

df_new = df.copy()

#df_new = df_new[(df_new['Team'] == 'Millwall') & (df_new['Opponent'] == 'Portsmouth')]

pd.set_option('display.max_columns', None) 
df_new.head()

def team_results(team):
    
    team_data = df_new[df_new['Team'] == team]
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Create a copy of the league table based on the competition
    if (team_data['Comp'] == 'Premier-League').any():
        table_data = pl_table.copy()
    elif (team_data['Comp'] == 'Championship').any():
        table_data = ch_table.copy()
    else:
        return None  # Return None if the competition is not recognized

    # Initialize Home and Away columns
    table_data['Home'] = ''
    #table_data['Home Points'] = ''  # Initialize Home Points
    table_data['Away'] = ''
    #table_data['Away Points'] = ''  # Initialize Away Points

    # Map results to the league table
    for index, row in team_data.iterrows():
        if row['Venue'] == 'Home':
            # Update the Home column with the score and points
            points = row['Points']
            table_data.loc[table_data['Squad'] == row['Opponent'], 'Home'] += f"{points} ({row['Score']})"
        else:
            # Update the Away column with the score and points
            points = row['Points']
            table_data.loc[table_data['Squad'] == row['Opponent'], 'Away'] += f"{points} ({row['Score']})"

    # Apply styling based on points
    def highlight_points(value):
        # Extract the points from the value
        points = value.split(' ')[0]  # Get the first part before the space
        if points == '3':
            return 'background-color: green; color: white;'  # Green for 3 points
        elif points == '1':
            return 'background-color: yellow; color: black;'  # Yellow for 1 point
        elif points == '0':
            return 'background-color: red; color: white;'  # Red for 0 points
        return ''  # No formatting for other values

    # Apply the styling to the Home and Away Points columns
    styled_table = table_data.style.applymap(highlight_points, subset=['Home', 'Away'])

    #styled_table = styled_table[['Position', 'Squad', 'Home', 'Home Points', 'Away', 'Away Points']]

    return styled_table

def average_xGD(team):

    team_data = df_new[df_new['Team'] == team]
    Gameweek = team_data['Gameweek'].values  # Convert to numpy array
    xGD = team_data['6 Game xGD avg'].values  # Convert to numpy array

    # Map xGD values to colors
    norm = (xGD + 2.5) / 5  # Scale xGD from -2.5 to 2.5 to [0, 1]
    colors = cm.RdYlGn(norm)  # Use RdYlGn colormap

    # Plot each segment with fixed color scale
    for i in range(len(Gameweek) - 1):
        plt.plot(Gameweek[i:i+2], xGD[i:i+2], color=colors[i], linewidth=3)

    plt.axhline(0, color='white', linewidth=0.8, linestyle='--')  # Add gridline at y=0
    plt.ylim(-2.75, 2.75)  # Set y-axis limits
    plt.xlabel('Gameweek', color='white')
    plt.ylabel('xGD', color='white')
    #plt.figure(figsize=(8, 4))
    #plt.title(f'6 Game xGD Moving Average | {team}')
    # Set background color to transparent
    plt.gca().set_facecolor('none')  # Make the axes background transparent
    plt.gcf().patch.set_facecolor('none')  # Make the figure background transparent
    # Set axis outlines (spines) to white
    for spine in plt.gca().spines.values():
        spine.set_color('white')  # Set the color of the spines to white
    # Change gridlines and ticks to white
    plt.tick_params(axis='both', colors='white')  # Set tick marks to white


    st.pyplot(plt)  # Call the show method

def possession_impact(team):

    team_data = df_new[df_new['Team'] == team]

    #filter data based on possession
    possession_51 = team_data[team_data['Poss'] >= 50]
    possession_49 = team_data[team_data['Poss'] <= 49]

    # Calculate average points and xGD for possession >= 50
    avg_points_51 = possession_51['Points'].astype(float).mean() if not possession_51.empty else 0
    avg_xGD_51 = possession_51['6 Game xGD avg'].mean() if not possession_51.empty else 0
    count_games_51 = possession_51['Gameweek'].count() if not possession_51.empty else 0
    count_wins_51 = possession_51['Result'].value_counts().get('W', 0) if not possession_51.empty else 0
    count_draws_51 = possession_51['Result'].value_counts().get('D', 0) if not possession_51.empty else 0
    count_losses_51 = possession_51['Result'].value_counts().get('L', 0) if not possession_51.empty else 0

    # Calculate average points and xGD for possession < 50
    avg_points_49 = possession_49['Points'].astype(float).mean() if not possession_49.empty else 0
    avg_xGD_49 = possession_49['6 Game xGD avg'].mean() if not possession_49.empty else 0
    count_games_49 = possession_49['Gameweek'].count() if not possession_51.empty else 0
    count_wins_49 = possession_49['Result'].value_counts().get('W', 0) if not possession_51.empty else 0
    count_draws_49 = possession_49['Result'].value_counts().get('D', 0) if not possession_51.empty else 0
    count_losses_49 = possession_49['Result'].value_counts().get('L', 0) if not possession_51.empty else 0

    # Create a DataFrame to display the results in a single row
    results_df = pd.DataFrame({
        'Possession': ['< 50', '>= 50'],
        'Games': [count_games_49, count_games_51],
        'Wins': [count_wins_49, count_wins_51],
        'Draws': [count_draws_49, count_draws_51],
        'Losses': [count_losses_49, count_losses_51],
        'Points per Game': [avg_points_49, avg_points_51],
        'xGD': [avg_xGD_49, avg_xGD_51]
    })

    # Define a function to apply color mapping for Points per Game
    def color_points(value):
        if value < 1:
            return 'background-color: red; color: white;'
        elif 1 <= value < 1.8:
            return 'background-color: yellow; color: black;'
        elif value >= 1.8:
            return 'background-color: green; color: white;'
        return ''

    # Define a function to apply color mapping for xGD
    def color_xgd(value):
        if value < -0.2:
            return 'background-color: red; color: white;'
        elif -0.2 <= value < 0.5:
            return 'background-color: yellow; color: black;'
        elif value >= 0.5:
            return 'background-color: green; color: white;'
        return ''

    # Apply the styling to the DataFrame
    styled_df = results_df.style.applymap(color_points, subset=['Points per Game']) \
                                 .applymap(color_xgd, subset=['xGD'])

    return styled_df

def home_away(team):

    team_data = df_new[df_new['Team'] == team]

    #filter data based on possession
    home = team_data[team_data['Venue'] == 'Home']
    away = team_data[team_data['Venue'] == 'Away']

    # Calculate average points and xGD for possession >= 50
    avg_points_h = home['Points'].astype(float).mean() if not home.empty else 0
    avg_xGD_h = home['6 Game xGD avg'].mean() if not home.empty else 0
    count_games_h = home['Gameweek'].count() if not home.empty else 0
    count_wins_h = home['Result'].value_counts().get('W', 0) if not home.empty else 0
    count_draws_h = home['Result'].value_counts().get('D', 0) if not home.empty else 0
    count_losses_h = home['Result'].value_counts().get('L', 0) if not home.empty else 0

    # Calculate average points and xGD for possession < 50
    avg_points_a = away['Points'].astype(float).mean() if not away.empty else 0
    avg_xGD_a = away['6 Game xGD avg'].mean() if not away.empty else 0
    count_games_a = away['Gameweek'].count() if not away.empty else 0
    count_wins_a = away['Result'].value_counts().get('W', 0) if not away.empty else 0
    count_draws_a = away['Result'].value_counts().get('D', 0) if not away.empty else 0
    count_losses_a = away['Result'].value_counts().get('L', 0) if not away.empty else 0

    # Create a DataFrame to display the results in a single row
    results_df = pd.DataFrame({
        'Venue': ['Home', 'Away'],
        'Games': [count_games_h, count_games_a],
        'Wins': [count_wins_h, count_wins_a],
        'Draws': [count_draws_h, count_draws_a],
        'Losses': [count_losses_h, count_losses_a],
        'Points per Game': [avg_points_h, avg_points_a],
        'xGD': [avg_xGD_h, avg_xGD_a]
    })

    # Define a function to apply color mapping for Points per Game
    def color_points(value):
        if value < 1:
            return 'background-color: red; color: white;'
        elif 1 <= value < 1.8:
            return 'background-color: yellow; color: black;'
        elif value >= 1.8:
            return 'background-color: green; color: white;'
        return ''

    # Define a function to apply color mapping for xGD
    def color_xgd(value):
        if value < -0.2:
            return 'background-color: red; color: white;'
        elif -0.2 <= value < 0.5:
            return 'background-color: yellow; color: black;'
        elif value >= 0.5:
            return 'background-color: green; color: white;'
        return ''

    # Apply the styling to the DataFrame
    styled_df = results_df.style.applymap(color_points, subset=['Points per Game']) \
                                 .applymap(color_xgd, subset=['xGD'])

    return styled_df

## CREATE STREAMLIT DASHBOARD ##

# title, subheader and player/dashboard filter
st.set_page_config(page_title="Betting Dashboard -", layout="wide")
st.title(f"Betting Dashboard")
st.subheader("Filter by team to see their rolling xGD, results plotted against the current league table, effect of possession and home/away advantage on performances and results:")

unique_team = df_new['Team'].sort_values().unique()
team_filter = st.selectbox('Select a team:', unique_team, index=0)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f'6 Game xGD Moving Average | {team_filter}:')
    average_xGD(team_filter)

with col2:
    st.subheader(f'{team_filter} Results Table:')
    results_table = team_results(team_filter)
    st.dataframe(results_table)

col3, col4 = st.columns(2)

with col3:
    st.subheader(f'Effect of Possession on {team_filter}:')
    possession_table = possession_impact(team_filter)
    st.dataframe(possession_table)

with col4:
    st.subheader(f'Home/Away Performance for {team_filter}:')
    home_away_table = home_away(team_filter)
    st.dataframe(home_away_table)