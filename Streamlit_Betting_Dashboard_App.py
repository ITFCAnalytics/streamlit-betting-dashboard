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

#pl_table = pd.read_csv(f'{root}Premier League Table 2024-2025.csv')
pl_table_url = 'https://github.com/ITFCAnalytics/streamlit-betting-dashboard/raw/5c598eea521c1cac61ca030981a84b3a6e3ce584/Premier%20League%20Table%202025-2026.csv'
pl_table = pd.read_csv(pl_table_url)

#ch_table = pd.read_csv(f'{root}Championship Table 2024-2025.csv')
ch_table_url = 'https://github.com/ITFCAnalytics/streamlit-betting-dashboard/raw/5c598eea521c1cac61ca030981a84b3a6e3ce584/Championship%20Table%202025-2026.csv'
ch_table = pd.read_csv(ch_table_url)

#df = pd.read_csv(f'{root}Final FBRef Match Logs for 2024-2025.csv')
df_url = 'https://github.com/ITFCAnalytics/streamlit-betting-dashboard/raw/5c598eea521c1cac61ca030981a84b3a6e3ce584/Final%20FBRef%20Match%20Logs%20for%202025-2026.csv'
df = pd.read_csv(df_url)

df = df[df['Round'].str.contains('Matchweek', case=True)]

#columns = ['Team', 'Opponent', 'Venue', 'Result']
points_conditions = [
    (df['Result'] == 'W'),
    (df['Result'] == 'D'),
    (df['Result'] == 'L')
]
points_results = [3, 1, 0]

#df_filtered = df[columns]
df['Points'] = np.select(points_conditions, points_results)
df['Gameweek'] = df.groupby('Team').cumcount() + 1
df['Score'] = df['GF'].astype(str) + '-' + df['GA'].astype(str)
df['5 Game xGD avg'] = df.groupby('Team')['xGD'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['5 Game xGF avg'] = df.groupby('Team')['xG'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['5 Game xGA avg'] = df.groupby('Team')['xGA'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['Formation'] = df['Formation'].replace('5-3-2', '3-5-2').replace('3-5-1-1', '3-5-2').replace('5-4-1', '3-4-3').replace('4-1-4-1', '4-3-3').replace('4-4-1-1', '4-2-3-1').replace('4-5-1', '4-3-3').replace('3-2-4-1', '4-3-3').replace('3-1-4-2', '3-5-2').replace('4-1-3-2', '4-4-2')
df['Opp Formation'] = df['Opp Formation'].replace('5-3-2', '3-5-2').replace('3-5-1-1', '3-5-2').replace('5-4-1', '3-4-3').replace('4-1-4-1', '4-3-3').replace('4-4-1-1', '4-2-3-1').replace('4-5-1', '4-3-3').replace('3-2-4-1', '4-3-3').replace('3-1-4-2', '3-5-2').replace('4-1-3-2', '4-4-2')

df_new = df.copy()

#df_new = df_new[(df_new['Team'] == 'Millwall') & (df_new['Opponent'] == 'Portsmouth')]

pd.set_option('display.max_columns', None)

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
    # Clear the current figure to avoid overlaying plots
    plt.clf()
    
    # Filter data for the specified team
    team_data = df_new[df_new['Team'] == team]
    
    # Check if there is data for the team
    if team_data.empty:
        st.write(f"No data available for {team}.")
        return
    
    Gameweek = team_data['Gameweek'].values  # Convert to numpy array
    xGD = team_data['5 Game xGD avg'].values  # Convert to numpy array

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
    plt.gca().set_facecolor('none')  # Make the axes background transparent
    plt.gcf().patch.set_facecolor('none')  # Make the figure background transparent

    # Set axis outlines (spines) to white
    for spine in plt.gca().spines.values():
        spine.set_color('white')  # Set the color of the spines to white

    # Change gridlines and ticks to white
    plt.tick_params(axis='both', colors='white')  # Set tick marks to white

    st.pyplot(plt)  # Call the show method

def average_xGF_xGA(team):
    # Clear the current figure to avoid overlaying plots
    plt.clf()
    
    # Filter data for the specified team
    team_data = df_new[df_new['Team'] == team]
    
    # Check if there is data for the team
    if team_data.empty:
        st.write(f"No data available for {team}.")
        return
    
    Gameweek = team_data['Gameweek'].values  # Convert to numpy array
    xGF = team_data['5 Game xGF avg'].values  # Convert to numpy array
    xGA = team_data['5 Game xGA avg'].values  # Convert to numpy array

    # Plot each segment for xGF
    for i in range(len(Gameweek) - 1):
        plt.plot(Gameweek[i:i+2], xGF[i:i+2], color='green', linewidth=3)

    # Add linear trend line for xGF starting from Gameweek 5
    valid_indices_GF = Gameweek >= 5
    if valid_indices_GF.any():  # Check if there are valid indices
        zGF = np.polyfit(Gameweek[valid_indices_GF], xGF[valid_indices_GF], 1)  # Fit a linear polynomial
        pGF = np.poly1d(zGF)  # Create a polynomial function
        plt.plot(Gameweek[valid_indices_GF], pGF(Gameweek[valid_indices_GF]), color='green', linestyle='--', linewidth=1.5, label='Trend Line xGF')

    # Plot each segment for xGA
    for i in range(len(Gameweek) - 1):
        plt.plot(Gameweek[i:i+2], xGA[i:i+2], color='red', linewidth=3)

    # Add linear trend line for xGA starting from Gameweek 5
    valid_indices_GA = Gameweek >= 5
    if valid_indices_GA.any():  # Check if there are valid indices
        zGA = np.polyfit(Gameweek[valid_indices_GA], xGA[valid_indices_GA], 1)  # Fit a linear polynomial
        pGA = np.poly1d(zGA)  # Create a polynomial function
        plt.plot(Gameweek[valid_indices_GA], pGA(Gameweek[valid_indices_GA]), color='red', linestyle='--', linewidth=1.5, label='Trend Line xGA')

    plt.axhline(0, color='white', linewidth=0.8, linestyle='--')  # Add gridline at y=0
    plt.ylim(0, 4)  # Set y-axis limits
    plt.xlabel('Gameweek', color='white')
    plt.ylabel('xG', color='white')
    plt.gca().set_facecolor('none')  # Make the axes background transparent
    plt.gcf().patch.set_facecolor('none')  # Make the figure background transparent

    # Set axis outlines (spines) to white
    for spine in plt.gca().spines.values():
        spine.set_color('white')  # Set the color of the spines to white

    # Change gridlines and ticks to white
    plt.tick_params(axis='both', colors='white')  # Set tick marks to white

    plt.legend()  # Add legend to the plot
    st.pyplot(plt)  # Call the show method

import numpy as np
import pandas as pd
from matplotlib import cm

def possession_impact(team):
    team_data = df_new[df_new['Team'] == team]

    # Filter data based on possession
    possession_51 = team_data[team_data['Poss'] >= 50]
    possession_49 = team_data[team_data['Poss'] <= 49]

    # Calculate averages and counts
    avg_points_51 = possession_51['Points'].astype(float).mean() if not possession_51.empty else 0
    avg_xGD_51 = possession_51['xGD'].mean() if not possession_51.empty else 0
    count_games_51 = possession_51['Gameweek'].count() if not possession_51.empty else 0
    count_wins_51 = possession_51['Result'].value_counts().get('W', 0) if not possession_51.empty else 0
    count_draws_51 = possession_51['Result'].value_counts().get('D', 0) if not possession_51.empty else 0
    count_losses_51 = possession_51['Result'].value_counts().get('L', 0) if not possession_51.empty else 0

    avg_points_49 = possession_49['Points'].astype(float).mean() if not possession_49.empty else 0
    avg_xGD_49 = possession_49['xGD'].mean() if not possession_49.empty else 0
    count_games_49 = possession_49['Gameweek'].count() if not possession_49.empty else 0
    count_wins_49 = possession_49['Result'].value_counts().get('W', 0) if not possession_49.empty else 0
    count_draws_49 = possession_49['Result'].value_counts().get('D', 0) if not possession_49.empty else 0
    count_losses_49 = possession_49['Result'].value_counts().get('L', 0) if not possession_49.empty else 0

    # Create DataFrame
    results_df = pd.DataFrame({
        'Possession': [' > 50%', '< 50%'],
        'Games': [count_games_51, count_games_49],
        'Wins': [count_wins_51, count_wins_49],
        'Draws': [count_draws_51, count_draws_49],
        'Losses': [count_losses_51, count_losses_49],
        'Points per Game': [avg_points_51, avg_points_49],
        'xGD': [avg_xGD_51, avg_xGD_49]
    })

    # Function to compute contrasting text color
    def text_color(r, g, b):
        # Perceived brightness: formula for contrast
        brightness = (r*299 + g*587 + b*114)/1000
        return 'black' if brightness > 125 else 'white'

    # Color mapping for Points per Game
    def color_points(value):
        norm_value = (value - 0.5) / 2  # Scale Points from 0.5 to 2.5
        norm_value = np.clip(norm_value, 0, 1)
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    # Color mapping for xGD
    def color_xgd(value):
        norm_value = (value + 1.4) / 2.8  # Scale xGD from -1.4 to 1.4
        norm_value = np.clip(norm_value, 0, 1)
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    # Apply styling
    styled_df = results_df.style.applymap(color_points, subset=['Points per Game']) \
                                 .applymap(color_xgd, subset=['xGD'])

    return styled_df

def home_away(team):
    team_data = df_new[df_new['Team'] == team]

    # Filter data based on home/away
    home = team_data[team_data['Venue'] == 'Home']
    away = team_data[team_data['Venue'] == 'Away']

    # Calculate stats
    avg_points_h = home['Points'].astype(float).mean() if not home.empty else 0
    avg_xGD_h = home['xGD'].mean() if not home.empty else 0
    count_games_h = home['Gameweek'].count() if not home.empty else 0
    count_wins_h = home['Result'].value_counts().get('W', 0) if not home.empty else 0
    count_draws_h = home['Result'].value_counts().get('D', 0) if not home.empty else 0
    count_losses_h = home['Result'].value_counts().get('L', 0) if not home.empty else 0

    avg_points_a = away['Points'].astype(float).mean() if not away.empty else 0
    avg_xGD_a = away['xGD'].mean() if not away.empty else 0
    count_games_a = away['Gameweek'].count() if not away.empty else 0
    count_wins_a = away['Result'].value_counts().get('W', 0) if not away.empty else 0
    count_draws_a = away['Result'].value_counts().get('D', 0) if not away.empty else 0
    count_losses_a = away['Result'].value_counts().get('L', 0) if not away.empty else 0

    # DataFrame
    results_df = pd.DataFrame({
        'Venue': ['Home', 'Away'],
        'Games': [count_games_h, count_games_a],
        'Wins': [count_wins_h, count_wins_a],
        'Draws': [count_draws_h, count_draws_a],
        'Losses': [count_losses_h, count_losses_a],
        'Points per Game': [avg_points_h, avg_points_a],
        'xGD': [avg_xGD_h, avg_xGD_a]
    })

    # Helper for text color
    def text_color(r, g, b):
        brightness = (r*299 + g*587 + b*114)/1000
        return 'black' if brightness > 125 else 'white'

    # Color functions
    def color_points(value):
        norm_value = (value - 0.5) / 2
        norm_value = np.clip(norm_value, 0, 1)
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_xgd(value):
        norm_value = (value + 1.4) / 2.8
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    styled_df = results_df.style.applymap(color_points, subset=['Points per Game']) \
                                .applymap(color_xgd, subset=['xGD'])
    return styled_df

def opp_formation(team):
    team_data = df_new[df_new['Team'] == team]
    unique_formations = team_data['Opp Formation'].unique()
    results = []

    for formation in unique_formations:
        formation_data = team_data[team_data['Opp Formation'] == formation]

        avg_points = formation_data['Points'].astype(float).mean() if not formation_data.empty else 0
        avg_xGD = formation_data['xGD'].mean() if not formation_data.empty else 0
        count_games = formation_data['Gameweek'].count() if not formation_data.empty else 0
        count_wins = formation_data['Result'].value_counts().get('W', 0) if not formation_data.empty else 0
        count_draws = formation_data['Result'].value_counts().get('D', 0) if not formation_data.empty else 0
        count_losses = formation_data['Result'].value_counts().get('L', 0) if not formation_data.empty else 0

        results.append({
            'Oppo Formation': formation,
            'Games': count_games,
            'Wins': count_wins,
            'Draws': count_draws,
            'Losses': count_losses,
            'Points per Game': avg_points,
            'xGD': avg_xGD
        })

    results_df = pd.DataFrame(results)

    def text_color(r, g, b):
        brightness = (r*299 + g*587 + b*114)/1000
        return 'black' if brightness > 125 else 'white'

    def color_points(value):
        norm_value = (value - 0.5) / 2
        color = cm.RdYlGn(np.clip(norm_value, 0, 1))
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_xgd(value):
        norm_value = (value + 1.4) / 2.8
        color = cm.RdYlGn(np.clip(norm_value, 0, 1))
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    styled_df = results_df.style.applymap(color_points, subset=['Points per Game']) \
                                .applymap(color_xgd, subset=['xGD'])
    return styled_df

def get_formation_analysis(team1, team2):
    team1_data = df_new[df_new['Team'] == team1]
    team2_data = df_new[df_new['Team'] == team2]

    team1_formation = team1_data['Formation'].mode().iloc[0]
    team1_formation_count = team1_data['Formation'].value_counts().iloc[0]
    team2_formation = team2_data['Formation'].mode().iloc[0]
    team2_formation_count = team2_data['Formation'].value_counts().iloc[0]

    def text_color(r, g, b):
        brightness = (r*299 + g*587 + b*114)/1000
        return 'black' if brightness > 125 else 'white'

    def color_points(value):
        norm_value = value / 3
        color = cm.RdYlGn(np.clip(norm_value, 0, 1))
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_xgd(value):
        norm_value = (value + 1.4) / 2.8
        color = cm.RdYlGn(np.clip(norm_value, 0, 1))
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def calc_vs(df, opp_formation):
        results = []
        df_vs = df[df['Opp Formation'] == opp_formation]
        for formation in df_vs['Formation'].unique():
            f_data = df_vs[df_vs['Formation'] == formation]
            results.append({
                'Formation': formation,
                'Oppo Formation': opp_formation,
                'Games': f_data['Gameweek'].count(),
                'Wins': f_data['Result'].value_counts().get('W', 0),
                'Draws': f_data['Result'].value_counts().get('D', 0),
                'Losses': f_data['Result'].value_counts().get('L', 0),
                'Points per Game': f_data['Points'].astype(float).mean(),
                'xGD': f_data['xGD'].mean()
            })
        return pd.DataFrame(results)

    team1_formations = calc_vs(team1_data, team2_formation)
    team2_formations = calc_vs(team2_data, team1_formation)

    if not team1_formations.empty:
        team1_formations = team1_formations.style.applymap(color_points, subset=['Points per Game']) \
                                                 .applymap(color_xgd, subset=['xGD'])
    if not team2_formations.empty:
        team2_formations = team2_formations.style.applymap(color_points, subset=['Points per Game']) \
                                                 .applymap(color_xgd, subset=['xGD'])

    return {
        'team1_formation': team1_formation,
        'team1_formation_count': team1_formation_count,
        'team2_formation': team2_formation,
        'team2_formation_count': team2_formation_count,
        'team1_formations': team1_formations,
        'team2_formations': team2_formations
    }

def match_logs(team):

    team_data = df_new[df_new['Team'] == team]
    team_data['Points'] = pd.to_numeric(team_data['Points'], errors='coerce')

    filtered_df = pd.DataFrame(['Gameweek', 'Venue', 'Opponent', 'Formation', 'Opp Formation', 
                                'Score', 'Points', 'xG', 'xGA', 'xGD', 'Poss']).T
    filtered_df = filtered_df.rename(columns=filtered_df.iloc[0])
    filtered_df = filtered_df.reindex(filtered_df.index.drop(0))

    team_data_filtered = team_data[filtered_df.columns]
    final_df = pd.concat([filtered_df, team_data_filtered], ignore_index=True)

    def text_color(r, g, b):
        brightness = (r*299 + g*587 + b*114)/1000
        return 'black' if brightness > 125 else 'white'

    def color_points(value):
        if value < 1:
            r, g, b = 255, 0, 0
        elif 1 <= value < 1.8:
            r, g, b = 255, 255, 0
        else:
            r, g, b = 0, 128, 0
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_xgd(value):
        norm_value = (value + 1.4) / 2.8
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_xgf(value):
        norm_value = (value - 1)
        norm_value = np.clip(norm_value, 0, 1)
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_xga(value):
        norm_value = (3 - value) / 3
        norm_value = np.clip(norm_value, 0, 1)
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    def color_poss(value):
        norm_value = (value - 30) / 40
        norm_value = np.clip(norm_value, 0, 1)
        color = cm.RdYlGn(norm_value)
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        font = text_color(r, g, b)
        return f'background-color: rgb({r},{g},{b}); color: {font}'

    styled_df = final_df.style.applymap(color_points, subset=['Points']) \
                              .applymap(color_xgd, subset=['xGD']) \
                              .applymap(color_poss, subset=['Poss']) \
                              .applymap(color_xgf, subset=['xG']) \
                              .applymap(color_xga, subset=['xGA'])

    return styled_df

## CREATE STREAMLIT DASHBOARD ##

# title, subheader and player/dashboard filter
st.set_page_config(page_title="Betting Dashboard -", layout="wide")
st.title(f"Betting Dashboard")
st.subheader("Select the 'One Team View' to filter by team to see their rolling xGD, results plotted against the current league table, effect of possession and home/away advantage on performances and results. Or filter by the 'Two Team Comparison' to see how teams may match up against each other:")
function_filter = st.radio("Select a view to apply:", ("One Team View", "Two Team Comparison"))

if function_filter == "One Team View":
    unique_team = df_new['Team'].sort_values().unique()
    team_filter = st.selectbox('Select a team:', unique_team, index=0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f'5 Game xGD Moving Average | {team_filter}:')
        average_xGD(team_filter)

    with col2:
        st.subheader(f'5 Game xGF vs xGA Moving Average | {team_filter}:')
        average_xGF_xGA(team_filter)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f'Effect of Possession on {team_filter}:')
        possession_table = possession_impact(team_filter)
        st.table(possession_table)

    with col4:
        st.subheader(f'Effect of Opposition Formation on {team_filter}:')
        opp_formation_table = opp_formation(team_filter)
        st.table(opp_formation_table)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader(f'Home/Away Performance for {team_filter}:')
        home_away_table = home_away(team_filter)
        st.table(home_away_table)

    with col6:
        st.subheader(f'{team_filter} Results Table:')
        results_table = team_results(team_filter)
        st.table(results_table)

    col7 = st.columns(1)[0]

    with col7:
        with st.expander(f'{team_filter} Match Logs:', expanded=True):
            match_logs_table = match_logs(team_filter)
            st.table(match_logs_table)

else:
    col1_select, col2_select = st.columns(2)
    
    with col1_select:
        team1 = st.selectbox('Select first team:', df_new['Team'].sort_values().unique(), index=0)
    
    with col2_select:
        # Filter out the first team from the second dropdown
        team2_options = df_new[df_new['Team'] != team1]['Team'].sort_values().unique()
        team2 = st.selectbox('Select second team:', team2_options, index=0)
    
    st.title(f"{team1} vs {team2} Comparison")
    
    # Get formation analysis
    formation_analysis = get_formation_analysis(team1, team2)
    
    # Display formation analysis at the top
    st.subheader("Formation Analysis")
    formation_col1, formation_col2 = st.columns(2)
    
    with formation_col1:
        st.write(f"{team1} most used formation: {formation_analysis['team1_formation']} ({formation_analysis['team1_formation_count']} times)")
        st.write(f"Performance against a {formation_analysis['team2_formation']}:")
        st.table(formation_analysis['team1_formations'])
    
    with formation_col2:
        st.write(f"{team2} most used formation: {formation_analysis['team2_formation']} ({formation_analysis['team2_formation_count']} times)")
        st.write(f"Performance against a {formation_analysis['team1_formation']}:")
        st.table(formation_analysis['team2_formations'])
    
    st.subheader("Click the tabs below to see the metrics for each of the team's selected:")

    # Create tabs for each team's detailed analysis
    tab1, tab2 = st.tabs([team1, team2])
    
    # Team 1 tab
    with tab1:
        col1_t1, col2_t1 = st.columns(2)
        
        with col1_t1:
            st.subheader(f'5 Game xGD Moving Average | {team1}:')
            average_xGD(team1)

        with col2_t1:
            st.subheader(f'5 Game xGF vs xGA Moving Average | {team1}:')
            average_xGF_xGA(team1)
        
        col3_t1, col4_t1 = st.columns(2)
        
        with col3_t1:
            st.subheader(f'Effect of Possession on {team1}:')
            possession_table = possession_impact(team1)
            st.table(possession_table)
        
        with col4_t1:
            st.subheader(f'Effect of Opposition Formation on {team1}:')
            opp_formation_table = opp_formation(team1)
            st.table(opp_formation_table)

        col5_t1, col6_t1 = st.columns(2)

        with col5_t1:
            st.subheader(f'Home/Away Performance for {team1}:')
            home_away_table = home_away(team1)
            st.table(home_away_table)

        with col6_t1:
            st.subheader(f'{team1} Results Table:')
            results_table = team_results(team1)
            st.table(results_table)

        col7_t1 = st.columns(1)[0]

        with col7_t1:
            with st.expander(f'{team1} Match Logs:', expanded=True):
                match_logs_table = match_logs(team1)
                st.table(match_logs_table)
    
    # Team 2 tab
    with tab2:
        col1_t2, col2_t2 = st.columns(2)
        
        with col1_t2:
            st.subheader(f'5 Game xGD Moving Average | {team2}:')
            average_xGD(team2)

        with col2_t2:
            st.subheader(f'5 Game xGF vs xGA Moving Average | {team2}:')
            average_xGF_xGA(team2)
        
        col3_t2, col4_t2 = st.columns(2)
        
        with col3_t2:
            st.subheader(f'Effect of Possession on {team2}:')
            possession_table = possession_impact(team2)
            st.table(possession_table)
        
        with col4_t2:
            st.subheader(f'Effect of Opposition Formation on {team2}:')
            opp_formation_table = opp_formation(team2)
            st.table(opp_formation_table)

        col5_t2, col6_t2 = st.columns(2)

        with col5_t2:
            st.subheader(f'Home/Away Performance for {team2}:')
            home_away_table = home_away(team2)
            st.table(home_away_table)

        with col6_t2:
            st.subheader(f'{team2} Results Table:')
            results_table = team_results(team2)
            st.table(results_table)

        col7_t2 = st.columns(1)[0]

        with col7_t2:
            with st.expander(f'{team2} Match Logs:', expanded=True):
                match_logs_table = match_logs(team2)
                st.table(match_logs_table)