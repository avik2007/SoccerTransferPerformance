import pandas as pd
import os
import numpy as np

def main():
    print("Loading datasets...")
    # 1. Load the DataFrames
    c_d = os.path.dirname(os.path.abspath(__file__))
    data_path = c_d + '/../TPP_Data/KaggleFootballData/'   
    output_path = c_d + '/../TPP_Data/TPP_ProcessedDatasets/'
    
    try:
        appearances = pd.read_csv(os.path.join(data_path, 'appearances.csv'))
        clubs = pd.read_csv(os.path.join(data_path, 'clubs.csv'))
        competitions = pd.read_csv(os.path.join(data_path, 'competitions.csv'))
        games = pd.read_csv(os.path.join(data_path, 'games.csv'))
        transfers = pd.read_csv(os.path.join(data_path, 'transfers.csv'))
        # NEW: Load players to get Date of Birth
        players = pd.read_csv(os.path.join(data_path, 'players.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Processing dates and calculating stints...")
    
    # 2. Preprocessing: Get Dates into Appearances
    appearances_with_date = appearances.merge(
        games[['game_id', 'date']], 
        on='game_id', 
        how='left'
    ).rename(columns={'date_x':'date'}).drop(columns=['date_y'], errors='ignore')
    # Note: 'date' might be duplicated depending on merge, strictly keeping one
    #if 'date' not in appearances_with_date.columns:
        # Fallback if rename failed
    #    print('nope')
    #    appearances_with_date = appearances.merge(games[['game_id', 'date']], on='game_id', how='left')
    #appearances_with_date.head()
    appearances_with_date['date'] = pd.to_datetime(appearances_with_date['date'])

    # --- THE STINT LOGIC ---
    appearances_with_date = appearances_with_date.sort_values(['player_id', 'player_club_id', 'date'])
    appearances_with_date['prev_game_date'] = appearances_with_date.groupby(['player_id', 'player_club_id'])['date'].shift(1)
    appearances_with_date['days_since_last'] = (appearances_with_date['date'] - appearances_with_date['prev_game_date']).dt.days

    STINT_THRESHOLD = 250 # This is helps to throw out loan spells
    appearances_with_date['is_new_stint'] = appearances_with_date['days_since_last'] > STINT_THRESHOLD
    appearances_with_date['is_new_stint'] = appearances_with_date['is_new_stint'].fillna(True) 

    appearances_with_date['stint_id'] = appearances_with_date.groupby(['player_id', 'player_club_id'])['is_new_stint'].cumsum()

    # 3. Aggregation: Create Player-Club-Stint Stats
    player_club_stats = appearances_with_date.groupby(['player_id', 'player_club_id', 'stint_id', 'player_name']).agg(
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        minutes_played=('minutes_played', 'sum'),
        yellow_cards=('yellow_cards', 'sum'),
        red_cards=('red_cards', 'sum'),
        total_appearances=('game_id', 'count'),
        first_app=('date', 'min'),
        last_app=('date', 'max')
    ).reset_index()

    print("Merging club and country information...")

    # 4. Merging: Add Context
    stats_with_club_info = player_club_stats.merge(
        clubs[['club_id', 'domestic_competition_id', 'name']], 
        left_on='player_club_id',
        right_on='club_id',
        how='left'
    ).rename(columns={'name': 'club_name'})

    final_player_stats = stats_with_club_info.merge(
        competitions[['competition_id', 'country_name', 'name']], 
        left_on='domestic_competition_id',
        right_on='competition_id',
        how='left'
    ).rename(columns={'name': 'competition_name'})

    final_player_stats = final_player_stats.drop(columns=['club_id', 'domestic_competition_id', 'competition_id'])

    print("Filtering for Premier League arrivals...")

    # 5. Create 'epl_arrivals_with_country'
    
    # A. Clean Fees and Dates
    transfers['fee_cleaned'] = pd.to_numeric(transfers['transfer_fee'], errors='coerce').fillna(0)
    
    # Standardize 'transfer_date'. Often it is not in transfers.csv, only 'season' is.
    # However, if your dataset version has it, we convert it.
    # If not, we cannot accurately calculate age to the day, so we might need to approximate.
    if 'transfer_date' not in transfers.columns:
        # Fallback: Create a dummy date based on season (e.g. July 1st of that season)
        transfers['transfer_date'] = pd.to_datetime(transfers['season'].astype(str) + '-07-01')
    else:
        transfers['transfer_date'] = pd.to_datetime(transfers['transfer_date'])

    is_loan_end = transfers['transfer_fee'].astype(str).str.contains('End of loan', case=False, na=False)
    clean_transfers = transfers[~is_loan_end].copy()

    # B. Identify BUYING Club
    transfers_buying = clean_transfers.merge(
        clubs[['club_id', 'domestic_competition_id', 'name']],
        left_on='to_club_id',
        right_on='club_id',
        how='left',
        suffixes=('', '_buying')
    ).rename(columns={'name': 'buying_club_name', 'domestic_competition_id': 'buying_league_id'})

    # C. Identify SELLING Club
    transfers_full = transfers_buying.merge(
        clubs[['club_id', 'domestic_competition_id', 'name']],
        left_on='from_club_id',
        right_on='club_id',
        how='left',
        suffixes=('', '_selling')
    ).rename(columns={'name': 'selling_club_name', 'domestic_competition_id': 'selling_league_id'})

    # D. Competition info for BUYING league
    transfers_full = transfers_full.merge(
        competitions[['competition_id', 'name', 'country_name']],
        left_on='buying_league_id',
        right_on='competition_id',
        how='left'
    ).rename(columns={'name': 'buying_league_name', 'country_name': 'buying_country'})

    # E. Filter for EPL Arrivals
    epl_mask = (transfers_full['buying_league_id'] == 'GB1') | \
               ((transfers_full['buying_country'] == 'England') & (transfers_full['buying_league_name'].str.contains('Premier League', na=False)))

    epl_arrivals_with_country = transfers_full[epl_mask].copy()

    # F. Add Origin League Context
    epl_arrivals_with_country = epl_arrivals_with_country.merge(
        competitions[['competition_id', 'name', 'country_name']],
        left_on='selling_league_id',
        right_on='competition_id',
        how='left',
        suffixes=('', '_origin')
    ).rename(columns={'name': 'origin_league_name', 'country_name': 'origin_country'})

    # --- NEW STEP G: ADD AGE ---
    # Merge with players.csv to get date_of_birth
    epl_arrivals_with_country = epl_arrivals_with_country.merge(
        players[['player_id', 'date_of_birth']],
        on='player_id',
        how='left'
    )
    
    # Convert DOB to datetime
    epl_arrivals_with_country['date_of_birth'] = pd.to_datetime(epl_arrivals_with_country['date_of_birth'], errors='coerce')
    
    # Calculate Age at Transfer
    # (Transfer Date - DOB) / 365.25
    epl_arrivals_with_country['age_at_transfer'] = (
        (epl_arrivals_with_country['transfer_date'] - epl_arrivals_with_country['date_of_birth']).dt.days / 365.25
    ).round(1)

    # Use the calculated age, filling NaNs with the 'age' column if it existed in transfers (rare)
    if 'age' in epl_arrivals_with_country.columns:
        epl_arrivals_with_country['age'] = epl_arrivals_with_country['age_at_transfer'].fillna(epl_arrivals_with_country['age'])
    else:
        epl_arrivals_with_country['age'] = epl_arrivals_with_country['age_at_transfer']

    # Select columns
    cols_to_keep = [
        'player_id', 'player_name', 'age', 'date_of_birth', # Added DOB for verification
        'season', 'transfer_date', 
        'transfer_fee', 'fee_cleaned', 'market_value_in_eur',
        'from_club_id', 'to_club_id',
        'buying_club_name', 'buying_league_name',
        'selling_club_name', 'origin_league_name', 'origin_country'
    ]
    
    epl_arrivals_with_country = epl_arrivals_with_country[[c for c in cols_to_keep if c in epl_arrivals_with_country.columns]]

    # 6. Save
    print("Saving outputs to CSV...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    final_player_stats.to_csv(output_path + 'final_player_stats.csv', index=False)
    epl_arrivals_with_country.to_csv(output_path + 'epl_arrivals_with_country.csv', index=False)

    print("Success! Files created.")

if __name__ == "__main__":
    main()