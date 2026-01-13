import pandas as pd

def main():
    print("Loading datasets...")
    # 1. Load the DataFrames
    # Ensure these CSV files are in the same directory as this script
    try:
        appearances = pd.read_csv('appearances.csv')
        clubs = pd.read_csv('clubs.csv')
        competitions = pd.read_csv('competitions.csv')
        games = pd.read_csv('games.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Processing dates and aggregating stats...")
    
    # 2. Preprocessing: Get Dates into Appearances
    # We merge appearances with games to attach the specific match date to every appearance
    appearances_with_date = appearances.merge(
        games[['game_id', 'date']], 
        on='game_id', 
        how='left'
    )
    
    # Convert date column to datetime objects for accurate sorting and min/max calculations
    appearances_with_date['date'] = pd.to_datetime(appearances_with_date['date'])

    # 3. Aggregation: Create Player-Club Stats
    # Group by player and the specific club they played for to get historical stints
    player_club_stats = appearances_with_date.groupby(['player_id', 'player_club_id']).agg(
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        minutes_played=('minutes_played', 'sum'),
        yellow_cards=('yellow_cards', 'sum'),
        red_cards=('red_cards', 'sum'),
        total_appearances=('game_id', 'count'),
        first_app=('date', 'min'),  # Earliest appearance date
        last_app=('date', 'max')    # Latest appearance date
    ).reset_index()

    print("Merging club and country information...")

    # 4. Merging: Add Context (Country and Competition)
    # Step A: Merge with clubs to get the domestic_competition_id and club name
    stats_with_club_info = player_club_stats.merge(
        clubs[['club_id', 'domestic_competition_id', 'name']], 
        left_on='player_club_id',
        right_on='club_id',
        how='left'
    ).rename(columns={'name': 'club_name'})

    # Step B: Merge with competitions to get country_name and competition_name
    final_player_stats = stats_with_club_info.merge(
        competitions[['competition_id', 'country_name', 'name']], 
        left_on='domestic_competition_id',
        right_on='competition_id',
        how='left'
    ).rename(columns={'name': 'competition_name'})

    # Clean up: Drop the ID columns used for merging as they are no longer needed
    final_player_stats = final_player_stats.drop(
        columns=['club_id', 'domestic_competition_id', 'competition_id']
    )

    print("Filtering for Premier League arrivals...")

    # 5. Create 'epl_arrivals_with_country'
    # Filter for rows where the country is England and the competition looks like the Premier League
    epl_mask = (
        (final_player_stats['country_name'] == 'England') &
        (final_player_stats['competition_name'].str.contains('Premier League', case=False, na=False))
    )
    
    epl_arrivals_with_country = final_player_stats[epl_mask].copy()

    # Sort by the first appearance date to see the timeline of arrivals
    epl_arrivals_with_country = epl_arrivals_with_country.sort_values(by='first_app')

    # 6. Save to CSV
    print("Saving outputs to CSV...")
    final_player_stats.to_csv('final_player_stats.csv', index=False)
    epl_arrivals_with_country.to_csv('epl_arrivals_with_country.csv', index=False)

    print("Success! 'final_player_stats.csv' and 'epl_arrivals_with_country.csv' have been created.")

if __name__ == "__main__":
    main()
