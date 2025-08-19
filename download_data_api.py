import requests
import pandas as pd
import os
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create data directory if it doesn't exist
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class ESPNCollegeFootballAPI:
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
        
    def get_scoreboard(self, year, week=None, season_type=2):
        """
        Get scoreboard data for specific year and week
        Using different ESPN API endpoints and date-based filtering
        """
        
        # Try multiple ESPN API approaches
        api_attempts = [
            self._try_calendar_based_approach(year, week, season_type),
            self._try_direct_date_approach(year, week, season_type),
            self._try_games_endpoint(year, week, season_type),
            self._try_alternative_scoreboard(year, week, season_type)
        ]
        
        for i, attempt_func in enumerate(api_attempts):
            try:
                print(f"Trying API approach {i+1} for {year} week {week}")
                data = attempt_func()
                
                if data and self._validate_year_data(data, year):
                    print(f"✅ API approach {i+1} succeeded for {year}")
                    return data
                else:
                    print(f"❌ API approach {i+1} failed validation for {year}")
                    
            except Exception as e:
                print(f"API approach {i+1} failed with error: {e}")
                continue
        
        print(f"All API approaches failed for {year} week {week}")
        return None
    
    def _try_calendar_based_approach(self, year, week, season_type):
        """Try using calendar endpoint to find games"""
        def attempt():
            url = f"{self.base_url}/calendar"
            params = {'year': year, 'seasontype': season_type}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            calendar_data = response.json()
            
            # Find the specific week's date range
            target_dates = self._extract_week_dates(calendar_data, week)
            if not target_dates:
                return None
            
            # Get games for those specific dates
            return self._get_games_by_dates(target_dates)
        
        return attempt
    
    def _try_direct_date_approach(self, year, week, season_type):
        """Try calculating dates based on typical college football schedule"""
        def attempt():
            # Calculate approximate dates for the week
            start_date, end_date = self._calculate_week_dates(year, week, season_type)
            
            url = f"{self.base_url}/scoreboard"
            params = {
                'dates': f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        return attempt
    
    def _try_games_endpoint(self, year, week, season_type):
        """Try using games endpoint with different parameters"""
        def attempt():
            url = f"{self.base_url}/games"
            params = {
                'year': year,
                'seasontype': season_type,
                'week': week,
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        return attempt
    
    def _try_alternative_scoreboard(self, year, week, season_type):
        """Try alternative scoreboard parameters"""
        def attempt():
            url = f"{self.base_url}/scoreboard"
            
            # Try with group parameter (FBS = 80)
            params = {
                'groups': '80',  # FBS teams
                'year': year,
                'seasontype': season_type,
                'week': week
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        return attempt
    
    def _calculate_week_dates(self, year, week, season_type):
        """Calculate approximate start/end dates for a given week"""
        import datetime as dt
        
        # College football typically starts in late August/early September
        if season_type == 1:  # Preseason
            season_start = dt.date(year, 8, 15)  # Mid August
        elif season_type == 2:  # Regular season
            season_start = dt.date(year, 8, 25)  # Late August
        else:  # Postseason
            season_start = dt.date(year, 12, 15)  # Mid December
        
        # Find the first Saturday of the season
        days_ahead = 5 - season_start.weekday()  # Saturday = 5
        if days_ahead <= 0:
            days_ahead += 7
        first_saturday = season_start + dt.timedelta(days=days_ahead)
        
        # Calculate week start (typically Thursday before)
        week_start = first_saturday + dt.timedelta(weeks=week-1, days=-2)
        week_end = week_start + dt.timedelta(days=6)
        
        return week_start, week_end
    
    def _extract_week_dates(self, calendar_data, target_week):
        """Extract date ranges from calendar data for specific week"""
        # This would parse ESPN's calendar format to find date ranges
        # Implementation depends on ESPN's calendar structure
        return None  # Placeholder
    
    def _get_games_by_dates(self, date_range):
        """Get games for specific date range"""
        # Implementation for date-based game retrieval
        return None  # Placeholder
    
    def _validate_year_data(self, data, expected_year):
        """Validate that the returned data is actually from the expected year"""
        if not data or 'events' not in data:
            return False
            
        events = data.get('events', [])
        if not events:
            return False
        
        valid_events = 0
        total_checked = 0
        
        for event in events[:10]:  # Check first 10 events
            event_date = event.get('date')
            if event_date:
                try:
                    event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                    event_year = event_datetime.year
                    
                    total_checked += 1
                    if event_year == expected_year:
                        valid_events += 1
                    
                except Exception as e:
                    continue
        
        if total_checked > 0:
            validation_rate = valid_events / total_checked
            print(f"Validation: {valid_events}/{total_checked} events from correct year ({validation_rate:.1%})")
            return validation_rate >= 0.7  # Require 70% to be from correct year
        
        return False
    
    def get_season_weeks(self, year, season_type=2):
        """Get reasonable week counts based on season type and year"""
        
        # Standard week counts by season type
        standard_weeks = {
            1: 4,   # Preseason
            2: 15,  # Regular season (can vary 12-17)
            3: 6    # Postseason
        }
        
        # Adjustments by year (some seasons had different lengths)
        if season_type == 2:  # Regular season adjustments
            if year == 2020:
                return 16  # COVID-19 affected season
            elif year <= 2013:
                return 14  # Shorter seasons in earlier years
            elif year >= 2014:
                return 15  # Standard modern length
        
        return standard_weeks.get(season_type, 15)
    
    def get_comprehensive_teams(self):
        """Get comprehensive list of FBS college football teams using multiple approaches"""
        print("Fetching comprehensive team list...")
        
        all_teams = {}
        
        # Method 1: Try teams endpoint with FBS group
        teams_1 = self._get_teams_by_group()
        if teams_1:
            for team in teams_1:
                all_teams[team['team_id']] = team
        
        # Method 2: Extract teams from conference data
        teams_2 = self._get_teams_from_conferences()
        if teams_2:
            for team in teams_2:
                all_teams[team['team_id']] = team
        
        # Method 3: Extract teams from recent games data
        teams_3 = self._get_teams_from_games()
        if teams_3:
            for team in teams_3:
                all_teams[team['team_id']] = team
        
        # Method 4: Add known missing major teams manually
        teams_4 = self._add_known_major_teams()
        if teams_4:
            for team in teams_4:
                all_teams[team['team_id']] = team
        
        print(f"Found {len(all_teams)} total teams using multiple methods")
        return list(all_teams.values())
    
    def _get_teams_by_group(self):
        """Method 1: Get teams using FBS group parameter"""
        teams = []
        
        # Try different group IDs for FBS teams
        group_ids = ['80', '1', 'fbs']  # Different ways ESPN might categorize FBS
        
        for group_id in group_ids:
            try:
                url = f"{self.base_url}/teams"
                params = {'groups': group_id, 'limit': 200}
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                extracted_teams = self._extract_teams_from_response(data)
                if extracted_teams:
                    print(f"Method 1 (group {group_id}): Found {len(extracted_teams)} teams")
                    teams.extend(extracted_teams)
                    break  # If we got teams, use this group
                    
            except Exception as e:
                print(f"Group {group_id} failed: {e}")
                continue
        
        return teams
    
    def _get_teams_from_conferences(self):
        """Method 2: Get teams by fetching conference data"""
        teams = []
        
        try:
            # Get conference standings which should include all FBS teams
            url = f"{self.base_url}/standings"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'children' in data:
                for conference in data['children']:
                    if 'standings' in conference:
                        for standing_group in conference['standings']:
                            if 'entries' in standing_group:
                                for entry in standing_group['entries']:
                                    team = entry.get('team')
                                    if team:
                                        team_info = self._format_team_info(team)
                                        if team_info:
                                            teams.append(team_info)
            
            print(f"Method 2 (conferences): Found {len(teams)} teams")
            
        except Exception as e:
            print(f"Conference method failed: {e}")
        
        return teams
    
    def _get_teams_from_games(self):
        """Method 3: Extract teams from recent games data"""
        teams = {}
        
        try:
            # Get current season games to extract team data
            current_year = datetime.now().year
            data = self.get_scoreboard(current_year, season_type=2)
            
            if data and 'events' in data:
                for event in data['events']:
                    competition = event.get('competitions', [{}])[0]
                    competitors = competition.get('competitors', [])
                    
                    for competitor in competitors:
                        team = competitor.get('team')
                        if team:
                            team_info = self._format_team_info(team)
                            if team_info:
                                teams[team_info['team_id']] = team_info
            
            print(f"Method 3 (games): Found {len(teams)} teams")
            
        except Exception as e:
            print(f"Games extraction method failed: {e}")
        
        return list(teams.values())
    
    def _add_known_major_teams(self):
        """Method 4: Add known major FBS teams that might be missing"""
        # This is a fallback for major teams that should definitely be included
        known_teams = [
            # SEC
            {'team_id': 333, 'name': 'Alabama Crimson Tide', 'abbreviation': 'ALA', 'conference_id': 8},
            {'team_id': 61, 'name': 'Georgia Bulldogs', 'abbreviation': 'UGA', 'conference_id': 8},
            {'team_id': 57, 'name': 'Florida Gators', 'abbreviation': 'FLA', 'conference_id': 8},
            {'team_id': 2, 'name': 'Auburn Tigers', 'abbreviation': 'AUB', 'conference_id': 8},
            {'team_id': 99, 'name': 'LSU Tigers', 'abbreviation': 'LSU', 'conference_id': 8},
            {'team_id': 142, 'name': 'Tennessee Volunteers', 'abbreviation': 'TENN', 'conference_id': 8},
            
            # Big Ten
            {'team_id': 194, 'name': 'Michigan Wolverines', 'abbreviation': 'MICH', 'conference_id': 5},
            {'team_id': 194, 'name': 'Ohio State Buckeyes', 'abbreviation': 'OSU', 'conference_id': 5},
            {'team_id': 213, 'name': 'Penn State Nittany Lions', 'abbreviation': 'PSU', 'conference_id': 5},
            {'team_id': 275, 'name': 'Wisconsin Badgers', 'abbreviation': 'WIS', 'conference_id': 5},
            
            # Big 12
            {'team_id': 251, 'name': 'Texas Longhorns', 'abbreviation': 'TEX', 'conference_id': 4},
            {'team_id': 201, 'name': 'Oklahoma Sooners', 'abbreviation': 'OU', 'conference_id': 4},
            {'team_id': 66, 'name': 'Iowa State Cyclones', 'abbreviation': 'ISU', 'conference_id': 4},
            
            # ACC
            {'team_id': 52, 'name': 'Florida State Seminoles', 'abbreviation': 'FSU', 'conference_id': 1},
            {'team_id': 150, 'name': 'Clemson Tigers', 'abbreviation': 'CLEM', 'conference_id': 1},
            {'team_id': 120, 'name': 'Miami Hurricanes', 'abbreviation': 'MIA', 'conference_id': 1},
            
            # Pac-12
            {'team_id': 30, 'name': 'USC Trojans', 'abbreviation': 'USC', 'conference_id': 9},
            {'team_id': 26, 'name': 'UCLA Bruins', 'abbreviation': 'UCLA', 'conference_id': 9},
            {'team_id': 24, 'name': 'Stanford Cardinal', 'abbreviation': 'STAN', 'conference_id': 9},
            {'team_id': 12, 'name': 'Arizona Wildcats', 'abbreviation': 'ARIZ', 'conference_id': 9},
        ]
        
        formatted_teams = []
        for team_data in known_teams:
            formatted_team = {
                'team_id': team_data['team_id'],
                'name': team_data['name'],
                'short_name': team_data['name'].split()[-1],  # Last word
                'abbreviation': team_data['abbreviation'],
                'nickname': team_data['name'].split()[-1],
                'mascot': team_data['name'].split()[-1],
                'color': '000000',  # Default
                'alternate_color': 'ffffff',  # Default
                'logo': None,
                'conference_id': team_data.get('conference_id'),
                'location': team_data['name'].split()[0],  # First word
                'venue_id': None,
                'venue_name': None
            }
            formatted_teams.append(formatted_team)
        
        print(f"Method 4 (known teams): Added {len(formatted_teams)} major teams")
        return formatted_teams
    
    def _extract_teams_from_response(self, data):
        """Extract team data from ESPN API response"""
        teams = []
        
        if 'sports' in data:
            for sport in data['sports']:
                for league in sport.get('leagues', []):
                    for team_wrapper in league.get('teams', []):
                        team = team_wrapper.get('team')
                        if team:
                            team_info = self._format_team_info(team)
                            if team_info:
                                teams.append(team_info)
        
        return teams
    
    def _format_team_info(self, team):
        """Format team data consistently"""
        try:
            return {
                'team_id': team.get('id'),
                'name': team.get('displayName'),
                'short_name': team.get('shortDisplayName'),
                'abbreviation': team.get('abbreviation'),
                'nickname': team.get('nickname'),
                'mascot': team.get('name'),
                'color': team.get('color'),
                'alternate_color': team.get('alternateColor'),
                'logo': team.get('logo'),
                'conference_id': team.get('conferenceId'),
                'location': team.get('location'),
                'venue_id': team.get('venue', {}).get('id') if team.get('venue') else None,
                'venue_name': team.get('venue', {}).get('fullName') if team.get('venue') else None
            }
        except Exception as e:
            print(f"Error formatting team: {e}")
            return None

def process_teams_data(teams_data):
    """Process teams data into a flat structure for CSV"""
    # If we got data from comprehensive teams method, it's already processed
    if isinstance(teams_data, list) and teams_data and 'team_id' in teams_data[0]:
        return teams_data
    
    # Otherwise, process the ESPN response format
    teams = []
    if teams_data and 'sports' in teams_data:
        for sport in teams_data['sports']:
            for league in sport.get('leagues', []):
                for team in league.get('teams', []):
                    team_info = {
                        'team_id': team.get('team', {}).get('id'),
                        'name': team.get('team', {}).get('displayName'),
                        'short_name': team.get('team', {}).get('shortDisplayName'),
                        'abbreviation': team.get('team', {}).get('abbreviation'),
                        'nickname': team.get('team', {}).get('nickname'),
                        'mascot': team.get('team', {}).get('name'),
                        'color': team.get('team', {}).get('color'),
                        'alternate_color': team.get('team', {}).get('alternateColor'),
                        'logo': team.get('team', {}).get('logo'),
                        'conference_id': team.get('team', {}).get('conferenceId'),
                        'location': team.get('team', {}).get('location'),
                        'venue_id': team.get('team', {}).get('venue', {}).get('id'),
                        'venue_name': team.get('team', {}).get('venue', {}).get('fullName')
                    }
                    teams.append(team_info)
    
    return teams

def process_scoreboard_data(scoreboard_data, expected_year, expected_week, season_type):
    """Process scoreboard data with strict year validation"""
    games = []
    skipped_games = 0
    
    if not scoreboard_data or 'events' not in scoreboard_data:
        return games
    
    for event in scoreboard_data['events']:
        # Strict date validation
        event_date = event.get('date')
        if not event_date:
            skipped_games += 1
            continue
            
        try:
            event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            event_year = event_datetime.year
            
            # Only process events from the exact expected year
            if event_year != expected_year:
                skipped_games += 1
                continue
                
        except Exception as e:
            print(f"Error parsing date {event_date}: {e}")
            skipped_games += 1
            continue
        
        # Only include completed games for historical years
        current_year = datetime.now().year
        is_completed = event.get('status', {}).get('type', {}).get('completed', False)
        
        if expected_year < current_year and not is_completed:
            skipped_games += 1
            continue
        
        # Skip games without US venues for historical data
        venue_state = event.get('competitions', [{}])[0].get('venue', {}).get('address', {}).get('state')
        if expected_year < current_year and not venue_state:
            skipped_games += 1
            continue
        
        # Get actual week from event
        actual_week = expected_week
        week_info = event.get('week', {})
        if isinstance(week_info, dict) and 'number' in week_info:
            actual_week = week_info['number']
        elif isinstance(week_info, int):
            actual_week = week_info
        
        # Build game info
        game_info = {
            'season': expected_year,
            'week': actual_week,
            'season_type': season_type,
            'game_id': event.get('id'),
            'date': event.get('date'),
            'name': event.get('name'),
            'short_name': event.get('shortName'),
            'completed': is_completed,
            'status_description': event.get('status', {}).get('type', {}).get('description'),
            'clock': event.get('status', {}).get('clock', 0),
            'period': event.get('status', {}).get('period', 0),
            'venue_name': event.get('competitions', [{}])[0].get('venue', {}).get('fullName'),
            'venue_city': event.get('competitions', [{}])[0].get('venue', {}).get('address', {}).get('city'),
            'venue_state': venue_state,
            'venue_indoor': event.get('competitions', [{}])[0].get('venue', {}).get('indoor'),
            'neutral_site': event.get('competitions', [{}])[0].get('neutralSite'),
            'conference_competition': event.get('competitions', [{}])[0].get('conferenceCompetition'),
            'attendance': event.get('competitions', [{}])[0].get('attendance'),
        }
        
        # Process team information
        competition = event.get('competitions', [{}])[0]
        competitors = competition.get('competitors', [])
        competitors_sorted = sorted(competitors, key=lambda x: x.get('homeAway', 'away'))
        
        for i, competitor in enumerate(competitors_sorted):
            team_prefix = f"team_{i+1}"
            team = competitor.get('team', {})
            
            game_info.update({
                f"{team_prefix}_id": team.get('id'),
                f"{team_prefix}_name": team.get('displayName'),
                f"{team_prefix}_abbreviation": team.get('abbreviation'),
                f"{team_prefix}_mascot": team.get('name'),
                f"{team_prefix}_score": competitor.get('score'),
                f"{team_prefix}_home_away": competitor.get('homeAway'),
                f"{team_prefix}_winner": competitor.get('winner'),
                f"{team_prefix}_conference": team.get('conferenceId'),
            })
            
            # Add rankings if available
            if competitor.get('curatedRank'):
                game_info[f"{team_prefix}_rank"] = competitor.get('curatedRank', {}).get('current')
            
            # Add records
            records = competitor.get('records', [])
            for record in records:
                record_type = record.get('type', 'overall')
                game_info[f"{team_prefix}_record_{record_type}"] = record.get('summary')
        
        games.append(game_info)
    
    print(f"Processed {len(games)} games, skipped {skipped_games} games")
    return games

def save_to_csv(data, filename):
    """Save data to CSV file"""
    if not data:
        return None
        
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.DataFrame(data)
    
    # If file exists, append; otherwise create new
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # Remove duplicates based on game_id if it exists
        if 'game_id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['game_id'], keep='last')
        combined_df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    return filepath

def download_season_data(year, season_type=2):
    """Download all data for a specific season"""
    espn_api = ESPNCollegeFootballAPI()
    
    print(f'Starting season {year}...')
    
    # Get number of weeks in season
    max_weeks = espn_api.get_season_weeks(year, season_type)
    print(f'Total weeks in season: {max_weeks}')
    
    all_games = []
    
    for week in range(1, max_weeks + 1):
        print(f'Downloading season {year}, week {week}...')
        
        # Add delay to be respectful to ESPN's servers
        time.sleep(0.15)
        
        try:
            scoreboard_data = espn_api.get_scoreboard(year, week, season_type)
            if scoreboard_data:
                games = process_scoreboard_data(scoreboard_data, year, week, season_type)
                all_games.extend(games)
                print(f"Week {week}: Found {len(games)} games")
            else:
                print(f"Week {week}: No data returned")
                
        except Exception as e:
            error_msg = f"Error downloading {year} week {week}: {str(e)}"
            print(error_msg)
    
    # Save all games for this season
    if all_games:
        save_to_csv(all_games, f'games_{year}_season{season_type}.csv')
        print(f"Saved {len(all_games)} total games for {year} season {season_type}")
    else:
        print(f"No games found for {year} season {season_type}")
        
    return len(all_games)

def download_historical_data(start_year, end_year, season_types=[2]):
    """Download historical data for multiple seasons"""
    
    total_games = 0
    
    try:
        for year in range(start_year, end_year + 1):
            for season_type in season_types:
                season_name = {1: 'preseason', 2: 'regular', 3: 'postseason'}[season_type]
                print(f"\n{'='*60}")
                print(f"Downloading {season_name} season {year}...")
                print(f"{'='*60}")
                
                games_count = download_season_data(year, season_type)
                total_games += games_count
                
                print(f"✅ Completed {season_name} season {year}: {games_count} games")
        
        print(f'Download completed! Total games: {total_games}')
        
        # Also download current teams data
        print(f"\n{'='*60}")
        print("Downloading teams data...")
        print(f"{'='*60}")
        
        espn_api = ESPNCollegeFootballAPI()
        teams_data = espn_api.get_comprehensive_teams()
        if teams_data:
            processed_teams = process_teams_data(teams_data)
            save_to_csv(processed_teams, 'teams.csv')
            print(f"✅ Saved {len(processed_teams)} teams")
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")

def download_teams_data():
    """Download only teams data"""
    print("Downloading teams data...")
    
    espn_api = ESPNCollegeFootballAPI()
    teams_data = espn_api.get_comprehensive_teams()
    
    if teams_data:
        # Remove duplicates
        unique_teams = {}
        for team in teams_data:
            if team['team_id'] and team['team_id'] not in unique_teams:
                unique_teams[team['team_id']] = team
        
        final_teams = list(unique_teams.values())
        save_to_csv(final_teams, 'teams.csv')
        
        print(f'Successfully downloaded {len(final_teams)} teams')
        return len(final_teams)
    else:
        print('No teams data could be retrieved')
        return 0
