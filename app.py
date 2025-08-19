from flask import Flask, request, jsonify, g, send_file
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sqlite3
import threading
import time
from download_data_api import ESPNCollegeFootballAPI, download_teams_data, download_season_data

model_loaded = False

app = Flask(__name__)

# Configuration
app.config['DATA_DIR'] = 'data'
app.config['MODEL_PATH'] = 'football_score_predictor.pkl'
app.config['DATABASE'] = 'college_football.db'
app.config['FRONTEND_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with teams and games tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                short_name TEXT,
                abbreviation TEXT,
                mascot TEXT,
                conference_id INTEGER,
                location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                season INTEGER,
                week INTEGER,
                season_type INTEGER,
                date TEXT,
                team_1_id INTEGER,
                team_1_score INTEGER,
                team_1_home_away TEXT,
                team_2_id INTEGER,
                team_2_score INTEGER,
                team_2_home_away TEXT,
                completed BOOLEAN,
                venue_name TEXT,
                venue_state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_1_id) REFERENCES teams (team_id),
                FOREIGN KEY (team_2_id) REFERENCES teams (team_id)
            )
        ''')
        
        # Team stats table (for caching historical performance)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                team_id INTEGER,
                season INTEGER,
                games_played INTEGER,
                wins INTEGER,
                losses INTEGER,
                avg_score REAL,
                avg_allowed REAL,
                home_record TEXT,
                away_record TEXT,
                conference_record TEXT,
                current_rank INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, season),
                FOREIGN KEY (team_id) REFERENCES teams (team_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_teams_from_csv(self):
        """Load teams from CSV into database"""
        csv_path = os.path.join(app.config['DATA_DIR'], 'teams.csv')
        if not os.path.exists(csv_path):
            return False
        
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(self.db_path)
        
        for _, row in df.iterrows():
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO teams 
                (team_id, name, short_name, abbreviation, mascot, conference_id, location)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get('team_id'),
                row.get('name'),
                row.get('short_name'),
                row.get('abbreviation'),
                row.get('mascot'),
                row.get('conference_id'),
                row.get('location')
            ))
        
        conn.commit()
        conn.close()
        return True
    
    def search_teams(self, query):
        """Search teams by name, abbreviation, or mascot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT team_id, name, abbreviation, mascot, location
            FROM teams 
            WHERE name LIKE ? OR abbreviation LIKE ? OR mascot LIKE ? OR location LIKE ?
            ORDER BY name
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'team_id': row[0],
                'name': row[1],
                'abbreviation': row[2],
                'mascot': row[3],
                'location': row[4]
            }
            for row in results
        ]
    
    def get_team_by_id(self, team_id):
        """Get team information by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT team_id, name, abbreviation, mascot, location, conference_id
            FROM teams WHERE team_id = ?
        ''', (team_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'team_id': result[0],
                'name': result[1],
                'abbreviation': result[2],
                'mascot': result[3],
                'location': result[4],
                'conference_id': result[5]
            }
        return None
    
    def get_team_stats(self, team_id, season=None):
        """Get team statistics for a season"""
        if season is None:
            season = datetime.now().year
        
        # Calculate stats from games data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get games where team played
        cursor.execute('''
            SELECT 
                CASE WHEN team_1_id = ? THEN team_1_score ELSE team_2_score END as team_score,
                CASE WHEN team_1_id = ? THEN team_2_score ELSE team_1_score END as opp_score,
                CASE WHEN team_1_id = ? THEN team_1_home_away ELSE team_2_home_away END as home_away,
                completed
            FROM games 
            WHERE (team_1_id = ? OR team_2_id = ?) AND season = ? AND completed = 1
        ''', (team_id, team_id, team_id, team_id, team_id, season))
        
        games = cursor.fetchall()
        conn.close()
        
        if not games:
            return {
                'wins': 0, 'losses': 0, 'games_played': 0,
                'avg_score': 21.0, 'avg_allowed': 21.0,
                'home_record': '0-0', 'away_record': '0-0'
            }
        
        wins = sum(1 for g in games if g[0] > g[1])
        losses = len(games) - wins
        avg_score = sum(g[0] for g in games) / len(games)
        avg_allowed = sum(g[1] for g in games) / len(games)
        
        home_games = [g for g in games if g[2] == 'home']
        away_games = [g for g in games if g[2] == 'away']
        
        home_wins = sum(1 for g in home_games if g[0] > g[1])
        home_losses = len(home_games) - home_wins
        away_wins = sum(1 for g in away_games if g[0] > g[1])
        away_losses = len(away_games) - away_wins
        
        return {
            'wins': wins,
            'losses': losses,
            'games_played': len(games),
            'avg_score': avg_score,
            'avg_allowed': avg_allowed,
            'home_record': f'{home_wins}-{home_losses}',
            'away_record': f'{away_wins}-{away_losses}'
        }

# Your existing model classes (updated versions from previous response)
class EnsembleFootballPredictor:
    def __init__(self):
        self.pipelines_team1 = {
            'rf': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'gb': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]),
            'lr': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
        }
        
        self.pipelines_team2 = {
            'rf': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'gb': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]),
            'lr': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
        }
        
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X, y1, y2):
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        for name, pipeline in self.pipelines_team1.items():
            pipeline.fit(X, y1)
        
        for name, pipeline in self.pipelines_team2.items():
            pipeline.fit(X, y2)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.feature_names and hasattr(X, 'columns'):
            if list(X.columns) != self.feature_names:
                X = X[self.feature_names]
        
        if hasattr(X, 'replace'):
            X = X.replace([np.inf, -np.inf], np.nan)
        
        team1_preds = []
        team2_preds = []
        
        for pipeline in self.pipelines_team1.values():
            team1_preds.append(pipeline.predict(X))
        
        for pipeline in self.pipelines_team2.values():
            team2_preds.append(pipeline.predict(X))
        
        team1_pred = np.mean(team1_preds, axis=0)
        team2_pred = np.mean(team2_preds, axis=0)
        
        return team1_pred, team2_pred

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_trained_at = None
        self.db_manager = DatabaseManager(app.config['DATABASE'])
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        model_path = app.config['MODEL_PATH']
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.model_trained_at = datetime.fromtimestamp(os.path.getmtime(model_path))
                print(f"Loaded existing model from {model_path}")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
        
        # Train new model
        return self.train_new_model()
    
    def train_new_model(self):
        """Train a new model from scratch"""
        print("Training new model...")
        
        try:
            # Load and process all game data
            X, y1, y2 = self.prepare_training_data()
            
            if len(X) < 100:  # Need sufficient data
                print(f"Insufficient training data: {len(X)} games")
                return False
            
            # Train model
            self.model = EnsembleFootballPredictor()
            self.model.fit(X, y1, y2)
            
            # Save model
            joblib.dump(self.model, app.config['MODEL_PATH'])
            self.model_trained_at = datetime.now()
            
            print(f"Model trained successfully with {len(X)} games")
            return True
            
        except Exception as e:
            print(f"Model training failed: {e}")
            return False
    
    def prepare_training_data(self):
        """Prepare training data from CSV files"""
        # Load all game CSV files
        csv_files = glob.glob(os.path.join(app.config['DATA_DIR'], "games_*.csv"))
        
        if not csv_files:
            raise ValueError("No game data files found")
        
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Process features (simplified version of your existing code)
        df_processed = self.engineer_features(combined_df)
        X, y1, y2 = self.prepare_features(df_processed)
        
        return X, y1, y2
    
    def engineer_features(self, df):
        """Simplified feature engineering"""
        df = df.copy()
        
        # Only completed games
        df = df[df['completed'] == True].copy()
        df = df.dropna(subset=['team_1_score', 'team_2_score'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Parse records
        def parse_record(record_str):
            if pd.isna(record_str) or record_str == '':
                return 0, 0
            try:
                wins, losses = str(record_str).split('-')
                return int(wins), int(losses)
            except:
                return 0, 0
        
        df['team_1_wins'], df['team_1_losses'] = zip(*df['team_1_record_total'].fillna('0-0').apply(parse_record))
        df['team_2_wins'], df['team_2_losses'] = zip(*df['team_2_record_total'].fillna('0-0').apply(parse_record))
        
        df['team_1_win_pct'] = df['team_1_wins'] / (df['team_1_wins'] + df['team_1_losses'] + 0.001)
        df['team_2_win_pct'] = df['team_2_wins'] / (df['team_2_wins'] + df['team_2_losses'] + 0.001)
        
        # Home field
        df['team_1_is_home'] = (df['team_1_home_away'] == 'home').astype(int)
        df['team_2_is_home'] = (df['team_2_home_away'] == 'home').astype(int)
        
        # Conference game
        df['conference_game'] = df['conference_competition'].fillna(False).astype(int)
        
        # Rankings
        df['team_1_ranked'] = (~pd.isna(df['team_1_rank'])).astype(int)
        df['team_2_ranked'] = (~pd.isna(df['team_2_rank'])).astype(int)
        df['team_1_rank_filled'] = df['team_1_rank'].fillna(99)
        df['team_2_rank_filled'] = df['team_2_rank'].fillna(99)
        
        # Venue info
        df['venue_indoor_int'] = df['venue_indoor'].fillna(False).astype(int)
        df['neutral_site_int'] = df['neutral_site'].fillna(False).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix"""
        feature_columns = [
            'week', 'season_type', 'month', 'day_of_week',
            'team_1_wins', 'team_1_losses', 'team_1_win_pct',
            'team_2_wins', 'team_2_losses', 'team_2_win_pct',
            'team_1_is_home', 'team_2_is_home',
            'conference_game', 'venue_indoor_int', 'neutral_site_int',
            'team_1_ranked', 'team_2_ranked', 'team_1_rank_filled', 'team_2_rank_filled'
        ]
        
        # Add missing columns with defaults
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df_clean = df.dropna(subset=['team_1_score', 'team_2_score']).copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        X = df_clean[feature_columns]
        y1 = df_clean['team_1_score']
        y2 = df_clean['team_2_score']
        
        return X, y1, y2
    
    def predict_game(self, team1_id, team2_id, game_info=None):
        """Predict score for a specific matchup"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Get team information
        team1_info = self.db_manager.get_team_by_id(team1_id)
        team2_info = self.db_manager.get_team_by_id(team2_id)
        
        if not team1_info or not team2_info:
            raise ValueError("One or both teams not found")
        
        # Get team stats
        current_season = game_info.get('season', datetime.now().year) if game_info else datetime.now().year
        team1_stats = self.db_manager.get_team_stats(team1_id, current_season)
        team2_stats = self.db_manager.get_team_stats(team2_id, current_season)
        
        # Prepare feature vector
        features = pd.DataFrame([{
            'week': game_info.get('week', 1) if game_info else 1,
            'season_type': game_info.get('season_type', 2) if game_info else 2,
            'month': game_info.get('month', 9) if game_info else 9,
            'day_of_week': game_info.get('day_of_week', 5) if game_info else 5,
            'team_1_wins': team1_stats['wins'],
            'team_1_losses': team1_stats['losses'],
            'team_1_win_pct': team1_stats['wins'] / max(1, team1_stats['wins'] + team1_stats['losses']),
            'team_2_wins': team2_stats['wins'],
            'team_2_losses': team2_stats['losses'],
            'team_2_win_pct': team2_stats['wins'] / max(1, team2_stats['wins'] + team2_stats['losses']),
            'team_1_is_home': 1 if game_info and game_info.get('team1_home_away') == 'home' else 0,
            'team_2_is_home': 1 if game_info and game_info.get('team2_home_away') == 'home' else 0,
            'conference_game': 1 if team1_info.get('conference_id') == team2_info.get('conference_id') else 0,
            'venue_indoor_int': game_info.get('indoor', 0) if game_info else 0,
            'neutral_site_int': game_info.get('neutral_site', 0) if game_info else 0,
            'team_1_ranked': game_info.get('team1_rank') is not None if game_info else 0,
            'team_2_ranked': game_info.get('team2_rank') is not None if game_info else 0,
            'team_1_rank_filled': game_info.get('team1_rank', 99) if game_info else 99,
            'team_2_rank_filled': game_info.get('team2_rank', 99) if game_info else 99,
        }])
        
        # Make prediction
        team1_score, team2_score = self.model.predict(features)
        
        return {
            'team1': {
                'id': team1_id,
                'name': team1_info['name'],
                'predicted_score': round(team1_score[0]),
                'stats': team1_stats
            },
            'team2': {
                'id': team2_id,
                'name': team2_info['name'],
                'predicted_score': round(team2_score[0]),
                'stats': team2_stats
            },
            'prediction_details': {
                'winner': team1_info['name'] if team1_score[0] > team2_score[0] else team2_info['name'],
                'margin': abs(round(team1_score[0] - team2_score[0])),
                'total_points': round(team1_score[0] + team2_score[0]),
                'confidence': self._calculate_confidence(team1_score[0], team2_score[0])
            }
        }
    
    def _calculate_confidence(self, score1, score2):
        """Calculate prediction confidence based on point spread"""
        spread = abs(score1 - score2)
        if spread >= 14:
            return 'High'
        elif spread >= 7:
            return 'Medium'
        else:
            return 'Low'

# Initialize global objects
model_manager = ModelManager()

# Background task to update data
def update_data_background():
    """Background task to periodically update team and game data"""
    while True:
        try:
            # Update teams data daily
            print("Updating teams data...")
            download_teams_data()
            model_manager.db_manager.load_teams_from_csv()
            
            # Update current season data
            current_year = datetime.now().year
            print(f"Updating {current_year} season data...")
            download_season_data(current_year, season_type=2)
            
            # Retrain model if new data available
            print("Checking if model needs retraining...")
            model_manager.train_new_model()
            
        except Exception as e:
            print(f"Background update failed: {e}")
        
        # Wait 24 hours
        time.sleep(24 * 60 * 60)

# API Routes

def load_model_on_start():
    """Initialize model before first request"""
    if not hasattr(g, 'model_initialized'):
        model_manager.load_or_train_model()
        model_manager.db_manager.load_teams_from_csv()
        g.model_initialized = True

@app.route('/api/teams/search', methods=['GET'])
def search_teams():
    """Search for teams by name, abbreviation, or location"""
    query = request.args.get('q', '').strip()
    
    if not query or len(query) < 2:
        return jsonify({'error': 'Query must be at least 2 characters'}), 400
    
    try:
        teams = model_manager.db_manager.search_teams(query)
        return jsonify({
            'teams': teams,
            'count': len(teams)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams/<int:team_id>', methods=['GET'])
def get_team(team_id):
    """Get detailed information about a specific team"""
    try:
        team = model_manager.db_manager.get_team_by_id(team_id)
        if not team:
            return jsonify({'error': 'Team not found'}), 404
        
        # Get current season stats
        season = request.args.get('season', datetime.now().year, type=int)
        stats = model_manager.db_manager.get_team_stats(team_id, season)
        
        return jsonify({
            'team': team,
            'stats': stats,
            'season': season
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_game():
    """Predict the outcome of a game between two teams"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        team1_id = data.get('team1_id')
        team2_id = data.get('team2_id')
        
        if not team1_id or not team2_id:
            return jsonify({'error': 'Both team1_id and team2_id are required'}), 400
        
        if team1_id == team2_id:
            return jsonify({'error': 'Teams cannot play themselves'}), 400
        
        # Optional game information
        game_info = {
            'week': data.get('week', 1),
            'season': data.get('season', datetime.now().year),
            'season_type': data.get('season_type', 2),
            'month': data.get('month', 9),
            'day_of_week': data.get('day_of_week', 5),
            'team1_home_away': data.get('team1_home_away', 'home'),
            'team2_home_away': data.get('team2_home_away', 'away'),
            'indoor': data.get('indoor', 0),
            'neutral_site': data.get('neutral_site', 0),
            'team1_rank': data.get('team1_rank'),
            'team2_rank': data.get('team2_rank')
        }
        
        # Make prediction
        prediction = model_manager.predict_game(team1_id, team2_id, game_info)
        
        return jsonify({
            'prediction': prediction,
            'game_info': game_info,
            'model_trained_at': model_manager.model_trained_at.isoformat() if model_manager.model_trained_at else None
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict/simple', methods=['GET'])
def predict_game_simple():
    """Simple GET endpoint for quick predictions"""
    try:
        team1_id = request.args.get('team1_id', type=int)
        team2_id = request.args.get('team2_id', type=int)
        
        if not team1_id or not team2_id:
            return jsonify({'error': 'Both team1_id and team2_id are required'}), 400
        
        # Use default game info
        game_info = {
            'week': request.args.get('week', 1, type=int),
            'season': request.args.get('season', datetime.now().year, type=int),
            'team1_home_away': request.args.get('team1_location', 'home'),
            'team2_home_away': request.args.get('team2_location', 'away')
        }
        
        prediction = model_manager.predict_game(team1_id, team2_id, game_info)
        
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matchup', methods=['GET'])
def get_matchup_by_names():
    """Get prediction by team names instead of IDs"""
    try:
        team1_name = request.args.get('team1', '').strip()
        team2_name = request.args.get('team2', '').strip()
        
        if not team1_name or not team2_name:
            return jsonify({'error': 'Both team1 and team2 parameters are required'}), 400
        
        # Search for teams
        team1_results = model_manager.db_manager.search_teams(team1_name)
        team2_results = model_manager.db_manager.search_teams(team2_name)
        
        if not team1_results:
            return jsonify({'error': f'Team "{team1_name}" not found'}), 404
        if not team2_results:
            return jsonify({'error': f'Team "{team2_name}" not found'}), 404
        
        # Use first match for each team
        team1 = team1_results[0]
        team2 = team2_results[0]
        
        # Get game info from query params
        game_info = {
            'week': request.args.get('week', 1, type=int),
            'season': request.args.get('season', datetime.now().year, type=int),
            'team1_home_away': request.args.get('location', 'neutral').lower(),
            'team2_home_away': 'away' if request.args.get('location', 'neutral').lower() == 'home' else 'home',
            'neutral_site': 1 if request.args.get('location', 'neutral').lower() == 'neutral' else 0
        }
        
        prediction = model_manager.predict_game(team1['team_id'], team2['team_id'], game_info)
        
        return jsonify({
            'prediction': prediction,
            'teams_found': {
                'team1_matches': team1_results[:3],  # Show top 3 matches
                'team2_matches': team2_results[:3]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model status and training information"""
    try:
        return jsonify({
            'model_loaded': model_manager.model is not None,
            'model_trained_at': model_manager.model_trained_at.isoformat() if model_manager.model_trained_at else None,
            'model_type': type(model_manager.model).__name__ if model_manager.model else None,
            'database_path': app.config['DATABASE'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Manually trigger model retraining"""
    try:
        success = model_manager.train_new_model()
        
        if success:
            return jsonify({
                'message': 'Model retrained successfully',
                'trained_at': model_manager.model_trained_at.isoformat()
            })
        else:
            return jsonify({'error': 'Model retraining failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/update', methods=['POST'])
def update_data():
    """Manually trigger data update"""
    try:
        data = request.get_json() or {}
        year = data.get('year', datetime.now().year)
        
        # Update teams data
        download_teams_data()
        model_manager.db_manager.load_teams_from_csv()
        
        # Update season data
        download_season_data(year, season_type=2)
        
        return jsonify({
            'message': f'Data updated successfully for {year}',
            'updated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conferences', methods=['GET'])
def get_conferences():
    """Get list of conferences and their teams"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conference_id, COUNT(*) as team_count,
                   GROUP_CONCAT(name, ', ') as teams
            FROM teams 
            WHERE conference_id IS NOT NULL
            GROUP BY conference_id
            ORDER BY team_count DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        conferences = []
        for row in results:
            conferences.append({
                'conference_id': row[0],
                'team_count': row[1],
                'teams': row[2].split(', ')[:10] if row[2] else []  # Show first 10 teams
            })
        
        return jsonify({'conferences': conferences})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/team/<int:team_id>/history', methods=['GET'])
def get_team_history(team_id):
    """Get historical performance for a team"""
    try:
        years = request.args.get('years', 5, type=int)
        current_year = datetime.now().year
        
        history = []
        for year in range(current_year - years, current_year + 1):
            stats = model_manager.db_manager.get_team_stats(team_id, year)
            if stats['games_played'] > 0:
                history.append({
                    'season': year,
                    **stats
                })
        
        team = model_manager.db_manager.get_team_by_id(team_id)
        
        return jsonify({
            'team': team,
            'history': history
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/popular-matchups', methods=['GET'])
def get_popular_matchups():
    """Get suggested popular matchups"""
    popular_teams = [
        {'id': 333, 'name': 'Alabama Crimson Tide'},
        {'id': 61, 'name': 'Georgia Bulldogs'},
        {'id': 194, 'name': 'Michigan Wolverines'},
        {'id': 213, 'name': 'Penn State Nittany Lions'},
        {'id': 251, 'name': 'Texas Longhorns'},
        {'id': 201, 'name': 'Oklahoma Sooners'},
        {'id': 52, 'name': 'Florida State Seminoles'},
        {'id': 150, 'name': 'Clemson Tigers'},
        {'id': 30, 'name': 'USC Trojans'},
        {'id': 99, 'name': 'LSU Tigers'}
    ]
    
    matchups = [
        {'team1': popular_teams[0], 'team2': popular_teams[1], 'rivalry': 'SEC Championship rivals'},
        {'team1': popular_teams[2], 'team2': popular_teams[3], 'rivalry': 'Big Ten rivals'},
        {'team1': popular_teams[4], 'team2': popular_teams[5], 'rivalry': 'Red River Rivalry'},
        {'team1': popular_teams[6], 'team2': popular_teams[7], 'rivalry': 'ACC rivals'},
        {'team1': popular_teams[0], 'team2': popular_teams[9], 'rivalry': 'SEC West rivals'},
    ]
    
    return jsonify({
        'popular_teams': popular_teams,
        'suggested_matchups': matchups
    })

@app.errorhandler(404)
def not_found(error):
    # Check if this is an API request
    if request.path.startswith('/api/') or request.path.startswith('/health'):
        return jsonify({'error': 'Endpoint not found'}), 404
    else:
        # For non-API requests, try to serve the main page
        return send_file('index.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_manager.model is not None,
        'database_connected': os.path.exists(app.config['DATABASE'])
    })

# Documentation endpoint
@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API documentation"""
    docs = {
        'endpoints': {
            'GET /api/teams/search?q=<query>': 'Search for teams by name, abbreviation, or location',
            'GET /api/teams/<team_id>': 'Get detailed team information and stats',
            'GET /api/teams/<team_id>?season=<year>': 'Get team stats for specific season',
            'POST /api/predict': 'Predict game outcome (requires JSON body with team1_id, team2_id)',
            'GET /api/predict/simple?team1_id=<id>&team2_id=<id>': 'Simple prediction with default settings',
            'GET /api/matchup?team1=<name>&team2=<name>': 'Predict by team names instead of IDs',
            'GET /api/model/status': 'Get model status and information',
            'POST /api/model/retrain': 'Manually trigger model retraining',
            'POST /api/data/update': 'Manually trigger data update',
            'GET /api/conferences': 'Get list of conferences and teams',
            'GET /api/stats/team/<team_id>/history': 'Get historical team performance',
            'GET /api/popular-matchups': 'Get suggested popular team matchups',
            'GET /health': 'Health check endpoint'
        },
        'example_requests': {
            'search_teams': '/api/teams/search?q=alabama',
            'get_team': '/api/teams/333',
            'simple_prediction': '/api/predict/simple?team1_id=333&team2_id=61',
            'name_based_prediction': '/api/matchup?team1=Alabama&team2=Georgia&location=neutral',
            'detailed_prediction': {
                'url': '/api/predict',
                'method': 'POST',
                'body': {
                    'team1_id': 333,
                    'team2_id': 61,
                    'week': 12,
                    'season': 2024,
                    'team1_home_away': 'home',
                    'team2_home_away': 'away',
                    'team1_rank': 1,
                    'team2_rank': 3
                }
            }
        },
        'response_format': {
            'prediction': {
                'team1': {
                    'id': 'int',
                    'name': 'string',
                    'predicted_score': 'int',
                    'stats': 'object'
                },
                'team2': {
                    'id': 'int',
                    'name': 'string', 
                    'predicted_score': 'int',
                    'stats': 'object'
                },
                'prediction_details': {
                    'winner': 'string',
                    'margin': 'int',
                    'total_points': 'int',
                    'confidence': 'string (High/Medium/Low)'
                }
            }
        }
    }
    
    return jsonify(docs)

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/styles.css')
def serve_styles():
    """Serve CSS file"""
    return send_file('styles.css', mimetype='text/css')

@app.route('/script.js')
def serve_script():
    """Serve JavaScript file"""
    return send_file('script.js', mimetype='application/javascript')

# Add CORS support for API calls
from flask_cors import CORS
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/health": {"origins": "*"}
})

def initialize_model():
    """Initialize model once"""
    global model_loaded
    if not model_loaded:
        print("Loading model on application startup...")
        model_manager.load_or_train_model()
        model_manager.db_manager.load_teams_from_csv()
        model_loaded = True
        print("Model loaded successfully!")

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    
    # Initialize database and load model ONCE
    print("Initializing application...")
    initialize_model()  # Add this line
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
