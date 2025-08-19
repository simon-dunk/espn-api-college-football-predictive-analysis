import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class CollegeFootballScorePredictor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_columns = ['team_1_score', 'team_2_score']
        
    def load_and_prepare_data(self):
        """Load games data and prepare for modeling"""
        print("Loading and preparing data...")
        
        # Load all game files
        all_dfs = []
        games_files = [f for f in os.listdir(self.data_dir) if f.startswith('games_') and f.endswith('.csv')]
        
        for filename in games_files:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            all_dfs.append(df)
        
        if not all_dfs:
            raise FileNotFoundError("No game data files found")
        
        # Combine all data
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter out future seasons and incomplete games
        current_year = pd.Timestamp.now().year
        df = df[df['season'] < current_year]
        df = df[df['completed'] == True]
        
        # Remove games with missing scores
        df = df.dropna(subset=['team_1_score', 'team_2_score'])
        
        print(f"Loaded {len(df)} completed games from seasons {sorted(df['season'].unique())}")
        return df
    
    def create_team_stats(self, df):
        """Create historical team statistics for each team"""
        print("Creating team statistics...")
        
        # Sort by date to ensure chronological order
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        team_stats = {}
        
        # Initialize team stats
        all_teams = set(df['team_1_id'].unique()) | set(df['team_2_id'].unique())
        for team_id in all_teams:
            team_stats[team_id] = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'points_scored': 0,
                'points_allowed': 0,
                'home_wins': 0,
                'home_games': 0,
                'away_wins': 0,
                'away_games': 0,
                'conf_wins': 0,
                'conf_games': 0,
                'last_5_wins': 0,
                'recent_scores': [],
                'recent_allowed': []
            }
        
        # Create rolling statistics for each game
        team_1_stats = []
        team_2_stats = []
        
        for idx, row in df.iterrows():
            team_1_id = row['team_1_id']
            team_2_id = row['team_2_id']
            
            # Get current stats before this game
            t1_stats = team_stats[team_1_id].copy()
            t2_stats = team_stats[team_2_id].copy()
            
            # Calculate derived stats
            for stats in [t1_stats, t2_stats]:
                stats['win_pct'] = stats['wins'] / max(stats['games_played'], 1)
                stats['avg_points_scored'] = stats['points_scored'] / max(stats['games_played'], 1)
                stats['avg_points_allowed'] = stats['points_allowed'] / max(stats['games_played'], 1)
                stats['home_win_pct'] = stats['home_wins'] / max(stats['home_games'], 1)
                stats['away_win_pct'] = stats['away_wins'] / max(stats['away_games'], 1)
                stats['conf_win_pct'] = stats['conf_wins'] / max(stats['conf_games'], 1)
                stats['last_5_win_pct'] = stats['last_5_wins'] / 5
                stats['recent_avg_scored'] = np.mean(stats['recent_scores'][-5:]) if stats['recent_scores'] else 0
                stats['recent_avg_allowed'] = np.mean(stats['recent_allowed'][-5:]) if stats['recent_allowed'] else 0
            
            team_1_stats.append(t1_stats)
            team_2_stats.append(t2_stats)
            
            # Update stats after this game
            team_1_score = row['team_1_score']
            team_2_score = row['team_2_score']
            team_1_won = team_1_score > team_2_score
            is_conf_game = row.get('conference_competition', False)
            
            # Update team 1 stats
            team_stats[team_1_id]['games_played'] += 1
            team_stats[team_1_id]['points_scored'] += team_1_score
            team_stats[team_1_id]['points_allowed'] += team_2_score
            team_stats[team_1_id]['recent_scores'].append(team_1_score)
            team_stats[team_1_id]['recent_allowed'].append(team_2_score)
            
            if team_1_won:
                team_stats[team_1_id]['wins'] += 1
            else:
                team_stats[team_1_id]['losses'] += 1
            
            if row['team_1_home_away'] == 'home':
                team_stats[team_1_id]['home_games'] += 1
                if team_1_won:
                    team_stats[team_1_id]['home_wins'] += 1
            else:
                team_stats[team_1_id]['away_games'] += 1
                if team_1_won:
                    team_stats[team_1_id]['away_wins'] += 1
            
            if is_conf_game:
                team_stats[team_1_id]['conf_games'] += 1
                if team_1_won:
                    team_stats[team_1_id]['conf_wins'] += 1
            
            # Update last 5 games for team 1
            if len(team_stats[team_1_id]['recent_scores']) >= 5:
                recent_wins = sum(1 for i in range(-5, 0) 
                                if len(team_stats[team_1_id]['recent_scores']) > abs(i) and 
                                   team_stats[team_1_id]['recent_scores'][i] > team_stats[team_1_id]['recent_allowed'][i])
                team_stats[team_1_id]['last_5_wins'] = recent_wins
            
            # Update team 2 stats (similar logic)
            team_stats[team_2_id]['games_played'] += 1
            team_stats[team_2_id]['points_scored'] += team_2_score
            team_stats[team_2_id]['points_allowed'] += team_1_score
            team_stats[team_2_id]['recent_scores'].append(team_2_score)
            team_stats[team_2_id]['recent_allowed'].append(team_1_score)
            
            if not team_1_won:
                team_stats[team_2_id]['wins'] += 1
            else:
                team_stats[team_2_id]['losses'] += 1
            
            if row['team_2_home_away'] == 'home':
                team_stats[team_2_id]['home_games'] += 1
                if not team_1_won:
                    team_stats[team_2_id]['home_wins'] += 1
            else:
                team_stats[team_2_id]['away_games'] += 1
                if not team_1_won:
                    team_stats[team_2_id]['away_wins'] += 1
            
            if is_conf_game:
                team_stats[team_2_id]['conf_games'] += 1
                if not team_1_won:
                    team_stats[team_2_id]['conf_wins'] += 1
            
            # Update last 5 games for team 2
            if len(team_stats[team_2_id]['recent_scores']) >= 5:
                recent_wins = sum(1 for i in range(-5, 0) 
                                if len(team_stats[team_2_id]['recent_scores']) > abs(i) and 
                                   team_stats[team_2_id]['recent_scores'][i] > team_stats[team_2_id]['recent_allowed'][i])
                team_stats[team_2_id]['last_5_wins'] = recent_wins
        
        return team_1_stats, team_2_stats
    
    def engineer_features(self, df):
        """Create features for machine learning"""
        print("Engineering features...")
        
        # Get team statistics
        team_1_stats, team_2_stats = self.create_team_stats(df)
        
        # Create feature matrix
        features = []
        
        # Basic game information
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
        df['season_type'] = pd.to_numeric(df['season_type'], errors='coerce')
        
        # Time-based features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        for idx, row in df.iterrows():
            feature_row = {}
            
            # Game context features
            feature_row['week'] = row['week']
            feature_row['season_type'] = row['season_type']
            feature_row['month'] = row['month']
            feature_row['day_of_week'] = row['day_of_week']
            feature_row['neutral_site'] = 1 if row.get('neutral_site') else 0
            feature_row['conference_game'] = 1 if row.get('conference_competition') else 0
            feature_row['indoor_venue'] = 1 if row.get('venue_indoor') else 0
            
            # Team rankings (if available)
            feature_row['team_1_rank'] = row.get('team_1_rank', 26)  # Unranked = 26
            feature_row['team_2_rank'] = row.get('team_2_rank', 26)
            feature_row['rank_difference'] = feature_row['team_1_rank'] - feature_row['team_2_rank']
            
            # Home field advantage
            feature_row['team_1_home'] = 1 if row['team_1_home_away'] == 'home' else 0
            feature_row['team_2_home'] = 1 if row['team_2_home_away'] == 'home' else 0
            
            # Team statistics
            t1_stats = team_1_stats[idx]
            t2_stats = team_2_stats[idx]
            
            # Team 1 stats
            for stat_name, stat_value in t1_stats.items():
                if stat_name not in ['recent_scores', 'recent_allowed']:
                    feature_row[f'team_1_{stat_name}'] = stat_value
            
            # Team 2 stats
            for stat_name, stat_value in t2_stats.items():
                if stat_name not in ['recent_scores', 'recent_allowed']:
                    feature_row[f'team_2_{stat_name}'] = stat_value
            
            # Relative team strength features
            feature_row['win_pct_diff'] = t1_stats['win_pct'] - t2_stats['win_pct']
            feature_row['avg_scored_diff'] = t1_stats['avg_points_scored'] - t2_stats['avg_points_scored']
            feature_row['avg_allowed_diff'] = t1_stats['avg_points_allowed'] - t2_stats['avg_points_allowed']
            feature_row['recent_scored_diff'] = t1_stats['recent_avg_scored'] - t2_stats['recent_avg_scored']
            feature_row['recent_allowed_diff'] = t1_stats['recent_avg_allowed'] - t2_stats['recent_avg_allowed']
            
            # Conference strength (if same conference)
            if row.get('team_1_conference') == row.get('team_2_conference'):
                feature_row['conf_win_pct_diff'] = t1_stats['conf_win_pct'] - t2_stats['conf_win_pct']
            else:
                feature_row['conf_win_pct_diff'] = 0
            
            features.append(feature_row)
        
        feature_df = pd.DataFrame(features)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        print(f"Created {len(feature_df.columns)} features")
        return feature_df
    
    def prepare_features_and_targets(self, df):
        """Prepare final feature matrix and target variables"""
        print("Preparing features and targets...")
        
        # Create features
        feature_df = self.engineer_features(df)
        
        # Get targets
        targets = df[self.target_columns].copy()
        
        # Remove rows with invalid targets
        valid_idx = targets.notna().all(axis=1)
        feature_df = feature_df[valid_idx]
        targets = targets[valid_idx]
        
        # Store feature names
        self.feature_names = feature_df.columns.tolist()
        
        print(f"Final dataset: {len(feature_df)} games with {len(self.feature_names)} features")
        return feature_df, targets
    
    def train_model(self, X, y):
        """Train the score prediction model"""
        print("Training model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Try different models
        models = {
            'Random Forest': MultiOutputRegressor(RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            )),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0))
        }
        
        best_model = None
        best_score = float('inf')
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
                        
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'model': model
            }
            
            print(f"{name} Results:")
            print(f"  MAE: {mae:.2f} points")
            print(f"  RMSE: {rmse:.2f} points") 
            print(f"  R²: {r2:.3f}")
            
            # Track best model (lowest MAE)
            if mae < best_score:
                best_score = mae
                best_model = model
        
        self.model = best_model
        
        # Feature importance for Random Forest
        if isinstance(best_model.estimators_[0], RandomForestRegressor):
            self.analyze_feature_importance(X, best_model)
        
        return results
    
    def analyze_feature_importance(self, X, model):
        """Analyze and display feature importance"""
        print("\nFeature Importance Analysis:")
        print("="*50)
        
        # Get importance from both estimators (team_1_score, team_2_score)
        importance_1 = model.estimators_[0].feature_importances_
        importance_2 = model.estimators_[1].feature_importances_
        
        # Average importance across both outputs
        avg_importance = (importance_1 + importance_2) / 2
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']:<30}: {row['importance']:.4f}")
    
    def save_model(self, filename='college_football_score_predictor.pkl'):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'target_columns': self.target_columns
        }
        
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filename='college_football_score_predictor.pkl'):
        """Load a trained model"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.label_encoders = model_data['label_encoders']
        self.target_columns = model_data['target_columns']
        
        print(f"Model loaded from {filepath}")
    
    def predict_game_score(self, game_features):
        """Predict scores for a single game"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Ensure features are in correct order
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(game_features.get(feature_name, 0))
        
        # Scale features
        X_scaled = self.scaler.transform([feature_vector])
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        return {
            'team_1_predicted_score': round(prediction[0]),
            'team_2_predicted_score': round(prediction[1]),
            'predicted_winner': 'Team 1' if prediction[0] > prediction[1] else 'Team 2',
            'predicted_margin': abs(prediction[0] - prediction[1])
        }
    
    def train_full_pipeline(self):
        """Complete training pipeline"""
        print("="*60)
        print("COLLEGE FOOTBALL SCORE PREDICTION MODEL")
        print("="*60)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Create features and targets
        X, y = self.prepare_features_and_targets(df)
        
        # Train model
        results = self.train_model(X, y)
        
        # Save model
        model_path = self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        return results, model_path

class GamePredictor:
    """Helper class for making predictions on new games"""
    
    def __init__(self, model_path='data/college_football_score_predictor.pkl'):
        self.predictor = CollegeFootballScorePredictor()
        self.predictor.load_model(model_path.split('/')[-1])
        self.team_stats_cache = {}
    
    def load_current_team_stats(self):
        """Load current season team statistics"""
        # This would load the most recent team statistics
        # For now, we'll create a simplified version
        pass
    
    def predict_upcoming_game(self, team_1_id, team_2_id, game_info=None):
        """Predict score for an upcoming game"""
        
        # Default game info
        if game_info is None:
            game_info = {
                'week': 10,
                'season_type': 2,
                'month': 10,
                'day_of_week': 5,  # Saturday
                'neutral_site': 0,
                'conference_game': 0,
                'indoor_venue': 0,
                'team_1_home': 1,
                'team_2_home': 0
            }
        
        # You would need to implement logic to get current team stats
        # For demonstration, using placeholder values
        team_features = {
            **game_info,
            'team_1_rank': 15,
            'team_2_rank': 20,
            'rank_difference': -5,
            'team_1_games_played': 8,
            'team_1_wins': 6,
            'team_1_win_pct': 0.75,
            'team_1_avg_points_scored': 28.5,
            'team_1_avg_points_allowed': 21.2,
            'team_2_games_played': 8,
            'team_2_wins': 5,
            'team_2_win_pct': 0.625,
            'team_2_avg_points_scored': 24.8,
            'team_2_avg_points_allowed': 23.1,
            'win_pct_diff': 0.125,
            'avg_scored_diff': 3.7,
            'avg_allowed_diff': -1.9
        }
        
        # Add any missing features with default values
        for feature_name in self.predictor.feature_names:
            if feature_name not in team_features:
                team_features[feature_name] = 0
        
        return self.predictor.predict_game_score(team_features)

def train_model():
    """Main function to train the model"""
    predictor = CollegeFootballScorePredictor()
    results, model_path = predictor.train_full_pipeline()
    return predictor, results, model_path

def load_trained_model(model_path='data/college_football_score_predictor.pkl'):
    """Load a pre-trained model"""
    predictor = CollegeFootballScorePredictor()
    predictor.load_model(model_path.split('/')[-1])
    return predictor

# Example usage and testing
if __name__ == "__main__":
    # Train the model
    print("Training College Football Score Prediction Model...")
    predictor, results, model_path = train_model()
    
    # Test predictions
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    # Example prediction
    game_predictor = GamePredictor(model_path)
    
    sample_prediction = game_predictor.predict_upcoming_game(
        team_1_id=123,  # Team 1
        team_2_id=456,  # Team 2
        game_info={
            'week': 12,
            'season_type': 2,
            'month': 11,
            'day_of_week': 5,  # Saturday
            'neutral_site': 0,
            'conference_game': 1,
            'indoor_venue': 0,
            'team_1_home': 1,
            'team_2_home': 0,
            'team_1_rank': 8,
            'team_2_rank': 15
        }
    )
    
    print("Sample Prediction:")
    print(f"  Team 1 Predicted Score: {sample_prediction['team_1_predicted_score']}")
    print(f"  Team 2 Predicted Score: {sample_prediction['team_2_predicted_score']}")
    print(f"  Predicted Winner: {sample_prediction['predicted_winner']}")
    print(f"  Predicted Margin: {sample_prediction['predicted_margin']:.1f}")
