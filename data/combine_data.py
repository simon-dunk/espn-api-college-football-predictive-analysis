import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_all_data(data_dir=""):
    """Load all CSV files from the data directory"""
    csv_files = glob.glob(os.path.join(data_dir, "games_*.csv"))
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded {file}: {len(df)} games")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total games loaded: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError("No data files found")

def engineer_features(df):
    """Create features for prediction"""
    df = df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Only use completed games
    df = df[df['completed'] == True].copy()
    
    # Remove games with missing scores
    df = df.dropna(subset=['team_1_score', 'team_2_score'])
    
    # Parse records (assuming format like "5-2")
    def parse_record(record_str):
        if pd.isna(record_str) or record_str == '':
            return 0, 0
        try:
            wins, losses = record_str.split('-')
            return int(wins), int(losses)
        except:
            return 0, 0
    
    # Parse team records
    df['team_1_wins'], df['team_1_losses'] = zip(*df['team_1_record_total'].apply(parse_record))
    df['team_2_wins'], df['team_2_losses'] = zip(*df['team_2_record_total'].apply(parse_record))
    
    # Calculate win percentages
    df['team_1_win_pct'] = df['team_1_wins'] / (df['team_1_wins'] + df['team_1_losses'] + 0.001)
    df['team_2_win_pct'] = df['team_2_wins'] / (df['team_2_wins'] + df['team_2_losses'] + 0.001)
    
    # Home field advantage
    df['team_1_is_home'] = (df['team_1_home_away'] == 'home').astype(int)
    df['team_2_is_home'] = (df['team_2_home_away'] == 'home').astype(int)
    
    # Conference game indicator
    df['conference_game'] = df['conference_competition'].astype(int)
    
    # Ranking features (handle NaN rankings)
    df['team_1_ranked'] = (~pd.isna(df['team_1_rank'])).astype(int)
    df['team_2_ranked'] = (~pd.isna(df['team_2_rank'])).astype(int)
    df['team_1_rank_filled'] = df['team_1_rank'].fillna(99)
    df['team_2_rank_filled'] = df['team_2_rank'].fillna(99)
    
    # Indoor/neutral site
    df['venue_indoor_int'] = df['venue_indoor'].astype(int)
    df['neutral_site_int'] = df['neutral_site'].astype(int)
    
    return df

def add_historical_features(df):
    """Add historical performance features"""
    df = df.sort_values(['date']).copy()
    
    # Initialize dictionaries to store team stats
    team_stats = {}
    
    # Lists to store historical features
    team_1_avg_score = []
    team_1_avg_allowed = []
    team_2_avg_score = []
    team_2_avg_allowed = []
    head_to_head_record = []
    
    for idx, row in df.iterrows():
        team_1_id = row['team_1_id']
        team_2_id = row['team_2_id']
        current_date = row['date']
        
        # Get historical stats for team 1
        if team_1_id in team_stats:
            t1_scores = [game['score'] for game in team_stats[team_1_id] if game['date'] < current_date]
            t1_allowed = [game['opp_score'] for game in team_stats[team_1_id] if game['date'] < current_date]
            team_1_avg_score.append(np.mean(t1_scores) if t1_scores else 21)  # Default average
            team_1_avg_allowed.append(np.mean(t1_allowed) if t1_allowed else 21)
        else:
            team_1_avg_score.append(21)
            team_1_avg_allowed.append(21)
        
        # Get historical stats for team 2
        if team_2_id in team_stats:
            t2_scores = [game['score'] for game in team_stats[team_2_id] if game['date'] < current_date]
            t2_allowed = [game['opp_score'] for game in team_stats[team_2_id] if game['date'] < current_date]
            team_2_avg_score.append(np.mean(t2_scores) if t2_scores else 21)
            team_2_avg_allowed.append(np.mean(t2_allowed) if t2_allowed else 21)
        else:
            team_2_avg_score.append(21)
            team_2_avg_allowed.append(21)
        
        # Head-to-head record (simplified)
        h2h = 0  # You could implement more sophisticated H2H tracking
        head_to_head_record.append(h2h)
        
        # Update team stats with current game
        if team_1_id not in team_stats:
            team_stats[team_1_id] = []
        if team_2_id not in team_stats:
            team_stats[team_2_id] = []
        
        team_stats[team_1_id].append({
            'date': current_date,
            'score': row['team_1_score'],
            'opp_score': row['team_2_score']
        })
        
        team_stats[team_2_id].append({
            'date': current_date,
            'score': row['team_2_score'],
            'opp_score': row['team_1_score']
        })
    
    # Add features to dataframe
    df['team_1_avg_score_hist'] = team_1_avg_score
    df['team_1_avg_allowed_hist'] = team_1_avg_allowed
    df['team_2_avg_score_hist'] = team_2_avg_score
    df['team_2_avg_allowed_hist'] = team_2_avg_allowed
    df['head_to_head'] = head_to_head_record
    
    return df

def prepare_features(df):
    """Prepare feature matrix and target variables with better missing value handling"""
    
    feature_columns = [
        'week', 'season_type', 'month', 'day_of_week',
        'team_1_wins', 'team_1_losses', 'team_1_win_pct',
        'team_2_wins', 'team_2_losses', 'team_2_win_pct',
        'team_1_is_home', 'team_2_is_home',
        'conference_game', 'venue_indoor_int', 'neutral_site_int',
        'team_1_ranked', 'team_2_ranked', 'team_1_rank_filled', 'team_2_rank_filled',
        'team_1_avg_score_hist', 'team_1_avg_allowed_hist',
        'team_2_avg_score_hist', 'team_2_avg_allowed_hist',
        'head_to_head'
    ]
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = [col for col in feature_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            df[col] = 0
    
    # Remove rows with missing target values only
    df_clean = df.dropna(subset=['team_1_score', 'team_2_score']).copy()
    
    # Replace infinite values with NaN, then handle them in the pipeline
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    X = df_clean[feature_columns]
    y1 = df_clean['team_1_score']  # Team 1 score
    y2 = df_clean['team_2_score']  # Team 2 score
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Missing values per feature:")
    print(X.isnull().sum())
    
    return X, y1, y2, df_clean

class FootballScorePredictor:
    def __init__(self):
        # Create pipelines that include imputation
        self.team1_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        self.team2_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X, y1, y2):
        # Store feature names for validation
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        # Train pipelines
        self.team1_pipeline.fit(X, y1)
        self.team2_pipeline.fit(X, y2)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate feature order if we have feature names
        if self.feature_names and hasattr(X, 'columns'):
            if list(X.columns) != self.feature_names:
                print("Warning: Feature order mismatch. Reordering...")
                X = X[self.feature_names]
        
        # Check for infinite values and replace them
        if hasattr(X, 'replace'):
            X = X.replace([np.inf, -np.inf], np.nan)
        
        team1_pred = self.team1_pipeline.predict(X)
        team2_pred = self.team2_pipeline.predict(X)
        
        return team1_pred, team2_pred
    
    def get_feature_importance(self, feature_names=None):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if feature_names is None:
            feature_names = self.feature_names or [f'feature_{i}' for i in range(len(self.team1_pipeline.named_steps['model'].feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'team1_importance': self.team1_pipeline.named_steps['model'].feature_importances_,
            'team2_importance': self.team2_pipeline.named_steps['model'].feature_importances_
        })
        
        importance_df['avg_importance'] = (importance_df['team1_importance'] + 
                                         importance_df['team2_importance']) / 2
        
        return importance_df.sort_values('avg_importance', ascending=False)

class EnsembleFootballPredictor:
    def __init__(self):
        # Multiple models for ensemble with imputation pipelines
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
        # Store feature names
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        # Train all pipelines
        for name, pipeline in self.pipelines_team1.items():
            pipeline.fit(X, y1)
        
        for name, pipeline in self.pipelines_team2.items():
            pipeline.fit(X, y2)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate feature order
        if self.feature_names and hasattr(X, 'columns'):
            if list(X.columns) != self.feature_names:
                print("Warning: Feature order mismatch. Reordering...")
                X = X[self.feature_names]
        
        # Handle infinite values
        if hasattr(X, 'replace'):
            X = X.replace([np.inf, -np.inf], np.nan)
        
        # Get predictions from all models
        team1_preds = []
        team2_preds = []
        
        for pipeline in self.pipelines_team1.values():
            team1_preds.append(pipeline.predict(X))
        
        for pipeline in self.pipelines_team2.values():
            team2_preds.append(pipeline.predict(X))
        
        # Average predictions (simple ensemble)
        team1_pred = np.mean(team1_preds, axis=0)
        team2_pred = np.mean(team2_preds, axis=0)
        
        return team1_pred, team2_pred

def evaluate_predictions(y_true, y_pred, team_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"{team_name} Score Prediction:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.3f}")
    print()

def predict_game_score(predictor, team1_data, team2_data, game_data):
    """
    Predict score for a new game with proper data validation
    
    Parameters:
    - predictor: trained FootballScorePredictor
    - team1_data: dict with team 1 information
    - team2_data: dict with team 2 information  
    - game_data: dict with game information
    """
    
    # Helper function to safely get numeric values
    def safe_get(data, key, default=0):
        value = data.get(key, default)
        if pd.isna(value) or value is None:
            return default
        return float(value)
    
    # Calculate win percentages safely
    team1_wins = safe_get(team1_data, 'wins', 0)
    team1_losses = safe_get(team1_data, 'losses', 0)
    team1_total_games = team1_wins + team1_losses
    team1_win_pct = team1_wins / max(1, team1_total_games) if team1_total_games > 0 else 0.5
    
    team2_wins = safe_get(team2_data, 'wins', 0)
    team2_losses = safe_get(team2_data, 'losses', 0)
    team2_total_games = team2_wins + team2_losses
    team2_win_pct = team2_wins / max(1, team2_total_games) if team2_total_games > 0 else 0.5
    
    # Create feature vector with proper validation
    features = pd.DataFrame([{
        'week': safe_get(game_data, 'week', 1),
        'season_type': safe_get(game_data, 'season_type', 2),
        'month': safe_get(game_data, 'month', 9),
        'day_of_week': safe_get(game_data, 'day_of_week', 5),
        'team_1_wins': team1_wins,
        'team_1_losses': team1_losses,
        'team_1_win_pct': team1_win_pct,
        'team_2_wins': team2_wins,
        'team_2_losses': team2_losses,
        'team_2_win_pct': team2_win_pct,
        'team_1_is_home': 1 if team1_data.get('home_away') == 'home' else 0,
        'team_2_is_home': 1 if team2_data.get('home_away') == 'home' else 0,
        'conference_game': safe_get(game_data, 'conference_game', 0),
        'venue_indoor_int': safe_get(game_data, 'indoor', 0),
        'neutral_site_int': safe_get(game_data, 'neutral_site', 0),
        'team_1_ranked': 1 if team1_data.get('rank') is not None and not pd.isna(team1_data.get('rank')) else 0,
        'team_2_ranked': 1 if team2_data.get('rank') is not None and not pd.isna(team2_data.get('rank')) else 0,
        'team_1_rank_filled': safe_get(team1_data, 'rank', 99),
        'team_2_rank_filled': safe_get(team2_data, 'rank', 99),
        'team_1_avg_score_hist': safe_get(team1_data, 'avg_score', 21),
        'team_1_avg_allowed_hist': safe_get(team1_data, 'avg_allowed', 21),
        'team_2_avg_score_hist': safe_get(team2_data, 'avg_score', 21),
        'team_2_avg_allowed_hist': safe_get(team2_data, 'avg_allowed', 21),
        'head_to_head': safe_get(game_data, 'h2h_record', 0)
    }])
    
    # Check for any remaining NaN values
    if features.isnull().any().any():
        print("Warning: NaN values detected in features:")
        print(features.isnull().sum())
        # Fill any remaining NaN values
        features = features.fillna(0)
    
    # Verify feature order matches training data
    expected_features = [
        'week', 'season_type', 'month', 'day_of_week',
        'team_1_wins', 'team_1_losses', 'team_1_win_pct',
        'team_2_wins', 'team_2_losses', 'team_2_win_pct',
        'team_1_is_home', 'team_2_is_home',
        'conference_game', 'venue_indoor_int', 'neutral_site_int',
        'team_1_ranked', 'team_2_ranked', 'team_1_rank_filled', 'team_2_rank_filled',
        'team_1_avg_score_hist', 'team_1_avg_allowed_hist',
        'team_2_avg_score_hist', 'team_2_avg_allowed_hist',
        'head_to_head'
    ]
    
    # Reorder features to match training order
    features = features[expected_features]
    
    print("Feature values for prediction:")
    for col in features.columns:
        print(f"  {col}: {features[col].iloc[0]}")
    
    # Make prediction
    team1_score, team2_score = predictor.predict(features)
    
    # Round to nearest integer (scores are whole numbers)
    team1_score = round(team1_score[0])
    team2_score = round(team2_score[0])
    
    return team1_score, team2_score

def safe_predict_example(predictor):
    try:
        team1_info = {
            'wins': 8, 'losses': 2, 'rank': 15, 'home_away': 'home',
            'avg_score': 28.5, 'avg_allowed': 18.2
        }
        team2_info = {
            'wins': 6, 'losses': 4, 'rank': None, 'home_away': 'away',
            'avg_score': 24.1, 'avg_allowed': 22.8
        }
        game_info = {
            'week': 12, 'season_type': 2, 'month': 11, 'day_of_week': 5,
            'conference_game': 1, 'indoor': 0, 'neutral_site': 0
        }
        predicted_score1, predicted_score2 = predict_game_score(predictor, team1_info, team2_info, game_info)
        print(f"Predicted Score: Team 1: {predicted_score1}, Team 2: {predicted_score2}")
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("Please check your input data and model training.")

def plot_predictions(y_true, y_pred, title):
    """Plot predicted vs actual scores"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title(f'{title} - Predicted vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = y_pred - y_true
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Score')
    plt.ylabel('Residuals')
    plt.title(f'{title} - Residuals')
    
    plt.tight_layout()
    plt.show()

def save_model(predictor, filepath):
    """Save the trained model"""
    joblib.dump(predictor, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a trained model"""
    return joblib.load(filepath)

def main():
    # Load data
    df = load_all_data()
    
    # Process data
    df_processed = engineer_features(df)
    print(f"Processed data shape: {df_processed.shape}")
    
    # Add historical features
    df_with_history = add_historical_features(df_processed)
    
    # Prepare features
    X, y1, y2, df_clean = prepare_features(df_with_history)
    
    # Split data
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y1, y2, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the model
    predictor = FootballScorePredictor()
    predictor.fit(X_train, y1_train, y2_train)
    
    # Make predictions
    y1_pred, y2_pred = predictor.predict(X_test)
    
    # Evaluate
    evaluate_predictions(y1_test, y1_pred, "Team 1")
    evaluate_predictions(y2_test, y2_pred, "Team 2")
    
    # Feature importance
    importance = predictor.get_feature_importance(X.columns)
    print("Top 10 Most Important Features:")
    print(importance.head(10))
    
    # Run the safe prediction example
    # safe_predict_example(predictor)
    
    # Plot results
    plot_predictions(y1_test, y1_pred, "Team 1 Scores")
    plot_predictions(y2_test, y2_pred, "Team 2 Scores")
    
    # Calculate total points prediction accuracy
    total_actual = y1_test + y2_test
    total_predicted = y1_pred + y2_pred
    print("Total Points Prediction:")
    evaluate_predictions(total_actual, total_predicted, "Total Game")
    
    # Winner prediction accuracy
    actual_winners = (y1_test > y2_test).astype(int)
    predicted_winners = (y1_pred > y2_pred).astype(int)
    winner_accuracy = (actual_winners == predicted_winners).mean()
    print(f"Winner Prediction Accuracy: {winner_accuracy:.3f}")
    
    # Train ensemble model
    ensemble_predictor = EnsembleFootballPredictor()
    ensemble_predictor.fit(X_train, y1_train, y2_train)
    
    # Evaluate ensemble
    y1_pred_ens, y2_pred_ens = ensemble_predictor.predict(X_test)
    print("Ensemble Model Results:")
    evaluate_predictions(y1_test, y1_pred_ens, "Team 1")
    evaluate_predictions(y2_test, y2_pred_ens, "Team 2")
    
    # Winner accuracy for ensemble
    predicted_winners_ens = (y1_pred_ens > y2_pred_ens).astype(int)
    winner_accuracy_ens = (actual_winners == predicted_winners_ens).mean()
    print(f"Ensemble Winner Prediction Accuracy: {winner_accuracy_ens:.3f}")
    
    # Save the best model
    save_model(ensemble_predictor, "football_score_predictor.pkl")
    
    # Example of loading and using the model
    # loaded_predictor = load_model("football_score_predictor.pkl")

if __name__ == "__main__":
    main()
