import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.index_scaler = MinMaxScaler(feature_range=(0, 100))  # For composite index scaling
        self.recent_accuracy = None
        self.recent_metrics = None
        self.feature_columns = None  # Store feature columns used during training
        self.important_features = None  # Store important features for composite index
    
    def _handle_infinite_values(self, df):
        """
        Handle infinite values in DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with potential infinite values
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with infinite values replaced
        """
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Replace inf and -inf with NaN
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # For each column, replace NaN with the column median or 0 if median is unavailable
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    df_clean[col].fillna(0, inplace=True)
                else:
                    df_clean[col].fillna(median_val, inplace=True)
        
        return df_clean
        
    def train_model(self, X_train, y_train, recent_data=None):
        """
        Train the XGBoost classifier.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Features for training
        y_train : pandas.Series
            Target variable for training
        recent_data : pandas.DataFrame, optional
            Recent data for calculating accuracy metrics (default is None)
        """
        try:
            # Store feature columns used in training
            self.feature_columns = X_train.columns.tolist()
            
            # Clean input data to handle infinite values and NaNs
            X_train = self._handle_infinite_values(X_train)
            
            # Split data into train and validation sets
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_split)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Define XGBoost classifier with multi-core training
            self.model = xgb.XGBClassifier(
                n_estimators=500,           # Increased from 100
                learning_rate=0.05,         # Decreased from 0.1
                max_depth=6,                # Increased from 5
                random_state=42,
                subsample=0.8,              # Added: Use 80% of data per tree
                colsample_bytree=0.8,       # Added: Use 80% of features per tree
                min_child_weight=3,         # Added: Control overfitting
                gamma=0.1,                  # Added: Minimum loss reduction
                reg_alpha=0.1,              # Added: L1 regularization
                reg_lambda=1.0,             # Added: L2 regularization
                use_label_encoder=False,
                eval_metric=['logloss', 'error'],
                n_jobs=6,                   # Use 6 CPU cores for parallel processing
                tree_method='hist',         # Faster histogram-based algorithm
                scale_pos_weight=1.0,       # Balance positive and negative weights
                verbosity=1                 # Show some training information
            )
            
            # Create evaluation set for early stopping
            eval_set = [(X_train_scaled, y_train_split), (X_val_scaled, y_val)]
            
            # Train the model with early stopping
            self.model.fit(
                X_train_scaled, 
                y_train_split,
                eval_set=eval_set,
                early_stopping_rounds=50,  # Added here instead of in constructor
                verbose=True
            )
            
            # Get best iteration
            best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.model.n_estimators
            print(f"Best iteration: {best_iteration}")
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"Validation accuracy: {accuracy:.4f}")
            print(classification_report(y_val, y_pred, zero_division=0))
            
            # Retrain on the full dataset with the optimal number of trees
            X_train_full_scaled = self.scaler.transform(X_train)
            
            # Create a new model with the best number of trees but NO early stopping
            final_model = xgb.XGBClassifier(
                n_estimators=best_iteration,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric=['logloss', 'error'],
                n_jobs=6,
                tree_method='hist',
                scale_pos_weight=1.0,
                verbosity=0
            )
            
            # Fit the final model without early stopping
            final_model.fit(X_train_full_scaled, y_train)
            
            # Replace the model with the final version
            self.model = final_model
            
            # Get feature importance and store important features for composite index
            feature_importance = self.get_feature_importance()
            top_n = min(10, len(feature_importance))
            self.important_features = feature_importance.head(top_n)['Feature'].tolist()
            
            # Calculate accuracy metrics on recent data if provided
            if recent_data is not None:
                try:
                    # Clean recent data as well
                    if 'Target' in recent_data.columns:
                        y_recent = recent_data['Target']
                        recent_data_clean = self._handle_infinite_values(recent_data)
                        recent_data_clean['Target'] = y_recent  # Preserve original targets
                        self.calculate_recent_accuracy(recent_data_clean)
                    else:
                        self.calculate_recent_accuracy(recent_data)
                    print(f"Calculated recent accuracy with {len(recent_data)} days of recent data")
                except Exception as e:
                    print(f"Warning: Could not calculate recent accuracy: {str(e)}")
                    # Continue even if we can't calculate recent accuracy
        
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            raise
    
    def calculate_recent_accuracy(self, recent_data):
        """
        Calculate accuracy metrics for the most recent data.
        
        Parameters:
        -----------
        recent_data : pandas.DataFrame
            Recent data (last 120 days) for calculating accuracy metrics
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        # Extract target
        if 'Target' not in recent_data.columns:
            raise ValueError("Target column not found in recent data")
        
        y_recent = recent_data['Target']
        
        # Extract only the features that were used during training
        if self.feature_columns is None:
            raise ValueError("No feature columns stored from training")
        
        # Create a DataFrame with the same columns as used in training (filled with zeros)
        X_recent = pd.DataFrame(0, index=recent_data.index, columns=self.feature_columns)
        
        # Fill in the values for features that exist in both DataFrames
        for col in self.feature_columns:
            if col in recent_data.columns:
                X_recent[col] = recent_data[col]
        
        # Clean data to handle infinite values
        X_recent = self._handle_infinite_values(X_recent)
        
        # Scale features using the same scaler used during training
        X_recent_scaled = self.scaler.transform(X_recent)
        
        # Make predictions
        y_pred = self.model.predict(X_recent_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_recent, y_pred)
        conf_matrix = confusion_matrix(y_recent, y_pred)
        
        # Store metrics
        self.recent_accuracy = accuracy
        
        # Calculate additional metrics
        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store additional metrics
            self.recent_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'up_precision': precision,  # Precision for predicting price up
                'down_precision': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Precision for predicting price down
                'up_count': int(tp + fn),  # Total days price went up
                'down_count': int(tn + fp),  # Total days price went down
                'correct_predictions': int(tp + tn),  # Total correct predictions
                'total_predictions': len(y_recent)  # Total predictions
            }
        
        print(f"Calculated recent accuracy metrics using {len(y_recent)} days of data")
        return accuracy
    
    def predict(self, features):
        """
        Make a prediction for the given features.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features for prediction
        
        Returns:
        --------
        int
            Prediction (1 for price up, 0 for price down)
        float
            Probability of price going up
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        # Ensure we have all required features
        if self.feature_columns is not None:
            # Create a DataFrame with the same columns as used in training (filled with zeros)
            prediction_features = pd.DataFrame(0, index=features.index, columns=self.feature_columns)
            
            # Fill in the values for features that exist in both DataFrames
            for col in self.feature_columns:
                if col in features.columns:
                    prediction_features[col] = features[col]
            
            features = prediction_features
        
        # Clean data to handle infinite values
        features = self._handle_infinite_values(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return prediction, probability
    
    def calculate_composite_index(self, df):
        """
        Calculate a composite prediction index from 0 to 100.
        0 = Strong sell signal, 100 = Strong buy signal.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with features
            
        Returns:
        --------
        pandas.Series
            Composite index values
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        try:
            # Create a copy to avoid modifying the original
            data = df.copy()
            
            # Initialize composite index components
            components = []
            
            # 1. Model predictions (if enough features available)
            if self.feature_columns is not None:
                try:
                    # Prepare features for prediction
                    model_features = pd.DataFrame(0, index=data.index, columns=self.feature_columns)
                    for col in self.feature_columns:
                        if col in data.columns:
                            model_features[col] = data[col]
                    
                    # Clean data to handle infinite values
                    model_features = self._handle_infinite_values(model_features)
                    
                    # Get prediction probabilities
                    features_scaled = self.scaler.transform(model_features)
                    prediction_probs = self.model.predict_proba(features_scaled)[:, 1]  # Probability of class 1 (up)
                    
                    # Add to components (weight: 40%)
                    components.append(pd.Series(prediction_probs * 100, index=data.index, name="Model_Prediction").mul(0.4))
                except Exception as e:
                    print(f"Warning: Could not include model predictions in composite index: {str(e)}")
            
            # 2. Technical indicators component (weight: 60%)
            tech_components = []
            
            # Add RSI (normalized to 0-100 scale, 30->0, 70->100)
            if 'RSI' in data.columns:
                rsi_norm = pd.Series(data['RSI'].copy(), index=data.index, name="RSI_Component")
                # Convert RSI to a scale where oversold (30) = 0 and overbought (70) = 100
                rsi_norm = (rsi_norm - 30) * (100 / 40)
                rsi_norm = rsi_norm.clip(0, 100)
                tech_components.append(rsi_norm)
            
            # Add MACD histogram (normalized)
            if 'MACD_hist' in data.columns:
                # Calculate absolute max for normalization
                max_val = data['MACD_hist'].abs().max()
                if max_val > 0:
                    macd_norm = pd.Series(data['MACD_hist'].copy(), index=data.index, name="MACD_Component")
                    # Scale from -max to +max to 0-100
                    macd_norm = ((macd_norm / max_val) + 1) * 50
                    tech_components.append(macd_norm)
            
            # Add Bollinger Band position
            if 'BB_pct' in data.columns:
                bb_component = pd.Series(data['BB_pct'].copy() * 100, index=data.index, name="BB_Component")
                tech_components.append(bb_component)
            
            # Add Stochastic Oscillator
            if 'Stoch_K' in data.columns:
                stoch_component = pd.Series(data['Stoch_K'].copy(), index=data.index, name="Stoch_Component")
                tech_components.append(stoch_component)
            
            # Add MFI if available
            if 'MFI' in data.columns:
                mfi_component = pd.Series(data['MFI'].copy(), index=data.index, name="MFI_Component")
                tech_components.append(mfi_component)
            
            # Add moving average crossover signal
            if 'MA_Cross_Signal' in data.columns:
                ma_cross = pd.Series(data['MA_Cross_Signal'].copy(), index=data.index, name="MA_Cross_Component")
                # Convert -1/1 to 0/100
                ma_cross = ((ma_cross + 1) / 2) * 100
                tech_components.append(ma_cross)
            
            # Add custom composite signal if available
            if 'Composite_Signal' in data.columns:
                custom_signal = pd.Series(data['Composite_Signal'].copy() * 100, index=data.index, name="Custom_Component")
                tech_components.append(custom_signal)
            
            # If we have technical components, combine them and add to main components list
            if tech_components:
                tech_df = pd.concat(tech_components, axis=1)
                # Replace any NaN or infinite values
                tech_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                tech_df.fillna(50, inplace=True)  # Fill with neutral value
                tech_component = tech_df.mean(axis=1).mul(0.6)  # Weight: 60%
                components.append(tech_component)
            
            # Combine all components
            if not components:
                # If no components are available, return a default series
                default_index = pd.Series(50, index=data.index, name="Composite_Index")
                return default_index
            
            composite_index = pd.concat(components, axis=1).sum(axis=1)
            
            # Ensure the index is between 0 and 100
            composite_index = composite_index.clip(0, 100)
            
            # Apply smoothing (5-period moving average)
            composite_index = composite_index.rolling(window=5).mean().fillna(50)
            
            return composite_index
            
        except Exception as e:
            print(f"Error calculating composite index: {str(e)}")
            # Return a default series in case of error
            return pd.Series(50, index=df.index, name="Composite_Index")
    
    def save_model(self, model_path="stock_predictor_model.pkl"):
        """
        Save the trained model and scaler to disk.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to save the model (default is "stock_predictor_model.pkl")
        """
        if self.model is None:
            raise ValueError("Model is not trained. Cannot save untrained model.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'index_scaler': self.index_scaler,
            'recent_accuracy': self.recent_accuracy,
            'recent_metrics': self.recent_metrics,
            'feature_columns': self.feature_columns,
            'important_features': self.important_features
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {model_path}")
        
    def load_model(self, model_path="stock_predictor_model.pkl"):
        """
        Load a trained model and scaler from disk.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to load the model from (default is "stock_predictor_model.pkl")
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        # Load index scaler if available
        if 'index_scaler' in model_data:
            self.index_scaler = model_data['index_scaler']
        
        # Load feature columns if available
        if 'feature_columns' in model_data:
            self.feature_columns = model_data['feature_columns']
        
        # Load important features if available
        if 'important_features' in model_data:
            self.important_features = model_data['important_features']
        
        # Load metrics if available
        if 'recent_accuracy' in model_data:
            self.recent_accuracy = model_data['recent_accuracy']
        
        if 'recent_metrics' in model_data:
            self.recent_metrics = model_data['recent_metrics']
        
        print(f"Model loaded from {model_path}")
        
    def get_feature_importance(self):
        """
        Get the feature importance from the trained model.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        # Use stored feature columns
        if self.feature_columns is None:
            raise ValueError("No feature columns stored")
        
        feature_names = self.feature_columns
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return feature_importance