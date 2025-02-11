import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
import os
from scipy.optimize import differential_evolution
from scipy.stats import trim_mean
import glob
from fuzzy_controller import TrainableFuzzyController

class FuzzyControllerTrainer:
    def __init__(self, training_data_dir: str):
        """
        Initialize trainer with data from both high and low cognitive load states
        
        Args:
            training_data_dir: Directory containing the training data CSV files
        """
        self.data = self._load_training_data(training_data_dir)
        self.controller = TrainableFuzzyController()
    
    def _load_training_data(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess all training data"""
        # Find the most recent files for each type
        files = glob.glob(os.path.join(data_dir, '*.csv'))
        if not files:
            raise ValueError(f"No CSV files found in {data_dir}")
            
        latest_files = {
            'baseline': None,
            'high_load': None,
            'low_load': None
        }
        
        # Find the most recent file for each type
        for file in files:
            basename = os.path.basename(file)
            for data_type in latest_files.keys():
                if data_type in basename:
                    if latest_files[data_type] is None or os.path.getmtime(file) > os.path.getmtime(latest_files[data_type]):
                        latest_files[data_type] = file
        
        # Load and preprocess each dataset
        datasets = {}
        for data_type, filepath in latest_files.items():
            if filepath and os.path.exists(filepath):
                print(f"Loading {data_type} data from {filepath}")
                df = pd.read_csv(filepath)
                df = self._preprocess_data(df)
                datasets[data_type] = df
            else:
                print(f"Warning: No data found for {data_type}")
        
        return datasets
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data to remove outliers and normalize values"""
        processed_df = df.copy()
        
        # Remove outliers using IQR method
        for col in ['Theta_pow', 'Alpha_pow', 'Beta_pow', 'Gamma_pow']:
            if col in processed_df.columns:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                processed_df = processed_df[
                    (processed_df[col] >= Q1 - 1.5 * IQR) & 
                    (processed_df[col] <= Q3 + 1.5 * IQR)
                ]
        
        return processed_df
    
    def _calculate_ideal_speed(self, cognitive_load: float, data_type: str) -> float:
        """
        Calculate ideal speed based on cognitive load and data type
        
        Args:
            cognitive_load: Beta/(Theta + Alpha) ratio
            data_type: 'baseline', 'high_load', or 'low_load'
        """
        if data_type == 'high_load':
            # Slower speeds for high cognitive load
            base_speed = 1.5
            min_speed = 0.75
            midpoint = 0.7
        elif data_type == 'low_load':
            # Faster speeds for low cognitive load
            base_speed = 2.5
            min_speed = 1.5
            midpoint = 0.4
        else:  # baseline
            # Medium speeds for baseline
            base_speed = 2.0
            min_speed = 1.0
            midpoint = 0.5
        
        steepness = 5
        speed = base_speed - (base_speed - min_speed) / (
            1 + np.exp(-steepness * (cognitive_load - midpoint))
        )
        return np.clip(speed, self.controller.config.min_speed, 
                      self.controller.config.max_speed)
    
    def objective_function(self, params):
        """Objective function that considers all states"""
        # Unpack parameters
        speed_rules = {
            'very_low':  params[0],
            'low':       params[1],
            'medium':    params[2],
            'high':      params[3],
            'very_high': params[4]
        }
        beta_weight = params[5]
        beta_spread = params[6]
        ratio_spread = params[7]
        
        # Update controller parameters
        self.controller.params.speed_rules = speed_rules
        self.controller.params.beta_weight = beta_weight
        self.controller.params.ratio_weight = 1 - beta_weight
        self.controller.params.beta_spread = beta_spread
        self.controller.params.ratio_spread = ratio_spread
        
        total_error = 0
        total_samples = 0
        
        # Calculate error for each dataset
        for data_type, df in self.data.items():
            if df is None or df.empty:
                continue
                
            for _, row in df.iterrows():
                target_speed = self.controller.update(
                    row['Theta_pow'],
                    row['Alpha_pow'],
                    row['Beta_pow']
                )
                
                beta_ratio = row['Beta_pow'] / (row['Theta_pow'] + row['Alpha_pow'])
                ideal_speed = self._calculate_ideal_speed(beta_ratio, data_type)
                
                # Calculate weighted error
                error = (target_speed - ideal_speed) ** 2
                
                # Add higher penalty for deviations in high cognitive load state
                if data_type == 'high_load':
                    error *= 1.5
                
                total_error += error
                total_samples += 1
        
        return total_error / total_samples if total_samples > 0 else float('inf')
    
    def train(self):
        """Train the controller using all available data"""
        bounds = [
            (1.5, 2.5),   # very_low speed
            (1.25, 2.0),  # low speed
            (0.9, 1.1),   # medium speed
            (0.5, 0.8),   # high speed
            (0.3, 0.6),   # very_high speed
            (0.5, 0.8),   # beta_weight
            (0.5, 1.5),   # beta_spread
            (0.1, 0.3)    # ratio_spread
        ]
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=50,
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating='deferred',
            workers=-1  # Use all available CPU cores
        )
        
        if result.success:
            print("\nTraining successful!")
            print(f"Final error: {result.fun:.6f}")
            
            # Update controller with optimal parameters
            optimal_params = result.x
            self.controller.params.speed_rules = {
                'very_low':  optimal_params[0],
                'low':       optimal_params[1],
                'medium':    optimal_params[2],
                'high':      optimal_params[3],
                'very_high': optimal_params[4]
            }
            self.controller.params.beta_weight = optimal_params[5]
            self.controller.params.ratio_weight = 1 - optimal_params[5]
            self.controller.params.beta_spread = optimal_params[6]
            self.controller.params.ratio_spread = optimal_params[7]
            
            print("\nOptimized Parameters:")
            print(f"Speed Rules: {self.controller.params.speed_rules}")
            print(f"Beta Weight: {self.controller.params.beta_weight:.2f}")
            print(f"Ratio Weight: {self.controller.params.ratio_weight:.2f}")
            print(f"Beta Spread: {self.controller.params.beta_spread:.2f}")
            print(f"Ratio Spread: {self.controller.params.ratio_spread:.2f}")
        else:
            print("\nTraining failed:", result.message)
        
        return self.controller

def train_controller(training_data_dir: str, output_params_path: str):
    """
    Train controller using all available data and save parameters
    
    Args:
        training_data_dir: Directory containing training data CSV files
        output_params_path: Path to save optimized parameters JSON
    """
    print(f"\nStarting controller training...")
    print(f"Loading data from: {training_data_dir}")
    print(f"Parameters will be saved to: {output_params_path}")
    
    trainer = FuzzyControllerTrainer(training_data_dir)
    trained_controller = trainer.train()
    
    if trained_controller is not None:
        params_dict = {
            'beta_centers': trained_controller.params.beta_centers,
            'ratio_centers': trained_controller.params.ratio_centers,
            'speed_rules': trained_controller.params.speed_rules,
            'beta_weight': trained_controller.params.beta_weight,
            'ratio_weight': trained_controller.params.ratio_weight,
            'beta_spread': trained_controller.params.beta_spread,
            'ratio_spread': trained_controller.params.ratio_spread
        }
        
        # Just write directly to the file - no need for makedirs() if it's just a filename
        with open(output_params_path, 'w') as f:
            json.dump(params_dict, f, indent=4)
        print(f"\nOptimized parameters saved to: {output_params_path}")
    
    return trained_controller

if __name__ == "__main__":
    import argparse
    import time
    start = time.time()    
    parser = argparse.ArgumentParser(description='Train Fuzzy Controller from collected EEG data')
    parser.add_argument('--data_dir', type=str, default='training_data',
                      help='Directory containing training data CSV files')
    parser.add_argument('--output', type=str, default='optimized_parameters.json',
                      help='Output path for optimized parameters JSON')
    
    args = parser.parse_args()
    
    try:
        trained_controller = train_controller(
            args.data_dir,
            args.output
        )
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    print(time.time() - start)