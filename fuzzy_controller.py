# fuzzy_controller.py

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict
import json
from scipy.optimize import differential_evolution
from scipy.stats import trim_mean

@dataclass
class ControllerConfig:
    min_speed: float = 0.75    
    max_speed: float = 1.75    
    default_speed: float = 1.0 
    step_size: float = 0.25   

@dataclass
class FuzzyParameters:
    # Beta power centers (based on percentile analysis)
    beta_centers: Dict[str, float] = field(default_factory=lambda: {
        'very_low':  3.0,
        'low':       4.0,
        'medium':    4.5,
        'high':      5.0,
        'very_high': 6.0
    })
    
    # Beta ratio centers (based on percentile analysis)
    ratio_centers: Dict[str, float] = field(default_factory=lambda: {
        'very_low':  0.33,
        'low':       0.50,
        'medium':    0.67,
        'high':      0.85,
        'very_high': 0.98
    })
    
    # Speed rules (will be optimized)
    speed_rules: Dict[str, float] = field(default_factory=lambda: {
        'very_low':  2.0,
        'low':       1.5,
        'medium':    1.0,
        'high':      0.75,
        'very_high': 0.5
    })
    
    beta_weight: float = 0.6
    ratio_weight: float = 0.4
    beta_spread: float = 0.8
    ratio_spread: float = 0.15

class TrainableFuzzyController:
    def __init__(self, params: FuzzyParameters = None):
        self.config = ControllerConfig()
        self.params = params if params else FuzzyParameters()
        self.current_speed = self.config.default_speed
        self.window_size = 5
        self.beta_history = []
        self.ratio_history = []
    
    def _membership(self, value, center, spread):
        return np.exp(-((value - center) ** 2) / (2 * spread ** 2))
    
    def _get_smoothed_value(self, value, history):
        """Improved smoothing with outlier rejection"""
        history.append(value)
        if len(history) > self.window_size:
            history.pop(0)
        if len(history) >= 3:
            return trim_mean(history, 0.2)
        return np.mean(history)
    
    def update(self, theta_power, alpha_power, beta_power):
        """Calculate target speed based on EEG metrics with improved outlier handling"""
        beta_power = np.clip(beta_power, 2.5, 8.3)
        
        if (theta_power + alpha_power) > 0:
            beta_ratio = beta_power / (theta_power + alpha_power)
            beta_ratio = np.clip(beta_ratio, 0, 1.68)
        else:
            beta_ratio = 0.4
        
        beta = self._get_smoothed_value(beta_power, self.beta_history)
        ratio = self._get_smoothed_value(beta_ratio, self.ratio_history)
        
        beta_memberships = {
            label: self._membership(beta, center, self.params.beta_spread)
            for label, center in self.params.beta_centers.items()
        }
        
        ratio_memberships = {
            label: self._membership(ratio, center, self.params.ratio_spread)
            for label, center in self.params.ratio_centers.items()
        }
        
        combined_memberships = {}
        for label in self.params.speed_rules.keys():
            combined_memberships[label] = (
                self.params.beta_weight * beta_memberships[label] +
                self.params.ratio_weight * ratio_memberships[label]
            )
        
        total_membership = sum(combined_memberships.values())
        if total_membership > 0:
            target_speed = sum(
                membership * self.params.speed_rules[label] 
                for label, membership in combined_memberships.items()
            ) / total_membership
        else:
            target_speed = self.config.default_speed
            
        return np.clip(target_speed, self.config.min_speed, self.config.max_speed)

    @classmethod
    def load_parameters(cls, filepath):
        """Load parameters from a JSON file"""
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        params = FuzzyParameters()
        params.beta_centers = params_dict['beta_centers']
        params.ratio_centers = params_dict['ratio_centers']
        params.speed_rules = params_dict['speed_rules']
        params.beta_weight = params_dict['beta_weight']
        params.ratio_weight = params_dict['ratio_weight']
        params.beta_spread = params_dict['beta_spread']
        params.ratio_spread = params_dict['ratio_spread']
        return cls(params)

    def save_parameters(self, filepath):
        """Save current parameters to a JSON file"""
        params_dict = {
            'beta_centers': self.params.beta_centers,
            'ratio_centers': self.params.ratio_centers,
            'speed_rules': self.params.speed_rules,
            'beta_weight': self.params.beta_weight,
            'ratio_weight': self.params.ratio_weight,
            'beta_spread': self.params.beta_spread,
            'ratio_spread': self.params.ratio_spread
        }
        with open(filepath, 'w') as f:
            json.dump(params_dict, f, indent=4)