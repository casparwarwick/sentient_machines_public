##########################################
# Shared Classifier Classes
##########################################

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

## Logistic regression
class LRClassifier:
    """Logistic Regression classifier."""
    
    def __init__(self, normalize=None):
        self.lr = None
        self.layer = None
        self.scaler = None
        self.normalize = normalize
        
    def train(self, acts, labels, ridge_penalty=None):
        """Train the Logistic Regression classifier."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if enabled
        if self.normalize:
            self.scaler = StandardScaler()
            acts_numpy = self.scaler.fit_transform(acts_numpy)
        
        if ridge_penalty is None:
            penalty = None
        else:
            penalty = 'l2'
        
        self.lr = LogisticRegression(
            penalty=penalty, 
            C=1.0/ridge_penalty if ridge_penalty else 1.0,
            fit_intercept=True,
            max_iter=1000
        )
        self.lr.fit(acts_numpy, labels.numpy())
    
    def predict(self, acts):
        """Predict labels for new activations."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if scaler was fitted during training
        if self.scaler is not None:
            acts_numpy = self.scaler.transform(acts_numpy)
            
        return self.lr.predict(acts_numpy)
    
    def predict_proba(self, acts):
        """Predict probabilities for new activations."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if scaler was fitted during training
        if self.scaler is not None:
            acts_numpy = self.scaler.transform(acts_numpy)
            
        return self.lr.predict_proba(acts_numpy)

## Mass Mean Classifier from Marks and Tegmark
class MMClassifier:
    """Mass Mean classifier."""
    
    def __init__(self, normalize=None):
        self.direction = None
        self.lr = None
        self.layer = None
        self.scaler = None
        self.normalize = normalize
        
    def train(self, acts, labels):
        """Train the Mass Mean classifier."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if enabled
        if self.normalize:
            self.scaler = StandardScaler()
            acts_numpy = self.scaler.fit_transform(acts_numpy)
            acts = torch.tensor(acts_numpy)
        
        pos_mask = labels == 1
        neg_mask = labels == -1
        
        pos_acts = acts[pos_mask]
        neg_acts = acts[neg_mask]
        
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        
        self.direction = (pos_mean - neg_mean).numpy()
        
        projections = acts.numpy() @ self.direction
        projections = projections.reshape(-1, 1)
        
        self.lr = LogisticRegression(penalty=None, fit_intercept=True)
        self.lr.fit(projections, labels.numpy())
    
    def predict(self, acts):
        """Predict labels for new activations."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if scaler was fitted during training
        if self.scaler is not None:
            acts_numpy = self.scaler.transform(acts_numpy)
            
        projections = acts_numpy @ self.direction
        projections = projections.reshape(-1, 1)
        return self.lr.predict(projections)
    
    def predict_proba(self, acts):
        """Predict probabilities for new activations."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if scaler was fitted during training
        if self.scaler is not None:
            acts_numpy = self.scaler.transform(acts_numpy)
            
        projections = acts_numpy @ self.direction
        projections = projections.reshape(-1, 1)
        return self.lr.predict_proba(projections)

## Training of Truth and Polarity Direction (TTPD) classifier by Bürger et al. 
# Some helper functions
def learn_truth_directions(acts_centered, labels, polarities):
    """Learn truth directions t_g and t_p using OLS."""
    has_negations = not torch.allclose(polarities.float(), torch.tensor(1.0), atol=1e-8)
    
    if not has_negations:
        X = labels.float().reshape(-1, 1)
    else:
        X = torch.column_stack([labels.float(), labels.float() * polarities.float()])
    
    acts_centered_float = acts_centered.float()
    solution = torch.linalg.inv(X.T @ X) @ X.T @ acts_centered_float
    
    if not has_negations:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]
    
    return t_g, t_p

def learn_polarity_direction(acts, polarities):
    """Learn polarity direction using logistic regression."""
    polarity_labels = (polarities + 1) / 2
    
    lr = LogisticRegression(penalty=None, fit_intercept=True)
    lr.fit(acts.numpy(), polarity_labels.numpy())
    
    return lr.coef_[0]

class TTPDClassifier:
    """TTPD (Truth and Polarity Direction) classifier."""
    
    def __init__(self, normalize=None):
        self.t_g = None
        self.polarity_direc = None
        self.lr = None
        self.layer = None
        self.scaler = None
        self.normalize = normalize
        
    def train(self, acts, labels, polarities):
        """Train the TTPD classifier."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if enabled
        if self.normalize:
            self.scaler = StandardScaler()
            acts_numpy = self.scaler.fit_transform(acts_numpy)
            acts = torch.tensor(acts_numpy)
        
        acts_centered = acts - acts.mean(dim=0)
        
        self.t_g, self.t_p = learn_truth_directions(acts_centered, labels, polarities)
        self.t_g = self.t_g.numpy()
        
        self.polarity_direc = learn_polarity_direction(acts, polarities)
        
        acts_2d = self._project_acts(acts)
        
        self.lr = LogisticRegression(penalty=None, fit_intercept=True)
        self.lr.fit(acts_2d, labels.numpy())
        
    def _project_acts(self, acts):
        """Project activations onto truth and polarity directions."""
        acts_numpy = acts.numpy()
        
        # Apply normalization if scaler was fitted during training
        if self.scaler is not None:
            acts_numpy = self.scaler.transform(acts_numpy)
            
        proj_t_g = acts_numpy @ self.t_g
        proj_p = acts_numpy @ self.polarity_direc.T
        acts_2d = np.column_stack([proj_t_g, proj_p])
        return acts_2d
    
    def predict(self, acts):
        """Predict labels for new activations."""
        acts_2d = self._project_acts(acts)
        return self.lr.predict(acts_2d)
    
    def predict_proba(self, acts):
        """Predict probabilities for new activations."""
        acts_2d = self._project_acts(acts)
        return self.lr.predict_proba(acts_2d)
