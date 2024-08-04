# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:35:18 2019

@author: A.Goumilevski
"""
from enum import unique
from enum import Enum

@unique
class SolverMethod(Enum):
    """Solver algorithms."""
    LBJ               = 1  # Juillard, Laxton, McAdam, Pioro algorithm
    ABLR              = 2  # Armstrong, Black, Laxton, Rose algorithm
    Villemot          = 3  # Villemot Sebastien  (DYNARE Toolbox algorithm)
    Benes             = 4  # Jaromir Benes (IRIS Toolbox algorithm)
    BinderPesaran     = 5  # Binder and Pesaran algorithm
    AndersonMoore     = 6  # Anderson and Moore algorithm
    
@unique
class FilterAlgorithm(Enum):
    """Kalman filter algorithms."""
    Diffuse            = 1 # Diffuse Kalman filter (multivariate and univariate) with missing observations
    Durbin_Koopman     = 2 # Non-diffuse variant of Kalman filter
    Non_Diffuse_Filter = 3 # Non Diffuse variant of Durbin-Koopman Kalman filter
    Particle           = 4 # Particle filter
    Unscented          = 5 # Unscented Kalman filter
     
    
@unique
class SmootherAlgorithm(Enum):
    """Kalman smoother algorithms."""
    BrysonFrazier        = 1 # Bryson, Frazier    
    Diffuse              = 2 # Diffuse Kalman Smoother (multivariate and univariate) with missing observations
    Durbin_Koopman       = 3 # Non-diffuse variant of Kalman smoother


@unique
class InitialCondition(Enum):
    """Endogenous variables starting values algorithms."""
    StartingValues = 1 # Starting values are used
    SteadyState    = 2 # Steady-state values are used as starting values
    History        = 3 # Starting values are read from a history file
    
@unique
class PriorAssumption(Enum):
    """Assumptions about starting value of error covariance matrix."""
    Diffuse        = 1 # Diffuse prior for covariance matrices (Pinf and Pstar)
    StartingValues = 2 # Starting values for covariance matrices (Pinf with diagonal values of 1.E6 on diagonl and Pstar=TQT')
    Equilibrium    = 3 # Equilibrium error covariance matrices obtained by discrete Lyapunov solver by using stable part ot transition and shock matrices
    Asymptotic     = 4 # Asymptotic values for error covariance matrices; it is obtained by solving discrete Riccati equation. 
  
@unique
class BoundaryConditions(Enum):
    """Boundary Conditions."""
    FixedBoundaryCondition           = 1 # Fixed condition at right boundary
    ZeroDerivativeBoundaryCondition  = 2 # Zero derivative condition at right boundary
    
class BoundaryCondition:
    Condition = BoundaryConditions.ZeroDerivativeBoundaryCondition
    #Condition = BoundaryConditions.FixedBoundaryCondition