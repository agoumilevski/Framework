# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13, 2018
Driver program of Python framework.

@author: A.Goumilevski
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from misc.termcolor import cprint
from model.factory import import_model
from model.model import Model
from utils.util import simulationRange,getPeriods
from utils.prettyTable import PrettyTable
from utils.util import saveToDatabase,saveToExcel
from numeric.solver import linear_solver,nonlinear_solver
from numeric.filters.filter import linear_filter,nonlinear_filter
from graphs.util import plot,plotSteadyState,plotImage
from graphs.util import plotEigenValues,plotDecomposition
from utils.getData import fetchData
from numeric.solver import util
from utils.util import getOutputFolderPath

out_dir = getOutputFolderPath()
figures_dir = os.path.abspath(os.path.abspath(os.path.join(out_dir,'graphs')))
xl_dir = os.path.abspath(os.path.abspath(os.path.join(out_dir,'data')))
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
if not os.path.exists(xl_dir):
    os.makedirs(xl_dir)

def setParameters(
        model, order, Solver, Filter, Smoother, 
        Prior, InitCondition, SamplingMethod=None
        ):
    """
    Set model parameters.
    
    Parameters:
        :param model: Model object.
        :type fname: Model.
        :param order: Approximation order of solution of non-linear system of equations.
        :type order: int.
        :param Solver: Solver algorithm.
        :type Solver: str.
        :param Filter: KF filter algorithm.
        :type Filter: str.
        :param Smoother: KF smoother algorithm.
        :type Smoother: str.
        :param Prior: Error covariance matrix estimation method.
        :type Prior: str.
        :param InitCondition: Endogenous variables initial condition method.
        :type InitCondition: str.
        :param SamplingMethod: Markov Chain Monte Carlo sampling algorithm.
        :type SamplingMethod: str.
        :returns:  Instance of Model class.
    """
    from model.settings import SmootherAlgorithm, FilterAlgorithm
    from model.settings import InitialCondition, PriorAssumption

    if not Filter is None:
        if len(Filter) == 0:
            model.FILTER = None    
        elif Filter.upper() == "DURBIN_KOOPMAN":         # Non-diffuse Kalman Filter
            model.FILTER = FilterAlgorithm.Durbin_Koopman
        elif Filter.upper() == "NON_DIFFUSE_FILTER":     # Non diffuse Durbin-Koopman Kalman Filter
            model.FILTER = FilterAlgorithm.Non_Diffuse_Filter
        elif Filter.upper() == "DIFFUSE":                # Diffuse Kalman Filter (transcription of Dynare code to Python)
            model.FILTER = FilterAlgorithm.Diffuse        
        elif Filter.upper() == "PARTICLE":               # Particle Filter
            model.FILTER = FilterAlgorithm.Particle
        elif Filter.upper() == "UNSCENTED":              # Unscented Kalman Filter
            model.FILTER = FilterAlgorithm.Unscented
        else:
            cprint(f"Filter algorithm {Filter} is not implemented... Resetting to Diffuse algorithm.","red") 
            model.FILTER = FilterAlgorithm.Diffuse

    if not Smoother is None:
        if len(Smoother) == 0:
            model.SMOOTHER = None
        elif Smoother.upper() == "BRYSONFRAZIER":
            model.SMOOTHER = SmootherAlgorithm.BrysonFrazier  
        elif Smoother.upper() == "DURBIN_KOOPMAN":         # Non-diffuse Kalman Smoother
            model.SMOOTHER = SmootherAlgorithm.Durbin_Koopman
        elif Smoother.upper() == "DIFFUSE":
            model.SMOOTHER = SmootherAlgorithm.Diffuse   
        else:
            cprint(f"Smoother algorithm {Smoother} is not implemented... Resetting to Diffuse algorithm.","red")
            model.SMOOTHER = SmootherAlgorithm.Diffuse  
        if model.FILTER.value == FilterAlgorithm.Diffuse.value:
            if not model.SMOOTHER.value == SmootherAlgorithm.Diffuse.value:
                cprint(f"Smoother {model.SMOOTHER} should be set to {SmootherAlgorithm.Diffuse}... Resetting.\n","red")
                model.SMOOTHER = SmootherAlgorithm.Diffuse 
        elif model.FILTER.value == FilterAlgorithm.Durbin_Koopman.value:
            if not model.SMOOTHER.value == SmootherAlgorithm.Durbin_Koopman.value:
                cprint(f"Smoother {model.SMOOTHER} should be set to {SmootherAlgorithm.Durbin_Koopman}... Resetting.\n","red")
                model.SMOOTHER = SmootherAlgorithm.Durbin_Koopman 
        
    if not Prior is None:
        if Prior.upper() == "STARTINGVALUES":
            model.PRIOR = PriorAssumption.StartingValues
        elif Prior.upper() == "EQUILIBRIUM":
            model.PRIOR = PriorAssumption.Equilibrium
        elif Prior.upper() == "DIFFUSE":
            model.PRIOR = PriorAssumption.Diffuse
        elif Prior.upper() == "ASYMPTOTIC":
            model.PRIOR = PriorAssumption.Asymptotic

    if not InitCondition is None:
        if InitCondition.upper() == "STEADYSTATE":
            model.INITIAL_CONDITION = InitialCondition.SteadyState
        elif InitCondition.upper() == "STARTINGVALUES":
            model.INITIAL_CONDITION = InitialCondition.StartingValues
        elif InitCondition.upper() == "HISTORY":
            model.INITIAL_CONDITION = InitialCondition.History
        else:
            cprint(f"Initial condition {InitCondition} is not implemented... Resetting to Starting values.","red")
            model.INITIAL_CONDITION = InitialCondition.StartingValues
        
    return model
        

def importModel(fname, order=1, hist=None, boundary_conditions=None, exog=None, InitCondition=None,
                Prior=None, estimate=False, Solver=None, Filter=None, Smoother=None,
                shocks_file_path=None, steady_state_file_path=None, measurement_file_path=None, 
                calibration_file_path=None, use_cache=False,anticipate=None,
                model_info=False,graph_info=False,bSparse=False):
    """
    Import model.
    
    Parameters:
        :param fname: The path to yaml file name.
        :type fname: str.
        :param order: Approximation order of solution of the non-linear system of equations.
        :type order: int.
        :param hist: Path to the history excel file. 
        :type hist: str.
        :param boundary_conditions: Path to the boundary conditions excel file.  This file contains initial and terminal conditions.
        :type boundary_conditions: str.
        :param exog: List of exogenous variables.
        :type exog: list.
        :param InitCondition: Endogenous variables initial condition method.
        :type InitCondition: str.
        :param Prior: Error covariance matrix estimation method.
        :type Prior: str.
        :param estimate: If True estimate model parameters.
        :type estimate: bool.
        :param Solver: Solver algorithm.
        :type Solver: str.
        :param Filter: KF filter algorithm.
        :type Filter: str.
        :param Smoother: KF smoother algorithm.
        :type Smoother: str.
        :param shocks_file_path: Path to shock file.
        :type shocks_file_path: str.
        :param steady_state_file_path: Path to steady-state file.
        :type steady_state_file_path: str.
        :param measurement_file_path: Path to a file with measurement data.
        :type measurement_file_path: str. 
        :param calibration_file_path: Path to calibration files or a file.
        :type calibration_file_path: list or str.
        :param use_cache: If True reads previously saved model from a file of model dump.
        :type use_cache: bool.
        :param anticipate: If True future shocks are anticipated.
        :type anticipate: bool.
        :param model_info: If True creates a pdf/latex model file.
        :type model_info: bool.
        :param graph_info: If True displays graph of model equations. The default is False.
        :type graph_info:  bool, optional
        :param bSparse: Use sparse matrix algebra.
        :type bSparse: bool.
        
        :returns:  Instance of Model class.
    """
    from utils.interface import loadModel
    from preprocessor.util import updateFiles
    from model.settings import SolverMethod
    from model.settings import InitialCondition
    
    name, ext = os.path.splitext(fname)
    model_path = fname.replace(name+ext,name+".bin")
    
    model_file_exist = os.path.exists(model_path)
    if model_file_exist and use_cache:
        
        model = loadModel(model_path,
                          shocks_file_path=shocks_file_path,
                          steady_state_file_path=steady_state_file_path,
                          calibration_file_path=calibration_file_path)
        
        model.anticipate = anticipate
        model.bSparse = bSparse
        if not Solver is None:
            if Solver.upper() == "VILLEMOT":
                model.SOLVER = SolverMethod.Villemot
                model.isLinear = True
            elif Solver.upper() == "ANDERSONMOORE":
                model.SOLVER = SolverMethod.AndersonMoore
                model.isLinear = True
            elif Solver.upper() == "BINDERPESARAN":
                model.SOLVER = SolverMethod.BinderPesaran
                model.isLinear = True
            elif Solver.upper() == "BENES":
                model.SOLVER = SolverMethod.Benes
                model.isLinear = True
            elif Solver.upper() == "LBJ":
                model.SOLVER = SolverMethod.LBJ
                model.isLinear = False
            elif Solver.upper() == "ABLR":
                model.SOLVER = SolverMethod.ABLR
                model.isLinear = False
            elif Solver.upper() == "LBJAX":
                model.SOLVER = SolverMethod.LBJax
                model.isLinear = False
                model.jaxdiff = True
            elif Solver.upper() == "BLOCK":
                model.SOLVER = SolverMethod.Block
                model.isLinear = False
            elif Solver.upper() == "DM":
                model.SOLVER = SolverMethod.DM
                model.isLinear = False
            else:
                model.SOLVER = SolverMethod.LBJ
                model.isLinear = False
        else:
            model.SOLVER = SolverMethod.LBJ
            model.isLinear = False
            
        if not InitCondition is None:
            if InitCondition.upper() == "STEADYSTATE":
                model.INITIAL_CONDITION = InitialCondition.SteadyState
            elif InitCondition.upper() == "STARTINGVALUES":
                model.INITIAL_CONDITION = InitialCondition.StartingValues
            elif InitCondition.upper() == "HISTORY":
                model.INITIAL_CONDITION = InitialCondition.History
            else:
                cprint(f"Initial condition {InitCondition} is not implemented... Resetting to Starting values.","red")
                model.INITIAL_CONDITION = InitialCondition.StartingValues

        # Update python generated files   
        path = os.path.dirname(os.path.abspath(__file__))
        updateFiles(model,path+"/preprocessor")
        
    else:       
        smodel = import_model(fname=fname,order=order,hist=hist,
                              boundary_conditions=boundary_conditions,
                              exog=exog,shocks_file_path=shocks_file_path,
                              steady_state_file_path=steady_state_file_path,
                              calibration_file_path=calibration_file_path,
                              measurement_file_path=measurement_file_path)
        smodel.bSparse = bSparse
        infos = {
            'name' : smodel.name,
            'filename' : fname
        }
        
        if not Solver is None:
            if Solver.upper() == "VILLEMOT":
                smodel.SOLVER = SolverMethod.Villemot
                smodel.isLinear = True
            elif Solver.upper() == "ANDERSONMOORE":
                smodel.SOLVER = SolverMethod.AndersonMoore
                smodel.isLinear = True
            elif Solver.upper() == "BINDERPESARAN":
                smodel.SOLVER = SolverMethod.BinderPesaran
                smodel.isLinear = True
            elif Solver.upper() == "BENES":
                smodel.SOLVER = SolverMethod.Benes
                smodel.isLinear = True
            elif Solver.upper() == "LBJ":
                smodel.SOLVER = SolverMethod.LBJ
                smodel.isLinear = False
            elif Solver.upper() == "LBJAX":
                smodel.SOLVER = SolverMethod.LBJax
                smodel.isLinear = False
                smodel.jaxdiff = True
            elif Solver.upper() == "ABLR":
                smodel.SOLVER = SolverMethod.ABLR
                smodel.isLinear = False
            elif Solver.upper() == "BLOCK":
                smodel.SOLVER = SolverMethod.Block
                smodel.isLinear = False
            elif Solver.upper() == "DM":
                smodel.SOLVER = SolverMethod.DM
                smodel.isLinear = False
            else:
                smodel.SOLVER = SolverMethod.LBJ
                smodel.isLinear = False
        else:
            smodel.SOLVER = SolverMethod.LBJ
            smodel.isLinear = False
            
        model = Model(interface=smodel,anticipate=anticipate,infos=infos)
        model.estimate = estimate
        
        # Set model parameters
        model = setParameters(model,order,Solver,Filter,Smoother,Prior,InitCondition)
        simulationRange(model)
        
        # Serialize model into file
        from utils.interface import saveModel
        saveModel(model_path,model)
    
    print()
    print('='*max(15,len(model.name)))
    print(model.name)
    print('='*max(15,len(model.name)))
    print()
    print(model)
    
    if model_info:
        from misc.text2latex import saveDocument
        saveDocument(model)
        
    if graph_info:
        print(model)
        #   Create model equations graph
        img_file_name = "Equations_Graph.png"
        from info.graphs import createGraph, createClusters
        createGraph(model,img_file_name)
        cluster_file_name = "Minimum_Spanning_Tree.png"
        createClusters(model,cluster_file_name)
    
    return model


def findEigenValues(fname=None,model=None,steady_state=None):
    """
    Find eigen values of system of equations at steady state.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param model: Model object.
        :type model: Model.
        :param Output: Boolean variable.  If set saves steady state solution to excel file and sqlite database.
        :type Output: bool.
        
        :returns: Steady state solution.
    """ 
    path = os.path.dirname(os.path.abspath(__file__))
    
    # Import model
    if model is None:
        if fname is None:
            fname = os.path.abspath(os.path.join(path,'../models/template.yaml'))
        model = importModel(fname)
     
    eig = util.find_eigen_values(model,steady_state)
    ev  = np.sort(abs(eig))
    if len(ev) > 0 and model.count == 1:
        nev = sum(abs(e) >= 1+1.e-10 for e in ev)
        print()
        if nev > 0:
            cprint(f"{nev} eigen values of model transition matrix are larger than one.","green")
            # from model.settings import PriorAssumption
            # if model.PRIOR == PriorAssumption.Equilibrium:
            #     cprint("Equilibrium prior assumption is not applicable for non-stationary models... Please use different assumption!.","red")
        else:
            cprint("Stationary model","green")
        # print()
        # cprint('Number of unstable roots: %d' % nev,'blue')
        # cprint('Eigen Values:','blue')
        # cprint(['{:.2e}'.format(abs(x)) for x in ev],'blue')

    return  eig


def findSteadyStateSolution(fname=None,model=None,Output=False,excel_dir=None,debug=False):
    """
    Find steady state solution.
    
    This function uses variables starting values as initial condition for an iterative algorithm.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param model: Model object.
        :type model: Model.
        :param Output: Boolean variable.  If set saves steady state solution to excel file and sqlite database.
        :type Output: bool.
        :param excel_dir: Pth to directory where excel steady state file will be saved.
        :type excel_dir: str.
        
        :returns: Steady state solution.
    """ 
    path = os.path.dirname(os.path.abspath(__file__))
    
    # Import model
    if model is None:
        if fname is None:
            fname = os.path.abspath(os.path.join(path,'../models/template.yaml'))
        model = importModel(fname)
     
    # Get parameter names
    par_names = model.symbols['parameters']
    # Get parameter values
    param_values = model.calibration['parameters']
    if np.ndim(param_values) == 2:
        param_values = param_values[:,0]
    # Get variables
    var = model.symbols['variables']
    n = len(var)
    if 'new_variables' in model.symbols:
        new_vars = model.symbols['new_variables']
    else:
        new_vars = []
    
    # Find steady-state
    if model.isLinear:
        ss, ss_growth = linear_solver.find_steady_state(model)
    else:
        ss, ss_growth = nonlinear_solver.find_steady_state(model)
    
    # model.steady_state  = dict(zip(var,ss))
    # model.steady_growth = dict(zip(var,ss_growth))
        
    if debug and model.count == 1:
        cprint('Steady-State Solution:','blue')
        sv = [n+"="+str(round(v,4)) for n,v in zip(var,ss) if  n not in new_vars][:n]
        sv.sort()
        cprint(sv,'blue')
        print()
    
    if Output :
        # Save results in excel file
        if not excel_dir is None and os.path.exists(excel_dir):
            xl_dir = excel_dir
        if fname is None:
            ex_name = os.path.abspath(os.path.join(xl_dir,'Steady_State.csv'))
        else:
            name, ext = os.path.splitext(fname)
            ex_name = os.path.abspath(os.path.join(xl_dir,os.path.basename(name) + '_Steady_State.csv'))
        with open(ex_name, 'w') as f:
            f.writelines(','.join(var) + ',' + ','.join(par_names) + '\n')
            f.writelines(','.join(str(x) for x in ss) + ',' + ','.join(str(x) for x in param_values) +'\n')
              
        par_names = ["PARAM_" + s for s in par_names]
        columns=np.append(par_names,var)                  
        if len(columns) < 1000:
            # Save steady-state solution to Python database
            data_dir = os.path.abspath(os.path.join(path,'../data'))
            dbfilename = os.path.abspath(os.path.join(data_dir,'db/ss.sqlite'))
            sqlitedir = os.path.dirname(dbfilename)
            if not os.path.isdir(sqlitedir):
                os.makedirs(sqlitedir)
            # SQLITE is case insensitive.  Therefore, append parameter names with PAR_ prefix.
            data = np.append(param_values,ss)
            saveToDatabase(dbfilename=dbfilename,data=data,columns=columns)
        
    return  ss, ss_growth


def findSteadyStateSolutions(fname=None,model=None,number_of_steps=10,par_range={},Plot=False,Output=False):
    """
    Find steady state solution for a range of parameters.
    
    The parameters range is defined in the "options" sections of YAML model file.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param model: Model object.
        :type model: Model.
        :param number_of_steps: Number of steps.
        :type number_of_steps: int.
        :param par_range: Parameters range.
        :type par_range: dictionary.
        :param Plot: Boolean variable.  If set to True shows graphs.
        :type Plot: bool.
        :param Output: Boolean variable.  If set to True saves graphs.
        :type Output: bool.
        
        :returns: List of steady state arrays for given parameter range.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    
    if model is None:
        if fname is None:
            fname = os.path.abspath(os.path.join(path,'../models/template.yaml'))
        # Import model
        model = importModel(fname)
       
    # Get variables   
    s = model.symbols['variables']
    # Get parameters
    par_names = model.symbols['parameters']
    
    arr_ss = []; par_ss = []; data = []
    par_values = model.calibration['parameters']
    for j,par in enumerate(par_names):
        arr = []; arr2 = []
        if par in par_range.keys():
            rng = par_range[par]
        elif par in model.options:
            rng = model.options[par]
        else:
            rng = None
        if not rng is None:
            for i in range(11):
                new_par = rng[0] + (rng[1]-rng[0])/number_of_steps*i
                # Set new calibration parameter's value
                model.set_calibration(par,new_par)
                if model.isLinear:
                    ss, ss_growth = linear_solver.find_steady_state(model)
                else:
                    ss, ss_growth  = nonlinear_solver.find_steady_state(model)
                arr.append(np.append(new_par,ss))
                new_par_values = par_values.copy()
                new_par_values[j] = new_par
                arr2.append(np.append(new_par_values,ss))
            arr_ss.append(arr)
            par_ss.append(par)
            data.append(arr2)
            # Restore model calibration parameter's value
            model.set_calibration(par,par_values[j])
            
        model.calibration['parameters'] = par_values
            
    if len(data) == 0:
        return
        
    if Plot:
        if len(arr_ss) > 0:
            cprint("Steady-State Solutions","green")
            plotSteadyState(path_to_dir=figures_dir,s=s,arr_ss=arr_ss,par_ss=par_ss)
                
    if Output:

        columns = par_names + s
        
        # Save results in excel file
        if fname is None:
            ex_name = os.path.abspath(os.path.join(path,'../data/Steady_State.csv'))
        else:
            name, ext = os.path.splitext(fname)
            ex_name = os.path.abspath(os.path.join(path,'../data/' + os.path.basename(name) + '_Steady_State.csv'))
        with open(ex_name, 'w') as f:
            f.writelines( ','.join(columns) + '\n')
            for p in range(len(data)):
                yIter = np.array(data[p])
                dim1,dim2 = yIter.shape
                for it in range(dim1):
                    y = yIter[it,:]
                    f.writelines(",".join(map(str, y)) + ",".join(map(str, par_values)) + '\n')

        # Save steady-state solution in Python database
        if len(columns) < 2000:
            dbfilename = os.path.abspath(os.path.join(path,'../data/db/ss_param_range.sqlite'))
            data = np.array(data)
            # Save results in Python database
            saveToDatabase(dbfilename=dbfilename,data=data,columns=columns)
            
    return arr_ss,par_ss,par_names 


def getImpulseResponseFunctions(fname,Plot=False,Output=False):
    """
    Get impulse response functions (IRF).
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :type Plot: bool.
        :param Output: Boolean variable.  If set then saves graphs.
        :type Output: bool.
        
        :returns: IRF for given parameters.
    """
    rng_date,y  = run(fname=fname,irf=True,Plot=Plot,Output=Output)
    return rng_date,y


def optimize(fpath=None,Output=False,plot_variables=None,model_info=False):
    """
    Call main driver program.
    
    Runs model optimization.
    
    Parameters:
        :param fpath: Path to model file.
        :type fpath: str.        
        :param Output: If set outputs simulation results to excel file and Python sqlite database.
        :type Output: bool.
        :param output_dir: Path to output directory.
        :type output_dir: str.
        :param plot_variables: Names of variables to plot.
        :type plot_variables: list.
        :param model_info: If True creates a pdf/latex model file.
        :type model_info: bool.
        :returns: Optimization results.
    """
    from numeric.optimization.optimize import run
    run(fpath=fpath,Output=Output,plot_variables=plot_variables,model_info=model_info)
  

def kalman_filter(fname=None,model=None,y0=None,T=-1,output_variables=None,fout=None,Plot=False,
                  decomp_variables=None,anticipate=None,Output=False,output_dir=None,hist=None,
                  meas=None,InitCondition=None,Prior=None, calibration_file_path=None,
                  Solver=None,Filter=None,Smoother=None):
    """
    Runs Kalman filtering.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param model: Model object.
        :type model: Model.
        :param y0: Starting (or guessed) values of the solution.
        :type y0: numpy array.
        :param T: Number of time periods.
        :type T: int.
        :param output_variables: output variables.
        :type output_variables: list.  
        :param Plot: Boolean variable.If this flag is raised then plots graphs.
        :type Plot: bool.
        :param decomp_variables: List of decomposition variables.
        :type decomp_variables: list.
        :param anticipate: It True then future shocks are anticipated.
        :type anticipate: bool.
        :param Output: If set outputs simulation results to excel file and Python sqlite database.
        :type Output: bool.
        :param output_dir: Path to output directory.
        :type output_dir: str.
        :param hist: Path to history excel file. It contains starting and terminal values of endogenous variables.
        :type hist: str.
        :param boundary_conditions: Path to the boundary conditions excel file.  This file contains initial and terminal conditions.
        :type boundary_conditions: str.
        :param exog: List of exogenous variables.
        :type exog: list.
        :param meas: Path to a file with measurement data.
        :type meas: str. 
        :param InitCondition: (StartingValues,SteadyState) 
           If StartingValues then use starting values
           If SteadyState then use steady state as starting values
        :type InitialCondition: str.
        :param Prior: (StartingValues,DiffusePrior,Equilibrium) If Rosenberg then applies algorithm to find initial values of state variables.
           If StartingValues then use starting values for covariance matrices
           If SteadyState then use steady state for covariance matrices
           If Equilibrium then solve Lyapunov equation to find steady state covariance matrices.
        :type Prior: str.
        :param estimate_Posterior: If this flag is raised, then calibrate model parameters and apply Kalman Filter with the new calibrated parameters.
        :type estimate_Posterior: bool. 
        :param fout: Path to output excel file.
        :type fout: str.
        :param calibration_file_path: Path to a calibration file or files.
        :type calibration_file_path: str or list.
        :param Solver: Name of numerical method to solve the system of equations.
        :type Solver: str.
        :param Filter: Name of Kalman filter algorithm.
        :type Filter: str.
        :param Smoother: Name of Kalman filter smoother algorithm.
        :type Smoother: str.
        :param calibration_file_path: If True the transition equations, variables, shocks are combined with the measurement ones.
        :type calibration_file_path: bool.
        
        :returns: Kalman filter results.
    """
    
    out = run(fname=fname,model=model,y0=y0,T=T,output_variables=output_variables,Output=Output,
              output_dir=output_dir,fout=fout,Plot=Plot,decomp_variables=decomp_variables,
              anticipate=anticipate,hist=hist,meas=meas,InitCondition=InitCondition,
              Prior=Prior,calibration_file_path=calibration_file_path,Solver=Solver,
              Filter=Filter,Smoother=Smoother,runKalmanFilter=True)
    
    return out
    
    
def estimate(fname=None,model=None,y0=None,output_variables=None,fout=None,Plot=False,
             Output=False,output_dir=None,meas=None,Prior=None,runKalmanFilter=True):
    """
    Estimates model parameters.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param model: Model object.
        :type model: Model.
        :param y0: Starting (or guessed) values of the solution.
        :type y0: numpy array.
        :param fout: Path to output excel file.
        :type fout: str.
        :param Plot: Boolean variable.If this flag is raised then plots graphs.
        :type Plot: bool.
        :param decomp_variables: List of decomposition variables.
        :param Output: If set outputs simulation results to excel file and Python sqlite database.
        :type Output: bool. 
        :param output_dir: Path to output directory.
        :type output_dir: str.
        :param meas: Path to a file with measurement data.
        :type meas: str. 
        :type Prior: str.
        :param runKalmanFilter: If True runs Kalman filter after estimation, otherwise - runs forecast,
        :type runKalmanFilter: bool.
        
        
        :returns: Model estimation.
    """    
    out = run(fname=fname,model=model,y0=y0,fout=fout,output_variables=output_variables,
              Output=Output,output_dir=output_dir,meas=meas,Prior=Prior,
              Plot=Plot,runKalmanFilter=runKalmanFilter)
    
    return out


def run(fname=None,model=None,y0=None,order=1,T=-1,Tmax=1.e6,irf=False,prefix=None,output_variables=None,Plot=False,decomp_variables=None,
        anticipate=None,PlotSurface=False,Plot3D=False,Output=False,output_dir=None,hist=None,boundary_conditions=None,exog=None,meas=None,
        InitCondition=None,Prior=None,fout=None,MULT=1,shocks_file_path=None,steady_state_file_path=None,
        calibration_file_path=None,Solver=None,Filter=None,Smoother=None,header=None,
        opt_ss_continue=True,graph_info=False,use_cache=False,model_info=False,Sparse=False,runKalmanFilter=False):
    """
    Main driver program.
    
    It runs simulations, finds steady state solution, plots graphs, and saves results to excel files and Python sqlite databases.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param model: Model object.
        :type model: Model.
        :param y0: Starting (or guessed) values of the solution.
        :type y0: numpy array.
        :param order: Approximation order of non-linear system of equations.
        :type order: int.
        :param T: Number of time periods.
        :type T: int.
        :param Tmax: Maximum number of periods to display in graphs.
        :type Tmax: int.
        :param irf: IRF variable.
        :type irf: int.
        :param prefix: Prefix name.  It is used to plot and output variables which name starts with prefix.
        :type prefix: str.
        :param output_variables: output variables.
        :type output_variables: list.  
        :param Plot: Boolean variable.If this flag is raised then plots graphs.
        :type Plot: bool.
        :param decomp_variables: List of decomposition variables.
        :type decomp_variables: list.
        :param anticipate: It True then future shocks are anticipated.
        :type anticipate: bool.
        :param PlotSurface: Boolean variable. If this flag is raised then plots 2D graphs.
        :type PlotSurface: bool. 
        :param Plot3D: Boolean variable. If this flag is raised then plots 3D graphs.
        :type Plot3D: bool.
        :param Output: If set outputs simulation results to excel file and Python sqlite database.
        :type Output: bool.    
        :param output_dir: Path to output directory.
        :type output_dir: str.            
        :param hist: Path to history excel file. It contains starting and terminal values of endogenous variables.
        :type hist: str.
        :param boundary_conditions: Path to the boundary conditions excel file.  This file contains initial and terminal conditions.
        :type boundary_conditions: str.
        :param exog: List of exogenous variables.
        :type exog: list.
        :param meas: Path to a file with measurement data.
        :type meas: str. 
        :param InitCondition: (StartingValues,SteadyState) 
           If StartingValues is set, then use starting values
           If SteadyState is set, then use steady state as starting values
        :param Prior: (StartingValues,DiffusePrior,Equilibrium) If Rosenberg then applies algorithm to find initial values of state variables.
           If StartingValues then use starting values for covariance matrices
           If SteadyState then use steady state for covariance matrices
           If Equilibrium then solve Lyapunov equation to find steady state covariance matrices.
        :type Prior: str.
        :param fout: Path to output excel file.
        :type fout: str.
        :param MULT: Multiplier defining terminal time.  If set greater than one than 
                     solution will be computed for this extended time range interval.
        :type MULT: float.   
        :param shocks_file_path: Path to shock file.
        :type shocks_file_path: str.
        :param steady_state_file_path: Path to steady-state file.
        :type steady_state_file_path: str.
        :param calibration_file_path: Path to a calibration file or files.
        :type calibration_file_path: str or list.
        :param Solver: Name of numerical method to solve the system of equations.
        :type Solver: str.
        :param Filter: Name of Kalman filter algorithm.
        :type Filter: str.
        :param Smoother: Name of Kalman filter smoother algorithm.
        :type Smoother: str.
        :param header: Graph header.
        :type header: str.
        :param opt_ss_continue: If set and steady state solution is invalid, then find steady state and continue simulations.
        :type opt_ss_continue: bool.
        :param graph_info: If True shows model equations graph.
        :type graph_info: bool.
        :param calibration_file_path: If True the transition equations, variables, shocks are combined with the measurement ones.
        :type calibration_file_path: bool.
        :param use_cache: If True reads previously saved model from a file of model dump.
        :type use_cache: bool.
        :param model_info: If True creates a pdf/latex model file.
        :type model_info: bool.
        :param Sparse: If True use sparse matrices algebra.
        :type Sparse: bool.
        :param runKalmanFilter: If True runs Kalman Filter.
        :type runKalmanFilter: bool.
        
        :returns: Simulation results.
    """
    global figures_dir, xl_dir
    from model.settings import InitialCondition,BoundaryConditions,BoundaryCondition
    
    if not output_dir is None:
        figures_dir = os.path.abspath(os.path.abspath(os.path.join(output_dir,'graphs')))
        xl_dir = os.path.abspath(os.path.abspath(os.path.join(output_dir,'data')))
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        if not os.path.exists(xl_dir):
            os.makedirs(xl_dir)
        

    if model is None:
        if fname is None:
            fname = os.path.abspath(os.path.join(os.getcwd(),'../models/template.yaml'))
        # Import model
        model = importModel(fname,hist=hist,boundary_conditions=boundary_conditions,order=order,exog=exog,
                            InitCondition=InitCondition,Prior=Prior,estimate=estimate,
                            Solver=Solver,Filter=Filter,Smoother=Smoother,
                            shocks_file_path=shocks_file_path,
                            steady_state_file_path=steady_state_file_path,
                            calibration_file_path=calibration_file_path,
                            measurement_file_path=meas,model_info=model_info,
                            graph_info=graph_info, anticipate=anticipate,
                            use_cache=use_cache,bSparse=Sparse)
        
    else:
        # Set model parameters
        model = setParameters(model,order,Solver,Filter,Smoother,Prior,InitCondition)
        model.anticipate = anticipate
            
    model.count += 1
    meas_df = None
    
    # Save model info into pdf file
    if model_info:
        from misc.text2latex import saveDocument
        saveDocument(model)
        
    # Create model equations graph
    if graph_info:
        from info.graphs import createGraph
        from info.graphs import createClusters
        #from info.graphs import getInfo
        #getInfo(model)
        img_file_name = "Equations_Graph.png"
        createGraph(model,img_file_name)
        img_file_name = "Minimum_Spanning_Tree.png"
        createClusters(model,img_file_name)
    
    epsilonhat,etahat = None,None
#    print('Variables:')
    variables = model.symbols['variables']
    n = len(variables)


#    print('shocks:')
    shock_names = model.symbols['shocks']
    shock_values = model.calibration['shocks']
#    print(shock_names)
    
#    print('Parameters:')
    par_names = model.symbols['parameters']
    par_values = model.calibration['parameters']
#    pv = [n+"="+str(v) for n,v in zip(par_names,par_values)]
#    pv.sort()       
#    print(pv)
#    print()

    # Build dictionary of shock leads and lags
    from utils.equations import getMaxLeadsLags
    # Gets maximum of leads and minimum of lags of equations shock variables
    max_lead_shock,min_lag_shock,_,_,shock_lead,shock_lag = \
            getMaxLeadsLags(eqs=model.symbolic.equations,variables=shock_names)
    if max_lead_shock > min_lag_shock:
        shockLeadsLags = {**shock_lag,**shock_lead}
        LeadsLags = {x:[0] for x in range(len(shock_names))}
        for x in shockLeadsLags:
            ind = x.index("(")
            key = x[:ind]
            key = shock_names.index(key)
            leadlag = x[1+ind:-1]
            leadlag = int(leadlag)
            LeadsLags[key] = sorted(LeadsLags[key] + [leadlag])
        util.shockLeadsLags = LeadsLags    

    
    if 'Npaths' in model.options:
        Npaths = model.options['Npaths']
    else:
        Npaths = 1
    
    if T < 1:
        if 'T' in model.options:
            T = model.options['T']
        else:
            T = 101
        
    if 'frequency' in model.options:
        freq = model.options['frequency']
    else:
        freq = 0
    skip = 0
    
    start,end,rng,rng_date,start_filter,end_filter,filter_rng,Time,T = simulationRange(model,freq=freq,T=T)   
    periods = getPeriods(model,T,rng)
         
    util.PERMANENT_SHOCK = 'shock_values' in model.options and not 'periods' in model.options
    if boundary_conditions is None:
        BoundaryCondition.Condition = BoundaryConditions.ZeroDerivativeBoundaryCondition
    else:
        BoundaryCondition.Condition = BoundaryConditions.FixedBoundaryCondition
    
    if meas is None:
        meas = model.symbolic.measurement_file_path
        
    if bool(meas) and 'measurement_variables' in model.symbols:
        measurement_variables = model.symbols['measurement_variables']
        measurement_shocks = model.symbols['measurement_shocks'] if 'measurement_shocks' in model.symbols else None
        measurement_equations = model.symbolic.measurement_equations
        Q,H = util.getCovarianceMatricies(measurement_variables,measurement_shocks,measurement_equations,shock_names,model)
        ext = meas.split(".")[-1].lower()
        if ext == 'xlsx' or ext == 'xls':
            meas_df = pd.read_excel(meas,header=0,index_col=0,parse_dates=True)
        else:
            meas_df = pd.read_csv(filepath_or_buffer=meas,sep=',',header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
       
        if 'frequency' in model.data_sources:
            obs_freq = model.data_sources['frequency']
        else:
            obs_freq = 'AS'
        meas_df  = fetchData(model.data_sources,meas_df=meas_df,fpath=meas,freq=obs_freq)
        if not meas_df is None:    
            if not filter_rng is None:
                fstart = rng[0]
                fend = filter_rng[-1]
                mask = (meas_df.index >= fstart.to_timestamp()) & (meas_df.index <= fend.to_timestamp())
                meas_df = meas_df.loc[mask]
                mask = rng_date < fstart.to_timestamp()
                if np.any(mask): # Insert nan rows to dataframe for periods less than start of filter range 
                    tmp_rng = rng_date[mask]
                    skip = len(tmp_rng)
                    tmp_df = pd.DataFrame(index=tmp_rng,columns=meas_df.columns)
                    meas_df = pd.concat((tmp_df,meas_df))
                rng_meas = meas_df.index
    
            df_columns = list(meas_df.columns)
            common = [x for x in measurement_variables if x in df_columns]
            if len(common) > 0:
                meas = meas_df[measurement_variables]
                meas = meas.values.astype('float64')
            else:
                arr = []
                for v in measurement_variables:
                    if "_" in v:
                        ind = v.index("_")
                        v = v[1+ind:]
                    if v in df_columns:
                        arr.append(meas_df[v].values)
                    else:
                        arr.append([np.nan]*len(meas_df))
                meas = np.array(arr).T
                        
            missing_obs = np.isnan(meas)
            ind_non_missing = []
            for obs in missing_obs:
                ind_non_missing.append([i for i,x in enumerate(obs) if not x])
    else:
        meas = None  
        rng_meas = None
        measurement_variables = None
        Q = None; H = None
  
    if not model.steady_state is None and model.count == 1:
        err = util.checkSteadyState(model,variables)
        err2 = np.nansum(err*err)
        if err2 > 1.e-10:
            if opt_ss_continue:
                model.steady_state = None
                cprint("Incorrect steady state: the sum of the equations squared residuals is, {0:.1e}.\n".format(err2),'red')
            else:
                from gui.dialog import showDialog 
                msg = "  The sum of the equations squared residuals is: {0:.1e}. \nDo you want to continue?".format(err2)
                yes_no = showDialog(msg=msg,title="Incorrect steady state")
                if yes_no == "yes":
                    model.steady_state = None
                elif yes_no == "no":   
                    y = None
                    cprint("Incorrect steady state: the sum of the squared equations residuals is, err={0:.1e}.\nExitting ...".format(err2),'red')
                    return rng_date,y,variables,rng,periods,model
                
    ss = None; ev = []
    if not bool(model.symbolic.bellman):
        if model.INITIAL_CONDITION == InitialCondition.SteadyState:
            if model.steady_state is None:
                #print("Finding a steady-state solution. It uses starting variable values.")
                #print()
                ss, growth = findSteadyStateSolution(fname=fname,model=model,Output=False)
                model.steady_state = ss
                model.steady_growth = growth
            elif isinstance(model.steady_state,dict):
                ss = []
                for i in range(n):
                    v = variables[i]
                    if v in model.steady_state:
                        ss.append(model.steady_state[v])  
                    else:
                        cprint("Steady state variable {} is missing. Please correct steady-state variables. Exitting...".format(v),'red')
                        return
            else:
                ss = model.steady_state
            
            # Find eigen values
            ev = findEigenValues(model=model,steady_state=ss)
            model.ev = ev
            
        # If stationary solution is given then use it as a starting value
        if model.INITIAL_CONDITION == InitialCondition.SteadyState and not ss is None:
            y0 = ss
        elif model.INITIAL_CONDITION == InitialCondition.History and not hist is None:
            # Set starting values
            model.setStartingValues(hist=hist)
            
    # Assign starting values
    var = model.calibration['variables']
    if y0 is None:
        y0 = np.copy(var)
            
    ###     Check if model has a stationary solution.
    # # Model is stationary if all eigen values are less than one
    # bStationary1 = np.all(abs(ev)<1.0)
    # # Or if the norm of functions at steady state is small...
    # func = model.functions["f_static"](ss,par_values,shock_values)
    # err = np.linalg.norm(func)
    # bStationary2 = err < 1.e-6
    # bStationary = bStationary1 and bStationary2
                
    if model.GENERATE_CPP_CODE:
        from utils.util import create_config_file
        create_config_file(T,variables,y0,shock_names,shock_values,par_names,par_values,model.options)
                
    if model.isLinear is None:
        model.isLinear = False
    
    lrx_filter = []; hp_filter = []
          
    if bool(model.symbolic.bellman):
        from numeric.dp import bellman
        iterations,y,max_f,elapsed = bellman.simulate(model=model)
        rng_date = np.arange(len(y))
        
    elif runKalmanFilter and not (Q is None or H is None or meas is None): 
        # Apply Kalman filter
        if model.isLinear:
            iterations,y,epsilonhat,etahat,max_f,elapsed = \
                linear_filter(model=model,T=T,periods=periods,y0=y0,Qm=Q,Hm=H,
                              obs=meas,MULT=MULT,skip=skip,missing_obs=missing_obs,
                              ind_non_missing=ind_non_missing)
        else:
            iterations,y,epsilonhat,etahat,max_f,elapsed = \
                nonlinear_filter(model=model,T=T,periods=periods,y0=y0,Qm=Q,Hm=H,
                                 obs=meas,MULT=MULT,skip=skip,missing_obs=missing_obs,
                                 ind_non_missing=ind_non_missing)
    
        # Apply LRX and HP filters
        from numeric.filters.filters import LRXfilter, HPfilter
        
        nm = len(measurement_variables)
        for i in range(nm):
            meas_vars = [x.lower() for x in measurement_variables]
            mv = meas_vars[i]
            if "_meas" in mv:
                var_name = measurement_variables[i][:-5]
            elif "obs_" in mv:
                var_name = measurement_variables[i][4:]
            else:
                var_name = None
            if var_name in variables:
                ind = variables.index(var_name)
                pr = ss[ind] if not ss is None else None
                y_meas = meas[skip:,i]
                y_meas = y_meas[~np.isnan(y_meas)]
                if len(y_meas) > 0:
                    v1 = LRXfilter(y=y_meas,lmbd=100,w1=1,w2=0.1,w3=0,dprior=0.1,prior=pr)[0]
                    v2 = HPfilter(data=y_meas,lmbd=100)[0]
                    for j in range(skip):
                        v1 = np.insert(v1,0,np.nan)
                        v2 = np.insert(v2,0,np.nan)
                else:
                    v1 = np.zeros(len(meas))
                    v2 = np.zeros(len(meas))
                lrx_filter.append(v1)
                hp_filter.append(v2)
                
        lrx_filter = np.array(lrx_filter,dtype=object)
        hp_filter = np.array(hp_filter,dtype=object)
        
    else:            
        # Solve equations by iterations by applying Newton's method.
        if model.isLinear:
            iterations,y,yIter,max_f,elapsed = \
                linear_solver.simulate(model=model,T=T,periods=periods,
                                       steady_state=ss,y0=y0,Npaths=Npaths)
        else:
            iterations,y,yIter,max_f,elapsed = \
                nonlinear_solver.simulate(model=model,T=T,periods=periods,
                                          steady_state=ss,y0=y0,Npaths=Npaths,MULT=MULT)
        
        if irf:
            if model.steady_state is None:
                # Set shocks to zero
                if 'shock_values' in model.options:
                    shock_values = model.options.get('shock_values')
                    model.options['shock_values'] = shock_values*0
                    
                # Solve equations
                if model.isLinear:     
                    iterations_ss,y_ss,yIter_ss,max_f_ss,elapsed_ss = \
                        linear_solver.simulate(model=model,T=T,periods=periods,
                                               steady_state=ss,y0=y0,Npaths=Npaths)
                else:
                    iterations_ss,y_ss,yIter_ss,max_f_ss,elapsed_ss = \
                        nonlinear_solver.simulate(model=model,T=T,periods=periods,
                                                  y0=y0,Npaths=Npaths,MULT=MULT)
                
                if 'shock_values' in model.options:
                    model.options['shock_values'] = shock_values
                
            else:
                if isinstance(model.steady_state,dict):
                    steady_state = []
                    for k in variables:
                        steady_state.append(model.steady_state[k])
                else:
                    steady_state = model.steady_state
                        
                y_ss = np.tile(steady_state,T).reshape(T,-1)
                
            # Compute deviations from the steady state
            arr = []
            for i in range(len(y)):
                arr.append(y[i]-y_ss[i])
            y = arr
            
    if Output:    
        print("\nElapsed time: %.2f (seconds)" % elapsed)
        if not model.isLinear:
            print(f"Number of iterations: {iterations}; error: {max_f:.1e}")
        print()
    
        if n < 10:
            # Print results
            output_data = np.array(y[-1])
            nrow = len(output_data)
            if not output_variables is None:
                indices = [i for i, e in enumerate(variables)]
                output_data =  output_data[:,indices]
            column_names = ["Time"]+variables
            if len(shock_values) > 1:
                nd = np.ndim(shock_values)
                if nd == 2:
                    column_names += shock_names
                    nrow = min(len(output_data),len(shock_values))
                    output_data = np.column_stack((output_data[:nrow],shock_values[:nrow]))
#                elif nd == 1:
#                    column_names += shock_names
#                    nrow = len(output_data)
#                    shock_values = np.repeat((shock_values,shock_values))
#                    output_data = np.column_stack((output_data,shock_values))
            if len(lrx_filter) > 0:
                column_names += ["LRX_"+x[:-5] for x,y in zip(measurement_variables,meas_vars) if "_meas" in y] + \
                                ["LRX_"+x[4:] for x,y in zip(measurement_variables,meas_vars) if "obs_" in y]
                filtered = np.empty((nrow,nm))
                filtered[:] = np.nan
                filtered[0:len(lrx_filter.T[:nrow])] = lrx_filter.T[:nrow]
                output_data = np.column_stack((output_data[:nrow],filtered))
            if len(hp_filter) > 0:
                column_names += ["HP_"+x[:-5] for x,y in zip(measurement_variables,meas_vars)if "_meas" in y] + \
                                ["HP_"+x[4:] for x,y in zip(measurement_variables,meas_vars) if "obs_" in y]
                filtered = np.empty((nrow,nm))
                filtered[:] = np.nan
                filtered[0:len(lrx_filter.T[:nrow])] = hp_filter.T[:nrow]
                output_data = np.column_stack((output_data[:nrow],filtered))
            pTable = PrettyTable(column_names)
            for i in range(nrow-2):
                if rng_date is None:
                    date = rng[i]
                else:    
                    if isinstance(rng_date[i],dt.date):
                        date = rng_date[i].strftime("%m-%d-%Y")
                    else:
                        date = rng_date[i]
                row = [date] + list(output_data[i,:])
                pTable.add_row(row)
            pTable.float_format = "4.2" 
            print(pTable)
            print()

        # Save results in csv file
        if fout is None or not os.path.exists(fout):
            fout = os.path.abspath(os.path.join(xl_dir,'Data.csv'))
        else:
            fout = os.path.abspath(os.path.join(xl_dir,os.path.basename(fout)))
            
        saveToExcel(fname=fout,data=y,variable_names=variables,par_values=par_values,
                    par_names=par_names,output_variables=output_variables,
                    rng=rng_date,Npaths=Npaths)
        
        # if n < 10000:  #Python sqlite database has a limited number of columns
        #     # Save solution to Python database
        #     data_dir = os.path.dirname(os.path.abspath(fout))
        #     if shocks_file_path is None:
        #         dbfilename = os.path.abspath(os.path.join(data_dir,'db/sol.sqlite'))
        #     else:
        #         name,ext = os.path.splitext(os.path.basename(shocks_file_path))
        #         dbfilename = os.path.abspath(os.path.join(data_dir,'db',name+'.sqlite'))
        #     sqlitedir = os.path.dirname(dbfilename)
        #     if not os.path.isdir(sqlitedir):
        #         os.makedirs(sqlitedir)
        #     if rng_date is None:
        #         saveToDatabase(dbfilename=dbfilename,data=np.array(yy),columns=s)
        #     else:
        #         saveToDatabase(dbfilename=dbfilename,data=np.array(yy),columns=np.append(["PATH","DATE"],s),dates=rng_date)

    
    if graph_info:
        plotImage(path_to_dir=figures_dir,fname=img_file_name)
            
    if Plot:

        # Plot eigen values
        if len(ev) > 0 and max(abs(ev)) < 10: 
            plotEigenValues(path_to_dir=figures_dir,ev=ev)
        print("Plotting Endogenous Variables")
        var_labels = model.symbols.get("variables_labels",{})
        plot(path_to_dir=figures_dir,data=y,variable_names=variables,
             meas_values=meas_df,meas_variables=measurement_variables,
             lrx_filter=lrx_filter,hp_filter=hp_filter,Tmax=Tmax,
             output_variables=output_variables,var_labels=var_labels,
             prefix=prefix,rng=rng,rng_meas=rng_meas,irf=irf,header=header,
             Npaths=Npaths,save=Output) # ,steady_state=ss

        if not decomp_variables is None:
            print("Plotting Decomposition of Endogenous Variables")
            plotDecomposition(path_to_dir=figures_dir,model=model,y=np.squeeze(y[-1]),
                              variables_names=variables,decomp_variables=decomp_variables,periods=periods,
                              isKF=not meas is None,header=header,rng=rng,
                              save=Output)

    if runKalmanFilter:
        return rng_date,y,epsilonhat,etahat
    else:
        return rng_date,y
 
