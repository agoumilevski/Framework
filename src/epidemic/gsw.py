#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:25:58 2021

@author: A.Goumilevski
"""
import os
#from utils.util import simulationRange

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(os.path.abspath(path + "../..")))

CALIBRATE = True

if __name__ == '__main__':
    """
    The main test program.
    """
    from driver import importModel, run as simulate

    fname = "COVID19/gsw_model.yaml" # GSW model"
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, '../models', fname))
    
    # Path to data
    meas = os.path.abspath(os.path.join(working_dir, '../data/COVID19/country_data.xlsx'))
    
    output_variables = ['y','kpf','lab','c','unempl','inve','pinf','r','labstar'] 
    decomp = ['y','lab','inve','pinf']
    
    # Create model object with perfect foresight solver
    model = importModel(fname=file_path,model_info=False)
    model.anticipate = True
    
    par_names  = model.symbols["parameters"]
    par_values = model.calibration["parameters"]
    par = dict(zip(par_names,par_values))
        
    model.options["periods"] = [1]
    model.options["shock_values"] = [90,0,0,0,0,0,0,0,0]

    fout = 'data/COVID19/US_Lockdown_1Q.csv' 
    simulate(model=model,meas=None,output_variables=output_variables,decomp_variables=decomp,fout=fout,Output=True,Plot=True,Tmax=9)
    
    # Two periods shocks
    model.options["periods"] = [1,2]
    model.options["shock_values"] = [[90,0,0,0,0,0,0,0,0],[90,0,0,0,0,0,0,0,0]]

    fout = 'data/COVID19/US_Lockdown.csv' 
    simulate(model=model,meas=None,output_variables=output_variables,decomp_variables=decomp,fout=fout,Output=True,Plot=True,Tmax=9)
   