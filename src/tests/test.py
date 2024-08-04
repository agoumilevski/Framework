"""
Program for testing various models.

Created on Tue Mar 13 15:58:11 2018
@author: A.Goumilevski
"""
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/..")
sys.path.append(working_dir)
os.chdir(working_dir)

if __name__ == '__main__':
    """
    The main test program.
    """
    from driver import run

    fname = 'models/TOY/JLMP98.yaml'   # Simple monetary policy examplez
    #fname = 'models/TOY/RBC.yaml'      # Simple RBC model
    #fname = 'models/TOY/RBC1.yaml'    # Simple RBC model with stochastic shocks
    fout = 'data/test.csv' # Results are saved in this file
    decomp = ['PDOT','RR','RS','Y'] # List of variables for which decomposition plots are produced
    output_variables = None #['pie','r','y','ystar'] #['PDOT','RR','RS','Y','PIE','LGDP','G','L_GDP','L_GDP_GAP']
    
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, '..', fname))
    
    # Function that runs simulations, model parameters estimation, MCMC sampling, etc...
    rng_date,yy = \
    run(fname=file_path,fout=fout,decomp_variables=decomp,
             output_variables=output_variables,
             Output=True,Plot=True,Solver="LBJ",
             graph_info=False,use_cache=False,Sparse=False)


    
