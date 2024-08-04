import numpy as np

def get_compiled_function_and_jacobian(f_dynamic,params,y,shock,t=0,order=1,debug=False):
    """
    Returns function and Jacobian.
    
    Parameters:
        :param f_dynamic: Model object.
        :type f_dynamic: Model.
        :param params: array of parameters.
        :type params: numpy.array.
        :param y: Array of values of endogenous variables.
        :type y: numpy.array
        :param shock: The values of shocks.
        :type shock: numpy.array
        :param t: Time index.
        :type t: int
        :param order: Order of partial derivatives of the system of equations.
        :type order: int
        :returns: Function , Jacobian, Hessian, and third order derivatives matrices.
    """
    # Build the approximation point to take the derivatives
    yy = np.concatenate([y[2],y[1],y[0],shock])
    
    if debug:
        from preprocessor.f_dynamic import f_dynamic
        # m = dict(zip(model.symbols["variables"],y2))
        # print(m)
        # print()
        derivatives = f_dynamic(yy,params,order=order)
    else:
        derivatives = f_dynamic(yy,params,order=order)
        
    return derivatives
       

def correct(n,max_lead_shock,min_lag_shock,indexEndogVariables,derivative):
    """
    Set elements of not swapped array to zero.

    Parameters
    ----------
    n : int
        number of endogenous variables.
    max_lead_shock : int
        Minimum lag shock number.
    min_lag_shock : int
        Mimum lead shock number.
    indexEndogVariables : numpy ndarray
        Index Of EndogVariables.
    derivative : numpy ndarray
        partial derivatives array.

    Returns
    -------
    Corrected derivatives array.

    """                                  
    if max_lead_shock == 0 and min_lag_shock == 0:  
        derivative[np.array(indexEndogVariables)-n] = 0
        derivative[np.array(indexEndogVariables)+2*n] = 0
    elif max_lead_shock < 0 and min_lag_shock == 0:  
        derivative[np.array(indexEndogVariables)] = 0
    elif max_lead_shock == 0 and min_lag_shock > 0:  
        derivative[np.array(indexEndogVariables)+2*n] = 0
        
    return derivative
            
            
def flip(n,order,indEndog,indExog,max_lead_shock,min_lag_shock,derivatives):
    """
    Swap indices of endogenous and exogenous variables.

    Parameters:
        :param order: Order of partial derivatives of the system of equations.
        :type order: int
        :param endogVariables: list
        :type endogVariables: Endogenous variables to be flipped
        :param exogVariables: list
        :type exogVariables: Exogenous variables to be flipped
        :returns: Function, Jacobian, Hessian, and third order derivatives matrices.
    """
    #TODO: Need testing
    if indEndog is None or indExog is None:
        return derivatives
    
    elif not len(indEndog) == len(indExog):
        from misc.termcolor import cprint
        cprint(f"Number of flipped endogenous varaibles {len(indEndog)} and exogenous variables {len(indExog)} is different.","red")
        return derivatives
    
    if order == 0:
        return derivatives
    
    elif order == 1:
        tmp0 = derivatives[0]
        tmp1 = derivatives[1]
        tmp1[:,indExog],tmp1[:,indEndog] = tmp1[:,indEndog],tmp1[:,indExog]
        return tmp0,tmp1
    
    elif order == 2:
        tmp0 = derivatives[0]
        tmp1 = derivatives[1]
        tmp2 = derivatives[2]
        tmp1[:,indExog],tmp1[:,indEndog] = tmp1[:,indEndog],tmp1[:,indExog]
        tmp2[:,indExog,indExog],tmp2[:,indEndog,indEndog] = tmp2[:,indEndog,indEndog],tmp2[:,indExog,indExog]
        return tmp0,tmp1,tmp2
        
    elif order == 3:
        tmp0 = derivatives[0]
        tmp1 = derivatives[1]
        tmp2 = derivatives[2]
        tmp3 = derivatives[3]
        tmp1[:,indExog],tmp1[:,indEndog] = tmp1[:,indEndog],tmp1[:,indExog]
        tmp2[:,indExog,indExog],tmp2[:,indEndog,indEndog] = tmp2[:,indEndog,indEndog],tmp2[:,indExog,indExog]
        tmp3[:,indExog,indExog,indExog],tmp3[:,indEndog,indEndog,indEndog] = tmp3[:,indEndog,indEndog,indEndog],tmp3[:,indExog,indExog,indExog]        
        return tmp0,tmp1,tmp2,tmp3
    
    else:
        return derivatives
        
    
def get_function_and_jacobian(model,params=None,y=None,shock=None,t=0,order=1,bSparse=False,debug=False):
    """
    Returns function and jacobian.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param params: array of parameters.
        :type params: numpy.array.
        :param y: Array of values of endogenous variables.
        :type y: numpy.array
        :param shock: The values of shocks.
        :type shock: numpy.array
        :param t: Time index.
        :type t: int
        :param order: Order of partial derivatives of the system of equations.
        :type order: int
        :param bSparse: If this flag is raised then call f_sparse function; otherwise - f_dynamic
        :type bSparse: bool
        :param debug: If this flag is raised then compiles function from files; otherwise use the cache.
        :type debug: bool
        :returns: Function, Jacobian, Hessian, and third order derivatives matrices.
    
    .. note::
        When 'debug' flag is raised and different models are instantiated, there could 
        be inconsistency between the cached function 'f_func' or 'f_sparse' and the current 
        source file. Please use with caution only when model is not changed in the same run.
        
    """
    from numeric.solver.util import getParameters
    params = getParameters(parameters=params,model=model,t=t)
    
    #n = len(model.symbols["variables"])
    if shock is None: 
        n_shocks = len(model.symbols['shocks'])
        x = list(np.zeros(n_shocks))
        #x = list(model.calibration['shocks'])
        for i in range(model.min_lag_shock,model.max_lead_shock):
            x += x
    else:
        if np.ndim(shock) == 2:
            x = np.diag(shock)
        else:
            x = shock
    x = np.atleast_1d(x)
    
 
    # Build the approximation point to take the derivatives
    if y is None: 
        y1 = y2 = y3 = model.calibration['variables']
    else:
        y1 = y[2]; y2 = y[1]; y3 = y[0]
        
    yy = np.concatenate([y1,y2,y3,x])
 
    if model.autodiff or model.jaxdiff:
        f_func = model.functions["func"]
        bHasAttr  = hasattr(f_func,"py_func")
        if bHasAttr:
            f_0 = f_func.py_func(yy,params)
        else:
            f_0 = f_func(yy,params)
        if order == 0:
            derivatives = f_0       
        elif order == 1:
            f_jacob = model.functions["f_jacob"]
            if bHasAttr:
                f_1 = f_jacob.py_func(yy,params) 
            else:
                f_1 = f_jacob(yy,params) 
            derivatives = f_0, f_1
        elif order == 2:
            f_jacob = model.functions["f_jacob"]
            if bHasAttr:
                f_1 = f_jacob.py_func(yy,params) 
            else:
                f_1 = f_jacob(yy,params) 
            f_hessian = model.functions["f_hessian"]
            if bHasAttr:
                f_2 = f_hessian.py_func(yy,params)
            else:
                f_2 = f_hessian(yy,params)
            derivatives = f_0, f_1, f_2
        elif order == 3:
            f_jacob = model.functions["f_jacob"]
            if bHasAttr:
                f_1 = f_jacob.py_func(yy,params) 
            else:
                f_1 = f_jacob(yy,params) 
            f_hessian = model.functions["f_hessian"]
            if bHasAttr:
                f_2 = f_hessian.py_func(yy,params)
            else:
                f_2 = f_hessian(yy,params)
            f_tensor = model.functions["f_tensor"]
            if bHasAttr:
                f_3 = f_tensor.py_func(yy,params)
            else:
                f_3 = f_tensor(yy,params)
            derivatives = f_0, f_1, f_2, f_3
                 
    else:
        
        if debug:
            if bSparse:
                from preprocessor.f_sparse import f_sparse
                bHasAttr  = hasattr(f_sparse,"py_func")
                if bHasAttr:
                    derivatives = f_sparse.py_func(yy,params,order=order)
                else:
                    derivatives = f_sparse(yy,params,order=order)
            else:
                from preprocessor.f_dynamic import f_dynamic
                bHasAttr  = hasattr(f_dynamic,"py_func")
                if bHasAttr:
                    derivatives = f_dynamic.py_func(yy,params,order=order)
                else:
                    derivatives = f_dynamic(yy,params,order=order)
        else:
            if bSparse:
                func = model.functions['f_sparse'] 
            else:
                func = model.functions['f_dynamic']
            bHasAttr  = hasattr(func,"py_func")
            if bHasAttr:
                derivatives = func.py_func(yy,params,order=order)
            else:
                derivatives = func(yy,params,order=order)
            
    return derivatives
    # return flip(n,order,model.indexOfFlippedEndogVariables,model.indexOfFlippedExogVariables,
    #             model.max_lead_shock,model.min_lag_shock,derivatives)
 
  
def get_function(model,y,func=None,shock=None,t=0):
    """
    Returns a function object.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param y: Array of values of endogenous variables.
        :type y: numpy.array
        :order func: Model equations function.
        :order func: Function
        :param shock: The values of shocks.
        :type shock: numpy.array
        :param t: Time index.
        :type t: int
        :returns: Function.
    """
    from numeric.solver.util import getParameters
    params = getParameters(model=model,t=t)
    
    if shock is None: 
        x = model.calibration['shocks']
    else:
        x = shock
 
    # Build the approximation point to take the derivatives
    yy = np.concatenate([y[2,:],y[1,:],y[0,:],x])        
 
    if func is None:
        func = model.functions['f_dynamic']
    
    bHasAttr   = hasattr(func,"py_func")
    if bHasAttr:
        f = func.py_func(yy,params,order=0)
    else:
        f = func(yy,params,order=0)
 
    return f

