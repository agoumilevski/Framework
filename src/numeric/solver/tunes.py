"""
Implements judgemental adjustments to endogenous variables paths.

Created on Fri Oct 23, 2020
Part of the code translated from Matlab Iris to Python.

@author: A.Goumlevski
"""
import numpy as np
import scipy.linalg as la


def forecast_with_tunes(model,Nperiods,y,T,Re,C,shocks,has_imaginary_shocks,has_imaginary_values,scale=complex(1,0)):
    """
    Forecast with judgmental adjustments (conditional forecasts).
    This is a solution for a 'fixed path' of endogenous variables.
    
    For references please see Michal Andrle's algorithm described in an unpubished paper:     
    https://michalandrle.weebly.com/uploads/1/3/9/2/13921270/iris_simulate.pdf
    
    
    Parameters:
        model : Model.
            Model object.
        Nperiods : numpy.ndarray.
            Number of time periods.
        y : numpy.ndarray.
            The solution with no tunes.
        T : numpy.ndarray.
            Transition matrix.
        Re : numpy.ndarray.
            Matrix of coefficiets of forward shocks.
        C : numpy.ndarray.
            Matrix of constants.
        shocks : list.
            Shocks at different time periods.
        has_imaginary_shocks : bool.
            Flag set to True if ay of the shocks is a complex number, and to False other otherwise.
        has_imaginary_values : bool.
            Flag set to True if any of the endogenous variables path is a complex number, and to False other otherwise.
        scale : numpy.complex.
            Scaling factor for standard deviations of expected and un-expected shocks.
            If model.anticipate flag is set to True , then real part of scale is a multiplier
            for standard devitions of expected shocks and the imaginary part is a multiplier
            for standard devitions of un-expected shocks.
            If model.anticipate flag is set to False, then it is wise versa.  
           
    Returns:
        model : Model.
            Model object.
        y : numpy array.
            Numerical solution with tunes.
            
    """
    y, adjusted_shocks = mixed(model=model,x=y,T=T,Re=Re,C=C,shocks=shocks,Nperiods=Nperiods,scale=scale)

    if "periods" in model.options:
        del model.options["periods"]
    periods = None
        
    return y,adjusted_shocks,model
        

def mixed(model,x,T,Re,C,shocks,Nperiods,scale=complex(1,0)):
    """
    Forecast with judgmental adjustments (conditional forecasts).
    
    This is a generic case with a mixture of anticipated and un-anticipated shocks.  
    This soltion also can handle conditional shocks.
    
    If model anticipate flag is set to True, then real numbers describe anticipated shocks
    and imaginary numbers describe un-anticipated shocks.
    If model anticipate flag is set to False, then real numbers describe un-anticipated shocks
    and imaginary numbers describe anticipated shocks.
    
    Parameters:
        model : Model.
            `Model' object.
        x : numpy.ndarray.
            Unadjusted values of endogenous variables.
        T : numpy array.
            Transition matrix.
        Re : numpy.ndarray.
            Matrix of coefficiets of shocks.
        C : numpy.ndarray.
            Matrix of constants.
        shocks : list.
            Shocks at different time periods.
        Nperiods : int.
            Simulation time range.
        scale : numpy.complex.
            Scaling factor for standard deviations of expected and un-expected shocks.
            If model.anticipate flag is set to True , then real part of scale is a multiplier
            for standard devitions of expected shocks and the imaginary part is a multiplier
            for standard devitions of un-expected shocks.
            If model.anticipate flag is set to False, then it is wise versa.    

    Returns:
        x : numpy array.
            Solution of the system.
        ea : numpy array.
            Anticipated shocks.
        eu : numpy array.
            Un-anticipated shocks.

    """
    last = 0
    # Get time periods of user's judgmental adjusstments.
    if bool(model.mapSwap):
        rng  = sorted(model.mapSwap.keys())
        last = rng[-1]
    # else:
    #     return x, shocks
        
    if bool(model.condShocks):
        cond_rng  = sorted(model.condShocks.keys())
        last = max(last,cond_rng[-1])
        
    shock_values = model.calibration['shocks']
    n = len(shock_values)
    for i in range(n-1,-1,-1):
        shk = shock_values[i]
        if np.isscalar(shk):
            if not shk==0:
                break
        else:    
            if not all(shk==0):
                break
    last = max(last,i+1)
    
    if last == 0:
        return x, shocks
            
    
    variables = model.symbols['variables']
    shock_var = model.symbols['shocks']
    nx        = len(variables) # Number of endogenous variables
    ne        = len(shock_var) # Number of shocks
    nt        = 1+nx+2*ne*last # Total number of unknowns
    
    
    # Indices of exogenized variables and endogenized shocks.
    indExog1,indEndog1,valExog,valEndog,indExp,indUnExp,indCond,indCondNotExog,valCond = \
        getIndices(model.mapSwap,model.condShocks,model.anticipate,nx,ne,last)

    # Indices of exogenized data points and endogenized shocks.
    if bool(model.mapSwap):
        _indExog  = [False] *  nx    + indExog1
        indEndog  = [False] * (1+nx) + indEndog1 
    else:
        _indExog  = [False] * ((1+nx)*last)
        indEndog  = [False] *  nt
        
    indExog = indExog1
    not_indExog  = np.logical_not(indExog)
    not_indEndog = np.logical_not(indEndog)
        
    # Get swapped system matrices
    Mb,M = computeMatrices(model=model,T=T,Re=Re,C=C,indExog=_indExog,indEndog=indEndog,nx=nx,ne=ne,last=last)

    # Flip over expected and un-expected shocks if anticipate flag is set to False.
    if model.anticipate:
        eu1 = np.imag(shocks[:last])
        ea1 = np.real(shocks[:last])
    else:
        eu1 = np.real(shocks[:last])
        ea1 = np.imag(shocks[:last])
              
    inp = np.array([1]+list(x[0])+list(np.ravel(eu1))+list(np.ravel(ea1)))
    
    # Swap exogenized outputs and endogenized inputs.
    # rhs = [inp(~endi);outp(exi)]
    tmp1 = inp[not_indEndog]
    tmp2 = valExog[indExog]
    tmp  = tmp1.tolist() + tmp2.tolist()
    rhs  = np.array(tmp)
    # lhs = [outp(~exi);inp(endi)]
    # tmp1 = valExog[not_indExog]
    # tmp2 = inp[indEndog]
    # lhs  = tmp1.tolist() + tmp2.tolist()
    
    # Solve the swapped system.
    lhs  = M  @ rhs
    # Values of variables at the end of adjustments.
    xadj = Mb @ rhs
    
    # For conditional shocks minimize the difference between LHS and RHS... 
    # Use KF equations for update step where observables are the user's judgemental values 
    # of endogenous variables and the standard deviations of these exogenized variables are zero.
    if bool(model.condShocks):
        
        # Create correlation matrices of expected and un-expected shocks.
        # Set the standard deviations of endogenized shocks to zero. Otherwise an anticipated 
        # endogenized shock would have a non-zero unanticipated standard deviation, and vice versa.
        stdAntShk              = np.ones(shape=(ne,last))
        stdAntShk[indExp]      = 0
        stdAntShk[indUnExp]    = 0
        stdUnAntShk            = np.ones(shape=(ne,last))
        stdUnAntShk[indExp]    = 0
        stdUnAntShk[indUnExp]  = 0
        stdAntShk             *= np.imag(scale)
        stdUnAntShk           *= np.real(scale)
        # For anticipated shocks swap standard deviations
        if model.anticipate:
            stdAntShk,stdUnAntShk = stdUnAntShk,stdAntShk
            
        # Compute the covariance matrix of the RHS in the swapped system.
        P     = calcPrhs(stdUnAntShk,stdAntShk,indEndog,indExog,nx,ne,nt,last)
        Z     = M[indCondNotExog]
        # Prediction error
        err   = valCond - lhs[indCondNotExog]
        # Update mean forecast
        PZt   = P   @ Z.T
        F     = Z   @ PZt
        K     = PZt @ la.pinv(F)
        delta = K   @ err
        rhs  += delta
        lhs  += M @ delta
        xadj += Mb @ delta
        del P
  
    # Release memory allocated to large arrays.
    del Mb, M
    
    outp = np.zeros(nx*last)
    inp  = np.zeros(nt)
    n1   = sum(not_indExog)
    n2   = sum(not_indEndog)
    outp[not_indExog] = lhs[:n1]
    outp[indExog]     = rhs[n2:]
    inp[not_indEndog] = rhs[:n2]
    inp[indEndog]     = lhs[n1:]
    
    # Un-anticipated shocks
    eu  = np.reshape(inp[nx+1:nx+1+ne*last],(last,ne))
    # Anticipated shocks.
    ea  = np.reshape(inp[nx+1+ne*last:],(last,ne))
    
    # Combine un-expected and expected shocks
    shocks = shocks.astype(complex)
    for t in range(last):
        for i in range(ne):
            if model.anticipate:
                shocks[t,i] = complex(0,1)*eu[t,i] + ea[t,i]
            else:
                shocks[t,i] = eu[t,i] + complex(0,1)*ea[t,i]

    # Forecasted solution.
    x[1:last+1] = np.reshape(outp,(last,nx))
    # Compute solution for the rest of time range
    for t in range(last,Nperiods+1):
        x[t+1] = T @ x[t] + C
        for j in range(t,Nperiods+1):
            x[t+1] += np.real(Re[j-t]) @ np.real(shocks[j])
             
    return x, shocks


def getIndices(mapSwap,condShocks,anticipate,nx,ne,last,debug=False):
    """
    Return indices of exogenized variables and endogenized shocks and judgemental adjustments.

    Parameters:
        mapSwap : dict.
            Map of indices and values of exogenized variables and endogenized shocks.
        condShocks : dict.
            Map of conditinal shocks.
        anticipate : bool.
            True if shocks are anticipated.
        nx : int
            The number of endogenous variables.
        ne : int
            The number of shocks.
        last : int
            The last time period of adjustments.
        
    Returns:
        Indices of exogenized variables and endogenized shocks and values of judgemental adjustments.
    
    """ 
    indExp   = np.array([False] * (ne*last))
    indUnExp = np.array([False] * (ne*last))
    indEndog = [False] * (2*ne*last)
        
    if bool(mapSwap):
        index0={}; index1={}; index2={}; index3={}; indExog={}; values1={}; values2={}
        rng = sorted(mapSwap.keys())
        for t in rng:
            vals = mapSwap[t]
            ind0=[]; ind1=[]; ind2=[]; ind3=[]; val1=[]; val2=[]
            for val in vals:
                (i0,i1,v1,i2,v2) = val
                ind0.append(i0)
                ind1.append(i2)
                val1.append(v1)
                val2.append(v2)
                if np.imag(v1) == 0:
                    ind2.append(np.nan)
                    ind3.append(i2)
                else:
                    ind2.append(i2)
                    ind3.append(np.nan)
                
            index0[t]  = ind0
            index1[t]  = ind1
            index2[t]  = ind2
            index3[t]  = ind3
            values1[t] = val1
            values2[t] = val2
            
        tmp1      = getData(m1=index3,n=ne)
        tmp2      = getData(m1=index2,n=ne)
        indExog   = getData(m1=index0,n=nx)
        valExog   = getData(m1=index0,n=nx,m2=values1)
        if len(indExog) < nx*last:
            indExog += [False] * (nx*last-len(indExog))
        for i in range(ne*last):
            ii = i + ne*last
            if anticipate:
                if i < len(tmp1):
                    indEndog[i]  = tmp2[i]
                if i < len(tmp2):
                    indEndog[ii]  = tmp1[i]
            else:
                if i < len(tmp1):
                    indEndog[i]  = tmp1[i]
                if i < len(tmp2):
                    indEndog[ii]  = tmp2[i]
        
        if condShocks:
            #tmp = [(x-1)*ne+np.arange(ne) for x in rng]
            #indx = [item for sublist in tmp for item in sublist]
            indx = []
            for t in rng:
                indx.extend((t-1)*ne+np.arange(ne))
            indUnExp[indx] =  tmp1
            indExp[indx]   =  tmp2
        
        tmp3      = np.reshape(a=getData(m1=index3,n=ne,m2=values2),newshape=-1,order='F')
        tmp4      = np.reshape(a=getData(m1=index2,n=ne,m2=values2),newshape=-1,order='F')
        valEndog  = np.array(list(tmp3) + list(tmp4))
        valExog   = np.reshape(a=valExog,newshape=-1,order='F')
        if len(valExog) < nx*last:
            valExog = np.array(list(valExog) + [np.nan] * (nx*last-len(valExog)))
        
    else:
        indExog   = [False] * (nx*last)
        indEndog  = [False] * (2*ne*last)
        indExp    = [False] * (ne*last)
        indUnExp  = [False] * (ne*last)
        valExog   = np.zeros(nx*last)
        valEndog  = np.zeros(2*ne*last)
        
    indExp    = np.reshape(a=indExp,newshape=(ne,last),order='F')
    indUnExp  = np.reshape(a=indUnExp,newshape=(ne,last),order='F')
        
    if condShocks:
        
        indCond = [False] * (nx*last)
        for t in range(last):
            tp1 = t + 1
            if tp1 in condShocks:
                ind = [a for a,b in condShocks[tp1]]
                for i in ind:
                    indCond[t*nx+i] = True
        # Index of conditions on transition variables excluding exogenized positions.
        # tmp = indCond[not_indExog]
        tmp = [x for x,y in zip(indCond,indExog) if not y]
        indCondNotExog = tmp + [False] * (len(indCond)-len(tmp))
        valCond = []
        for t in range(1,1+last):
            if t in condShocks:
                arr = [b for a,b in condShocks[t]]
                valCond.extend(arr)
        valCond = np.array(valCond)   
    else: 
        indCond = None; indCondNotExog = None; valCond = None
        
    if debug:
        print('\nIndExog:',np.where(indExog))
        print('\nIndEndog:',np.where(indEndog))
        print('\nIndUnExp:',np.where(indUnExp))
        print('\nIndExp:',np.where(indExp))
        print('\nValExog:',valExog[indExog])
        print('\nValEndog:',valEndog[indEndog])
        if bool(condShocks):
            print('\nIndCond:',np.where(indCond))
            print('\nIndCondNotExog:',np.where(indCondNotExog))
            print('\nCondValue:',valCond)
        
    return indExog,indEndog,valExog,valEndog,indExp,indUnExp,indCond,indCondNotExog,valCond


def computeMatrices(model,T,Re,C,indExog,indEndog,nx,ne,last):
    """
    Compute system solution matrices with swapped exogenized variables and endogenized shocks.

    Parameters:
        model : Model
            Model object.
        T : numpy.ndarray.
            Transition matrix of endogenous variables.
        Re : numpy.ndarray.
            Forward matrix of coefficients of shocks for the entire range of adjustments.
        C : numpy.ndarray.
            Array of constants.
        indExog : list.
            Indices of exogenized variables.
        indEndog : list.
            Indices of endogenized variables.
        nx : int
            Number of endogenous variables.
        ne : int
            Number of shocks.
        last : int
            Last period of adjustments.
    
        Returns:
            M : Numpy array.
                Swapped matrix of shocks.

    """
    # Constant.
    Mc = np.zeros((nx*last))
    mc = np.zeros(nx)
    # Multipliers on initial condition.
    M0 = np.zeros((nx*last, nx))
    m0 = np.eye(nx)
    # Multipliers on unexpected shocks.
    Mu = np.zeros((nx*last, ne*last))
    mu = np.zeros((nx, ne*last))
    # Multipliers on expected shocks.
    Me = np.zeros((nx*last, ne*last))
    me = np.zeros((nx, ne*last))
    
    for t in range(last):
        # Constants.
        mc = T @ mc + C
        Mc[t*nx:(t+1)*nx] = mc
        # Initial conditions.
        m0 = T @ m0
        M0[t*nx:(t+1)*nx] = m0
        # Unexpected shocks.
        mu = T @ mu
        mu[:, t*ne:(t+1)*ne] += Re[0]
        Mu[t*nx:(t+1)*nx]  = mu
        # Expected shocks.
        me = T @ me
        for j in range(t,last):
            me[:,j*ne:(j+1)*ne] += Re[j-t]
        Me[t*nx:(t+1)*nx] = me

    mc = np.reshape(mc,(nx,1))
    Mc = np.reshape(Mc,(nx*last,1))
    
    tmp1 = np.concatenate((mc,m0,mu,me),axis=1)
    tmp2 = np.concatenate((Mc,M0,Mu,Me),axis=1)
    M    = np.concatenate((tmp1,tmp2),axis=0)
    
    if np.any(indExog) or np.any(indEndog):
        # Swap endogenised and exogenised columns in I and M matrices.
        not_indExog  = np.logical_not(indExog)
        not_indEndog = np.logical_not(indEndog)
        if False:
            I    = np.eye(len(M))
            I1   = I[:, not_indExog]
            I2   = I[:, indExog]
            M1   = M[:, not_indEndog]
            M2   = M[:, indEndog]
            tmp1 = np.concatenate((M1,-I2),axis=1)
            tmp2 = np.concatenate((I1,-M2),axis=1)
            M    = la.solve(tmp2,tmp1)
        else:
            M11  = M[np.ix_(not_indExog, not_indEndog)]
            M12  = M[np.ix_(not_indExog, indEndog)]
            M21  = M[np.ix_(indExog, not_indEndog)]
            M22  = M[np.ix_(indExog, indEndog)]
            iM22 = la.inv(M22)
            tmp1 = np.concatenate((M11-M12@iM22@M21, M12@iM22),axis=1)
            tmp2 = np.concatenate((-iM22@M21, iM22),axis=1)
            M    = np.concatenate((tmp1,tmp2),axis=0)
    
    Mb = M[:nx]; M  = M[nx:]
    return Mb,M
 
    
def stdcorr2cov(stdDev,ne,T) -> np.array:
    """
    Create covariance matrix given standard deviations.

    Parameters:
        stdDev : numpy.ndarray
            Standard deviations.
        ne : int
            Number of shocks.
        T : int
            Number of time periods.

    Returns:
        cov : Numpy array
            Diagonal covariance matrix.

    """
    cov = np.zeros(shape=(ne,ne,T))
    for i in range(T):
        x = stdDev[:,i]**2
        for j in range(ne):
            cov[j,j,i] = x[j]
        
    return cov
    
    
def calcPrhs(stdUnAntShk,stdAntShk,indEndog,indExog,nx,ne,nt,last) -> np.array:
    """
    Compute MSE/Cov matrix of the RHS in the swapped system.

    Parameters:
        stdUnAntShk : numpy .ndarray.
            Covariance matrix of un-anticipated shocks.
        stdAntShk : numpy .ndarray.
            Covariance matrix of anticipated shocks.
        indEndog : list.
            Indices of endogenized variables.
        indExog : list.
            Indices of exogenized variables.
        nx : int.
            Number of endogenous variables.
        ne : int.
            Number of shocks.
        nt : int.
            Size of covariance matrix.
        last : int.
            Last period of adjustments.

    Returns:
        Prhs : Numpy array.
            Covariance matrix of shocks.

    """
    from scipy.linalg import block_diag
    
    P    = np.zeros(shape=(nt,nt))
    Pu   = stdcorr2cov(stdUnAntShk,ne,last)
    Pe   = stdcorr2cov(stdAntShk,ne,last)
    pos  = np.arange(1+nx,1+nx+ne)
    for i in range(last):
        P[np.ix_(pos, pos)] = Pu[:,:,i]
        pos += ne
   
    for i in range(last):
        P[np.ix_(pos, pos)] = Pe[:,:,i]
        pos += ne
   
    not_indEndog = np.logical_not(indEndog) 
    
    P = P[np.ix_(not_indEndog, not_indEndog)]
    # Add np.zeros for the std errors of exogenized data points.
    # if len(P) < nt:
    #     n = nt - len(P)
    if np.any(indExog):
        n = sum(indExog)
        P = block_diag(P, np.zeros(shape=(n,n)))
        
    return P


def hasImaginaryValues(m) -> bool:
    """
    Check if any shock at any time is a complex number.

    Parameters:
        m : dict.
            Map of indices and values of exogenized variables and endogenized shocks.

    Returns:
        imaginary_values : bool
            Returns True if any path has imaginary value.
    """
    imaginary_values = False

    rng = sorted(m.keys())
    for t in rng:
        vals = m[t]
        for val in vals:
            (i0,i1,v1,i2,v2) = val
            if np.any(np.iscomplex(v1)) or np.any(np.iscomplex(v2)):
                imaginary_values = True
                break
    
    return imaginary_values
    

def hasImaginaryShocks(shocks) -> bool:
    """
    Check if any shock at any time is a complex number.

    Parameters:
        shocks : Numpy array
            List of shocks.

    Returns:
        imaginary_shocks : bool
            Returns True if any shock is a complex.
    """
    imaginary_shocks = False
    for shk in shocks:
        if np.any(np.iscomplex(shk)):
            imaginary_shocks = True
            break
     
    return imaginary_shocks
    

def getData(m1,n,m2=None):
    """
    Get list of indices or 2d array or values.

    Parameters:
        m1 : dict
            Map with key as time period and index as value.
        n : int, optional
            The number of variables.
        m2 : dict, optional
            Map with time periods and values of variables. The default is None.

    Returns:
        arr : list or numpy array
            Indices of exogenized variables and endogenized shocks.
    """
    from misc.termcolor import cprint
    
    rng = sorted(m1.keys())
    T = rng[-1]
    
    if m2 is None:
        
        arr = []
        for t in range(T):
            tp1 = 1 + t
            if tp1 in m1:
                x = [True if i in m1[tp1] else False for i in range(n)]
            else:
                x = [False] * n
            arr += x
            
    else:
        
        arr = np.empty((n,T)); arr[:] = np.nan
        for t in range(T):
            tp1 = 1 + t
            if tp1 in m1:
                indices = np.array(m1[tp1])
                b = [False if np.isnan(x) else True for x in indices]
                if tp1 in m2:
                    v = m2[tp1]
                    if np.isscalar(v):
                        if bool(v):
                            if np.imag(v) == 0:
                                arr[indices,t] = v
                            else:
                                arr[indices,t] = np.imag(v)
                    else:
                        indexes = indices[b]
                        values = np.array(v)[b]
                        for ind,val in zip(indexes,values):
                            if not val is None and not np.isnan(val):
                                if np.imag(val) == 0:
                                    arr[ind,t] = np.real(val)
                                else:
                                    arr[ind,t] = np.imag(val)
                else:
                    cprint("Time period {} is missing in the value dictionary {}".format(t,m2.keys()),"red")
            else:
                pass
                #cprint("Time period {} is missing in the index dictionary {}".format(t,m1.keys()),"red")
        
    return arr    
    

def not_anticipated(mapSwap,y,shocks,T,R,C,Nperiods,rng):
    """
    Forecast with all un-anticipated shocks.
    
    Parameters:
        mapSwap : dict.
            Map of indices and values of exogenized variables and endogenized shocks.
        y : numpy.ndarray.
            The solution with no tunes.
        shocks : list.
            Shocks at different time periods.
        T : numpy.ndarray.
            Transition matrix.
        R : numpy.ndarray.
            Matrix of coefficiets of shocks.
        C : numpy.ndarray.
            Matrix of constants.
        Nperiods : numpy array.
            Number of time periods.
        rng : list.
            Time range of the tunes.

    Returns:
        y : numpy.ndarray.
            Numerical solution with tunes.
        shocks : numpy.ndarray.
            Shocks that bring path of endogenous variables to the desired level.
    """
    for t in rng:
        vals = mapSwap[t]
        ind0=[]; val1=[]; ind2=[]
        for val in vals:
            (i0,i1,v1,i2,v2) = val
            ind0.append(i0)
            ind2.append(i2)
            val1.append(v1)
        Rsh = R[np.ix_(ind0,ind2)]
        yd = y[t,ind0] - val1
        # Correct shocks.
        delta = la.solve(Rsh,yd)
        shocks[t-1,ind2] = -delta 
        
        # Recalculate solution with the new shocks.
        for tt in range(t-1,Nperiods+1):
            y[tt+1] = T @ y[tt] + C + np.real(R) @ np.real(shocks[tt])
  
    return y, shocks
         

def anticipated(mapSwap,y,shocks,T,Re,C,Nperiods,rng,total_nmbr_shocks,n_shocks):
    """
    Forecast with a fixed path of some of the endogenous variables.
    
    This is a solution when all shocks are anticipated.
    
    Parameters:
        mapSwap : dict.
            Map of indices and values of exogenized variables and endogenized shocks.
        y : numpy.ndarray.
            The solution with no tunes.
        shocks : list.
            Shocks at different time periods.
        T : numpy.ndarray.
            Transition matrix.
        Re : numpy.ndarray.
            Forward matrix of coefficiets of shocks.
        C : numpy.ndarray.
            Matrix of constants.
        Nperiods : int.
            Number of time periods.
        rng : list.
            Time range of the tunes.

    Returns:
        y : numpy.ndarray.
            Numerical solution with judgemental adjustments.
        shocks : numpy.ndarray.
            Shocks that bring path of endogenous variables to the desired level.
    """
    N = len(T)
    n_rng = max(rng)
    yds = []; js = 0
    total_nmbr_shocks = total_nmbr_shocks
    Rsh = np.empty(shape=(total_nmbr_shocks,total_nmbr_shocks)) 
    S = np.empty(shape=(n_rng,n_rng,N,n_shocks))
    # Build matrix S
    for t in range(n_rng):
        for k in range(n_rng):
            if t == 0:
                S[t,k] = Re[k]
            else:
                S[t,k] = T @ S[t-1,k] 
                if t > k:
                    S[t,k] += Re[t-k]
    for jj,j in enumerate(rng):
        values = mapSwap[j]
        n1 = len(values)
        ind1=[]; val1=[]
        for val in values:
            (i0,i1,v1,i2,v2) = val
            ind1.append(i0)  
            val1.append(v1) 
        j1 = np.arange(js,js+n1,dtype=int)
        js += n1
        ks = 0
        # Compute matrix of coefficients of shocks to account for user's judgmental adjustments.
        yd = y[j,ind1] - val1 
        yds.extend(yd)
        for k in rng:
            values = mapSwap[k]
            n2 = len(values)
            ind2=[]
            for val in values:
                (i0,i1,v1,i2,v2) = val
                ind2.append(i2)
            k1 = np.arange(ks,ks+n2,dtype=int)
            ks += n2
            temp = S[j,k]
            Rsh[np.ix_(j1,k1)] = temp[np.ix_(ind1,ind2)]
              
    y_delta = np.array(yds)
    # Find shocks that make the solution equal to the 'fixed path' of endogenous variables.   
    delta = la.solve(Rsh,y_delta)
    js = 0
    for j,r in enumerate(rng):
        vals = mapSwap[rng[j]]
        n1 = len(vals)
        ind=[]
        for val in vals:
            (i0,i1,v1,i2,v2) = val
            ind.append(i2)
        shocks[r-1,ind] -= delta[js:js+n1]
        js += n1
        
    # Recalculate solution with the new shocks.
    for t in range(Nperiods+1):
        y[t+1] = T @ y[t] + C
        for j in range(t,Nperiods+1):
            y[t+1] += np.real(Re[j-t]) @ np.real(shocks[j])
        
    return y, shocks
    