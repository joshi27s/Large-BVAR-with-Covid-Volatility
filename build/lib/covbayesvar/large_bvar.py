import numpy as np
from numpy.random import gamma
from numpy.linalg import eigvals, eig, solve
from scipy import linalg as la
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Callable, Tuple, Any, Union, Optional
from scipy.stats import multivariate_normal as mvnrnd
from scipy.optimize import fsolve
from scipy.special import gammaln, betaln, factorial


def beta_coef(x, mosd):
    """
    Computes the coefficients for the Beta distribution.

    Args:
        x (list): Contains alpha and beta values for the Beta distribution
        mosd (list): Contains mode and standard deviation values.

    Returns:
        list: List with two results that represent two coefficients
    """

    al = x[0]  # alpha parameter
    bet = x[1]  # beta parameter

    # mode and standard deviation of the beta distribution
    mode = mosd[0]
    sd = mosd[1]

    # compute the first and second results based on the parameters
    r1 = mode - (al - 1) / (al + bet - 2)
    r2 = sd - (al * bet / ((al + bet) ** 2 * (al + bet + 1))) ** 0.5

    return [r1, r2]


def bfgsi(H0, dg, dx):
    """
    Perform a Broyden-Fletcher-Goldfarb-Shanno (BFGS) update on the inverse Hessian matrix.

    Args:
        H0 (numpy.ndarray): The current estimate of the inverse Hessian matrix.
            Must be a square matrix.
        dg (numpy.ndarray): The previous change in the gradient (as a column vector).
            Must be a vector of the same dimension as one side of H0.
        dx (numpy.ndarray): The previous change in the variable x (as a column vector).
            Must be a vector of the same dimension as one side of H0.

    Returns:
        numpy.ndarray: The updated inverse Hessian matrix. If the update fails, the original H0 is returned.

    Notes:
        - The function uses the BFGS formula to compute the updated inverse Hessian.
        - The update may fail if the dot product of dg and dx is too close to zero,
          in which case a warning is printed.

    Example:
        >>> H0 = np.diag([1, 2, 3])
        >>> dg = np.array([0.1, -0.2, 0.3])
        >>> dx = np.array([0.4, -0.5, 0.6])
        >>> bfgsi(H0, dg, dx)
        """
    # Ensure dg and dx are column vectors
    dg = np.reshape(dg, (-1, 1))
    dx = np.reshape(dx, (-1, 1))

    # Compute the product of H0 and dg
    Hdg = np.dot(H0, dg)

    # Compute the dot product of dg and dx
    dgdx = np.dot(dg.T, dx)

    # Check if dgdx is not too small to avoid division by zero
    if np.abs(dgdx) > 1e-12:
        H = H0 + (1 + (np.dot(dg.T, Hdg) / dgdx)) * (dx @ dx.T) / dgdx - (dx @ Hdg.T + Hdg @ dx.T) / dgdx
    else:
        print("bfgs update failed.")
        print(f"|dg| = {np.sqrt(np.dot(dg.T, dg))} |dx| = {np.sqrt(np.dot(dx.T, dx))}")
        print(f"dg'*dx = {dgdx}")
        print(f"|H*dg| = {np.dot(Hdg.T, Hdg)}")
        H = H0

    return H


def bvarFcst(y, beta, hz):
    """
    Computes the forecasts for a vector autoregression (VAR) model at the specified forecast horizons.

    This function takes historical data (`y`), estimated VAR coefficients (`beta`), and a list of forecast
    horizons (`hz`). It then computes the forecasts for each variable in `y` at each horizon in `hz`.

    Args:
        y (numpy.ndarray): A 2D array of observed data with shape (T, n), where T is the number of time periods
                           and n is the number of variables in the VAR model.
        beta (numpy.ndarray): A 2D array of coefficients for the VAR model with shape (k, n), where k is the
                              number of coefficients for each variable (including the intercept and lags of all
                              variables) and n is the number of variables.
        hz (list or array-like): A list or array of integers representing the forecast horizons. For example,
                                 hz=[1, 2, 3] will compute forecasts for 1, 2, and 3 periods ahead.

    Returns:
        numpy.ndarray: A 2D array with shape (len(hz), n) containing the forecasted values for each variable
                       at each specified horizon. The rows correspond to the horizons in `hz`, and the columns
                       correspond to the variables in `y`.

    Example:
        >>> # Example usage
        >>> y = np.array([[1.2, 0.5], [1.3, 0.7], [1.1, 0.4]])  # Historical data (3 periods, 2 variables)
        >>> beta = np.array([[0.5, 0.3], [0.1, -0.1], [0.2, 0.0]])  # Coefficients (3 coefficients, 2 variables)
        >>> hz = [1, 2]  # Forecast for 1 and 2 periods ahead
        >>> forecast = bvarFcst(y, beta, hz)
        >>> print(forecast)  # Output: Forecasted values at horizons 1 and 2

        """
    k, n = beta.shape
    lags = (k - 1) // n
    T = y.shape[0]

    # Initialize forecast matrix with zeros at the end
    Y = np.vstack([y, np.zeros((max(hz), n))])

    # Compute the forecasts
    for tau in range(1, max(hz) + 1):
        # Calculating the Python equivalent indices for MATLAB's Y([T+tau-1:-1:T+tau-lags],:)'
        start_index = T + tau - 2  # Equivalent to T+tau-1 in MATLAB, minus 1 for Python's 0-based index
        end_index = T + tau - lags - 2  # Equivalent to T+tau-lags in MATLAB, minus 1 for Python's 0-based index

        # Selecting the rows and transposing
        selected_rows_transposed = Y[start_index:end_index:-1, :].T
        # Flatten the selected rows (while maintaining MATLAB's column-major order)
        flattened_rows = selected_rows_transposed.flatten(order='F')
        # Prepend 1 to the flattened array
        xT = np.hstack([1, flattened_rows])
        Y[T + tau - 1, :] = xT @ beta

    # Extract the forecasts at the specified horizons
    forecast = Y[T + np.array(hz) - 1, :]

    return forecast


def bvarGLP(y, lags, **kwargs):
    """
    Estimate the BVAR model of Giannone, Lenza and Primiceri (2015)

    Args:
        y (numpy.ndarray): Data matrix.
        lags (int): Number of lags in the VAR.
        **kwargs: Additional arguments to specify the BVAR model settings.
                  - mcmc (int)
                  - MCMCconst (int)
                  - MNpsi (int or float)
                  - sur (int)
                  - noc (int)
                  - Ndraws (int)
                  - hyperpriors (int)

    Example of dimensions of inputs:
        - lags = 13
        - y.shape = (544, 40)
        - varargs = (mcmc=1, MCMCconst=1, MNpsi=0, sur=1, noc=1, Ndraws=2000, hyperpriors=1)

    Returns:
        dict: Results of the BVAR estimation.

    """

    # Call the function to set the BVAR priors (equivalent to MATLAB's setpriors_covid)

    (r, mode, sd, priorcoef, MIN, MAX, hyperpriors, Vc, pos, mn, MNalpha, sur, noc, Fcast, hz, mcmc, M,
     N, const, MCMCfcast, MCMCstorecoeff, MCMCMsur, long_run) = set_priors(y, lags, **kwargs)

    # Data matrix manipulations
    #########################################################################
    # Dimensions
    TT, n = y.shape  # Number of rows and columns in the data matrix
    k = n * lags + 1  # Number of coefficients for each equation

    # Constructing the matrix of regressors
    #########################################################################
    x = np.zeros((TT, k))
    x[:, 0] = 1
    # Fill x with lagged values of y
    for i in range(1, lags + 1):
        x[:, 1 + (i - 1) * n:1 + i * n] = lag(y, i)

    y0 = np.mean(y[:lags, :], axis=0)  # Mean along the first lags for each variable

    x = x[lags:, :]  # Drop the first 'lags' rows
    y = y[lags:, :]  # Drop the first 'lags' rows

    # Check if 'Tpre' and 'Tpost' exist in the kwargs dictionary
    if 'Tpre' in kwargs and 'Tpost' in kwargs:
        Tpre = kwargs['Tpre']
        Tpost = kwargs['Tpost']
        y = np.concatenate([y[:Tpre, :], y[Tpost - 1:, :]], axis=0)
        x = np.concatenate([x[:Tpre, :], x[Tpost - 1:, :]], axis=0)
        r['setpriors']['Tpre'] = Tpre
        r['setpriors']['Tpost'] = Tpost

    T, n = y.shape  # Update dimensions

    # MN prior mean
    #########################################################################
    b = np.zeros((k, n))
    diagb = (np.ones(n)).reshape(-1, 1)

    # Check if 'pos' is provided and update diagb accordingly
    if pos is not None:
        pos = [int(p) for p in pos]  # Convert list of strings to integers
        diagb[pos] = 0

    # Fill in the diagonal starting from (1, 0)
    for i in range(n):
        if i < len(diagb):
            b[i + 1, i] = diagb[i]
        else:
            b[i + 1, i] = 1  # Ensure the last column also gets a 1

    # Starting values for the minimization
    #########################################################################

    lambda0 = 0.2  # std of MN prior
    theta0 = 1  # std of sur prior
    miu0 = 1  # std of noc prior
    alpha0 = 2  # lag-decaying parameter of the MN prior

    # Residual variance of AR(1) for each variable
    SS = np.zeros((n, 1))  # Initialize SS as a n x 1 zero matrix

    for i in range(n):
        # Prepare the independent variables for OLS regression
        # Include a column of ones for the intercept and the (1+i)th column of x
        X = np.concatenate([np.ones((T, 1)), x[:, i + 1].reshape(-1, 1)], axis=1)

        # Run OLS regression using your custom ols1 function
        ar1 = ols1(y[:, i].reshape(-1, 1), X)

        # Store the variance of the residuals
        SS[i] = ar1['sig2hatols']

    MIN['psi'] = SS / 100
    MAX['psi'] = SS * 100
    psi0 = SS

    if mn['psi'] == 1:
        inpsi = -np.log((MAX['psi'] - psi0) / (psi0 - MIN['psi']))
    elif mn['psi'] == 0:
        inpsi = []

    if mn['alpha'] == 1:
        inalpha = -np.log((MAX['alpha'] - alpha0) / (alpha0 - MIN['alpha']))
    elif mn['alpha'] == 0:
        inalpha = []

    if sur == 1:
        intheta = -np.log((MAX['theta'] - theta0) / (theta0 - MIN['theta']))
    elif sur == 0:
        intheta = []

    if noc == 1:
        inmiu = -np.log((MAX['miu'] - miu0) / (miu0 - MIN['miu']))
    elif noc == 0:
        inmiu = []

    # Perform the element-wise operation
    lambda_diff = -np.log((MAX['lambda'] - lambda0) / (lambda0 - MIN['lambda']))

    # Prepare the x0 vector by stacking the variables
    # We'll filter out any empty lists or None values
    x0_elements = [lambda_diff]  # Start with lambda_diff, which is not empty

    # Add other elements if they are not empty or None
    # Add other elements if they are not empty or None
    for element in [inpsi, intheta, inmiu, inalpha]:
        if isinstance(element, (np.ndarray, list)):  # Check if element is an array or list
            if isinstance(element, np.ndarray) and element.size > 0:  # Non-empty ndarray
                x0_elements.append(np.atleast_2d(element))
            elif isinstance(element, list) and len(element) > 0:  # Non-empty list
                x0_elements.append(np.atleast_2d(element))
        elif element is not None:  # Handle other non-None cases if applicable
            x0_elements.append(np.atleast_2d(element))


    # Concatenate the elements vertically to form the x0 array
    x0 = np.vstack(x0_elements)
    # Check if x0 is a scalar
    if np.isscalar(x0):
        # If x0 is a scalar, initialize H0 as 10 times the identity matrix
        H0 = 10 * np.eye(1)
    else:
        # If x0 is not a scalar, calculate its length and initialize H0 accordingly
        length_x0 = x0.shape[0]
        H0 = 10 * np.eye(length_x0)

    # Set your convergence criteria and max number of iterations
    crit = 1e-16
    nit = 1000

    # Prepare the extra arguments for the function
    varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, MCMCMsur,
                long_run]

    fh, xh, gh, H, itct, fcount, retcodeh = csminwel(logMLVAR_formin, x0, H0, None, crit, nit,
                                                     *varargin)

    # Call the logMLVAR_formin_covid function
    fh, betahat, sigmahat = logMLVAR_formin(xh, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0,
                                            hyperpriors, priorcoef, MCMCMsur, long_run)

    # Initialize the dictionary r
    r['lags'] = {'lags': lags}

    # Add postmax as a sub-dictionary within r
    r['postmax'] = {
        'betahat': betahat,
        'sigmahat': sigmahat,
        'itct': itct,
        'SSar1': SS,
        'logPost': -fh,
        'lambda': MIN['lambda'] + (MAX['lambda'] - MIN['lambda']) / (1 + np.exp(-xh[0])),
        'theta': MAX['theta'],
        'miu': MAX['miu']
    }

    if mn['psi'] == 1:
        # diagonal elements of the scale matrix of the IW prior on the residual variance
        r['postmax']['psi'] = MIN['psi'] + \
                              (MAX['psi'] - MIN['psi']).reshape((-1, 1)) / (1 + np.exp(-xh[1:n + 1])).reshape((-1, 1))
        if sur == 1:
            # std of sur prior at the peak
            r['postmax']['theta'] = MIN['theta'] + \
                                    (MAX['theta'] - MIN['theta']) / (1 + np.exp(-xh[n + 1]))
            if noc == 1:
                # std of noc prior at the peak
                r['postmax']['miu'] = MIN['miu'] + \
                                      (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[n + 2]))
        elif sur == 0:
            if noc == 1:
                # std of sur prior at the peak
                r['postmax']['miu'] = MIN['miu'] + \
                                      (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[n + 1]))
    elif mn['psi'] == 0:
        r['postmax']['psi'] = SS
        if sur == 1:
            # std of sur prior at the peak
            r['postmax']['theta'] = MIN['theta'] + \
                                    (MAX['theta'] - MIN['theta']) / (1 + np.exp(-xh[1]))
            if noc == 1:
                # std of sur prior at the peak
                r['postmax']['miu'] = MIN['miu'] + \
                                      (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[2]))
        elif sur == 0:
            if noc == 1:
                # std of sur prior at the peak
                r['postmax']['miu'] = MIN['miu'] + \
                                      (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[1]))

    if mn['alpha'] == 0:
        r['postmax']['alpha'] = 2
    elif mn['alpha'] == 1:
        # Lag-decaying parameter of the MN prior
        r['postmax']['alpha'] = MIN['alpha'] + \
                                (MAX['alpha'] - MIN['alpha']) / (1 + np.exp(-xh[-1]))

    ######################### forecasts at the posterior mode ##############################
    if Fcast == 1:
        Y = np.concatenate((y, np.zeros((hz[-1], n))))
        for tau in range(1, max(hz) + 1):
            xT = np.concatenate(([[1]], np.reshape(Y[T + tau - 2: T + tau - lags - 2: -1].T, (k - 1, 1),
                                                   order='F'))).T
            Y[T + tau - 1, :] = xT @ r['postmax']['betahat']

        r['postmax']['forecast'] = Y[T + np.array(hz) - 1, :]

    if mcmc == 1:

        # Jacobian of the transformation of the hyperparameters that has been used for the constrained maximization

        JJ = np.exp(xh) / (1 + np.exp(xh)) ** 2
        JJ[0] = (MAX['lambda'] - MIN['lambda']) * JJ[0]

        if mn['psi'] == 1:
            JJ[1:n + 1] = (MAX['psi'] - MIN['psi']) * JJ[1:n + 1]
            if sur == 1:
                JJ[n + 1] = (MAX['theta'] - MIN['theta']) * JJ[n + 1]
                if noc == 1:
                    JJ[n + 2] = (MAX['miu'] - MIN['miu']) * JJ[n + 2]
            elif sur == 0:
                if noc == 1:
                    JJ[n + 1] = (MAX['miu'] - MIN['miu']) * JJ[n + 1]
        elif mn['psi'] == 0:
            if sur == 1:
                JJ[1] = (MAX['theta'] - MIN['theta']) * JJ[1]
                if noc == 1:
                    JJ[2] = (MAX['miu'] - MIN['miu']) * JJ[2]
            elif sur == 0:
                if noc == 1:
                    JJ[1] = (MAX['miu'] - MIN['miu']) * JJ[1]

        if mn['alpha'] == 1:
            JJ[-1] = (MAX['alpha'] - MIN['alpha']) * JJ[-1]

        JJ = np.diagflat(JJ)
        HH = JJ @ H @ JJ.T

        # regularizing the Hessian (making sure it is positive definite)
        E, V = np.linalg.eigh(HH)
        E = np.diagflat(E)
        HH = V @ np.abs(E) @ V.T

        # recovering the posterior mode
        if mn['psi'] == 1:
            modepsi = r['postmax']['psi']
        elif mn['psi'] == 0:
            modepsi = []

        if mn['alpha'] == 1:
            modealpha = r['postmax']['alpha']
        else:
            modealpha = []

        if sur == 1:
            modetheta = r['postmax']['theta']
        else:
            modetheta = []

        if noc == 1:
            modemiu = r['postmax']['miu']
        else:
            modemiu = []

        postmode = np.concatenate(TypecastToArray([r['postmax']['lambda'],
                                                   modepsi, modetheta, modemiu, modealpha]), axis=0)

        if len(postmode) == 1:
            postmode = float(postmode[0])
        else:
            postmode = postmode[:, np.newaxis]

        P = np.zeros((M, len(xh)))
        logMLold = -10e15

        while logMLold == -10e15:
            # Ensure postmode is in the form of a one-dimensional array
            postmode_array = np.array([postmode])
            temp = np.random.multivariate_normal(postmode_array, HH * const ** 2, 1)
            P[0, :] = temp.flatten()
            logMLold, betadrawold, sigmadrawold = logMLVAR_formcmc(P[0, :].T[:, np.newaxis],
                                                                   y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                                   pos, mn, sur, noc, y0,
                                                                   max(MCMCfcast, MCMCstorecoeff),
                                                                   hyperpriors, priorcoef, MCMCMsur, long_run)

        # Initialize a key "mcmc" to store the draws of beta and sigma, if either MCMCstorecoeff or MCMCfcast is on
        if MCMCstorecoeff == 1 or MCMCfcast == 1:
            r['mcmc'] = {}

        # If MCMCstorecoeff is on, initialize beta and sigma matrices
        # burn in the first N draws, and retain the last M-N draws of the coefficients, variances and forecasts
        if MCMCstorecoeff == 1:
            r['mcmc']['beta'] = np.zeros((k, n, M - N))
            r['mcmc']['sigma'] = np.zeros((n, n, M - N))

        # If MCMCfcast is on, add the Dforecast matrix to the existing dictionary
        if MCMCfcast == 1:
            r['mcmc']['Dforecast'] = np.zeros((len(hz), n, M - N))

        #  Metropolis iterations
        count = 0
        for i in range(2, M + 1):  # Start from 1 to M-1, because Python is 0-based
            if i == 1000 * (i // 1000):
                print(f'Now running the {i}th MCMC iteration (out of {M})')

            # Calculate the covariance matrix
            cov_matrix = HH * (const ** 2)
            # Check if negative eigenvalues are present, and if so, then regularize
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            if np.any(eigenvalues <= 0):
                # Set negative eigenvalues to 0
                eigenvalues[eigenvalues < 0] = 0
                # Reconstruct the covariance matrix using the modified eigenvalues
                # ensure that the matrix is symmetric and positive semi-definite
                cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Draw candidate value
            P[i - 1, :] = np.random.multivariate_normal(mean=P[i - 2, :], cov=cov_matrix)
            logMLnew, betadrawnew, sigmadrawnew = logMLVAR_formcmc(P[i - 1, :].reshape(-1, 1),
                                                                   y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                                   pos, mn, sur, noc, y0,
                                                                   max(MCMCfcast, MCMCstorecoeff),
                                                                   hyperpriors, priorcoef, MCMCMsur, long_run)

            if logMLnew > logMLold:
                logMLold = logMLnew
                count += 1
            else:
                # generate random number between 0 and 1 from uniform distribution
                if np.random.rand() < np.exp(logMLnew - logMLold):

                    # will never reach so logMLold should never be very small
                    logMLold = logMLnew
                    count += 1
                else:
                    P[i - 1, :] = P[i - 2, :]

                    # if MCMCfcast is on, take a new draw of the VAR coefficients
                    # with the old hyperparameters if have rejected the new ones
                    # (the speed of this step could be probably improved)

                    if MCMCfcast == 1 or MCMCstorecoeff == 1:
                        _, betadrawnew, sigmadrawnew = logMLVAR_formcmc(P[i - 1, :].reshape(-1, 1),
                                                                        y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                                        pos, mn, sur, noc, y0,
                                                                        max(MCMCfcast, MCMCstorecoeff),
                                                                        hyperpriors, priorcoef, MCMCMsur, long_run)

            # stores draws of VAR coefficients if MCMCstorecoeff is on
            if i > N and MCMCstorecoeff == 1:
                r['mcmc']['beta'][:, :, i - N - 1] = betadrawnew
                r['mcmc']['sigma'][:, :, i - N - 1] = sigmadrawnew

            if i > N and MCMCfcast == 1:
                Y = np.concatenate((y, np.zeros((hz[-1], n))))
                for tau in range(1, max(hz) + 1):
                    xT = np.concatenate(([[1]], np.reshape(Y[T + tau - 2: T + tau - lags - 2: -1].T,
                                                           (k - 1, 1), order='F'))).T
                    Y[T + tau - 1, :] = xT @ betadrawnew + \
                                        np.random.multivariate_normal(
                                            np.zeros((1, n)).flatten(), sigmadrawnew)

                r['mcmc']['Dforecast'][:, :, i - N - 1] = Y[T + np.array(hz) - 1, :]

        # store the draws of the hyperparameters
        r['mcmc']['lambda'] = P[N:, 0]  # std MN prior

        if mn['psi'] == 1:
            # diagonal elements of scale matrix of IW prior on residual variance
            r['mcmc']['PSI'] = P[N:, 1:n + 1]
            if sur == 1:
                r['mcmc']['theta'] = P[N:, n + 1]  # std of sur prior
                if noc == 1:
                    r['mcmc']['miu'] = P[N:, n + 2]  # std of noc prior
            elif sur == 0:
                if noc == 1:
                    r['mcmc']['miu'] = P[N:, n + 1]  # std of noc prior
        elif mn['psi'] == 0:
            if sur == 1:
                r['mcmc']['theta'] = P[N:, 1]  # std of sur prior
                if noc == 1:
                    r['mcmc']['miu'] = P[N:, 2]  # std of noc prior
            elif sur == 0:
                if noc == 1:
                    r['mcmc']['miu'] = P[N:, 1]  # std of noc prior

        if mn['alpha'] == 1:
            # Lag-decaying parameter of the MN prior
            r['mcmc']['alpha'] = P[N:, -1]

        # calculates the acceptance rate by comparing each element of the lambda array with its previous element
        # and computing the mean of these comparisons.
        r['mcmc']['ACCrate'] = np.mean(r['mcmc']['lambda'][1:] != r['mcmc']['lambda'][:-1])

    return r


def bvarGLP_covid(y, lags, priors_params=None, **kwargs):
    """
    Estimate the BVAR model of Giannone, Lenza and Primiceri (2015), augmented
    for changes in volatility due to Covid (March 2020). Designed for monthly data.

    The path of common volatility is controlled by 3 hyperparameters and has
    the form `[eta(1) eta(2)*eta(3)**[0:end]]`.

    Args:
        y (numpy.ndarray): Data matrix.
        lags (int): Number of lags in the VAR.
        **kwargs: Additional arguments to specify the BVAR model settings.
                  - mcmc (int)
                  - MCMCconst (int)
                  - MNpsi (int or float)
                  - sur (int)
                  - noc (int)
                  - Ndraws (int)
                  - hyperpriors (int)
                  - Tcovid (int)

    Example of dimensions of inputs:
        >>> lags = 13
        >>> y.shape = (544, 40)
        >>> varargs = (mcmc=1, MCMCconst=1, MNpsi=0, sur=1, noc=1, Ndraws=2000, hyperpriors=1, Tcovid=507)

    Returns:
        dict: Results of the BVAR estimation.

    """

    # Call the function to set the BVAR priors (equivalent to MATLAB's setpriors_covid)

    (r, mode, sd, priorcoef, MIN, MAX, albet, mosd,  hyperpriors, Vc, pos, mn, MNalpha, Tcovid, sur, noc, Fcast,
        hz, mcmc, M, N, const, MCMCfcast, MCMCstorecoeff) = set_priors_covid(priors_params=priors_params, **kwargs)

    # Data matrix manipulations
    #########################################################################
    # Dimensions
    TT, n = y.shape  # Number of rows and columns in the data matrix
    k = n * lags + 1  # Number of coefficients for each equation

    # Constructing the matrix of regressors
    #########################################################################
    x = np.zeros((TT, k))
    x[:, 0] = 1
    # Fill x with lagged values of y
    for i in range(1, lags + 1):
        x[:, 1 + (i - 1) * n:1 + i * n] = lag(y, i)

    y0 = np.mean(y[:lags, :], axis=0)  # Mean along the first lags for each variable

    x = x[lags:, :]  # Drop the first 'lags' rows
    y = y[lags:, :]  # Drop the first 'lags' rows

    T, n = y.shape  # Update dimensions
    if Tcovid is not None:  # Check if Tcovid is valid
        Tcovid -= lags  # Perform the subtraction if Tcovid is valid
    else:
        Tcovid = None  # Keep Tcovid as None or handle it based on your logic

    # MN prior mean
    #########################################################################
    b = np.zeros((k, n))
    diagb = (np.ones(n)).reshape(-1, 1)

    # Check if 'pos' is provided and update diagb accordingly
    if pos is not None:
        diagb[pos] = 0

    # Fill in the diagonal starting from (1, 0)
    for i in range(n):
        if i < len(diagb):
            b[i + 1, i] = diagb[i]
        else:
            b[i + 1, i] = 1  # Ensure the last column also gets a 1

    # Starting values for the minimization
    #########################################################################

    lambda0 = 0.2  # std of MN prior
    theta0 = 1  # std of sur prior
    miu0 = 1  # std of noc prior
    alpha0 = 2  # lag-decaying parameter of the MN prior

    # Calculate 'aux' which measure volatility
    # Compute the mean of absolute differences from Tcovid to the end
    if Tcovid is None:
        # Handle the case where Tcovid is not provided or invalid
        aux = np.array([])
    else:
        # Select the slice from Tcovid-1 to T-1 (inclusive)
        y_post_Tcovid = y[Tcovid - 1:max(Tcovid + 1, T), :]
        y_pre_Tcovid = y[Tcovid - 2:max(Tcovid + 1, T) - 1, :]
        # Calculate mean_diff_post_Tcovid
        mean_diff_post_Tcovid = np.mean(np.abs(y_post_Tcovid - y_pre_Tcovid), axis=1)
        # Compute the mean of mean absolute differences before Tcovid
        mean_diff_pre_Tcovid = np.mean(
            np.abs(y[1:Tcovid - 1, :] - y[0:Tcovid - 2, :])
        )
        # Normalize the first mean by the second mean
        aux = mean_diff_post_Tcovid / mean_diff_pre_Tcovid

    # Check the length of 'aux' and define 'eta0' accordingly
    if aux.size == 0:
        eta0 = [] # Empty list for volatility hyperparameters
    elif aux.size == 2:
        eta0 = np.append(aux, [aux[0], 0.8])  # volatility hyperparameters
    elif aux.size >= 3:
        eta0 = (np.append(aux[:3], 0.8)).reshape(-1, 1)  # volatility hyperparameters

    # Residual variance of AR(1) for each variable
    SS = np.zeros((n, 1))  # Initialize SS as a n x 1 zero matrix

    for i in range(n):
        Tend = T  # Initialize Tend with the value of T

        # Check if Tcovid is not empty and update Tend accordingly
        if Tcovid is not None:
            Tend = Tcovid - 1

        # Perform OLS estimation only if Tend is valid
        ar1 = ols1(y[1:Tend, i], np.column_stack((np.ones((Tend - 1, 1)), y[0:Tend - 1, i])))

        # Update SS[i] with the estimated residual variance from the AR(1) model
        SS[i] = ar1['sig2hatols']  # Assuming ols1 returns a dictionary with key 'sig2hatols'

    # Calculations for inlambda and inHlambda
    inlambda = -np.log((MAX['lambda'] - lambda0) / (lambda0 - MIN['lambda']))
    inHlambda = (1 / (MAX['lambda'] - lambda0) + 1 / (lambda0 - MIN['lambda'])) ** 2 * (abs(lambda0) / 1) ** 2

    # Calculations for inalpha and inHalpha based on mn['alpha']
    if mn['alpha'] == 1:
        inalpha = -np.log((MAX['alpha'] - alpha0) / (alpha0 - MIN['alpha']))
        inHalpha = (1 / (MAX['alpha'] - alpha0) + 1 / (alpha0 - MIN['alpha'])) ** 2 * (abs(alpha0) / 1) ** 2
    elif mn['alpha'] == 0:
        inalpha = None
        inHalpha = None

    # Calculations for intheta and inHtheta based on sur
    if sur == 1:
        intheta = -np.log((MAX['theta'] - theta0) / (theta0 - MIN['theta']))
        inHtheta = (1 / (MAX['theta'] - theta0) + 1 / (theta0 - MIN['theta'])) ** 2 * (abs(theta0) / 1) ** 2
    else:
        intheta = None
        inHtheta = None

    # Calculations for inmiu and inHmiu based on noc
    if noc == 1:
        inmiu = -np.log((MAX['miu'] - miu0) / (miu0 - MIN['miu']))
        inHmiu = (1 / (MAX['miu'] - miu0) + 1 / (miu0 - MIN['miu'])) ** 2 * (abs(miu0) / 1) ** 2
    else:
        inmiu = None
        inHmiu = None

    # Calculations for ineta and inHeta based on Tcovid
    if Tcovid is not None:
        ncp = len(eta0)
        # Convert lists to numpy arrays with the same shape as eta0
        MAX_eta_transposed = MAX['eta'].reshape(-1, 1)
        MIN_eta_transposed = MIN['eta'].reshape(-1, 1)

        # Perform the element-wise operation
        ratio = (MAX_eta_transposed - eta0) / (eta0 - MIN_eta_transposed)
        ratio = np.where(ratio <= 0, 1e-8, ratio)  # Replace negative/zero values
        ineta = -np.log(ratio)
        inHeta = (1 / (MAX_eta_transposed - eta0) + 1 / (eta0 - MIN_eta_transposed)) ** 2 * (abs(eta0) / 1) ** 2
    else:
        ineta = None
        inHeta = None

    # Prepare the x0 vector by stacking the variables
    # We'll use a list comprehension to filter out any 'None' values
    # Filter out None elements and ensure each element is at least 2D
    x0_elements = [np.atleast_2d(x) for x in [inlambda, ineta, intheta, inmiu] if x is not None]

    # Concatenate the elements vertically to form the x0 array
    x0 = np.vstack(x0_elements)

    # Prepare the H0 diagonal matrix
    # Again, filtering out any 'None' values
    H0_elements = [np.atleast_2d(x) for x in [inHlambda, inHeta, inHtheta, inHmiu, inHalpha] if x is not None]
    H0 = np.diag(np.vstack(H0_elements).flatten())

    # Set your convergence criteria and max number of iterations
    crit = 1e-16
    nit = 1000

    # Prepare the extra arguments for the function
    varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]

    # Call the csminwel function
    fh, xh, gh, H, itct, fcount, retcodeh = csminwel(logMLVAR_formin_covid, x0, H0, None, crit, nit,
                                                     *varargin)

    # Call the logMLVAR_formin_covid function
    fh, betahat, sigmahat = logMLVAR_formin_covid(xh, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0,
                                                  hyperpriors, priorcoef, Tcovid)

    # Initialize the dictionary r
    r['lags'] = {'lags': lags}

    # Add postmax as a sub-dictionary within r
    r['postmax'] = {
        'betahat': betahat,
        'sigmahat': sigmahat,
        'itct': itct,
        'SSar1': SS,
        'logPost': -fh,
        'lambda': MIN['lambda'] + (MAX['lambda'] - MIN['lambda']) / (1 + np.exp(-xh[0])),
        'theta': MAX['theta'],
        'miu': MAX['miu'],
        'eta': np.array(MAX['eta']).T  # Transposing to match MATLAB's column vector
    }

    if Tcovid is not None:
        # covid-volatility hyperparameters
        r['postmax']['eta'] = (MIN_eta_transposed + (MAX_eta_transposed - MIN_eta_transposed) /
                               (1 + np.exp(-xh[1:ncp + 1])))

        if sur == 1:
            # std of sur prior at the peak
            r['postmax']['theta'] = MIN['theta'] + (MAX['theta'] - MIN['theta']) / (1 + np.exp(-xh[ncp + 1]))

            if noc == 1:
                # std of noc prior at the peak
                r['postmax']['miu'] = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[ncp + 2]))

        elif sur == 0:
            if noc == 1:
                # std of sur prior at the peak
                r['postmax']['miu'] = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[ncp + 1]))

    else:
        r['postmax']['eta'] = np.array([1, 1, 1])

        if sur == 1:
            # std of sur prior at the peak
            r['postmax']['theta'] = MIN['theta'] + (MAX['theta'] - MIN['theta']) / (1 + np.exp(-xh[1]))

            if noc == 1:
                # std of noc prior at the peak
                r['postmax']['miu'] = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[2]))

        elif sur == 0:
            if noc == 1:
                # std of sur prior at the peak
                r['postmax']['miu'] = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-xh[1]))

    if mn['alpha'] == 0:
        r['postmax']['alpha'] = 2
    elif mn['alpha'] == 1:
        # Lag-decaying parameter of the MN prior
        r['postmax']['alpha'] = MIN['alpha'] + (MAX['alpha'] - MIN['alpha']) / (1 + np.exp(-xh[-1]))

    # Check if Fcast is enabled
    if Fcast == 1:
        # Initialize the Y matrix
        Y = np.vstack([y, np.zeros((max(hz), n))])

        # Loop through all the forecast horizons
        for tau in range(1, max(hz) + 1):
            # Select the last 'lags' rows in reverse order and flatten the array
            # Adjust indices for zero-based indexing
            selected_Y = Y[T - tau - lags + 1:T - tau + 1][::-1].flatten()

            # Reshape the selected array to have k-1 rows
            # Note: The total number of elements must be equal to k-1
            xT = np.concatenate(([1], selected_Y[:k - 1]))

            # Generate the forecast
            Y[T + tau - 1, :] = np.dot(xT, r['postmax']['betahat'])

        # Store the forecasts
        r['postmax']['forecast'] = Y[T + np.array(hz) - 1, :]

    # Initialize mcmc; set to 1 for this example
    mcmc = 1

    # Check if MCMC is enabled
    if mcmc == 1:

        # Recovering the posterior mode
        if Tcovid is not None:
            modeeta = r['postmax']['eta']
        else:
            modeeta = None

        if mn['alpha'] == 1:
            modealpha = r['postmax']['alpha']
        elif mn['alpha'] == 0:
            modealpha = None

        if sur == 1:
            modetheta = r['postmax']['theta']  # Assuming modetheta is scalar
        elif sur == 0:
            modetheta = None

        if noc == 1:
            modemiu = r['postmax']['miu']  # Assuming modemiu is scalar
        elif noc == 0:
            modemiu = None

        # Filter out None elements and ensure each element is at least 2D
        postmode_elements = [np.atleast_2d(x) for x in [r['postmax']['lambda'], modeeta, modetheta, modemiu, modealpha]
                             if x is not None]

        # Concatenate the elements vertically to form the x0 array
        postmode = np.vstack(postmode_elements)

        # New computation of the inverse Hessian
        def fun(par):
            return logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                          pos, mn, sur, noc, y0, 0, hyperpriors, priorcoef, Tcovid)

        Hess, _ = hessian(fun, postmode)
        eigen_values, V = np.linalg.eig(Hess)
        E = np.diag(eigen_values)
        HH = -np.linalg.inv(Hess)

        if Tcovid is not None and T <= Tcovid + 1:
            HessNew = Hess.copy()
            HessNew[4, :] = 0
            HessNew[:, 4] = 0
            HessNew[4, 4] = -1
            HH = -np.linalg.inv(HessNew)
            HH[4, 4] = HH[2, 2]

        r['postmax']['HH'] = HH

        # Initialize variables
        P = np.zeros((M, len(xh)))
        LOGML = np.zeros((M, 1))
        logMLold = -1e15  # Initialize to a very small number

        # Starting value of the Metropolis algorithm
        while logMLold == -1e15:
            # Calculate the covariance matrix
            cov_matrix = HH * (const ** 2)
            # Check if negative eigenvalues are present, and if so, then regularize
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            if np.any(eigenvalues <= 0):
                # Set negative eigenvalues to 0
                eigenvalues[eigenvalues < 0] = 0
                # Reconstruct the covariance matrix using the modified eigenvalues
                # ensure that the matrix is symmetric and positive semi-definite
                cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # draw a random sample from the multivariate normal distribution and
            # populate the rows of matrix P one row at a time
            P[0, :] = np.random.multivariate_normal(mean=postmode.flatten(), cov=cov_matrix)

            logMLold, betadrawold, sigmadrawold = logMLVAR_formcmc_covid(
                P[0, :].reshape(-1, 1),
                y, x, lags, T, n, b,
                MIN, MAX, SS, Vc, pos,
                mn, sur, noc, y0,
                max([MCMCfcast, MCMCstorecoeff]),
                hyperpriors, priorcoef, Tcovid
            )
        LOGML[0] = logMLold

        # Initialize a key "mcmc" to store the draws of beta and sigma, if either MCMCstorecoeff or MCMCfcast is on
        if MCMCstorecoeff == 1 or MCMCfcast == 1:
            r['mcmc'] = {}

        # If MCMCstorecoeff is on, initialize beta and sigma matrices
        # burn in the first N draws, and retain the last M-N draws of the coefficients, variances and forecasts
        if MCMCstorecoeff == 1:
            r['mcmc']['beta'] = np.zeros((k, n, M - N))
            r['mcmc']['sigma'] = np.zeros((n, n, M - N))

        # If MCMCfcast is on, add the Dforecast matrix to the existing dictionary
        if MCMCfcast == 1:
            r['mcmc']['Dforecast'] = np.zeros((len(hz), n, M - N))

        count = 0

        for i in range(1, M):  # Start from 1 to M-1, because Python is 0-based
            if i == 1000 * (i // 1000):
                print(f'Now running the {i}th MCMC iteration (out of {M})')

            # Calculate the covariance matrix
            cov_matrix = HH * (const ** 2)

            # Check if negative eigenvalues are present, and if so, then regularize
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            if np.any(eigenvalues <= 0):
                # Set negative eigenvalues to 0
                eigenvalues[eigenvalues < 0] = 0
                # Reconstruct the covariance matrix using the modified eigenvalues
                # ensure that the matrix is symmetric and positive semi-definite
                cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Draw candidate value
            P[i, :] = np.random.multivariate_normal(mean=P[i - 1, :], cov=cov_matrix)

            logMLnew, betadrawnew, sigmadrawnew = logMLVAR_formcmc_covid(
                P[i, :].reshape(-1, 1),
                y, x, lags, T, n, b,
                MIN, MAX, SS, Vc, pos,
                mn, sur, noc, y0,
                max([MCMCfcast, MCMCstorecoeff]),
                hyperpriors, priorcoef, Tcovid
            )
            LOGML[i] = logMLnew

            if logMLnew > logMLold:
                logMLold = logMLnew
                count += 1
            else:
                if np.random.rand() < np.exp(logMLnew - logMLold):
                    logMLold = logMLnew
                    count += 1
                else:
                    P[i, :] = P[i - 1, :]
                    LOGML[i] = logMLold

                    if MCMCfcast == 1 or MCMCstorecoeff == 1:
                        _, betadrawnew, sigmadrawnew = logMLVAR_formcmc_covid(
                            P[i, :].reshape(-1, 1),
                            y, x, lags, T, n, b,
                            MIN, MAX, SS, Vc, pos,
                            mn, sur, noc, y0,
                            max([MCMCfcast, MCMCstorecoeff]),
                            hyperpriors, priorcoef, Tcovid
                        )

            if i > N and MCMCstorecoeff == 1:  # save the draws after burning in the initial N draws
                # Subtract 1 from (i - N) for zero-based indexing in python
                r['mcmc']['beta'][:, :, i - N] = betadrawnew
                r['mcmc']['sigma'][:, :, i - N] = sigmadrawnew

            if i > N and MCMCfcast == 1:
                Y = np.vstack([y, np.zeros((max(hz), n))])

                for tau in range(1, max(hz) + 1):
                    # select a potion of matrix Y: pick a subset of rows in
                    # reverse order: go backwards from T+tau-1 to T+tau-lags.
                    # then reshape the selected data into a column vector of
                    # size(k-1 x 1)
                    # The entire expression [1; ... ]': Creates a row vector xT
                    # that starts with 1 (for the intercept) and is followed by
                    # the reshaped data. It transposes the column vector into a row vector.

                    # Select the last 'lags' rows in reverse order and flatten the array
                    # Adjust indices for zero-based indexing
                    selected_Y = Y[T - tau - lags + 1:T - tau + 1][::-1].flatten()

                    # Reshape the selected array to have k-1 rows
                    # Note: The total number of elements must be equal to k-1
                    xT = np.concatenate(([1], selected_Y[:k - 1]))
                    if Tcovid is not None:
                        if T == Tcovid:  # if the current date (time) is the same as the covid date

                            #  set the scaling factor based on the value of tau
                            #  note that indexing in python starts from 0, instead of 1
                            #  scaling=P(i,2): If tau is 1, the scaling
                            #  factor is set to the third element of the ith row of P.

                            #  +(1+(P(i,3)-1)*P(i,4)^(tau-2)):
                            #  If tau is greater than or equal to 2,
                            #  the scaling factor is adjusted based on elements
                            #  4 and 5 of the ith row of P.
                            if tau == 1:
                                scaling = P[i, 2]  # Note: Indexing starts from 0 in Python
                            elif tau >= 2:
                                scaling = 1 + (P[i, 3] - 1) * P[i, 4] ** (tau - 2)
                        elif T > Tcovid:
                            scaling = 1 + (P[i, 3] - 1) * P[i, 4] ** (T - Tcovid + tau - 2)
                    else:
                        scaling = 1

                    # generate a vector "errors" by randomly sampling from a multivariate normal distribution
                    # with mean 0 and covariance matrix sigmadrawnew, then scale this vector
                    errors = np.random.multivariate_normal(mean=np.zeros(n), cov=sigmadrawnew) * scaling

                    # compute the value of Y at time T+tau -1
                    Y[T + tau - 1, :] = xT @ betadrawnew + errors  # Matrix multiplication

                # Y[T + hz - 1, :] selects the rows of Y corresponding to the indices T+hz
                # r['mcmc']['Dforecast'][:, :, i - N-1] refers to the (i-N)-th slice of the 3D array Dforecast
                # in the dictionary r['mcmc'].

                hz_array = np.array(hz)  # Ensure that hz is a NumPy array for vectorized operations

                # Update the forecast matrix for each element in hz
                r['mcmc']['Dforecast'][:, :, i - N] = Y[T + hz_array - 1, :]
        # Store draws of ML
        r['mcmc']['LOGML'] = LOGML[N:]

        # Store the draws of the hyperparameters

        #  Selects all elements of the first column from the N-th index to the end
        # retain the draws of lambda (after burning in N draws) in the key 'lambda' of the dictionary r['mcmc']
        r['mcmc']['lambda'] = P[N:, 0]  # std MN prior

        if Tcovid is not None:
            # Diagonal elements of the scale matrix of the IW prior on the residual variance
            # extract elements frm the N+1 row of matrix P to the last,
            #  and from the 2nd to (ncp + 1) -th columns of P
            r['mcmc']['eta'] = P[N:, 1:ncp + 1]

            if sur == 1:
                # std of sur prior
                #  extracts elements from the (N+1)-th row to the last row of the (ncp+2)-th column of P
                r['mcmc']['theta'] = P[N:, ncp + 1]

                if noc == 1:
                    # std of noc prior
                    # extracts elements from the (N+1)-th row to the last row of the (ncp+3)-th column of P
                    r['mcmc']['miu'] = P[N:, ncp + 2]

            elif sur == 0:
                if noc == 1:
                    # std of noc prior
                    r['mcmc']['miu'] = P[N:, ncp + 1]
        else:
            if sur == 1:
                # std of sur prior
                r['mcmc']['theta'] = P[N:, 1]
                if noc == 1:
                    # std of noc prior
                    r['mcmc']['miu'] = P[N:, 2]
            elif sur == 0:
                if noc == 1:
                    # std of noc prior
                    r['mcmc']['miu'] = P[N:, 1]

        if mn['alpha'] == 1:
            # Lag-decaying parameter of the MN prior
            # select all the retained draws of alpha (after burning in N draws) from (N+1)-th row
            # to the last row the last column of P (denote the last column of P by -1)
            r['mcmc']['alpha'] = P[N:, -1]

        # calculates the acceptance rate by comparing each element of the lambda array with its previous element
        # and computing the mean of these comparisons.
        r['mcmc']['ACCrate'] = np.mean(r['mcmc']['lambda'][1:] != r['mcmc']['lambda'][:-1])

    return r


def bvarIrfs(beta, sigma, nshock, hmax):
    """
   Computes Impulse Response Functions (IRFs) using Cholesky ordering.

   This function calculates IRFs up to a specified horizon based on the
   provided VAR model coefficients and covariance matrix of the error terms.
   The IRFs are generated using a Cholesky decomposition of the covariance matrix
   to apply a shock in a specified position.

   Args:
       beta (numpy.ndarray): Coefficient matrix of the VAR model. Size: [k, n],
           where k is the number of coefficients (including the intercept)
           and n is the number of variables in the VAR model.
       sigma (numpy.ndarray): Covariance matrix of the VAR model residuals. Size: [n, n],
           where n is the number of variables in the VAR model.
       nshock (int): Position of the variable to which the shock is applied.
           The position corresponds to the column number in 'beta'.
       hmax (int): Maximum horizon for the impulse response.

   Returns:
       numpy.ndarray: Computed impulse response functions. Size: [hmax, n],
           where hmax is the maximum horizon for the impulse response and
           n is the number of variables in the VAR model.

   Example:
       >>> beta = np.array([[0.2, 0.3], [0.1, 0.4], [0.5, 0.1], [0.3, 0.2], [0.1, 0.3]])
       >>> sigma = np.array([[0.5, 0.1], [0.1, 0.6]])
       >>> nshock = 1
       >>> hmax = 5
       >>> result_irf = bvar_irfs(beta, sigma, nshock, hmax)
       >>> print(result_irf)
       """
    k, n = beta.shape

    lags = int((k - 1) / n)

    # IRFs at the posterior mode
    cholVCM = np.linalg.cholesky(sigma)
    Y = np.zeros((lags + hmax, n))
    _in = lags
    vecshock = np.zeros((n, 1))
    vecshock[nshock - 1, 0] = 1
    for tau in range(1, hmax + 1):
        index_range = np.arange(_in + tau - 2, _in + tau - lags - 2, -1)
        # temp = Y[index_range, :].T
        xT = np.reshape(Y[index_range, :].T, (k - 1, 1), order='F').T
        Y[_in + tau - 1, :] = xT @ beta[1:, :] + (tau == 1) * (cholVCM @ vecshock).T

    irf = Y[_in:, :]

    return irf



def check_params(par):
    """
    Check the parameters for acceptability and fill in defaults for unspecified ones.

    Args:
        par (dict): A dictionary containing the parameters.

    Returns:
        dict: The dictionary with checked and updated parameters.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> par = {
            'DerivativeOrder': 2,
            'MethodOrder': 4,
            'RombergTerms': 2,is it
            'MaxStep': 100,
            'StepRatio': 2,
            'NominalStep': None,
            'Vectorized': 'no',
            'FixedStep': None,
            'Style': 'central'
        }

        >>> check_params(par)
    """

    # DerivativeOrder == 1 by default
    if par.get('DerivativeOrder') is None:
        par['DerivativeOrder'] = 1
    else:
        if not (1 <= par['DerivativeOrder'] <= 4):
            raise ValueError("DerivativeOrder must be one of [1, 2, 3, 4].")

    # MethodOrder == 2 by default
    if par.get('MethodOrder') is None:
        par['MethodOrder'] = 2
    else:
        if not (1 <= par['MethodOrder'] <= 4):
            raise ValueError("MethodOrder must be one of [1, 2, 3, 4].")
        if par['MethodOrder'] in [1, 3] and par.get('Style', '')[0].lower() == 'c':
            raise ValueError("MethodOrder==1 or 3 is not possible with central difference methods")

    # Style is 'central' by default
    valid_styles = ['central', 'forward', 'backward']
    if par.get('Style') is None:
        par['Style'] = 'central'
    else:
        if par['Style'].lower() not in valid_styles:
            raise ValueError(f"Invalid Style: {par['Style']}")

    # Vectorized == 'yes' by default
    if par.get('Vectorized') is None:
        par['Vectorized'] = 'yes'
    else:
        if par['Vectorized'].lower() not in ['yes', 'no']:
            raise ValueError(f"Invalid Vectorized: {par['Vectorized']}")

    # RombergTerms == 2 by default
    if par.get('RombergTerms') is None:
        par['RombergTerms'] = 2
    else:
        if not (0 <= par['RombergTerms'] <= 3):
            raise ValueError("RombergTerms must be one of [0, 1, 2, 3].")

    # FixedStep is None by default and must be > 0 if specified
    if par.get('FixedStep') is not None:
        if par['FixedStep'] <= 0:
            raise ValueError("FixedStep must be > 0.")

    # MaxStep == 10 by default and must be > 0 if specified
    if par.get('MaxStep') is None:
        par['MaxStep'] = 10
    else:
        if par['MaxStep'] <= 0:
            raise ValueError("MaxStep must be > 0.")

    return par


def cholred(S):
    """
    Compute the reduced Cholesky decomposition of a matrix.

    Args:
        S (array_like): Input matrix (n x n).

    Returns:
        array_like: Reduced Cholesky decomposition (n x n).
    """
    # Ensure symmetric matrix (required for eigh)

    # Compute eigenvalues (d) and eigenvectors (v)
    d, v = np.linalg.eigh((S + S.T) / 2)

    # Ensure eigenvalues are real numbers
    d = d[:, np.newaxis]
    d = np.real(d)

    # Scale value similar to the one in the MATLAB code
    scale = np.mean(np.diag(S)) * 1e-12

    # Find indices of eigenvalues greater than scale
    J = d > scale
    J = J.flatten()

    # Prepare a zero matrix of size S
    C = np.zeros(S.shape)

    # Compute the modified Cholesky decomposition
    C[J, :] = (v[:, J] @ np.sqrt(np.diagflat(d[J]))).T

    return C


def cols(x):
    """
    Return the number of columns in a matrix x.

    Args:
        x (array_like): Input matrix.

    Returns:
        int: Number of columns in x.
    """
    return x.shape[1]


def computeIrfs(beta, G, nshock, hmax):
    """
    Computes Impulse Response Functions (IRFs) up to a specified horizon.

    This function calculates IRFs based on the provided VAR model coefficients
    and a matrix G, which represent the effect of shocks in the model.

    Args:
        beta (numpy.ndarray): Coefficient matrix of the VAR model. Size: [k, n],
            where k is the number of coefficients (including the intercept)
            and n is the number of variables in the VAR model.
        G (numpy.ndarray): Matrix representing the effect of shocks in the model. Size: [n, n],
            where n is the number of variables in the VAR model.
        nshock (int): Position of the variable to which the shock is applied.
            The position corresponds to the column number in 'beta'.
        hmax (int): Maximum horizon for the impulse response.

    Returns:
        numpy.ndarray: Computed impulse response functions. Size: [hmax, n],
            where hmax is the maximum horizon for the impulse response and
            n is the number of variables in the VAR model.

    Example:
        >>> beta = np.array([[0.2, 0.3], [0.1, 0.4], ...]) # Replace with actual data
        >>> G = np.array([[...], [...]]) # Replace with actual data
        >>> nshock = 1
        >>> hmax = 24
        >>> result_irf = computeIrfs(beta, G, nshock, hmax)
        >>> print(result_irf)
    """
    k, n = beta.shape
    lags = int((k - 1) / n)

    # Initialize Y for storing IRFs
    Y = np.zeros((lags + hmax, n))
    _in = lags
    vecshock = np.zeros((n, 1))
    vecshock[nshock - 1, 0] = 1

    for tau in range(1, hmax + 1):
        index_range = np.arange(_in + tau - 2, _in + tau - lags - 2, -1)
        xT = np.reshape(Y[index_range, :].T, (k - 1, 1), order='F').T
        Y[_in + tau - 1, :] = xT @ beta[1:, :] + (tau == 1) * (G @ vecshock).T

    irf = Y[_in:, :]

    return irf


def csminit(fcn, x0, f0, g0, badg, H0, *varargin):
    """Performs a line search to find a suitable step size for optimization.

        This function conducts a line search to find a suitable step size (lambda)
        in the descent direction for optimization. It uses a combination of growing
        and shrinking strategies, adjusting lambda until certain improvement criteria are met.

        Args:
            fcn (callable): Function handle to the objective function.
            x0 (numpy.ndarray): Initial point, shape (n,).
            f0 (float): Function value at the initial point.
            g0 (numpy.ndarray): Gradient at the initial point, shape (n,).
            badg (int): Flag indicating if the gradient is bad (potentially inaccurate), scalar.
            H0 (numpy.ndarray): Approximate inverse Hessian or Hessian matrix at the initial point, shape (n, n).
            *varargin: Additional arguments passed to the target function.

        Returns:
            float: Best function value found during the line search.
            numpy.ndarray: Point corresponding to the best function value, shape (n,).
            int: Number of function evaluations.
            int: Return code indicating the termination condition.

        Note:
            This function is typically used within iterative optimization algorithms,
            such as quasi-Newton methods, to ensure that the step size is chosen to
            provide sufficient improvement in the objective function value.
        """

    # Constants
    ANGLE = 0.005  # Angle for line search
    THETA = 0.3  # Threshold for line search, 0 < THETA < 0.5
    FCHANGE = 1000  # Scaling factor for forceful changes
    MINLAMB = 1e-9  # Minimum value of the step size lambda
    MINDFAC = 0.01  # Minimum factor for changing the step size

    # Initialize function evaluation counter
    fcount = 0

    # Initialize step size lambda
    lambda_ = 1  # Used lambda_ to avoid conflict with Python's built-in lambda keyword

    # Initialize the current best estimate of x and its corresponding function value
    xhat = x0  # Initial point
    f = f0  # Function value at initial point
    fhat = f0  # Best function value, initialized to function value at initial point

    # Compute the norm of the initial gradient
    g = g0  # Initial gradient
    gnorm = np.linalg.norm(g)  # L2 norm of the gradient

    # Check if the gradient norm is below the threshold and not flagged as "bad"
    if (gnorm < 1e-12) and (not badg):
        retcode = 1  # Return code for gradient convergence
        dxnorm = 0
    else:
        # Compute the direction of descent (dx) using inverse Hessian (Gauss-Newton step)
        dx = -np.dot(H0, g)  # dx is (n,), H0 is (n, n), g is (n,)
        dxnorm = np.linalg.norm(dx)  # L2 norm of dx

        # Check for near-singular Hessian problem and rescale if needed
        if dxnorm > 1e12:
            print('Near-singular H problem.')
            dx = dx * FCHANGE / dxnorm

        # Compute the predicted directional derivative
        dfhat = np.matmul(dx.T, g0)

        # If gradient is not flagged as "bad," test for alignment of dx with gradient and correct if necessary
        if not badg:
            a = -dfhat / (gnorm * dxnorm)
            if a < ANGLE:
                # Correct the alignment if the angle is too low
                dx = dx - (ANGLE * dxnorm / gnorm + dfhat / (gnorm * gnorm)) * g
                # Rescale to keep the scale invariant to the angle correction
                dx = dx * dxnorm / np.linalg.norm(dx)
                dfhat = np.dot(dx.T, g)
                print(f'Correct for low angle: {a}')

        # Display the predicted improvement
        print(f'Predicted improvement: {-dfhat[0, 0] / 2:.9f}')

        # Initialization of variables for adjusting the length of step (lambda)
        # in the following loop
        done = False  # Flag to indicate if the step adjustment is done
        factor = 3  # Initial factor for changing lambda
        shrink = 1  # Flag to indicate if lambda should be shrunk
        lambdaMin = 0  # Minimum boundary for lambda
        lambdaMax = float('inf')  # Maximum boundary for lambda
        lambdaPeak = 0  # Peak value for lambda
        fPeak = f0  # Function value at the peak lambda
        lambdahat = 0  # Best lambda value

        # Start of loop to adjust step size (lambda) for line search
        while not done:
            # Adjust dx according to the size of x0
            if x0.shape[1] > 1:
                dxtest = x0 + dx.T * lambda_
            else:
                dxtest = x0 + dx * lambda_  # dxtest, x0, dx are 7x1 arrays, lambda is scalar

            # Evaluate the function at the new test point
            f = float(fcn(dxtest, *varargin)[0])  # f is scalar

            # Display the current lambda and function value
            print(f'lambda = {lambda_ :10.5g}; f = {f:20.7f}')

            # Update the best function value and corresponding x if improvement found
            if f < fhat:
                fhat = f
                xhat = dxtest
                lambdahat = lambda_

            # Increment function evaluation counter
            fcount += 1  # fcount is a scalar
            dfhat_value = dfhat.item()  # Extract the single value from the 2D array

            # Determine if the improvement signals are triggered to shrink or
            # grow lambda: shrinkSignal and growSignal are boolean variables
            shrinkSignal = (((not badg) and (f0 - f < max([-THETA * dfhat_value * lambda_, 0])))
                            or (badg and (f0 - f < 0)))
            growSignal = (not badg) and (lambda_ > 0) and (f0 - f > -(1 - THETA) * dfhat_value * lambda_)

            # Conditions to shrink lambda_
            if shrinkSignal and ((lambda_ > lambdaPeak) or (lambda_ < 0)):
                if (lambda_ > 0) and ((not shrink) or (lambda_ / factor <= lambdaPeak)):
                    shrink = True
                    factor = factor ** 0.6

                    while lambda_ / factor <= lambdaPeak:
                        factor = factor ** 0.6

                    if abs(factor - 1) < MINDFAC:
                        if abs(lambda_) < 4:
                            retcode = 2
                        else:
                            retcode = 7
                        done = True

                if (lambda_ < lambdaMax) and (lambda_ > lambdaPeak):
                    lambdaMax = lambda_

                lambda_ = lambda_ / factor

                if abs(lambda_) < MINLAMB:
                    if (lambda_ > 0) and (f0 <= fhat):
                        lambda_ = -lambda_ * (factor ** 6)
                    else:
                        retcode = 6 if lambda_ < 0 else 3
                        done = True

            # Conditions to grow lambda_
            elif (growSignal and lambda_ > 0) or (shrinkSignal and (lambda_ <= lambdaPeak) and (lambda_ > 0)):
                if shrink:
                    shrink = False
                    factor = factor ** 0.6

                    if abs(factor - 1) < MINDFAC:
                        retcode = 4 if abs(lambda_) < 4 else 7
                        done = True

                if (f < fPeak) and (lambda_ > 0):
                    fPeak = f
                    lambdaPeak = lambda_
                    if lambdaMax <= lambdaPeak:
                        lambdaMax = lambdaPeak * factor * factor

                lambda_ = lambda_ * factor

                if abs(lambda_) > 1e20:
                    retcode = 5
                    done = True

            else:
                done = True
                retcode = 7 if factor < 1.2 else 0

        # Display the norm of the descent direction dx
        print(f'Norm of dx {dxnorm:.5g}')

    return fhat, xhat, fcount, retcode


def csminwel(fcn: Callable, x0: np.ndarray, H0: np.ndarray,
             grad: Optional[Union[Callable, np.ndarray]], crit: float,
             nit: int, *varargin: Any) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Minimizes a function using a quasi-Newton method.

    Args:
        fcn (Callable): Objective function to be minimized.
        x0 (np.ndarray): Initial value of the parameter vector.
        H0 (np.ndarray): Initial value for the inverse Hessian. Must be positive definite.
        grad (Union[Callable, np.ndarray], optional): Either a function that calculates the gradient,
                                                      or a null array. If null, the program calculates a numerical gradient.
        crit (float): Convergence criterion. Iteration will cease when it's impossible to improve the function value by more than crit.
        nit (int): Maximum number of iterations.
        varargin (Any): Additional parameters that get handed off to `fcn` each time it is called.

    Returns:
        - Tuple[float, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
            - fh (float): The value of the function at the minimum.
            - xh (np.ndarray): The value of the parameters that minimize the function.
            - gh (np.ndarray): The gradient of the function at the minimum.
            - H (np.ndarray): The estimated inverse Hessian at the minimum.
            - itct (int): The total number of iterations performed.
            - fcount (int): The total number of function evaluations.
            - retcodeh (int): Return code that provides information about why the algorithm terminated.
    """

    # Get the dimensions of the initial guess x0
    nx, no = x0.shape
    nx = max(nx, no)  # Maximum of the dimensions as the number of variables

    # Verbose mode for displaying additional information
    Verbose = 1

    # Check if the gradient is provided or if it needs to be numerically computed
    NumGrad = grad is None  # NumGrad will be True if grad is None

    # Initialize flags and counters
    done = 0  # Done flag (0 for not done, 1 for done)
    itct = 0  # Iteration counter
    fcount = 0  # Function call counter

    # Evaluate the function at the initial guess
    f0 = float(fcn(x0, *varargin)[0])  # f0 is scalar

    # Check for a bad initial guess
    if f0 > 1e50:
        print("Bad initial parameter.")
        return None  # If the initial guess is bad, exit the function

    # If the gradient is not provided, compute it numerically
    if NumGrad:
        if grad is None or len(grad) == 0:
            g, badg = numgrad(fcn, x0, *varargin)  # g should be a NumPy array, badg is a flag
        else:
            # Check if any element of the provided gradient is zero
            badg = np.any(grad == 0)
            g = grad
    else:
        # If the gradient function is provided, evaluate it at the initial guess
        g, badg = grad(x0, *varargin)  # Here, grad is a callable function

    # Initialize the following variables
    retcode3 = 101  # Return code (not used in this portion of the code)
    x = x0.copy()  # Current solution x with the initial guess
    f = f0  # Current function value f with the initial function value
    H = H0.copy()  # Hessian (or its approximation) with the provided initial value
    cliff = 0  # Cliff flag (used to handle special cases in the optimization)

    # Start the main loop that continues until the 'done' flag is set to 1
    while not done:
        # Initialize empty gradient vectors for special cases in optimization
        g1 = []
        g2 = []
        g3 = []

        # Display debugging information
        print("-----------------")
        print("-----------------")

        print(f"f at the beginning of new iteration, {f:20.10f}")
        # -----------Comment out this line if the x vector is long----------------
        print("x = ", end="")
        print(" ".join([f"{xi:15.8g}" for xi in x.flatten()]))
        # -------------------------

        # Increment the iteration counter
        itct += 1

        # Call the csminit function to iterate using optimization algorithm
        # f1, fc, and retcode1 are 1x1 double or scalar, x1 is an array
        f1, x1, fc, retcode1 = csminit(fcn, x, f, g, badg, H, *varargin)

        # Increment the function call counter by the number of function calls made in csminit
        fcount += fc  # fcount is a scalar value

        # Check the return code from csminit
        if retcode1 != 1:
            if retcode1 == 2 or retcode1 == 4:
                # If retcode1 is 2 or 4, set flags indicating a "wall"
                wall1 = 1
                badg1 = 1
            else:
                # Compute the gradient at x1, either numerically or using the provided gradient function
                if NumGrad:
                    g1, badg1 = numgrad(fcn, x1, *varargin)
                else:
                    g1, badg1 = grad(x1, *varargin)  # g1 is a vector, and badg is a scalar

                # If the gradient is "bad", set the wall1 flag
                wall1 = badg1  # wall1 is scalar (or 1x1 double)

            # Check if we see a wall (special condition), and if the Hessian matrix is not 1D
            if wall1 and len(H.shape) > 1:  # Check special condition and dimension of Hessian
                # Perturb the Hessian matrix by adding random noise along its diagonal
                Hcliff = H + np.diag(np.diag(H) * np.random.rand(nx))
                print('Cliff. Perturbing search direction.')

                # Call csminit function to iterate using optimization algorithm
                f2, x2, fc, retcode2 = csminit(fcn, x, f, g, badg, Hcliff, *varargin)

                # Increment function call counter
                fcount += fc

                # Check if the new function value is less than the current one
                if f2 < f:
                    if retcode2 == 2 or retcode2 == 4:
                        wall2 = 1
                        badg2 = 1  # Set flags for special condition
                    else:
                        # Compute gradient at new x2
                        if NumGrad:
                            g2, badg2 = numgrad(fcn, x2, *varargin)
                        else:
                            g2, badg2 = grad(x2, *varargin)

                        wall2 = badg2

                    # Check if we hit the wall again
                    if wall2:
                        print("Cliff again. Try traversing")
                        if np.linalg.norm(x2 - x1) < 1e-13:
                            f3, x3, badg3, retcode3 = f, x, 1, 101
                        else:
                            # Compute the gradient based on the difference between f2 and f1
                            # and the distance between x2 and x1
                            gcliff = ((f2 - f1) / ((np.linalg.norm(x2 - x1)) ** 2)) * (x2 - x1)

                            if x0.shape[1] > 1:
                                gcliff = gcliff.T

                            # Call csminit with the computed gradient
                            f3, x3, fc, retcode3 = csminit(fcn, x, f, gcliff, 0, np.eye(nx),
                                                           *varargin)

                            # Increment function call counter
                            fcount += fc

                            # Compute gradient at x3 and check for special conditions
                            if retcode3 == 2 or retcode3 == 4:
                                wall3, badg3 = 1, 1
                            else:
                                if NumGrad:
                                    g3, badg3 = numgrad(fcn, x3, *varargin)
                                else:
                                    g3, badg3 = grad(x3, *varargin)

                                wall3 = badg3
                    else:
                        f3, x3, badg3, retcode3 = f, x, 1, 101
                else:
                    f3, x3, badg3, retcode3 = f, x, 1, 101
            else:
                # Normal iteration, no walls, or else 1D, else finished here
                f2, f3, badg2, badg3, retcode2, retcode3 = f, f, 1, 1, 101, 101

        else:
            # If retcode1 is 1, and we didn't encounter special conditions, keep previous values
            f2, f3, f1, retcode2, retcode3 = f, f, f, retcode1, retcode1

        # Determine the optimal gh, xh, and other related variables
        if f3 < f - crit and badg3 == 0:
            ih, fh = 3, f3
            xh, gh, badgh, retcodeh = x3, g3, badg3, retcode3
        elif f2 < f - crit and badg2 == 0:
            ih, fh = 2, f2
            xh, gh, badgh, retcodeh = x2, g2, badg2, retcode2
        elif f1 < f - crit and badg1 == 0:
            ih, fh = 1, f1
            xh, gh, badgh, retcodeh = x1, g1, badg1, retcode1
        else:
            # find the minimum among the function values
            fh = min(f1, f2, f3)
            ih = np.argmin([f1, f2, f3]) + 1  # +1 because MATLAB is 1-based indexing, Python is 0-based
            print(f"ih = {ih}")

            if ih == 1:
                xh = x1
            elif ih == 2:
                xh = x2
            elif ih == 3:
                xh = x3

            retcodei = [retcode1, retcode2, retcode3]
            # -1 because python has 0-based indexing, whereas Matlab has 1-based indexing
            retcodeh = retcodei[ih - 1]

            # Check if 'gh' exists in the local namespace
            if 'gh' in locals():
                nogh = len(gh) == 0  # Check if gh is empty
            else:
                nogh = True  # gh does not exist

            if nogh:
                if NumGrad:
                    gh, badgh = numgrad(fcn, xh, *varargin)
                else:
                    gh, badgh = grad(xh, *varargin)

            badgh = 1

        # Check if the algorithm is stuck (no significant improvement)
        stuck = abs(fh - f) < crit

        if (not badg) and (not badgh) and (not stuck):
            # Update the Hessian matrix using BFGS formula
            H = bfgsi(H, gh - g, xh - x)

        # Check termination conditions
        if Verbose:
            print("----")
            print(f"Improvement on iteration {itct} = {f - fh}")

            if itct > nit:
                print("iteration count termination")
                done = 1
            elif stuck:
                print("improvement < crit termination")
                done = 1

            rc = retcodeh

            # Print the various conditions that might have occurred
            if rc == 1:
                print("zero gradient")
            elif rc == 6:
                print("smallest step still improving too slow, reversed gradient")
            elif rc == 5:
                print("largest step still improving too fast")
            elif rc in (4, 2):
                print("back and forth on step length never finished")
            elif rc == 3:
                print("smallest step still improving too slow")
            elif rc == 7:
                print("warning: possible inaccuracy in H matrix")

        # Update the main variables for the next iteration or the final result
        f, x, g, badg = fh, xh, gh, badgh

    return fh, xh, gh, H, itct, fcount, retcodeh


def csolve(FUN, x, gradfun, crit, itmax, *varargin):
    """
        Finds the solution to a system of nonlinear equations using iterative methods.

        Args:
            FUN (function): The function representing the system of equations. Accepts a vector or matrix `x`.
            x (list or np.array): Initial guess for the solution.
            gradfun (function or None): Function to evaluate the gradient matrix. If None, a numerical gradient is used.
            crit (float): Tolerance level; if the sum of absolute values returned by FUN is less than this, the equation is considered solved.
            itmax (int): Maximum number of iterations.
            *varargin: Additional arguments passed on to FUN and gradfun.

        Returns:
            tuple: A tuple containing:
                - x (list or np.array): Solution to the system of equations.
                - rc (int): Return code indicating the status of the solution.
                    - 0: Normal solution.
                    - 1, 3: No solution (likely a numerical problem or discontinuity).
                    - 4: Termination due to reaching the maximum number of iterations.

        Example:
            def fun(x):
                return [x[0]**2 + x[1] - 3, x[0] + x[1]**2 - 3]

            x0 = [1, 1]
            crit = 1e-6
            itmax = 100

            x, rc = csolve(fun, x0, None, crit, itmax)
        """
    delta = 1e-6
    alpha = 1e-3
    verbose = 1
    analyticg = 1 if gradfun else 0
    nv = len(x)
    tvec = delta * np.eye(nv)
    done = 0
    f0 = FUN(x, *varargin) if varargin else FUN(x)
    af0 = np.sum(np.abs(f0))
    af00 = af0
    itct = 0
    while not done:
        if itct > 3 and af00 - af0 < crit * max(1, af0) and itct % 2 == 1:
            randomize = 1
        else:
            if not analyticg:
                grad = ((FUN(x * np.ones((1, nv)) + tvec, *varargin) - f0 * np.ones((1, nv))) /
                        delta) if varargin else (FUN(x * np.ones((1, nv)) + tvec) - f0 * np.ones((1, nv))) / delta
            else:
                grad = gradfun(x, *varargin)
            if np.linalg.cond(grad) < 1e-12:
                grad += tvec
            dx0 = -np.linalg.solve(grad, f0)
            randomize = 0
        if randomize:
            if verbose:
                print("\n Random Search")
            dx0 = np.linalg.norm(x) / np.random.randn(*x.shape)
        lambda_ = 1
        lambdamin = 1
        fmin = f0
        xmin = x
        afmin = af0
        dxSize = np.linalg.norm(dx0)
        factor = 0.6
        shrink = 1
        subDone = 0
        while not subDone:
            dx = lambda_ * dx0
            f = FUN(x + dx, *varargin) if varargin else FUN(x + dx)
            af = np.sum(np.abs(f))
            if af < afmin:
                afmin = af
                fmin = f
                lambdamin = lambda_
                xmin = x + dx
            if ((lambda_ > 0) and (af0 - af < alpha * lambda_ * af0)) or ((lambda_ < 0) and (af0 - af < 0)):
                if not shrink:
                    factor = factor ** 0.6
                    shrink = 1
                if abs(lambda_ * (1 - factor)) * dxSize > 0.1 * delta:
                    lambda_ = factor * lambda_
                elif (lambda_ > 0) and (factor == 0.6):
                    lambda_ = -0.3
                else:
                    subDone = 1
                    if lambda_ > 0:
                        if factor == 0.6:
                            rc = 2
                        else:
                            rc = 1
                    else:
                        rc = 3
            elif (lambda_ > 0) and (af - af0 > (1 - alpha) * lambda_ * af0):
                if shrink:
                    factor = factor ** 0.6
                    shrink = 0
                lambda_ = lambda_ / factor
            else:
                subDone = 1
                rc = 0
        itct += 1
        if verbose:
            print(f'\nitct {itct}, af {afmin}, lambda {lambdamin}, rc {rc}')
            print(f'   x  {xmin}')
            print(f'   f  {fmin}')
        x = xmin
        f0 = fmin
        af00 = af0
        af0 = afmin
        if itct >= itmax:
            done = 1
            rc = 4
        elif af0 < crit:
            done = 1
            rc = 0
    return x, rc


def derivest(fun, x0, varargin):
    """
    Estimate the n'th derivative of fun at x0 and provide an error estimate.

    Args:
        fun (callable): Function to differentiate. It should be vectorized.
        x0 (float or np.ndarray): Point(s) at which to differentiate fun.

    Keyword Args:
        DerivativeOrder (int): Specifies the derivative order estimated. Default is 1.
        MethodOrder (int): Specifies the order of the basic method used for the estimation. Default is 4.
        Style (str): Specifies the style of the basic method used for the estimation ('central', 'forward', 'backward'). Default is 'central'.
        RombergTerms (int): Number of Romberg terms for extrapolation. Default is 2.
        FixedStep (float): Fixed step size. Default is None.
        MaxStep (float): Specifies the maximum excursion from x0 that will be allowed. Default is 100.
        StepRatio (float): The ratio used between sequential steps. Default is 2.0000001.
        Vectorized (str): Whether the function is vectorized or not ('yes', 'no'). Default is 'yes'.

    Returns:
        der (float or np.ndarray): Derivative estimate for each element of x0.
        errest (float or np.ndarray): 95% uncertainty estimate of the derivative.
        finaldelta (float or np.ndarray): The final overall stepsize chosen.

    """
    if isinstance(x0, np.ndarray) and x0.shape == (1,):
        x0 = float(x0[0])

    # Define default parameters
    par = {
        'DerivativeOrder': 1,
        'MethodOrder': 4,
        'Style': 'central',
        'RombergTerms': 2,
        'FixedStep': None,
        'MaxStep': 100,
        'StepRatio': 2.0000001,  # To avoid integer multiples of the initial point for periodic functions
        'NominalStep': None,
        'Vectorized': 'yes'
    }

    # Calculate the number of keyword arguments
    na = len(varargin)

    # Check if kwargs has an even number of elements (it should, if it's a dictionary)
    if na % 2 == 1:
        raise ValueError("Property/value pairs must come as PAIRS of arguments.")

    # If kwargs is not empty, parse the property-value pairs
    if na > 0:
        par = parse_pv_pairs(par, varargin)

    # Check and possibly modify the parameters in 'par'
    par = check_params(par)

    if fun is None:
        # This could be a call to a function that prints the help message
        print("Help: Information about how to use 'derivest'")
        return
    elif callable(fun):
        # 'fun' is already a callable function, so no action is needed
        pass
    elif isinstance(fun, str):
        # 'fun' is a string, so attempt to convert it to a function
        fun = eval(fun)
    else:
        raise ValueError("'fun' must be a callable function or a string representing a function.")

    # No default for x0
    if x0 is None:
        raise ValueError('x0 was not supplied.')

    # Set the NominalStep in the parameter dictionary
    par['NominalStep'] = max(x0, 0.02)

    # Check if a single point was supplied
    x0 = np.array([[x0]])
    nx0 = np.shape(x0)
    n = np.prod(nx0)

    # Set the steps to use
    if par['FixedStep'] is None:
        # Basic sequence of steps, relative to a stepsize of 1
        delta = (par['MaxStep'] * (par['StepRatio'] ** np.arange(0, -26, -1))).reshape(-1, 1)
        ndel = len(delta)
    else:
        # Fixed, user-supplied absolute sequence of steps
        ndel = 3 + np.ceil(par['DerivativeOrder'] / 2) + par['MethodOrder'] + par['RombergTerms']
        if par['Style'][0].lower() == 'c':
            ndel = ndel - 2
        delta = par['FixedStep'] * (par['StepRatio'] ** -np.arange(0, ndel))

    # Convert ndel to integer as it may be a float due to np.ceil
    ndel = int(ndel)

    # Generate finite differencing rule in advance.
    # The rule is for a nominal unit step size and will
    # be scaled later to reflect the local step size.

    fdarule = 1  # Initialize fdarule
    if par['Style'].lower() == 'central':
        # For central rules, we will reduce the load by an
        # even or odd transformation as appropriate.
        if par['MethodOrder'] == 2:
            if par['DerivativeOrder'] == 1:
                # The odd transformation did all the work
                fdarule = 1
            elif par['DerivativeOrder'] == 2:
                # The even transformation did all the work
                fdarule = 2
            elif par['DerivativeOrder'] == 3:
                # The odd transformation did most of the work, but
                # we need to kill off the linear term
                fdarule = np.array([0, 1]) @ np.linalg.inv(fdamat(par['StepRatio'], 1, 2))
            elif par['DerivativeOrder'] == 4:
                # The even transformation did most of the work, but
                # we need to kill off the quadratic term
                fdarule = np.array([0, 1]) @ np.linalg.inv(fdamat(par['StepRatio'], 2, 2))
        else:
            # A 4th order method. We've already ruled out the 1st
            # order methods since these are central rules.
            if par['DerivativeOrder'] == 1:
                # The odd transformation did most of the work, but
                # we need to kill off the cubic term
                fdarule = np.array([1, 0]) @ np.linalg.inv(fdamat(par['StepRatio'], 1, 2))
            elif par['DerivativeOrder'] == 2:
                # The even transformation did most of the work, but
                # we need to kill off the quartic term
                fdarule = np.array([1, 0]) @ np.linalg.inv(fdamat(par['StepRatio'], 2, 2))
            elif par['DerivativeOrder'] == 3:
                # The odd transformation did much of the work, but
                # we need to kill off the linear & quintic terms
                fdarule = np.array([0, 1, 0]) @ np.linalg.inv(fdamat(par['StepRatio'], 1, 3))
            elif par['DerivativeOrder'] == 4:
                # The even transformation did much of the work, but
                # we need to kill off the quadratic and 6th order terms
                fdarule = np.array([0, 1, 0]) @ np.linalg.inv(fdamat(par['StepRatio'], 2, 3))
    elif par['Style'] in ['forward', 'backward']:
        # These two cases are identical, except at the very end,
        # where a sign will be introduced.

        # No odd/even transformation, but we already dropped
        # off the constant term
        if par['MethodOrder'] == 1:
            if par['DerivativeOrder'] == 1:
                # An easy one
                fdarule = 1
            else:
                # 2:4
                v = np.zeros(par['DerivativeOrder'])
                v[par['DerivativeOrder'] - 1] = 1
                fdarule = v / fdamat(par['StepRatio'], 0, par['DerivativeOrder'])
        else:
            # par['MethodOrder'] methods drop off the lower-order terms,
            # plus terms directly above DerivativeOrder
            v = np.zeros(par['DerivativeOrder'] + par['MethodOrder'] - 1)
            v[par['DerivativeOrder'] - 1] = 1
            fdarule = v / fdamat(par['StepRatio'], 0, par['DerivativeOrder'] + par['MethodOrder'] - 1)

        # Correct the sign for the 'backward' rule
        if par['Style'][0] == 'b':
            fdarule = -fdarule

    # check the type of fdarule, then decide whether to wrap in a NumPy array
    if isinstance(fdarule, (int, float)):
        fdarule = np.array([fdarule])
    elif isinstance(fdarule, np.ndarray):
        pass  # No need to wrap it again
    else:
        raise TypeError("Unsupported type for fdarule")

    # Number of elements in fdarule
    nfda = len(fdarule)

    # Ensure x0 is a 1D array
    x0 = np.ravel(x0)

    # Initialize f_x0 to have the same length as x0
    f_x0 = np.zeros_like(x0)

    # Will we need fun(x0)?
    if (par['DerivativeOrder'] % 2 == 0) or (not par['Style'].lower().startswith('central')):
        if par['Vectorized'].lower() == 'yes':
            f_x0 = fun(x0)
        else:
            # Not vectorized, so iterate with an integer index
            for j in range(len(x0)):
                val = x0[j]
                result = fun(val)

                if isinstance(result, tuple):
                    if len(result) > 0:
                        f_x0[j] = result[0]
                    else:
                        raise ValueError("Function returned an empty tuple.")
                else:
                    # If result is not a tuple, assign the value directly
                    f_x0[j] = result

    else:
        f_x0 = None

    # Initialize output arrays.
    der = np.zeros(nx0)
    errest = np.zeros(nx0)
    finaldelta = np.zeros(nx0)

    # Loop over the elements of x0
    for i in range(n):
        x0i = x0
        h = par['NominalStep']

        # A central, forward or backwards differencing rule?
        if par['Style'][0].lower() == 'c':
            # A central rule, so we will need to evaluate symmetrically around x0i
            if par['Vectorized'].lower() == 'yes':
                f_plusdel = fun(x0i + h * delta)
                f_minusdel = fun(x0i - h * delta)
            else:
                # Not vectorized, so loop
                f_minusdel = np.zeros_like(delta)
                f_plusdel = np.zeros_like(delta)

                for j, val in enumerate(delta):
                    # For f_plusdel
                    output_plus = fun(x0i + h * val)
                    # Check if output is a tuple with a non-None first element that's a list or np.ndarray
                    if isinstance(output_plus, tuple) and output_plus[0] is not None and \
                            isinstance(output_plus[0], (list, np.ndarray)):
                        f_plusdel[j] = output_plus[0][0]
                    elif np.ndim(output_plus) > 0:
                        f_plusdel[j] = output_plus[0]
                    else:
                        f_plusdel[j] = output_plus

                    # For f_minusdel
                    output_minus = fun(x0i - h * val)
                    # Check if output is a tuple with a non-None first element that's a list or np.ndarray
                    if isinstance(output_minus, tuple) and output_minus[0] is not None and \
                            isinstance(output_minus[0], (list, np.ndarray)):
                        f_minusdel[j] = output_minus[0][0]
                    elif np.ndim(output_minus) > 0:
                        f_minusdel[j] = output_minus[0]
                    else:
                        f_minusdel[j] = output_minus

            if par['DerivativeOrder'] in [1, 3]:
                # Odd transformation
                f_del = (f_plusdel - f_minusdel) / 2
            else:
                f_del = (f_plusdel + f_minusdel) / 2 - f_x0[i]

        elif par['Style'][0].lower() == 'f':
            # Forward rule
            if par['Vectorized'].lower() == 'yes':
                f_del = fun(x0i + h * delta) - f_x0[i]
            else:
                # Not vectorized, so loop
                f_del = np.zeros_like(delta)
                for j, val in enumerate(delta):
                    f_del[j] = fun(x0i + h * val) - f_x0[i]
        else:
            # Backward rule
            if par['Vectorized'].lower() == 'yes':
                f_del = fun(x0i - h * delta) - f_x0[i]
            else:
                # Not vectorized, so loop
                f_del = np.zeros_like(delta)
                for j, val in enumerate(delta):
                    f_del[j] = fun(x0i - h * val) - f_x0[i]
        # Check the size of f_del to ensure it was properly vectorized.
        f_del = f_del.reshape(-1, 1)
        if len(f_del) != ndel:
            raise ValueError("fun did not return the correct size result (fun must be vectorized)")

        # Apply the finite difference rule at each delta, scaling
        # as appropriate for delta and the requested DerivativeOrder.

        # First, decide how many of these estimates we will end up with.
        ne = ndel + 1 - nfda - par['RombergTerms']

        # Form the initial derivative estimates from the chosen
        # finite difference method.
        der_initial = (np.dot(vec2mat(f_del, ne, nfda), np.transpose(fdarule))).reshape(-1, 1)

        # Scale to reflect the local delta
        der_init = der_initial / (h * delta[:ne]) ** par['DerivativeOrder']

        # Each approximation that results is an approximation
        # of order par['DerivativeOrder'] to the desired derivative.
        # Additional (higher order, even or odd) terms in the
        # Taylor series also remain. Use a generalized (multi-term)
        # Romberg extrapolation to improve these estimates.

        if par['Style'].lower() == 'central':
            rombexpon = 2 * np.arange(1, par['RombergTerms'] + 1) + par['MethodOrder'] - 2
        else:
            rombexpon = np.arange(1, par['RombergTerms'] + 1) + par['MethodOrder'] - 1

        # Assuming rombextrap is defined elsewhere
        der_romb, errors = rombextrap(par['StepRatio'], der_init, rombexpon)

        # Choose which result to return

        if par['FixedStep'] is None:
            nest = len(der_romb)

            # Determine which values to trim based on the DerivativeOrder
            if par['DerivativeOrder'] in [1, 2]:
                trim = [0, 1, nest - 2, nest - 1]
            elif par['DerivativeOrder'] == 3:
                trim = list(range(0, 4)) + list(range(nest - 4, nest))
            elif par['DerivativeOrder'] == 4:
                trim = list(range(0, 6)) + list(range(nest - 6, nest))

            # Sort der_romb and get the corresponding indices
            if np.all(der_romb == der_romb[0]):
                tags = np.arange(len(der_romb))[:, np.newaxis]
            else:
                tags = np.argsort(der_romb, axis=0)
                der_romb = np.sort(der_romb, axis=0)

            # Delete elements from der_romb and tags arrays
            der_romb = (np.delete(der_romb, trim)).reshape(-1, 1)
            tags = (np.delete(tags, trim)).reshape(-1, 1)

            # Flatten tags for correct indexing
            tags_flattened = tags.flatten()

            # Reorder errors and trimdelta based on sorted tags
            errors = errors[tags_flattened].reshape(-1, 1)
            trimdelta = delta[tags_flattened].reshape(-1, 1)

            # Get the minimum error and its index
            errest[i], ind = np.min(errors), np.argmin(errors)

            # Update finaldelta and der
            finaldelta[i] = h * trimdelta[ind]
            der[i] = der_romb[ind]
        else:
            # If FixedStep is not None
            errest[i], ind = np.min(errors), np.argmin(errors)
            finaldelta[i] = h * delta[ind]
            der[i] = der_romb[ind]

    return der, errest, finaldelta


def disturbance_smoother_var(y, c, Z, G, C, B, H, s00, P00, T, n, ns, ne, SS):
    """
        Performs draws from the posterior of the disturbances and unobservable states
        of a state-space model. The model is defined by the input matrices and vectors.

        Args:
            y (np.ndarray): Observations, dimension T x n.
            c (np.ndarray): Constant term in observation equation, dimension n x T or n.
            Z (np.ndarray): Matrix in observation equation, dimension n x ns x T or n x ns.
            G (np.ndarray): Matrix in observation equation, dimension n x n x T or n x n.
            C (np.ndarray): Constant term in state equation, dimension ns.
            B (np.ndarray): Matrix in state equation, dimension ns x ns.
            H (np.ndarray): Matrix in state equation, dimension ns x ne x T or ns x ne.
            s00 (np.ndarray): Initial state, dimension ns.
            P00 (np.ndarray): Initial state covariance, dimension ns x ns.
            T (int): Number of time steps.
            n (int): Dimension of observation vector.
            ns (int): Dimension of state vector.
            ne (int): Dimension of disturbance vector.
            SS (str): Method for computing the smoother, either 'simulation' or 'kalman'.

        Returns:
            tuple:
                sdraw (np.ndarray): Draw of the state, dimension ns x T.
                epsdraw (np.ndarray): Draw of the disturbance, dimension ne x T for 'simulation'
                                      and an empty array for 'kalman'.

        Raises:
            ValueError: If SS is not 'simulation' or 'kalman'.
        """

    # Ensuring dimensions
    if Z.shape[2] == 1:
        Z = np.tile(Z, (1, 1, T))
    if c.shape[1] == 1:
        c = np.tile(c, (1, T))
    if G.shape[2] == 1:
        G = np.tile(G, (1, 1, T))
    if H.shape[2] == 1:
        H = np.tile(H, (1, 1, T))

    ind = np.isfinite(y)

    # Initialize matrices
    V = np.zeros((n, T))
    K = np.zeros((ns, n, T))
    HINV = np.zeros((n, n, T))
    SHAT = np.zeros((ns, T))
    SIG = np.zeros((ns, ns, T))
    shat = s00
    sig = P00

    for t in range(T):
        yt = y[t, ind[t, :]].reshape(-1, 1)
        ct = c[ind[t, :], t].reshape(-1, 1)
        Zt = Z[ind[t, :], :, t].squeeze()
        Gt = G[ind[t, :], :, t].squeeze()
        Ht = H[:, :, t].squeeze()

        shat, sig, v, k, hinv = kfilter_const(yt, ct, Zt, Gt, C, B, Ht, shat, sig)
        SHAT[:, t] = shat.squeeze()  # Use .squeeze() to convert shat to a one-dimensional array
        SIG[:, :, t] = sig
        V[ind[t, :], t] = v.squeeze()  # Apply .squeeze() here if necessary
        # Check if k has only one column
        if k.shape[1] == 1:
            K[:, ind[t, :], t] = k.squeeze().reshape(-1, 1)
        else:
            K[:, ind[t, :], t] = k.squeeze()
        # Apply .squeeze() with axis specified if necessary
        HINV[:, :, t][np.ix_(ind[t, :], ind[t, :])] = hinv

    # Disturbance smoother
    epshat = np.zeros((ne, T))
    r = np.zeros((ns, 1))
    for t in range(T - 1, -1, -1):  # iterate backwards
        Ht = H[:, :, t].squeeze()
        Zt = Z[ind[t, :], :, t].squeeze()
        # Check if k is 1D
        if Zt.ndim == 1:
            Zt = (Z[ind[t, :], :, t].squeeze()).reshape(1, -1)

        HINVt = HINV[:, :, t][np.ix_(ind[t, :], ind[t, :])]
        Vt = V[ind[t, :], t].reshape(-1, 1)

        epshat[:, t] = (Ht.T @ Zt.T @ HINVt @ Vt + Ht.T @ (np.eye(ns) - K[:, ind[t, :], t] @ Zt).T @ r).squeeze()
        r = B.T @ Zt.T @ HINVt @ Vt + B.T @ (np.eye(ns) - K[:, ind[t, :], t] @ Zt).T @ r

    if SS == 'kalman':
        # Smoothed states
        sdraw = np.zeros((ns, T))
        sdraw[:, 0] = (C + B @ s00 + H[:, :, 0].squeeze() @ epshat[:, 0].reshape(-1, 1)).squeeze()
        for t in range(1, T):
            sdraw[:, t] = (C + B @ sdraw[:, t - 1].reshape(-1, 1) +
                           H[:, :, t].squeeze() @ (epshat[:, t]).reshape(-1, 1)).squeeze()
        epsdraw = np.array([])  # Assuming epsdraw is not calculated for 'kalman' case

    elif SS == 'simulation':
        # Simulating new shocks, states, and observables
        epsplus = np.random.randn(ne, T)
        splus = np.zeros((ns, T))
        yplus = np.zeros((n, T))

        splus[:, 0] = (C + B @ s00 + H[:, :, 0].squeeze() @ epsplus[:, 0].reshape(-1, 1)).squeeze()
        yplus[ind[0, :], 0] = (Z[ind[0, :], :, 0].squeeze() @ splus[:, 0]).squeeze()

        for t in range(1, T):
            splus[:, t] = (C + B @ splus[:, t - 1].reshape(-1, 1) +
                           H[:, :, t].squeeze() @ epsplus[:, t].reshape(-1, 1)).squeeze()
            yplus[ind[t, :], t] = (Z[ind[t, :], :, t].squeeze() @ splus[:, t].reshape(-1, 1)).squeeze()

        # Kalman filter
        Vplus = np.zeros((n, T))
        Kplus = np.zeros((ns, n, T))
        HINVplus = np.zeros((n, n, T))
        SHATplus = np.zeros((ns, T))
        SIGplus = np.zeros((ns, ns, T))
        shat = s00
        sig = P00

        for t in range(T):
            ytplus = yplus[ind[t, :], t].reshape(-1, 1)
            ct = c[ind[t, :], t].reshape(-1, 1)
            Zt = Z[ind[t, :], :, t].squeeze()
            Gt = G[ind[t, :], :, t].squeeze()
            Ht = H[:, :, t].squeeze()

            shat, sig, v, k, hinv = kfilter_const(ytplus, ct, Zt, Gt, C, B, Ht, shat, sig)
            SHATplus[:, t] = shat.squeeze()
            SIGplus[:, :, t] = sig
            Vplus[ind[t, :], t] = v.squeeze()
            #Kplus[:, ind[t, :], t] = k.squeeze()
            Kplus[:, ind[t, :], t] = k.reshape((k.shape[0], -1))
            HINVplus[:, :, t][np.ix_(ind[t, :], ind[t, :])] = hinv

        # Disturbance smoother
        epshatplus = np.zeros((ne, T))
        r = np.zeros(ns)

        for t in range(T - 1, -1, -1):
            Ht = H[:, :, t].squeeze()
            Zt = Z[ind[t, :], :, t].squeeze().reshape(-1, 1)
            HINVplust = HINVplus[:, :, t][np.ix_(ind[t, :], ind[t, :])]
            Vplust = Vplus[ind[t, :], t].reshape(-1, 1)

            term1 = Ht.T @ (Zt @ (HINVplust @ Vplust))
            term2 = Ht.T @ (np.eye(ns) - Kplus[:, ind[t, :], t] @ Zt.T) @ r.reshape(-1, 1)
            epshatplus[:, t] = (term1 + term2).squeeze()

            # Calculate r
            Zt = Zt.reshape(-1, 1) if Zt.ndim == 1 else Zt
            r = r.reshape(-1, 1) if r.ndim == 1 else r
            # Correct matrix multiplications
            term1 = B.T @ Zt @ HINVplust @ Vplust
            term2 = B.T @ (np.eye(ns) - Kplus[:, ind[t, :], t] @ Zt.T) @ r
            # Ensure r is a column vector
            r = (term1 + term2).reshape(-1, 1)

        epsdraw = epshat + epsplus - epshatplus

        # Computation of state draws
        sdraw = np.zeros((ns, T))
        sdraw[:, 0] = (C + B @ s00 + H[:, :, 0].squeeze() @ epsdraw[:, 0].reshape(-1, 1)).squeeze()
        for t in range(1, T):
            sdraw[:, t] = (C + B @ sdraw[:, t - 1].reshape(-1, 1) +
                           H[:, :, t].squeeze() @ epsdraw[:, t].reshape(-1, 1)).squeeze()

    return sdraw, epsdraw


def drsnbrck(x):
    """
    Compute the derivative for the Rosenbrock problem.

    Parameters:
        x (numpy.array): Input vector of shape (2, 1)

    Returns:
        dr (numpy.array): Derivative of shape (2, 1)
        badg (int): Flag indicating if the gradient is bad (always 0)
    """
    dr = np.zeros((2, 1))
    dr[0, 0] = 2 * (x[0] - 1) - 8 * 105 * x[0] * (x[1] - x[0] ** 2) ** 3
    dr[1, 0] = 4 * 105 * (x[1] - x[0] ** 2) ** 3
    badg = 0

    return dr, badg


def fdamat(sr, parity, nterms):
    """
    Compute matrix for finite difference approximation (FDA) derivation.

    Args:
        sr (float): The ratio between successive steps.
        parity (int): The parity of the derivative terms.
            - 0: One-sided, all terms included but zeroth order
            - 1: Only odd terms included
            - 2: Only even terms included
        nterms (int): The number of terms in the series.

    Returns:
        numpy.ndarray: The FDA matrix.
    """
    srinv = 1. / sr

    if parity == 0:
        # single-sided rule
        i, j = np.mgrid[1:nterms + 1, 1:nterms + 1]
        c = 1. / factorial(np.arange(1, nterms + 1))
        mat = c[j - 1] * (srinv ** ((i - 1) * j))

    elif parity == 1:
        # odd order derivative
        i, j = np.mgrid[1:nterms + 1, 1:nterms + 1]
        c = 1. / factorial(np.arange(1, 2 * nterms, 2))
        mat = c[j - 1] * (srinv ** ((i - 1) * (2 * j - 1)))

    elif parity == 2:
        # even order derivative
        i, j = np.mgrid[1:nterms + 1, 1:nterms + 1]
        c = 1. / factorial(np.arange(2, 2 * nterms + 1, 2))
        mat = c[j - 1] * (srinv ** ((i - 1) * (2 * j)))

    return mat



def FIS(Y, Z, R, T, S):
    """
    Fixed Interval Smoother (FIS) based on Durbin and Koopman, 2001, p. 64-71.

    Args:
        Y (numpy.ndarray): Data array of shape (n, nobs), where n is the number of variables and nobs is the time dimension.
        Z (numpy.ndarray): System matrix \(Z\) of shape (n, m), where m is the dimension of the state vector.
        R (numpy.ndarray): System matrix \(R\) of shape (n, n).
        T (numpy.ndarray): Transition matrix \(T\) of shape (m, m).
        S (dict): Dictionary containing estimates from Kalman filter SKF.
            - S['Am']: Estimates \(a_{t|t-1}\) of shape (m, nobs).
            - S['Pm']: \(P_{t|t-1} = \text{Cov}(a_{t|t-1})\) of shape (m, m, nobs).

    Returns:
        dict: Dictionary containing smoothed estimates added to S:
            - S['AmT']: Smoothed estimates \(a_{t|T}\) of shape (m, nobs).
            - S['PmT']: \(P_{t|T} = \text{Cov}(a_{t|T})\) of shape (m, m, nobs).

    """
    # Get dimensions
    m, nobs = S['Am'].shape

    # Initialize smoothed estimates
    S['AmT'] = np.zeros((m, nobs))
    S['PmT'] = np.zeros((m, m, nobs))

    # Initial value for smoothed state
    S['AmT'][:, nobs - 1] = np.squeeze(S['Am'][:, nobs - 1])

    # Initialize r as zero vector
    r = np.zeros((m, 1))

    # Loop through observations in reverse time order
    for t in range(nobs, 0, -1):
        # Handling missing data
        y_t, Z_t, _, _ = MissData(Y[:, t - 1].reshape(-1, 1), Z, R, np.zeros((len(Y[:, t - 1]), 1)))

        # Extract the relevant matrices and vectors for the current time t
        ZF_t = np.array(S['ZF'][t - 1])  # Assuming S['ZF'] is a list of numpy arrays
        V_t = np.array(S['V'][t - 1])  # Assuming S['V'] is a list of numpy arrays

        # Update r according to the MATLAB formula
        r = np.dot(ZF_t, V_t) + np.dot((T @ (np.eye(m) - np.squeeze(S['Pm'][:, :, t - 1]) @ ZF_t @ Z_t)).T, r)

        # Update smoothed state estimate
        S['AmT'][:, t - 1] = S['Am'][:, t - 1] + np.dot(S['Pm'][:, :, t - 1], r).flatten()

    return S


def form_companion_matrices(betadraw, G, n, lags, TTfcst):
    """
    Forms the matrices of the VAR companion form.

    This function forms various matrices used in the VAR companion form, such as those for the
    observation and state equations. It takes into account a given forecast horizon and various other
    parameters.

    Args:
        betadraw (numpy.ndarray): Beta coefficients for the VAR model. Shape should be (1 + n * lags, n).
        G (numpy.ndarray): Matrix G in the state equation. Shape should be (n, n).
        n (int): The number of variables in the VAR model.
        lags (int): The number of lags in the VAR model.
        TTfcst (int): The forecast horizon.

    Returns:
        tuple: Tuple containing:
            - varc (numpy.ndarray): Vector of zeros of shape (n, TTfcst).
            - varZ (numpy.ndarray): 3D array with the Z matrix replicated TTfcst times. Shape is (n, n*lags, TTfcst).
            - varG (numpy.ndarray): 3D array of zeros of shape (n, n, TTfcst).
            - varC (numpy.ndarray): Vector containing the first n elements of betadraw. Shape is (n*lags, ).
            - varT (numpy.ndarray): State transition matrix. Shape is (n*lags, n*lags).
            - varH (numpy.ndarray): 3D array for the H matrix. Shape is (n*lags, n, TTfcst).
    """

    # Matrices of observation equation
    varc = np.zeros((n, TTfcst))
    varZ = np.zeros((n, n * lags))
    varZ[:, :n] = np.eye(n)
    varZ = np.repeat(varZ[:, :, np.newaxis], TTfcst, axis=2)
    varG = np.repeat(np.zeros((n, n))[:, :, np.newaxis], TTfcst, axis=2)

    # Matrices of state equation
    B = betadraw
    varC = np.zeros((n * lags, 1))
    varC[:n, 0] = B[0, :].T
    varT = np.vstack([B[1:, :].T, np.hstack([np.eye(n * (lags - 1)), np.zeros((n * (lags - 1), n))])])
    varH = np.zeros((n * lags, n, TTfcst))
    varH[:n, :, :] = np.repeat(G[:, :, np.newaxis], TTfcst, axis=2)

    return varc, varZ, varG, varC, varT, varH


def form_companion_matrices_covid(betadraw, G, etapar, tstar, n, lags, TTfcst):
    """
    Forms the matrices of the VAR companion form with COVID-related adjustments.

    Args:
        betadraw (numpy.ndarray): Beta coefficients for the VAR model. Shape should be (1 + n * lags, n).
        G (numpy.ndarray): Matrix G in the state equation. Shape should be (n, n).
        etapar (numpy.ndarray): Array containing the parameters for COVID adjustments. Shape should be (4,).
        tstar (int): The starting point of COVID effects in the forecast horizon.
        n (int): The number of variables in the VAR model.
        lags (int): The number of lags in the VAR model.
        TTfcst (int): The forecast horizon.

    Returns:
        tuple: Tuple containing:
            - varc (numpy.ndarray): Vector of zeros of shape (n, TTfcst).
            - varZ (numpy.ndarray): 3D array with the Z matrix replicated TTfcst times. Shape is (n, n*lags, TTfcst).
            - varG (numpy.ndarray): 3D array of zeros of shape (n, n, TTfcst).
            - varC (numpy.ndarray): Vector containing the first n elements of betadraw. Shape is (n*lags, ).
            - varT (numpy.ndarray): State transition matrix. Shape is (n*lags, n*lags).
            - varH (numpy.ndarray): 3D array for the H matrix with COVID-related adjustments. Shape is (n*lags, n, TTfcst).
    """
    # Matrices of observation equation
    varc = np.zeros((n, TTfcst))
    varZ = np.zeros((n, n * lags))
    varZ[:, :n] = np.eye(n)
    varZ = np.repeat(varZ[:, :, np.newaxis], TTfcst, axis=2)
    varG = np.repeat(np.zeros((n, n))[:, :, np.newaxis], TTfcst, axis=2)

    # Matrices of state equation
    B = betadraw
    varC = np.zeros((n * lags, 1))
    varC[:n, 0] = B[0, :].T
    varT = np.vstack([B[1:, :].T, np.hstack([np.eye(n * (lags - 1)), np.zeros((n * (lags - 1), n))])])
    varH = np.zeros((n * lags, n, TTfcst))

    for t in range(TTfcst):
        if t < tstar:
            varH[:n, :, t] = G
        elif t == tstar:
            varH[:n, :, t] = G * etapar[0]
        elif t == tstar + 1:
            varH[:n, :, t] = G * etapar[1]
        elif t == tstar + 2:
            varH[:n, :, t] = G * etapar[2]
        elif t > tstar + 2:
            varH[:n, :, t] = G * (1 + (etapar[2] - 1) * etapar[3] ** (t - tstar - 2))

    return varc, varZ, varG, varC, varT, varH


def gamma_coef(mode, sd, plotit):
    """
    Computes the coefficients of Gamma distribution coefficients and makes plots, if requested
    The parameters of the Gamma distribution are
    k = shape parameter: affects the PDF of the Gamma distribution, including skewness and mode
    theta = scale parameter: affects the spread of the distribution i.e. it shrinks or stretches the
    distribution along the x-axis.

    Args:
        mode (float): Mode of the Gamma distribution.
        sd (float): Standard deviation of the Gamma distribution.
        plotit (int): Flag to determine if the plot should be shown (1) or not (0).

    Returns:
        dict: Dictionary containing the 'k' and 'theta' parameters of the Gamma distribution.
    """
    # compute the k and theta parameters of the gamma distribution
    r_k = (2 + mode ** 2 / (sd ** 2) + np.sqrt((4 + mode ** 2 / sd ** 2) * mode ** 2 / sd ** 2)) / 2
    r_theta = np.sqrt(sd ** 2 / r_k)

    if plotit == 1:  # if we request to make plot
        xxx = np.arange(0, mode + 5 * sd, 0.000001)
        # plot and show the Gamma distribution
        plt.plot(xxx, (xxx ** (r_k - 1) * np.exp(-xxx / r_theta) * r_theta ** -r_k) / gamma(r_k), 'k--', linewidth=2)
        plt.show()
    # display the computed k and theta parameters
    return {'k': r_k, 'theta': r_theta}


def gradest(fun, x0):
    """
    Estimate the gradient vector of an analytical function of n variables.

    Args:
        fun (callable): Analytical function to differentiate. Must be a function
            of the vector or array x0.
        x0 (numpy.ndarray): Vector location at which to differentiate fun.
            If x0 is an nxm array, then fun is a function of n*m variables.

    Returns:
        tuple: A tuple containing:
            - grad (numpy.ndarray): Vector of first partial derivatives of fun.
                Will be a row vector of length x0.size.
            - err (numpy.ndarray): Vector of error estimates corresponding to each
                partial derivative in grad.
            - finaldelta (numpy.ndarray): Vector of final step sizes chosen for each
                partial derivative.

    Examples:
        >>> # Example using lambda functions in Python
        >>> grad, err, finaldelta = gradest(lambda x: np.sum(x ** 2), np.array([1, 2, 3]))

    """

    # Get the size of x0 so we can reshape later
    sx = x0.shape

    # Total number of derivatives we will need to take
    nx = x0.size

    # Initialize output arrays
    grad = np.zeros((1, nx))
    err = grad
    finaldelta = grad

    # Loop over each element in x0 to compute the gradient, error, and final delta
    for ind in range(nx):
        # Define a new function that swaps the element at index 'ind' with a new variable
        def fun_swapped(xi):
            return fun(swapelement(x0, ind, xi))

        # Optional parameters for derivest
        optional_params = ['DerivativeOrder', 1, 'Vectorized', 'no', 'MethodOrder', 2]

        # Call the derivest function to get the gradient, error, and final delta at this index
        grad[0, ind], err[0, ind], finaldelta[0, ind] = derivest(fun_swapped, x0[ind], optional_params)

    return grad, err, finaldelta


def hessian(fun, x0):
    """Compute the Hessian matrix of second partial derivatives for a scalar function.

        Given a scalar function of one or more variables, compute the Hessian matrix,
        a square matrix of second-order partial derivatives of the function. It is a
        generalization of the second derivative test for single-variable functions.

        Args:
            fun (callable): A scalar function that accepts a NumPy array and returns a scalar.
                            The function to differentiate must be a function of the vector or
                            array `x0`. The function does not need to be vectorized.
            x0 (np.ndarray): A NumPy array representing the point at which the Hessian matrix
                             is to be computed. If `x0` is an `n x m` array, then `fun` is
                             a function of `n * m` variables.

        Returns:
            tuple: A tuple containing:
                - hess (np.ndarray): An `n x n` symmetric matrix of second partial derivatives
                                      of `fun`, evaluated at `x0`.
                - err (np.ndarray): An `n x n` array of error estimates corresponding to each
                                     second partial derivative in `hess`.

        Raises:
            ValueError: If `fun` is not callable or if `x0` does not allow for the computation
                        of the Hessian matrix due to incompatible dimensions or data types.

        Examples:
            To use this function, define a scalar function of interest. For example, the
            Rosenbrock function, which is minimized at [1, 1]:

            >>> rosen = lambda x: (1 - x[0])**2 + 105 * (x[1] - x[0]**2)**2
            >>> hess, err = hessian(rosen, np.array([1, 1]))
            >>> print("Hessian matrix:\n", hess)
            >>> print("Error estimates:\n", err)

            The Hessian matrix and error estimates for the function at the point [1, 1]
            will be printed to the console.

        Notes:
            The `hessian` function is not a tool for frequent use on an expensive-to-evaluate
            objective function, especially in a large number of dimensions. Its computation
            will use roughly `O(6*n^2)` function evaluations for `n` parameters.

        See Also:
            `hessdiag`, `gradest`, and `rombextrap`: Auxiliary functions that are required
            for the computation of the Hessian matrix and must be defined elsewhere in the
            codebase.
        """

    # Define parameters
    params = {'StepRatio': 2.0000001, 'RombergTerms': 3}

    # Get the size of x0 so we can reshape later
    sx = x0.shape

    # Total number of derivatives we will need to take
    nx = np.size(x0)

    # Get the diagonal elements of the hessian (2nd partial derivatives wrt each variable)
    hess_diag, err_diag, _ = hessdiag(fun, x0)

    # Form the eventual hessian matrix, stuffing only the diagonals for now
    hess = np.diag(hess_diag)
    err = np.diag(err_diag)

    if nx < 2:
        # The hessian matrix is 1x1. All done
        return hess, err

    # Get the gradient vector to decide on intelligent step sizes for the mixed partials
    grad, graderr, stepsize = gradest(fun, x0)

    # Get params['RombergTerms']+1 estimates of the upper triangle of the hessian matrix
    dfac = (params['StepRatio'] ** (-np.arange(params['RombergTerms'] + 1))).reshape(-1, 1)
    for i in range(1, nx):
        for j in range(i):
            dij = (np.zeros(params['RombergTerms'] + 1)).reshape(-1, 1)
            for k in range(params['RombergTerms'] + 1):
                x0_perturb_plus_i_j = x0 + swap2(np.zeros_like(x0), i, dfac[k] * stepsize[0, i], j,
                                                 dfac[k] * stepsize[0, j])
                x0_perturb_minus_i_j = x0 + swap2(np.zeros_like(x0), i, -dfac[k] * stepsize[0, i], j,
                                                  -dfac[k] * stepsize[0, j])
                x0_perturb_plus_i_minus_j = x0 + swap2(np.zeros_like(x0), i, dfac[k] * stepsize[0, i], j,
                                                       -dfac[k] * stepsize[0, j])
                x0_perturb_minus_i_plus_j = x0 + swap2(np.zeros_like(x0), i, -dfac[k] * stepsize[0, i], j,
                                                       dfac[k] * stepsize[0, j])

                dij[k] = (fun(x0_perturb_plus_i_j)[0] + fun(x0_perturb_minus_i_j)[0] -
                          fun(x0_perturb_plus_i_minus_j)[0] - fun(x0_perturb_minus_i_plus_j)[0])

            dij = dij / (4 * np.prod(stepsize[0, [i, j]]))
            dij = dij / (dfac ** 2)

            # Romberg extrapolation step
            hess_ij, err_ij = rombextrap(params['StepRatio'], dij, [2, 4])

            hess[i, j] = hess_ij
            err[i, j] = err_ij
            hess[j, i] = hess[i, j]
            err[j, i] = err[i, j]

    return hess, err


def hessdiag(fun, x0):
    """
    Compute the diagonal elements of the Hessian matrix (vector of second partials)

    Parameters:
        fun: callable
            Scalar analytical function to differentiate.
        x0: np.ndarray
            Vector location at which to differentiate fun.
            If x0 is an nxm array, then fun is a function of n*m variables.

    Returns:
        HD: np.ndarray
            Vector of second partial derivatives of fun. These are the diagonal elements of the Hessian matrix.
        err: np.ndarray
            Vector of error estimates corresponding to each second partial derivative in HD.
        finaldelta: np.ndarray
        V   ector of final step sizes chosen for each second partial derivative.
    """

    # Get the size of x0 so we can reshape later
    sx = np.shape(x0)

    # Total number of derivatives we will need to take
    nx = np.prod(sx)

    # Initialize output variables
    HD = np.zeros(nx)
    err = np.zeros(nx)
    finaldelta = np.zeros(nx)

    # Loop through each element in x0
    for ind in range(nx):
        # Define a lambda function to swap elements in x0
        # Ensure xi is a scalar by using xi.item() if it's an array
        # Flatten the output of swapelement before reshaping
        fun_handle = lambda xi: fun(np.array(swapelement(x0.flatten().tolist(),
                                                         ind, xi.item()
                                                         if np.ndim(xi) > 0 else xi)).flatten().reshape(sx))
        extra_args = ['deriv', 2, 'vectorized', 'no']

        # Call derivest function
        HD[ind], err[ind], finaldelta[ind] = derivest(fun_handle, x0.flatten()[ind], extra_args)

    return HD, err, finaldelta


def kfilter_const(y, c, Z, G, C, T, H, shat, sig):
    """
    Kalman filter with constant variance for the state-space model.

    Args:
        y (np.array): Observation vector at time t. Shape (n, 1).
        c (float): Constant term in observation equation.
        Z (np.array): Observation loading matrix. Shape (n, m).
        G (np.array): Observation noise loading matrix. Shape (n, n).
        C (float): Constant term in state equation.
        T (np.array): State transition matrix. Shape (m, m).
        H (np.array): State noise loading matrix. Shape (m, m).
        shat (np.array): Prior state estimate. Shape (m, 1).
        sig (np.array): Prior state covariance matrix. Shape (m, m).

    Returns:
        tuple: Tuple containing the following elements:
            - shatnew (np.array): Updated state estimate. Shape (m, 1).
            - signew (np.array): Updated state covariance matrix. Shape (m, m).
            - v (np.array): Prediction error. Shape (n, 1).
            - k (np.array): Kalman gain. Shape (m, n).
            - sigmainv (np.array): Inverse of the innovation covariance. Shape (n, n).
    """
    # Reshape Z and G into row vectors if they are 1D arrays
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    if G.ndim == 1:
        G = G.reshape(1, -1)

    # Number of observations
    n = len(y)

    # Compute omega, the state covariance propagation
    omega = T @ sig @ T.T + H @ H.T
    # Calculate sigmainv with matrix inversion
    sigmainv = np.linalg.inv(Z @ omega @ Z.T + G @ G.T)

    # Compute Kalman gain
    k = omega @ Z.T @ sigmainv

    # Compute prediction error
    v = y - c - Z @ (C + T @ shat)

    # Update state estimate
    shatnew = C + T @ shat + k @ v

    # Update state covariance matrix
    signew = omega - k @ Z @ omega

    return shatnew, signew, v, k, sigmainv


def lag(x, n=1, v=0):
    """
    Create a matrix or vector of lagged values.

    Args:
        x (numpy.ndarray): Input matrix or vector (nobs x k).
        n (int, optional): Order of lag. Default is 1.
        v (int or float, optional): Initial values for lagged entries. Default is 0.

    Returns:
        numpy.ndarray: Matrix or vector of lags (nobs x k).

    Examples:
        >>> lag(x)  # Creates a matrix (or vector) of x, lagged 1 observation.
        >>> lag(x, n=2)  # Creates a matrix (or vector) of x, lagged 2 observations.
        >>> lag(x, n=2, v=999)  # Lagged with custom initial values of 999.

    Notes:
        If n <= 0, an empty array is returned.
    """
    if n < 1:
        return np.array([])

    rows, cols = x.shape
    zt = np.ones((n, cols)) * v

    # Trim and concatenate
    if n >= rows:
        return zt  # Only return initial values if n exceeds number of rows
    else:
        z = np.vstack((zt, x[:-n, :]))  # Stack initial values and trimmed x

    return z


def lag_matrix(Y, lags):
    """
    Create a matrix of lagged (time-shifted) series.

    Args:
        Y (np.ndarray): Time series data. Y may be a vector or a matrix.
                        If Y is a vector, it represents a single series.
                        If Y is a numObs-by-numSeries matrix, it represents
                        numObs observations of numSeries series.

        lags (list): List of integer delays or leads applied to each series in Y.
                     To include an unshifted copy of a series in the output, use a zero lag.

    Returns:
        np.ndarray: numObs-by-(numSeries*numLags) matrix of lagged versions of the series in Y.
                    Unspecified observations (presample and postsample data) are padded with NaN values.
    """

    # Ensure Y is a 2D numpy array
    Y = np.atleast_2d(Y)

    # Check if Y is a column vector, if so transpose it
    if Y.shape[0] == 1:
        Y = Y.T

    # Ensure lags is a list and convert it to a numpy array
    if not isinstance(lags, list):
        raise ValueError("lags must be a list of integers.")
    lags = np.array(lags)

    # Initialize the output lagged matrix
    numLags = len(lags)
    numObs, numSeries = Y.shape
    YLag = np.full((numObs, numSeries * numLags), np.nan)

    # Create lagged series
    for c in range(numLags):
        L = lags[c]
        columns = np.arange(numSeries * c, numSeries * (c + 1))

        if L > 0:  # Time delays
            YLag[L:, columns] = Y[:-L, :]
        elif L < 0:  # Time leads
            YLag[:L, columns] = Y[-L:, :]
        else:  # No shifts
            YLag[:, columns] = Y

    return YLag


def logMLVAR_formcmc(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, draw,
                     hyperpriors, priorcoef, MCMCMsur, long_run):
    """
    Compute the log-posterior (or logML if hyperpriors=0), and draws from the posterior distribution
    of the coefficients and of the covariance matrix of the residuals of the BVAR model by Giannone, Lenza,
    and Primiceri (2015).

    Args:
        par (np.ndarray): Parameters for the model, shaped (p, 1).
        y (np.ndarray): Output matrix, shaped (T, n).
        x (np.ndarray): Input matrix, shaped (T, k).
        lags (int): Number of lags in the VAR model.
        T (int): Number of time periods.
        n (int): Number of variables.
        b (np.ndarray): Prior mean for VAR coefficients, shaped (k, n).
        MIN (dict): Minimum hyperparameter values.
        MAX (dict): Maximum hyperparameter values.
        SS (np.ndarray): Sum of squares, shaped (n, 1).
        Vc (float): Prior variance for the constant.
        pos (np.ndarray): Position index (currently not used).
        mn (dict): Additional settings.
        sur (int): Indicator for Minnesota prior.
        noc (int): Indicator for no-cointegration prior.
        y0 (np.ndarray): Initial values for y, shaped (1, n).
        draw (int): Indicator for drawing from the posterior.
        hyperpriors (int): Indicator for using hyperpriors.
        priorcoef (dict): Coefficients for the prior.


        Returns:
        tuple: Contains the following elements -
            logML (float): The log marginal likelihood.
            betadraw (np.ndarray or None): Drawn VAR coefficients from the posterior, shaped (k, n).
            Returns None if 'draw' is set to 0.
            drawSIGMA (np.ndarray or None): Drawn covariance matrix from the posterior, shaped (n, n).
            Returns None if 'draw' is set to 0.

    """

    # Hyperparameters
    if isinstance(par, float):
        lambda_ = par
    else:
        lambda_ = par[0]
    d = n + 2

    # Initialize theta, miu, and eta from MIN dict
    theta = MIN['theta']
    miu = MIN['miu']

    # Conditional logic based on whether Tcovid is empty or not
    if mn['psi'] == 0:
        psi = SS * (d - n - 1)  # psi will be a column vector
        if sur == 1:
            theta = par[1]
            if noc == 1:
                miu = par[2]
        elif sur == 0:
            if noc == 1:
                miu = par[1]
    elif mn['psi'] == 1:
        psi = par[1:n + 1]
        if sur == 1:
            theta = par[n + 1]
            if noc == 1:
                miu = par[n + 2]
        elif sur == 0:
            if noc == 1:
                miu = par[n + 1]

    # Alpha hyperparameter logic based on mn dict
    if mn['alpha'] == 0:
        alpha = 2
    elif mn['alpha'] == 1:
        alpha = par[-1]

    # Check the type of each variable and convert to float if it's a ndarray of shape (1,)
    if isinstance(lambda_, np.ndarray) and lambda_.shape == (1,):
        lambda_ = float(lambda_)

    if isinstance(miu, np.ndarray) and miu.shape == (1,):
        miu = float(miu)

    if isinstance(theta, np.ndarray) and theta.shape == (1,):
        theta = float(theta)

    # check if the number of elements of [lambda; eta; theta; miu; alpha] fall
    # outside the bounds defined by MIN and MAX arrays
    # if any parameter is less than its corresponding min value or greater than
    # its corresponding max value, the criteria is met, yielding in True

    # Create column vectors for the scalar values
    lambda_vector = np.array([[lambda_]])
    theta_vector = np.array([[theta]])
    miu_vector = np.array([[miu]])
    alpha_vector = np.array([[alpha]])

    # Concatenate the parameters into a single column vector
    parameters = np.vstack((lambda_vector, psi, theta_vector, miu_vector, alpha_vector))

    # Create column vectors for MIN and MAX values
    MIN_values = np.vstack((np.array([[MIN['lambda']]]), MIN['psi'],
                            np.array([[MIN['theta']]]), np.array([[MIN['miu']]]), np.array([[MIN['alpha']]])))
    MAX_values = np.vstack((np.array([[MAX['lambda']]]), MAX['psi'],
                            np.array([[MAX['theta']]]), np.array([[MAX['miu']]]), np.array([[MAX['alpha']]])))

    # Check if any of the parameters are outside the bounds
    if np.any(parameters < MIN_values) or np.any(parameters > MAX_values):
        logML = -1e15  # Return a very low value of logML
        betadraw = None
        drawSIGMA = None
        return logML, betadraw, drawSIGMA

    else:
        # Priors
        k = 1 + n * lags  # Calculate k, the total number of coefficients for each variable
        omega = np.zeros((k, 1))  # Initialize omega as a kx1 zero vector
        omega[0] = Vc  # Set the first element to Vc

        for i in range(1, lags + 1):
            start_idx = 1 + (i - 1) * n
            end_idx = 1 + i * n
            # TODO: Note that lambda_ depends on the value obtained from the multivariate normal distribution,
            #  so the omega values will slightly differ from MATLAB's omega.
            omega[start_idx:end_idx] = (d - n - 1) * (lambda_ ** 2) * (1 / (i ** alpha)) / psi.reshape(-1, 1)

        # Prior scale matrix for the covariance of the shocks
        PSI = np.diagflat(psi)  # Create a diagonal matrix from psi

        Td = 0  # Initialize Td
        xdsur = np.array([])  # Initialize xdsur
        ydsur = np.array([])  # Initialize ydsur
        xdnoc = np.array([])  # Initialize xdnoc
        ydnoc = np.array([])  # Initialize ydnoc

        # Dummy observations if MCMCMsur == 1
        if MCMCMsur == 1:
            tightTHETA = 0.0001
            tightY0 = long_run

            # Create xdsur
            first_term_tightSUR = np.array([1 / tightTHETA]).reshape(1, 1)
            second_term_tightSUR = (1 / tightTHETA) * np.tile(tightY0, (1, lags))
            xdsur_tightSUR = np.hstack([first_term_tightSUR, second_term_tightSUR])

            # Create ydsur
            ydsur_tightSUR = (1 / tightTHETA) * tightY0

            # Append ydsur and xdsur to y and x
            y = np.vstack([y, ydsur_tightSUR])
            x = np.vstack([x, xdsur_tightSUR])

            # Update Td
            Td = 1

        # Dummy observations if sur and/or noc = 1
        # Handle sur==1 condition
        if sur == 1:
            first_term = np.array([1 / theta]).reshape(1, 1)  # Reshape to (1, 1)
            second_term = (1 / theta) * np.tile(y0, (1, lags))
            xdsur = np.hstack([first_term, second_term])
            ydsur = (1 / theta) * y0
            y = np.vstack([y, ydsur])
            x = np.vstack([x, xdsur])
            Td += 1

        if noc == 1:
            ydnoc = (1 / miu) * np.diagflat(y0)
            if pos:  # This will be False if pos is None or an empty list
                ydnoc[pos, pos] = 0
            diagonal_values = (1 / miu) * y0  # This should be 1x4
            diagonal_matrix = np.diagflat(diagonal_values)  # This should create a 4x4 diagonal matrix

            # Repeat the diagonal matrix for each lag
            repeated_matrix = np.tile(diagonal_matrix, (1, lags))  # This should be 4x(4*lags)

            # Now create xdnoc by stacking zeros and the repeated_matrix horizontally
            xdnoc = np.hstack([np.zeros((n, 1)), repeated_matrix])

            y = np.vstack([y, ydnoc])
            x = np.vstack([x, xdnoc])
            Td += n  # increment by n

        # Output calculations
        #############################################################################

        # Compute the posterior mode of the VAR coefficients (betahat)
        # This involves solving the linear system (x'x + diag(1/omega)) * betahat = x'y + diag(1/omega) * b
        # TODO: as omega values differ slightly from MATLAB's omega, betahat, epshat and subsequent terms that depend
        #  on betahat will also vary marginally
        betahat = np.linalg.solve(x.T @ x + np.diag(1 / omega.ravel()), x.T @ y +
                                  np.diag(1 / omega.ravel()) @ b)

        # Compute VAR residuals (epshat)
        epshat = y - x @ betahat

        # Update T with the number of dummy observations (Td)
        T += Td

        # Compute matrices aaa and bbb used in logML calculation
        # aaa is a weighted version of x'x and bbb is a weighted version of the residuals' covariance matrix
        aaa = np.diag(np.sqrt(omega.ravel())) @ (x.T @ x) @ np.diag(np.sqrt(omega.ravel()))
        term1 = np.diag(1 / np.sqrt(psi.ravel()))
        term2 = epshat.T @ epshat
        term3 = (betahat - b).T @ np.diag(1. / omega.ravel()) @ (betahat - b)
        bbb = term1 @ (term2 + term3) @ term1

        # extract the real parts of the eigenvalues of aaa and bbb
        # this is because eigenvalues can be complex numbers, which we don't want
        # if eigenvalues in eigaaa and eigbbb are less than 1e-12, set them to 0
        eigaaa = np.linalg.eigvals(aaa).real
        eigaaa[eigaaa < 1e-12] = 0
        eigaaa += 1  # increment the eigenvalues by 1

        eigbbb = np.linalg.eigvals(bbb).real
        eigbbb[eigbbb < 1e-12] = 0
        eigbbb += 1

        # Compute logML (log marginal likelihood)
        logML = (- n * T * np.log(np.pi) / 2 + np.sum(gammaln((T + d - np.arange(n)) / 2) -
                                                      gammaln((d - np.arange(n)) / 2)) - T * np.sum(np.log(psi)) / 2 -
                 n * np.sum(np.log(eigaaa)) / 2 - (T + d) * np.sum(np.log(eigbbb)) / 2)

        if sur == 1 or noc == 1:
            # Combine the dummy observations for y and x (yd and xd)
            yd = np.vstack([ydsur, ydnoc])
            xd = np.vstack([xdsur, xdnoc])

            # Prior mode of the VAR coefficients
            # Numerically stable according to the original Matlab code
            betahatd = b

            # Compute the VAR residuals at the prior mode
            epshatd = yd - np.matmul(xd, betahatd)

            # Compute matrices aaa and bbb for the dummy observations
            aaa = np.diag(np.sqrt(omega.ravel())) @ xd.T @ xd @ np.diag(np.sqrt(omega.ravel()))

            # Ensure psi and omega are 1D arrays for the calculations
            term1 = np.diag(1 / np.sqrt(psi.ravel()))
            term2 = epshatd.T @ epshatd
            term3 = (betahatd - b).T @ np.diag(1. / omega.ravel()) @ (betahatd - b)

            bbb = term1 @ (term2 + term3) @ term1

            # Compute eigenvalues and modify them as in the Matlab code
            eigaaa = eigvals(aaa).real
            eigaaa[eigaaa < 1e-12] = 0
            eigaaa += 1

            eigbbb = eigvals(bbb).real
            eigbbb[eigbbb < 1e-12] = 0
            eigbbb += 1

            # Compute the normalizing constant
            norm = (-n * Td * np.log(np.pi) / 2 + np.sum(
                gammaln((Td + d - np.arange(n)) / 2) - gammaln((d - np.arange(n)) / 2))
                    - Td * np.sum(np.log(psi.ravel())) / 2 - n * np.sum(np.log(eigaaa)) / 2 -
                    (Td + d) * np.sum(np.log(eigbbb)) / 2)

            # Update logML with the normalizing constant
            logML -= norm

        # Update logML based on hyperpriors
        if hyperpriors == 1:
            logML += log_gamma_pdf(lambda_, priorcoef['lambda']['k'], priorcoef['lambda']['theta'])
            if sur == 1:
                logML += log_gamma_pdf(theta, priorcoef['theta']['k'], priorcoef['theta']['theta'])
            if noc == 1:
                logML += log_gamma_pdf(miu, priorcoef['miu']['k'], priorcoef['miu']['theta'])
            if mn['psi'] == 1:
                logML += np.sum(log_ig2pdf(psi / (d - n - 1), priorcoef['alpha']['PSI'], priorcoef['beta']['PSI']))

        # If draw is off, set betadraw and drawSIGMA to empty lists
        if draw == 0:
            betadraw = []
            drawSIGMA = []

        # If draw is on, compute betadraw and drawSIGMA
        elif draw == 1:

            S = PSI + epshat.T @ epshat + (betahat - b).T @ np.diag((1. / omega).flatten()) @ (betahat - b)

            E, V = eig(S)
            # Create a diagonal matrix from the eigenvalues
            E_diag = np.diag(E)
            Sinv = V @ np.diag(1. / np.abs(E)) @ V.T
            eta = mvnrnd.rvs(np.zeros(n), Sinv, size=T + d)
            drawSIGMA = np.linalg.inv(eta.T @ eta)

            # Reduce Cholesky decomposition
            cholSIGMA = cholred((drawSIGMA + drawSIGMA.T) / 2)
            cholZZinv = cholred(solve(x.T @ x + np.diag(1. / omega.flatten()), np.eye(k)))

            # Generate betadraw
            betadraw = betahat + cholZZinv.T @ np.random.randn(*betahat.shape) @ cholSIGMA

    return logML, betadraw, drawSIGMA


def logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, draw,
                           hyperpriors, priorcoef, Tcovid=None):
    """
    Compute the log-posterior (or logML if hyperpriors=0), and draws from the posterior distribution
    of the coefficients and of the covariance matrix of the residuals of the BVAR model by Giannone, Lenza,
    and Primiceri (2015). The function also accounts for a change in volatility due to COVID-19.

    Args:
        par (np.ndarray): Parameters for the model, shaped (p, 1).
        y (np.ndarray): Output matrix, shaped (T, n).
        x (np.ndarray): Input matrix, shaped (T, k).
        lags (int): Number of lags in the VAR model.
        T (int): Number of time periods.
        n (int): Number of variables.
        b (np.ndarray): Prior mean for VAR coefficients, shaped (k, n).
        MIN (dict): Minimum hyperparameter values.
        MAX (dict): Maximum hyperparameter values.
        SS (np.ndarray): Sum of squares, shaped (n, 1).
        Vc (float): Prior variance for the constant.
        pos (np.ndarray): Position index (currently not used).
        mn (dict): Additional settings.
        sur (int): Indicator for Minnesota prior.
        noc (int): Indicator for no-cointegration prior.
        y0 (np.ndarray): Initial values for y, shaped (1, n).
        draw (int): Indicator for drawing from the posterior.
        hyperpriors (int): Indicator for using hyperpriors.
        priorcoef (dict): Coefficients for the prior.
        Tcovid (int, optional): Time index for COVID-19 structural break. Defaults to None.

        Returns:
        tuple: Contains the following elements -
            - logML (float): The log marginal likelihood.
            - betadraw (np.ndarray or None): Drawn VAR coefficients from the posterior, shaped (k, n).
            - Returns None if 'draw' is set to 0.
            - drawSIGMA (np.ndarray or None): Drawn covariance matrix from the posterior, shaped (n, n).
            - Returns None if 'draw' is set to 0.

    """

    # Hyperparameters
    lambda_ = par[0]  # Scalar value for lambda
    d = n + 2

    # Initialize theta, miu, and eta from MIN dict
    theta = MIN['theta']
    miu = MIN['miu']
    eta = np.array(MIN['eta']).reshape(-1, 1)  # Make sure eta is a column vector

    # Calculate psi
    psi = SS * (d - n - 1)  # psi will be a column vector

    # Conditional logic based on whether Tcovid is empty or not
    if Tcovid is None:
        if sur == 1:
            theta = par[1]
            if noc == 1:
                miu = par[2]
        elif sur == 0:
            if noc == 1:
                miu = par[1]
    else:
        ncp = 4  # Number of COVID parameters
        eta = par[1:ncp + 1].reshape(-1, 1)  # Update eta

        # Initialize invweights and update y and x based on it
        invweights = np.ones((T, 1))
        invweights[Tcovid - 1] = eta[0]
        invweights[Tcovid] = eta[1]
        if T > Tcovid + 1:
            invweights[Tcovid + 1:T, :] = 1 + (eta[2] - 1) * eta[3] ** np.arange(0, T - Tcovid - 1).reshape(-1, 1)

        y = np.diag(1. / invweights.ravel()) @ y
        x = np.diag(1. / invweights.ravel()) @ x

        if sur == 1:
            theta = par[ncp + 1]
            if noc == 1:
                miu = par[ncp + 2]
        elif sur == 0:
            if noc == 1:
                miu = par[ncp + 1]

    # Alpha hyperparameter logic based on mn dict
    if mn['alpha'] == 0:
        alpha = 2
    elif mn['alpha'] == 1:
        alpha = par[-1]

    # Check the type of each variable and convert to float if it's a ndarray of shape (1,)
    if isinstance(lambda_, np.ndarray) and lambda_.shape == (1,):
        lambda_ = float(lambda_)

    if isinstance(miu, np.ndarray) and miu.shape == (1,):
        miu = float(miu)

    if isinstance(theta, np.ndarray) and theta.shape == (1,):
        theta = float(theta)

    # check if the number of elements of [lambda; eta; theta; miu; alpha] fall
    # outside the bounds defined by MIN and MAX arrays
    # if any parameter is less than its corresponding min value or greater than
    # its corresponding max value, the criteria is met, yielding in True

    if np.any(np.concatenate([np.array([lambda_]), eta.ravel(), np.array([theta, miu, alpha])]) <
              np.concatenate([np.array([MIN['lambda']]), MIN['eta'],
                              np.array([MIN['theta'], MIN['miu'], MIN['alpha']])])) or \
            np.any(np.concatenate([np.array([lambda_]), eta.ravel(), np.array([theta, miu])]) >
                   np.concatenate([np.array([MAX['lambda']]), MAX['eta'], np.array([MAX['theta'], MAX['miu']])])):
        logML = -1e15  # Return a very low value of logML
        betadraw = None
        drawSIGMA = None
        return logML, betadraw, drawSIGMA

    else:
        # Priors
        k = 1 + n * lags  # Calculate k, the total number of coefficients for each variable
        omega = np.zeros((k, 1))  # Initialize omega as a kx1 zero vector
        omega[0] = Vc  # Set the first element to Vc

        for i in range(1, lags + 1):
            start_idx = 1 + (i - 1) * n
            end_idx = 1 + i * n
            omega[start_idx:end_idx] = (d - n - 1) * (lambda_ ** 2) * (1 / (i ** alpha)) / psi.reshape(-1, 1)

        # Prior scale matrix for the covariance of the shocks
        PSI = np.diagflat(psi)  # Create a diagonal matrix from psi

        Td = 0  # Initialize Td
        xdsur = np.array([])  # Initialize xdsur
        ydsur = np.array([])  # Initialize ydsur
        xdnoc = np.array([])  # Initialize xdnoc
        ydnoc = np.array([])  # Initialize ydnoc

        # Dummy observations if sur and/or noc = 1
        # Handle sur==1 condition
        if sur == 1:
            first_term = np.array([1 / theta]).reshape(1, 1)  # Reshape to (1, 1)
            second_term = (1 / theta) * np.tile(y0, (1, lags))
            xdsur = np.hstack([first_term, second_term])
            ydsur = (1 / theta) * y0
            y = np.vstack([y, ydsur])
            x = np.vstack([x, xdsur])
            Td = 1

        if noc == 1:
            ydnoc = (1 / miu) * np.diagflat(y0)
            if pos:  # This will be False if pos is None or an empty list
                ydnoc[pos, pos] = 0
            diagonal_values = (1 / miu) * y0  # This should be 1x4
            diagonal_matrix = np.diagflat(diagonal_values)  # This should create a 4x4 diagonal matrix

            # Repeat the diagonal matrix for each lag
            repeated_matrix = np.tile(diagonal_matrix, (1, lags))  # This should be 4x(4*lags)

            # Now create xdnoc by stacking zeros and the repeated_matrix horizontally
            xdnoc = np.hstack([np.zeros((n, 1)), repeated_matrix])

            y = np.vstack([y, ydnoc])
            x = np.vstack([x, xdnoc])
            Td += n  # increment by n

        # Output calculations
        #############################################################################

        # Compute the posterior mode of the VAR coefficients (betahat)
        # This involves solving the linear system (x'x + diag(1/omega)) * betahat = x'y + diag(1/omega) * b
        betahat = np.linalg.solve(x.T @ x + np.diag(1 / omega.ravel()), x.T @ y +
                                  np.diag(1 / omega.ravel()) @ b)

        # Compute VAR residuals (epshat)
        epshat = y - x @ betahat

        # Update T with the number of dummy observations (Td)
        T += Td

        # Compute matrices aaa and bbb used in logML calculation
        # aaa is a weighted version of x'x and bbb is a weighted version of the residuals' covariance matrix
        aaa = np.diag(np.sqrt(omega.ravel())) @ (x.T @ x) @ np.diag(np.sqrt(omega.ravel()))
        term1 = np.diag(1 / np.sqrt(psi.ravel()))
        term2 = epshat.T @ epshat
        term3 = (betahat - b).T @ np.diag(1. / omega.ravel()) @ (betahat - b)
        bbb = term1 @ (term2 + term3) @ term1

        # extract the real parts of the eigenvalues of aaa and bbb
        # this is because eigenvalues can be complex numbers, which we don't want
        # if eigenvalues in eigaaa and eigbbb are less than 1e-12, set them to 0
        eigaaa = np.linalg.eigvals(aaa).real
        eigaaa[eigaaa < 1e-12] = 0
        eigaaa += 1  # increment the eigenvalues by 1

        eigbbb = np.linalg.eigvals(bbb).real
        eigbbb[eigbbb < 1e-12] = 0
        eigbbb += 1

        # Compute logML (log marginal likelihood)
        logML = (- n * T * np.log(np.pi) / 2 + np.sum(gammaln((T + d - np.arange(n)) / 2) -
                                                      gammaln((d - np.arange(n)) / 2)) - T * np.sum(np.log(psi)) / 2 -
                 n * np.sum(np.log(eigaaa)) / 2 - (T + d) * np.sum(np.log(eigbbb)) / 2)

        if sur == 1 or noc == 1:
            # Combine the dummy observations for y and x (yd and xd)
            yd = np.vstack([ydsur, ydnoc])
            xd = np.vstack([xdsur, xdnoc])

            # Prior mode of the VAR coefficients
            # Numerically stable according to the original Matlab code
            betahatd = b

            # Compute the VAR residuals at the prior mode
            epshatd = yd - np.matmul(xd, betahatd)

            # Compute matrices aaa and bbb for the dummy observations
            aaa = np.diag(np.sqrt(omega.ravel())) @ xd.T @ xd @ np.diag(np.sqrt(omega.ravel()))

            # Ensure psi and omega are 1D arrays for the calculations
            term1 = np.diag(1 / np.sqrt(psi.ravel()))
            term2 = epshatd.T @ epshatd
            term3 = (betahatd - b).T @ np.diag(1. / omega.ravel()) @ (betahatd - b)

            bbb = term1 @ (term2 + term3) @ term1

            # Compute eigenvalues and modify them as in the Matlab code
            eigaaa = eigvals(aaa).real
            eigaaa[eigaaa < 1e-12] = 0
            eigaaa += 1

            eigbbb = eigvals(bbb).real
            eigbbb[eigbbb < 1e-12] = 0
            eigbbb += 1

            # Compute the normalizing constant
            norm = (-n * Td * np.log(np.pi) / 2 + np.sum(
                gammaln((Td + d - np.arange(n)) / 2) - gammaln((d - np.arange(n)) / 2))
                    - Td * np.sum(np.log(psi.ravel())) / 2 - n * np.sum(np.log(eigaaa)) / 2 -
                    (Td + d) * np.sum(np.log(eigbbb)) / 2)

            # Update logML with the normalizing constant
            logML -= norm

        # Account for re-weighting if Tcovid is not None
        if Tcovid is not None:
            logML -= n * np.sum(np.log(invweights))

        # Update logML based on hyperpriors
        if hyperpriors == 1:
            logML += log_gamma_pdf(lambda_, priorcoef['lambda']['k'], priorcoef['lambda']['theta'])
            if sur == 1:
                logML += log_gamma_pdf(theta, priorcoef['theta']['k'], priorcoef['theta']['theta'])
            if noc == 1:
                logML += log_gamma_pdf(miu, priorcoef['miu']['k'], priorcoef['miu']['theta'])
            if Tcovid is not None:
                logML -= 2 * np.log(eta[0]) + 2 * np.log(eta[1]) + 2 * np.log(eta[2]) + \
                         log_beta_pdf(eta[3], priorcoef['eta4']['alpha'], priorcoef['eta4']['beta'])

        # If draw is off, set betadraw and drawSIGMA to empty lists
        if draw == 0:
            betadraw = []
            drawSIGMA = []

        # If draw is on, compute betadraw and drawSIGMA
        elif draw == 1:

            S = PSI + epshat.T @ epshat + (betahat - b).T @ np.diag((1. / omega).flatten()) @ (betahat - b)

            E, V = eig(S)
            # Create a diagonal matrix from the eigenvalues
            E_diag = np.diag(E)
            Sinv = V @ np.diag(1. / np.abs(E)) @ V.T
            eta = mvnrnd.rvs(np.zeros(n), Sinv, size=T + d)
            drawSIGMA = np.linalg.inv(eta.T @ eta)

            # Reduce Cholesky decomposition
            cholSIGMA = cholred((drawSIGMA + drawSIGMA.T) / 2)
            cholZZinv = cholred(solve(x.T @ x + np.diag(1. / omega.flatten()), np.eye(k)))

            # Generate betadraw
            betadraw = betahat + cholZZinv.T @ np.random.randn(*betahat.shape) @ cholSIGMA

    return logML, betadraw, drawSIGMA


def logMLVAR_formin(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0,
                    hyperpriors, priorcoef, MCMCMsur, long_run):
    """
        Compute the log-posterior, posterior mode of the coefficients, and covariance matrix of the residuals for
         a BVAR model.

        This function implements the Bayesian Vector Autoregression (BVAR) model of Giannone, Lenza, and Primiceri
        (2015), extended to account for a change in volatility due to COVID-19.

        Args:
            par (array-like): Parameters for the model, shaped (p, 1).
            y (array-like): Output matrix, shaped (T, n).
            x (array-like): Input matrix, shaped (T, k).
            lags (int): Number of lags in the VAR model.
            T (int): Number of time periods.
            n (int): Number of variables.
            b (array-like): Prior mean for VAR coefficients, shaped (k, n).
            MIN (dict): Minimum hyperparameter values.
            MAX (dict): Maximum hyperparameter values.
            SS (array-like): Sum of squares, shaped (n, 1).
            Vc (float): Prior variance for the constant.
            pos (array-like): Position index (currently not used).
            mn (dict): Additional settings.
            sur (int): Indicator for Minnesota prior.
            noc (int): Indicator for no-cointegration prior.
            y0 (array-like): Initial values for y, shaped (1, n).
            hyperpriors (int): Indicator for using hyperpriors.
            priorcoef (dict): Coefficients for the prior.

        Returns:
            tuple: Contains logML, betahat, and sigmahat.
                - logML (float): Log marginal likelihood (or log-posterior if hyperpriors=0).
                - betahat (array-like): Posterior mode of the VAR coefficients, shaped (k, n).
                - sigmahat (array-like): Posterior mode of the covariance matrix, shaped (n, n).

        Example:
            >>> logML, betahat, sigmahat = logMLVAR_formin_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn,
            sur, noc, y0, hyperpriors, priorcoef, MCMCMsur, long_run)
        """

    # Hyperparameters
    lambda_ = MIN['lambda'] + (MAX['lambda'] - MIN['lambda']) / (1 + np.exp(-par[0]))
    d = n + 2

    # Conditional logic
    if mn['psi'] == 0:
        psi = SS * (d - n - 1)
        if sur == 1:
            theta = MIN['theta'] + (MAX['theta'] - MIN['theta']) / (1 + np.exp(-par[1]))
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[2]))
        elif sur == 0:
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[1]))

    elif mn['psi'] == 1:
        psi = MIN['psi'] + \
              (MAX['psi'] - MIN['psi']).reshape((-1, 1)) / (1 + np.exp(-par[1:n + 1])).reshape((-1, 1))
        if sur == 1:
            theta = MIN['theta'] + (MAX['theta'] - MIN['theta']) / (1 + np.exp(-par[n + 1]))
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[n + 2]))
        elif sur == 0:
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[n + 1]))

    if mn['alpha'] == 0:
        alpha = 2
    elif mn['alpha'] == 1:
        alpha = MIN['alpha'] + (MAX['alpha'] - MIN['alpha']) / (1 + np.exp(-par[-1]))

    ###################  Setting up the priors ###################
    k = 1 + n * lags
    omega = np.zeros((k, 1))

    # First element of omega
    omega[0] = Vc

    # Loop starts from 1 to lags
    for i in range(1, lags + 1):
        start_idx = 1 + (i - 1) * n
        end_idx = 1 + i * n
        omega[start_idx:end_idx] = ((d - n - 1) * (lambda_ ** 2) * (1 / (i ** alpha)) / psi).reshape(-1, 1)

    # Prior scale matrix for the covariance of the shocks
    PSI = np.diagflat(psi)

    # Initialize dummy observation variables
    Td = 0
    xdsur = np.array([])
    ydsur = np.array([])
    xdnoc = np.array([])
    ydnoc = np.array([])

    # Dummy observations if MCMCMsur == 1
    if MCMCMsur == 1:
        tightTHETA = 0.0001
        tightY0 = long_run

        # Create xdsur
        first_term_tightSUR = np.array([1 / tightTHETA]).reshape(1, 1)
        second_term_tightSUR = (1 / tightTHETA) * np.tile(tightY0, (1, lags))
        xdsur_tightSUR = np.hstack([first_term_tightSUR, second_term_tightSUR])

        # Create ydsur
        ydsur_tightSUR = (1 / tightTHETA) * tightY0

        # Append ydsur and xdsur to y and x
        y = np.vstack([y, ydsur_tightSUR])
        x = np.vstack([x, xdsur_tightSUR])

        # Update Td
        Td = 1

    # Handle sur==1 condition
    if sur == 1:
        first_term = np.array([1 / theta]).reshape(1, 1)  # Reshape to (1, 1)
        second_term = (1 / theta) * np.tile(y0, (1, lags))  # The shape should be (1, 4*lags)

        # Now try hstack
        xdsur = np.hstack([first_term, second_term])
        ydsur = (1 / theta) * y0
        y = np.vstack([y, ydsur])
        x = np.vstack([x, xdsur])
        Td += 1

    # Handle noc==1 condition
    if noc == 1:
        ydnoc = (1 / miu) * np.diagflat(y0)
        ydnoc[pos, pos] = 0
        diagonal_values = (1 / miu) * y0  # This should be 1x4
        diagonal_matrix = np.diagflat(diagonal_values)  # This should create a 4x4 diagonal matrix

        # Repeat the diagonal matrix for each lag
        repeated_matrix = np.tile(diagonal_matrix, (1, lags))  # This should be 4x(4*lags)

        # Now create xdnoc by stacking zeros and the repeated_matrix horizontally
        xdnoc = np.hstack([np.zeros((n, 1)), repeated_matrix])

        y = np.vstack([y, ydnoc])
        x = np.vstack([x, xdnoc])
        Td += n  # increment by n

    # Update T
    T += Td

    # Compute posterior mode of the VAR coefficients
    # Here omega is kx1, x'x is kxk, x'y is kxn, and b is kxn
    betahat = la.solve(x.T @ x + np.diag(1. / omega.ravel()), x.T @ y + np.diag(1. / omega.ravel()) @ b)

    # Compute VAR residuals
    # epshat will be of dimension Txn
    epshat = y - x @ betahat

    # Compute the posterior mode of the covariance matrix
    # sigmahat will be nxn
    sigmahat = (epshat.T @ epshat + PSI + (betahat - b).T @ np.diag(1. / omega.ravel()) @ (betahat - b)) / (
            T + d + n + 1)

    # Compute matrices aaa and bbb
    # aaa and bbb will be of dimensions kxk and nxn, respectively
    aaa = np.diag(np.sqrt(omega.ravel())) @ x.T @ x @ np.diag(np.sqrt(omega.ravel()))
    # Ensure psi and omega are 1D arrays
    psi_1D = psi.ravel()
    omega_1D = omega.ravel()

    # Compute the individual terms
    term1 = np.diag(1. / np.sqrt(psi_1D))
    term2 = epshat.T @ epshat
    term3 = (betahat - b).T @ np.diag(1. / omega_1D) @ (betahat - b)

    # Combine them all
    bbb = term1 @ (term2 + term3) @ term1

    # Compute eigenvalues and modify them as in the MATLAB code
    eigaaa = la.eigvals(aaa).real
    eigaaa[eigaaa < 1e-12] = 0
    eigaaa += 1

    eigbbb = la.eigvals(bbb).real
    eigbbb[eigbbb < 1e-12] = 0
    eigbbb += 1

    # Compute logML
    logML = (- n * T * np.log(np.pi) / 2 + np.sum(gammaln((T + d - np.arange(n)) / 2) -
                                                  gammaln((d - np.arange(n)) / 2)) -
             T * np.sum(np.log(psi)) / 2 - n * np.sum(np.log(eigaaa)) / 2 - (T + d) * np.sum(np.log(eigbbb)) / 2)

    # Check conditions for sur and/or noc
    if sur == 1 or noc == 1:
        yd = np.vstack([ydsur, ydnoc])
        xd = np.vstack([xdsur, xdnoc])

        # Since this is numerically more stable according to the original MATLAB code
        betahatd = b

        epshatd = yd - xd @ betahatd
        # Compute matrices aaa and bbb for the dummy observations
        # aaa and bbb will be of dimensions kxk and nxn, respectively
        aaa_dummy = np.diag(np.sqrt(omega.ravel())) @ xd.T @ xd @ np.diag(np.sqrt(omega.ravel()))
        # Ensure psi and omega are 1D arrays
        psi_1D_dummy = psi.ravel()
        omega_1D_dummy = omega.ravel()

        # Compute the individual terms
        term1_dummy = np.diag(1. / np.sqrt(psi_1D_dummy))
        term2_dummy = epshatd.T @ epshatd
        term3_dummy = (betahatd - b).T @ np.diag(1. / omega_1D_dummy) @ (betahatd - b)

        # Combine them all
        bbb_dummy = term1_dummy @ (term2_dummy + term3_dummy) @ term1_dummy

        # Compute eigenvalues and modify them as in the MATLAB code
        eigaaa_dummy = eigvals(aaa_dummy).real
        eigaaa_dummy[eigaaa_dummy < 1e-12] = 0
        eigaaa_dummy += 1

        eigbbb_dummy = eigvals(bbb_dummy).real
        eigbbb_dummy[eigbbb_dummy < 1e-12] = 0
        eigbbb_dummy += 1

        # Compute normalizing constant for the dummy observations
        norm_dummy = (- n * Td * np.log(np.pi) / 2 + np.sum(gammaln((Td + d - np.arange(n)) / 2) -
                                                            gammaln((d - np.arange(n)) / 2)) -
                      Td * np.sum(np.log(psi)) / 2 - n * np.sum(np.log(eigaaa_dummy)) / 2 -
                      (Td + d) * np.sum(np.log(eigbbb_dummy)) / 2)

        # Update logML with the normalizing constant for the dummy observations
        logML -= norm_dummy

    # Update logML based on hyperpriors
    if hyperpriors == 1:
        logML += log_gamma_pdf(lambda_, priorcoef['lambda']['k'], priorcoef['lambda']['theta'])
        if sur == 1:
            logML += log_gamma_pdf(theta, priorcoef['theta']['k'], priorcoef['theta']['theta'])
        if noc == 1:
            logML += log_gamma_pdf(miu, priorcoef['miu']['k'], priorcoef['miu']['theta'])
        if mn['psi'] == 1:
            logML = logML + np.sum(log_ig2pdf(psi / (d - n - 1), priorcoef['alpha']['PSI'], priorcoef['beta']['PSI']))

    # Finally, invert the sign of logML as in the original MATLAB code
    logML = -logML

    return logML, betahat, sigmahat


def logMLVAR_formin_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0,
                          hyperpriors, priorcoef, Tcovid=None):
    """
    Compute the log-posterior, posterior mode of the coefficients, and covariance matrix of the residuals for
     a BVAR model.

    This function implements the Bayesian Vector Autoregression (BVAR) model of Giannone, Lenza, and Primiceri
    (2015), extended to account for a change in volatility due to COVID-19.

    Args:
        par (array-like): Parameters for the model, shaped (p, 1).
        y (array-like): Output matrix, shaped (T, n).
        x (array-like): Input matrix, shaped (T, k).
        lags (int): Number of lags in the VAR model.
        T (int): Number of time periods.
        n (int): Number of variables.
        b (array-like): Prior mean for VAR coefficients, shaped (k, n).
        MIN (dict): Minimum hyperparameter values.
        MAX (dict): Maximum hyperparameter values.
        SS (array-like): Sum of squares, shaped (n, 1).
        Vc (float): Prior variance for the constant.
        pos (array-like): Position index (currently not used).
        mn (dict): Additional settings.
        sur (int): Indicator for Minnesota prior.
        noc (int): Indicator for no-cointegration prior.
        y0 (array-like): Initial values for y, shaped (1, n).
        hyperpriors (int): Indicator for using hyperpriors.
        priorcoef (dict): Coefficients for the prior.
        Tcovid (int, optional): Time index for COVID-19 structural break. Defaults to None.

    Returns:
        tuple: Contains logML, betahat, and sigmahat.
            - logML (float): Log marginal likelihood (or log-posterior if hyperpriors=0).
            - betahat (array-like): Posterior mode of the VAR coefficients, shaped (k, n).
            - sigmahat (array-like): Posterior mode of the covariance matrix, shaped (n, n).

    Example:
        >>> logML, betahat, sigmahat = logMLVAR_formin_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn,
        sur, noc, y0, hyperpriors, priorcoef, Tcovid)
    """

    # Hyperparameters
    lambda_ = MIN['lambda'] + (MAX['lambda'] - MIN['lambda']) / (1 + np.exp(-par[0]))
    d = n + 2
    psi = (SS * (d - n - 1)).reshape(-1, 1)  # psi will be a column vector

    # Conditional logic
    if Tcovid is None:
        if sur == 1:
            theta = MIN['theta'] + (MAX['theta'] - MIN['theta']) / (1 + np.exp(-par[1]))
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[2]))
        elif sur == 0:
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[1]))

    else:
        ncp = 4
        eta = (MIN['eta'].reshape(-1, 1) +
               (MAX['eta'].reshape(-1, 1) - MIN['eta'].reshape(-1, 1)) /
               (1 + np.exp(-par[1:ncp + 1].reshape(-1, 1))))

        invweights = np.ones((T, 1))  # Vector of s_t, shape is (T, 1)
        invweights[Tcovid - 1] = eta[0]
        invweights[Tcovid] = eta[1]
        if T > Tcovid + 1:
            invweights[Tcovid + 1:T, :] = (1 + (eta[2] - 1) * eta[3] ** np.arange(0, T - Tcovid - 1)).reshape(-1, 1)

        # Update y and x based on invweights
        y = np.diag(1. / invweights.ravel()) @ y
        x = np.diag(1. / invweights.ravel()) @ x

        if sur == 1:
            theta = MIN['theta'] + (MAX['theta'] - MIN['theta']) / (1 + np.exp(-par[ncp + 1]))
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[ncp + 2, 0]))
        elif sur == 0:
            if noc == 1:
                miu = MIN['miu'] + (MAX['miu'] - MIN['miu']) / (1 + np.exp(-par[ncp + 1]))

    if mn['alpha'] == 0:
        alpha = 2
    elif mn['alpha'] == 1:
        alpha = MIN['alpha'] + (MAX['alpha'] - MIN['alpha']) / (1 + np.exp(-par[-1]))

    # Setting up the priors
    k = 1 + n * lags
    omega = np.zeros((k, 1))

    # First element of omega
    omega[0] = Vc

    # Loop starts from 1 to lags
    for i in range(1, lags + 1):
        start_idx = 1 + (i - 1) * n
        end_idx = 1 + i * n
        omega[start_idx:end_idx] = ((d - n - 1) * (lambda_ ** 2) * (1 / (i ** alpha)) / psi).reshape(-1, 1)

    # Prior scale matrix for the covariance of the shocks
    PSI = np.diagflat(psi)

    # Initialize dummy observation variables
    Td = 0
    xdsur = np.array([])
    ydsur = np.array([])
    xdnoc = np.array([])
    ydnoc = np.array([])

    # Handle sur==1 condition
    if sur == 1:
        first_term = np.array([1 / theta]).reshape(1, 1)  # Reshape to (1, 1)
        second_term = (1 / theta) * np.tile(y0, (1, lags))  # The shape should be (1, 4*lags)

        # Now try hstack
        xdsur = np.hstack([first_term, second_term])
        ydsur = (1 / theta) * y0
        y = np.vstack([y, ydsur])
        x = np.vstack([x, xdsur])
        Td = 1

    # Handle noc==1 condition
    if noc == 1:
        ydnoc = (1 / miu) * np.diagflat(y0)
        ydnoc[pos, pos] = 0
        diagonal_values = (1 / miu) * y0  # This should be 1x4
        diagonal_matrix = np.diagflat(diagonal_values)  # This should create a 4x4 diagonal matrix

        # Repeat the diagonal matrix for each lag
        repeated_matrix = np.tile(diagonal_matrix, (1, lags))  # This should be 4x(4*lags)

        # Now create xdnoc by stacking zeros and the repeated_matrix horizontally
        xdnoc = np.hstack([np.zeros((n, 1)), repeated_matrix])

        y = np.vstack([y, ydnoc])
        x = np.vstack([x, xdnoc])
        Td += n  # increment by n

    # Update T
    T += Td

    # Compute posterior mode of the VAR coefficients
    # Here omega is kx1, x'x is kxk, x'y is kxn, and b is kxn
    betahat = la.solve(x.T @ x + np.diag(1. / omega.ravel()), x.T @ y + np.diag(1. / omega.ravel()) @ b)

    # Compute VAR residuals
    # epshat will be of dimension Txn
    epshat = y - x @ betahat

    # Compute the posterior mode of the covariance matrix
    # sigmahat will be nxn
    sigmahat = (epshat.T @ epshat + PSI + (betahat - b).T @ np.diag(1. / omega.ravel()) @ (betahat - b)) / (
            T + d + n + 1)

    # Compute matrices aaa and bbb
    # aaa and bbb will be of dimensions kxk and nxn, respectively
    aaa = np.diag(np.sqrt(omega.ravel())) @ x.T @ x @ np.diag(np.sqrt(omega.ravel()))
    # Ensure psi and omega are 1D arrays
    psi_1D = psi.ravel()
    omega_1D = omega.ravel()

    # Compute the individual terms
    term1 = np.diag(1. / np.sqrt(psi_1D))
    term2 = epshat.T @ epshat
    term3 = (betahat - b).T @ np.diag(1. / omega_1D) @ (betahat - b)

    # Combine them all
    bbb = term1 @ (term2 + term3) @ term1

    # Compute eigenvalues and modify them as in the MATLAB code
    eigaaa = la.eigvals(aaa).real
    eigaaa[eigaaa < 1e-12] = 0
    eigaaa += 1

    eigbbb = la.eigvals(bbb).real
    eigbbb[eigbbb < 1e-12] = 0
    eigbbb += 1

    # Compute logML
    logML = (- n * T * np.log(np.pi) / 2 + np.sum(gammaln((T + d - np.arange(n)) / 2) -
                                                  gammaln((d - np.arange(n)) / 2)) -
             T * np.sum(np.log(psi)) / 2 - n * np.sum(np.log(eigaaa)) / 2 - (T + d) * np.sum(np.log(eigbbb)) / 2)

    # Check conditions for sur and/or noc
    if sur == 1 or noc == 1:
        yd = np.vstack([ydsur, ydnoc])
        xd = np.vstack([xdsur, xdnoc])

        # Since this is numerically more stable according to the original MATLAB code
        betahatd = b

        epshatd = yd - xd @ betahatd
        # Compute matrices aaa and bbb for the dummy observations
        # aaa and bbb will be of dimensions kxk and nxn, respectively
        aaa_dummy = np.diag(np.sqrt(omega.ravel())) @ xd.T @ xd @ np.diag(np.sqrt(omega.ravel()))
        # Ensure psi and omega are 1D arrays
        psi_1D_dummy = psi.ravel()
        omega_1D_dummy = omega.ravel()

        # Compute the individual terms
        term1_dummy = np.diag(1. / np.sqrt(psi_1D_dummy))
        term2_dummy = epshatd.T @ epshatd
        term3_dummy = (betahatd - b).T @ np.diag(1. / omega_1D_dummy) @ (betahatd - b)

        # Combine them all
        bbb_dummy = term1_dummy @ (term2_dummy + term3_dummy) @ term1_dummy

        # Compute eigenvalues and modify them as in the MATLAB code
        eigaaa_dummy = eigvals(aaa_dummy).real
        eigaaa_dummy[eigaaa_dummy < 1e-12] = 0
        eigaaa_dummy += 1

        eigbbb_dummy = eigvals(bbb_dummy).real
        eigbbb_dummy[eigbbb_dummy < 1e-12] = 0
        eigbbb_dummy += 1

        # Compute normalizing constant for the dummy observations
        norm_dummy = (- n * Td * np.log(np.pi) / 2 + np.sum(gammaln((Td + d - np.arange(n)) / 2) -
                                                            gammaln((d - np.arange(n)) / 2)) -
                      Td * np.sum(np.log(psi)) / 2 - n * np.sum(np.log(eigaaa_dummy)) / 2 -
                      (Td + d) * np.sum(np.log(eigbbb_dummy)) / 2)

        # Update logML with the normalizing constant for the dummy observations
        logML -= norm_dummy

    # Account for re-weighting if Tcovid is not empty
    if Tcovid is not None:
        logML = logML - n * np.sum(np.log(invweights))

    # Update logML based on hyperpriors
    if hyperpriors == 1:
        logML += log_gamma_pdf(lambda_, priorcoef['lambda']['k'], priorcoef['lambda']['theta'])
        if sur == 1:
            logML += log_gamma_pdf(theta, priorcoef['theta']['k'], priorcoef['theta']['theta'])
        if noc == 1:
            logML += log_gamma_pdf(miu, priorcoef['miu']['k'], priorcoef['miu']['theta'])
        if Tcovid is not None:
            logML -= 2 * np.log(eta[0]) + 2 * np.log(eta[1]) + 2 * np.log(eta[2]) + \
                     log_beta_pdf(eta[3], priorcoef['eta4']['alpha'], priorcoef['eta4']['beta'])

    # Finally, invert the sign of logML as in the original MATLAB code
    logML = -logML

    return logML, betahat, sigmahat


def log_beta_pdf(x, al, bet):
    """
    Compute the log probability density function (PDF) of the Beta distribution.

    Args:
        x (float): Value at which to evaluate the PDF.
        al (float): Alpha parameter of the Beta distribution.
        bet (float): Beta parameter of the Beta distribution.

    Returns:
        float: Log PDF of the Beta distribution.
    """

    return (al - 1) * np.log(x) + (bet - 1) * np.log(1 - x) - betaln(al, bet)


def log_gamma_pdf(x, k, theta):
    """
    Computes the log of the Gamma probability density function (PDF) for given values.

    Args:
        x (float or numpy.ndarray): Points at which to evaluate the log of the Gamma PDF. Scalar or array.
        k (float): Shape parameter of the Gamma distribution. Scalar.
        theta (float): Scale parameter of the Gamma distribution. Scalar.

    Returns:
        float or numpy.ndarray: Log of the Gamma PDF evaluated at each point in `x`.
                                 The output will have the same shape as `x`.

    Example:
        >>> log_gamma_pdf(2.0, 1.0, 1.0)
        -2.0

        >>> log_gamma_pdf(np.array([2.0, 3.0]), 1.0, 1.0)
        array([-2., -3.])
    """
    # Compute the log of the Gamma PDF
    r = (k - 1) * np.log(x) - (x / theta) - k * np.log(theta) - gammaln(k)
    return r


def log_ig2pdf(x, alpha, beta):
    """
    Compute the log probability density function (PDF) of the Inverse Gamma distribution.

    Args:
        x (float): Value at which to evaluate the PDF.
        alpha (float): Shape parameter of the Inverse Gamma distribution.
        beta (float): Scale parameter of the Inverse Gamma distribution.

    Returns:
        float: Log PDF of the Inverse Gamma distribution.
    """

    return alpha * np.log(beta) - (alpha + 1) * np.log(x) - beta / x - gammaln(alpha)


def make_positive_definite(matrix):
    """
    Ensures that a given square matrix is positive definite.

    This function first checks if the matrix is already positive definite
    by looking at its eigenvalues. If any eigenvalues are negative, it then
    attempts to make the matrix positive definite by adding a small value to
    its diagonal elements and checking for positive definiteness via Cholesky
    decomposition.

    Args:
        matrix (numpy.ndarray): A square matrix.

    Returns:
        numpy.ndarray: A positive definite matrix.

    Raises:
        ValueError: If the matrix cannot be made positive definite
                after the maximum number of attempts.

    """
    n = matrix.shape[0]
    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        # The matrix is already positive definite
        return matrix

    # The matrix is not positive definite; proceed with regularization
    epsilon = 1e-02  # Starting value to add to the diagonal
    max_attempts = 10000  # Max attempts to avoid infinite loop

    for attempt in range(max_attempts):
        try:
            # Try Cholesky decomposition
            np.linalg.cholesky(matrix)
            return matrix  # Matrix is positive definite
        except np.linalg.LinAlgError:
            # Add epsilon to diagonal elements
            matrix += np.eye(n) * epsilon
            epsilon *= 10  # Increase epsilon

    raise ValueError("Unable to make the matrix positive definite.")


def MissData(y, C, R, c1):
    """
    Eliminates the rows in y, matrices C, R, and vector c1 that correspond to missing data (NaN) in y.

    Args:
        y (numpy.ndarray): Vector of observable data with dimensions (N, 1), where N is the number of observations.
        C (numpy.ndarray): Measurement matrix with dimensions (N, M), where M is the number of state variables.
        R (numpy.ndarray): Covariance matrix with dimensions (N, N).
        c1 (numpy.ndarray): Constant vector with dimensions (N, 1).

    Returns:
        tuple: A tuple containing updated `y`, `C`, `R`, and `c1` after eliminating rows corresponding to missing data.

        - y (numpy.ndarray): Updated vector with dimensions (N_new, 1), where N_new is the number of non-NaN entries in `y`.
        - C (numpy.ndarray): Updated matrix with dimensions (N_new, M).
        - R (numpy.ndarray): Updated covariance matrix with dimensions (N_new, N_new).
        - c1 (numpy.ndarray): Updated constant vector with dimensions (N_new, 1).
    """

    # Create a boolean array where each element is True if the corresponding element in y is not NaN
    ix = ~np.isnan(y)

    # Convert boolean array to integer array
    index_array = np.where(ix.flatten())[0]

    # Update y to only include rows where ix is True (i.e., remove NaN rows)
    y = y[ix.flatten()]

    # Update c1 to only include rows where ix is True (i.e., remove NaN rows)
    c1 = c1[ix.flatten()]

    # Update C to only include rows where ix is True (i.e., remove NaN rows)
    C = C[ix.flatten(), :]

    # Update R to only include rows and columns where ix is True (i.e., remove NaN rows and columns)
    R = R[np.ix_(index_array, index_array)]

    return y, C, R, c1


def numgrad(fcn: Callable, x: np.ndarray, *args: Any) -> Tuple[np.ndarray, int]:
    """
    Compute the numerical gradient of a given function using a central difference approximation.

    Args:
        fcn (Callable): Function whose gradient is to be computed.
        x (np.ndarray): Point at which the gradient is to be computed.
        args (Any): Additional arguments passed to the target function.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the numerical gradient at point x and a flag indicating
                                if any component of the gradient is bad.

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> x = np.array([1, 1])
        >>> g, badg = numgrad(f, x)
    """

    # Define perturbation value for finite difference calculation
    delta = 1e-6

    # Get the length of the input x
    n = len(x)

    # Create a matrix with delta along the diagonal, for perturbing each variable
    tvec = np.eye(n) * delta

    # Initialize the gradient vector
    g = np.zeros(n).reshape(-1, 1)

    # Evaluate the function at the initial point x
    f0 = fcn(x, *args)[0]

    # Flag to indicate if a bad gradient component is encountered
    badg = 0

    # Loop over each dimension to calculate the gradient
    for i in range(n):
        # Scaling factor for perturbation
        scale = 1

        # Select the appropriate perturbation vector
        if x.shape[0] > x.shape[1]:
            tvecv = tvec[i, :].reshape(1, -1)  # Reshape to make it a 2D row vector
        else:
            tvecv = tvec[:, i].reshape(1, -1)  # Reshape to make it a 2D row vector

        # Compute the gradient for the i-th component using central difference
        g0 = (fcn(x + scale * tvecv.T, *args)[0] - f0) / (scale * delta)

        # Check if the gradient component is within acceptable limits
        if abs(g0) < 1e15:
            g[i] = g0
        else:
            # If gradient component is bad, set it to 0 and flag the occurrence
            print('bad gradient ------------------------')
            g[i] = 0
            badg = 1

    return g, badg


def ols1(y, x):
    """
    Perform Ordinary Least Squares (OLS) regression.

    This function computes the OLS coefficients, fitted values, residuals, estimated
    variance of the residuals, and R-squared for a given set of observed dependent
    and independent variables.

    Args:
        y (numpy.ndarray): The dependent variable. Must be a column vector of shape `(nobs, 1)`.
        x (numpy.ndarray): The independent variables. Must be a matrix of shape `(nobs, nvar)`.

    Raises:
        ValueError: If `y` and `x` have different numbers of observations.

    Returns:
        dict: A dictionary containing the following keys:
            - "nobs": Number of observations.
            - "nvar": Number of independent variables.
            - "bhatols": OLS coefficient estimates.
            - "yhatols": Fitted values.
            - "resols": Residuals.
            - "sig2hatols": Estimated variance of residuals.
            - "sigbhatols": Estimated variance-covariance matrix of OLS coefficients.
            - "XX": X'X matrix used in OLS.
            - "R2": R-squared value.

    Example:
        >>> y = np.array([[1], [2], [3]])
        >>> x = np.array([[1, 1], [1, 2], [1, 3]])
        >>> result = ols1(y, x)
    """
    # Ensure x is a 2D array (nobs, nvar)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")

    # Ensure y is a 2D array (nobs, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Reshape y to a column vector if it's 1D
    elif y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y must be a 2D column vector with shape (nobs, 1)")

    # Check if the number of observations in y and x are the same
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same number of observations")

    # Get the number of observations and variables
    nobs, nvar = x.shape

    # Initialize a dictionary to store the results
    result = {"nobs": nobs, "nvar": nvar}

    # Compute the OLS coefficients using the formula: (X'X)^{-1}X'Y
    result["bhatols"] = np.linalg.lstsq(x.T @ x, x.T @ y, rcond=None)[0]

    # Compute the fitted values using the formula: X * bhat
    result["yhatols"] = (x @ result["bhatols"]).reshape(-1, 1)

    # Compute the residuals using the formula: Y - Yhat
    result["resols"] = (y - result["yhatols"]).reshape(-1, 1)

    # Compute the estimated variance of residuals using the formula: res' * res / (n - k)
    result["sig2hatols"] = (result["resols"].T @ result["resols"]) / (nobs - nvar)

    # Compute the estimated variance-covariance matrix of OLS coefficients
    result["sigbhatols"] = result["sig2hatols"] * np.linalg.inv(x.T @ x)

    # Compute X'X for reference
    result["XX"] = x.T @ x

    # Compute R-squared using the formula: Var(Yhat) / Var(Y)
    result["R2"] = np.var(result["yhatols"]) / np.var(y)

    return result


def parse_pv_pairs(default_params, pv_pairs):
    """
    Parses sets of property-value pairs and allows defaults.

    Args:
        default_params (dict): A dictionary with one field for every potential property-value pair.
                       Each field will contain the default value for that property.
                       If no default is supplied for a given property, then that field must be None.

        pv_pairs (list): A list of property-value pairs. Case is ignored when comparing properties to the list
                         of field names. Also, any unambiguous shortening of a field/property name is allowed.

    Returns:
        dict: A dictionary that reflects any updated property-value pairs in pv_pairs.

    Example:
        >>> default_params = {'DerivativeOrder': 1, 'MethodOrder': 4, 'RombergTerms': 2, 'MaxStep': 100,
                              'StepRatio': 2, 'NominalStep': None, 'Vectorized': 'yes', 'FixedStep': None, 'Style': 'central'}
        >>> pv_pairs = ['deriv', 2, 'vectorized', 'no']
        >>> updated_params = parse_pv_pairs(default_params, pv_pairs)
        >>> print(updated_params)

    """
    params = default_params.copy()
    npv = len(pv_pairs)
    n = npv // 2

    if npv % 2 != 0:
        raise ValueError("Property-value pairs must come in PAIRS.")

    if n <= 0:
        # Just return the defaults
        return params

    if not isinstance(params, dict):
        raise ValueError("No structure for defaults was supplied")

    # There was at least one pv pair. Process any supplied.
    propnames = list(params.keys())
    lpropnames = [name.lower() for name in propnames]

    for i in range(0, len(pv_pairs), 2):
        p_i = pv_pairs[i].lower()
        v_i = pv_pairs[i + 1]

        ind = lpropnames.index(p_i) if p_i in lpropnames else None

        if ind is None:
            ind = [j for j, name in enumerate(lpropnames) if name.startswith(p_i)]

            if len(ind) == 0:
                raise ValueError(f"No matching property found for: {pv_pairs[i]}")
            elif len(ind) > 1:
                raise ValueError(f"Ambiguous property name: {pv_pairs[i]}")
            else:
                ind = ind[0]
        p_i = propnames[ind]
        params[p_i] = v_i  # update the value

    return params


def plot_joint_marginal(YY, Y1CondLim, xlab, ylab, vis=False, LW=1.5):
    """
    Plots the joint distribution in the center, with marginals on the side.
    This version also plots a version conditioning on a set of limits on the first variable.

    Args:
        YY (array_like): Matrix containing the data to be plotted.
        Y1CondLim (array_like): Limits for the first variable's conditioning.
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        vis (str, optional): Figure visibility ('on' or 'off'). Defaults to 'off'.
        LW (float, optional): Line width. Defaults to 1.5.

    Returns:
        None: The function plots the joint distribution, marginals, and conditionals.
    """
    plotCond = True if Y1CondLim else False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis._axinfo['grid'].update({'color': 'lightgrey', 'linewidth': 0.5, 'linestyle': '--'})
    ax.yaxis._axinfo['grid'].update({'color': 'lightgrey', 'linewidth': 0.5, 'linestyle': '--'})
    ax.zaxis._axinfo['grid'].update({'color': 'lightgrey', 'linewidth': 0.5, 'linestyle': '--'})

    Y1Lim = np.quantile(YY[:, 0], [0, 1])
    Y2Lim = np.quantile(YY[:, 1], [0, 1])

    gridY1 = np.linspace(Y1Lim[0], Y1Lim[1], 100)
    gridY2 = np.linspace(Y2Lim[0], Y2Lim[1], 100)

    # Plot data
    ax.scatter(YY[:, 0], YY[:, 1], .001 * np.ones(YY[:, 0].shape), s=2.5, c='grey', alpha=.08)
    ax.set_xlim(Y1Lim)
    ax.set_ylim(Y2Lim)

    # Plot unconditional contour
    X, Y = np.meshgrid(gridY1, gridY2)
    values = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(YY.T)
    Z = kernel(values)
    Z = Z.reshape(X.shape)
    ax.contour(X, Y, Z, zdir='z', offset=-0.01, colors='k')

    # Plot conditionals
    if plotCond:
        YYCond = YY[(YY[:, 0] >= Y1CondLim[0]) & (YY[:, 0] <= Y1CondLim[1]), :]
        ax.scatter(YYCond[:, 0], YYCond[:, 1], .001 * np.ones(YYCond[:, 0].shape), s=2.5, c='r', alpha=.08)

        # Plot conditional marginals
        kde_Y1Cond = gaussian_kde(YYCond[:, 0])
        kde_Y2Cond = gaussian_kde(YYCond[:, 1])
        ax.plot(gridY1, Y2Lim[1] * np.ones_like(gridY1), kde_Y1Cond(gridY1) / max(kde_Y1Cond(gridY1)), lw=LW, color='r')
        ax.plot(Y1Lim[0] * np.ones_like(gridY2), gridY2, kde_Y2Cond(gridY2) / max(kde_Y2Cond(gridY2)), lw=LW, color='r')

    # Plot unconditional marginals
    kde_Y1 = gaussian_kde(YY[:, 0])
    kde_Y2 = gaussian_kde(YY[:, 1])
    ax.plot(gridY1, Y2Lim[1] * np.ones_like(gridY1), kde_Y1(gridY1) / max(kde_Y1(gridY1)), lw=LW, color='k')
    ax.plot(Y1Lim[0] * np.ones_like(gridY2), gridY2, kde_Y2(gridY2) / max(kde_Y2(gridY2)), lw=LW, color='k')

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel('Normalized')
    # Set the z-axis limit to prevent the contours from protruding
    ax.set_zlim(-0.01, 1)

    if not vis:
        plt.close()

    return fig, ax  # return the figure and axes objects


def plot_weighted_joint_and_marginals(YY, wStar, xlab, ylab, vis=False, LW=1.5, Y0=None):
    """
    Plots the joint distribution in the center, with marginals on the side.
    This version places Variable 1's marginal on the opposite side to avoid overlapping.

    Args:
        YY (array_like): Data to be plotted (2D array with columns as variables).
        wStar (array_like): Weights for the weighted contours and marginals.
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        vis (bool, optional): Whether to display the plot. Defaults to False.
        LW (float, optional): Line width for the marginals. Defaults to 1.5.
        Y0 (array_like, optional): Reference value to highlight. Defaults to None.

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """

    # Initialize figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Make grid lines lighter
    ax.xaxis._axinfo['grid'].update({'color': 'lightgrey', 'linewidth': 0.5, 'linestyle': '--'})
    ax.yaxis._axinfo['grid'].update({'color': 'lightgrey', 'linewidth': 0.5, 'linestyle': '--'})
    ax.zaxis._axinfo['grid'].update({'color': 'lightgrey', 'linewidth': 0.5, 'linestyle': '--'})

    # Compute limits for both variables
    Y1Lim = np.quantile(YY[:, 0], [0, 1])
    Y2Lim = np.quantile(YY[:, 1], [0, 1])

    # Generate grid for contours
    gridY1 = np.linspace(Y1Lim[0], Y1Lim[1], 100)
    gridY2 = np.linspace(Y2Lim[0], Y2Lim[1], 100)
    GridY1, GridY2 = np.meshgrid(gridY1, gridY2)
    Grid = np.vstack([GridY1.ravel(), GridY2.ravel()]).T

    # Plot data points
    ax.scatter(YY[:, 0], YY[:, 1], 0.001 * np.ones(len(YY)), s=wStar * 50, c='k', alpha=0.15)

    # Set limits for axes
    ax.set_xlim(Y1Lim)
    ax.set_ylim(Y2Lim)

    # Reference lines for marginals
    ax.plot([Y1Lim[0], Y1Lim[0]], Y2Lim, [0, 0], color='k', linewidth=1)
    ax.plot(Y1Lim, [Y2Lim[1], Y2Lim[1]], [0, 0], color='k', linewidth=1)

    # Plot unconditional contours
    kde = gaussian_kde(YY.T)
    Z = kde(Grid.T)
    Z = Z.reshape(GridY1.shape)
    ax.contour(GridY1, GridY2, Z, offset=0, linewidths=0.75, colors='k', linestyles='-')

    # Plot weighted contours
    kde_weighted = gaussian_kde(YY.T, weights=wStar)
    Z_weighted = kde_weighted(Grid.T)
    Z_weighted = Z_weighted.reshape(GridY1.shape)
    ax.contour(GridY1, GridY2, Z_weighted, offset=0, linewidths=0.75, colors='r', linestyles='-')

    # Marginals
    kde_Y1 = gaussian_kde(YY[:, 0])
    kde_Y2 = gaussian_kde(YY[:, 1])
    kde_Y1_weighted = gaussian_kde(YY[:, 0], weights=wStar)
    kde_Y2_weighted = gaussian_kde(YY[:, 1], weights=wStar)

    # Normalize marginals
    fY1 = kde_Y1(gridY1) / max(kde_Y1(gridY1))
    fY2 = kde_Y2(gridY2) / max(kde_Y2(gridY2))
    fY1_weighted = kde_Y1_weighted(gridY1) / max(kde_Y1_weighted(gridY1))
    fY2_weighted = kde_Y2_weighted(gridY2) / max(kde_Y2_weighted(gridY2))

    # Plot marginals
    ax.plot(gridY1, [Y2Lim[0]] * len(gridY1), fY1, linewidth=LW, color='k')  # Opposite side for Variable 1
    ax.plot([Y1Lim[0]] * len(gridY2), gridY2, fY2, linewidth=LW, color='k')  # Standard side for Variable 2
    ax.plot(gridY1, [Y2Lim[0]] * len(gridY1), fY1_weighted, linewidth=LW, color='r')
    ax.plot([Y1Lim[0]] * len(gridY2), gridY2, fY2_weighted, linewidth=LW, color='r')

    # Highlight reference value if provided
    if Y0 is not None:
        ax.scatter(Y0[0], Y0[1], 0.001, color='b', s=50)
        ax.plot([Y1Lim[0], Y0[0]], [Y0[1], Y0[1]], [0.001, 0.001], color='b', linestyle=':', linewidth=LW)
        ax.plot([Y0[0], Y0[0]], [Y0[1], Y2Lim[1]], [0.001, 0.001], color='b', linestyle=':', linewidth=LW)

    # Format axes
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel('Normalized')
    ax.set_xlim(Y1Lim)
    ax.set_ylim(Y2Lim)
    ax.set_zlim(0, 1)
    ax.view_init(30, 45)  # Set view angle

    if not vis:
        plt.close()

    return fig, ax




def printpdf(h, outfilename):
    """
    Saves the given figure as a PDF file with specified dimensions.

    Args:
        h (matplotlib.figure.Figure): The figure object to be saved.
        outfilename (str): The path and name of the output PDF file.

    Example:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 1])
        printpdf(fig, "output.pdf")
    """
    # Set the figure's paper size to its current size
    h.set_size_inches(h.get_figwidth(), h.get_figheight())
    h.savefig(outfilename, format='pdf', bbox_inches='tight')


def quantile_plot(time, quantiles, base_color=None, run_scenario_analysis=False, show_plot=False):
    """
    Plots a line chart with filled quantile bands.

    Args:
        Time (array-like): Time values for the x-axis.
        Quantiles (array-like): Quantiles to be plotted, either a 5 or 7 column matrix.
        baseColor (tuple, optional): RGB color value for the plot. Defaults to blue.

    Raises:
        ValueError: If the quantile matrix is not of size 5 or 7.

    Example:
        >>> Time = np.arange(0, 10, 0.1)
        >>> Quantiles = np.column_stack([np.sin(Time) * i for i in [0.5, 0.75, 1.0, 0.75, 0.5]])
        >>> quantile_plot(Time, Quantiles)
    """
    if base_color is None:
        base_color = np.array([44, 127, 184]) / 255  # Default blue color

    # Adjust the color intensity for the center line in case of 7-column matrix
    color_intensity_adjustment = 1.2 if quantiles.shape[1] == 7 else 1

    # Define a darker and bolder color for the center line
    dark_blue_color = np.array([30, 90, 150]) / 255  # Dark blue color
    adjusted_dark_blue_color = dark_blue_color * color_intensity_adjustment

    # Define a darker and bolder color for the center line
    dark_red_color = np.array([200, 30, 30]) / 255  # Dark red color
    adjusted_dark_red_color = dark_red_color * color_intensity_adjustment

    # Determine the quantiles to plot based on the number of columns
    if quantiles.shape[1] == 5:
        outer_bot, inner_bot, center, inner_top, outer_top = quantiles.T
    elif quantiles.shape[1] == 7:
        outer_bot, middle_bot, inner_bot, center, inner_top, middle_top, outer_top = quantiles.T
    else:
        raise ValueError("Quantiles matrix must have 5 or 7 columns.")

    # Plot the center line with adjusted color intensity
    if run_scenario_analysis:
        plt.plot(time, center, linewidth=2, color=adjusted_dark_red_color)
    else:
        plt.plot(time, center, linewidth=2, color=adjusted_dark_blue_color)

    # Fill operations for inner, middle (if applicable), and outer bands
    # Inner band
    inner_idx = ~np.isnan(inner_bot + inner_top)
    time_inner_flipped = np.concatenate([time[inner_idx], time[inner_idx][::-1]])
    quantiles_inner_flipped = np.concatenate([inner_bot[inner_idx], inner_top[inner_idx][::-1]])
    plt.fill(time_inner_flipped, quantiles_inner_flipped, color=base_color, alpha=0.5, edgecolor='none')

    if quantiles.shape[1] == 7:
        # Middle band
        middle_idx = ~np.isnan(middle_bot + middle_top)
        time_middle_flipped = np.concatenate([time[middle_idx], time[middle_idx][::-1]])
        quantiles_middle_flipped = np.concatenate([middle_bot[middle_idx], middle_top[middle_idx][::-1]])
        plt.fill(time_middle_flipped, quantiles_middle_flipped, color=base_color, alpha=0.25, edgecolor='none')

    # Outer band
    outer_idx = ~np.isnan(outer_bot + outer_top)
    time_outer_flipped = np.concatenate([time[outer_idx], time[outer_idx][::-1]])
    quantiles_outer_flipped = np.concatenate([outer_bot[outer_idx], outer_top[outer_idx][::-1]])
    plt.fill(time_outer_flipped, quantiles_outer_flipped, color=base_color, alpha=0.15, edgecolor='none')

    if show_plot:
        plt.show()


def wquantile(X, p, w):
    """
    Calculate weighted quantiles for each time period.

    Args:
        X (ndarray): Input data of shape (T, nDraws), where T is the number of time periods
                     and nDraws is the number of draws.
        p (float or list of floats): Target quantile(s) to compute (e.g., 0.5 for the median).
        w (ndarray): Weights of shape (nDraws,). Should sum to 1.

    Returns:
        ndarray: An array of shape (T, len(p)) containing the weighted quantiles for each time period.
    """
    T, _ = X.shape
    if isinstance(p, float):  # Ensure `p` is a list if a single float is provided
        p = [p]
    out = np.full((T, len(p)), np.nan)  # Initialize output array with NaN values

    for jQ, p_j in enumerate(p):  # Loop over each quantile
        for jt in range(T):       # Loop over each time period
            X_j = X[jt, :]        # Extract data for the current time period
            sorted_indices = np.argsort(X_j)  # Sort indices based on data
            sorted_X = X_j[sorted_indices]
            sorted_w = w[sorted_indices]  # Sort weights accordingly
            cumsum_w = np.cumsum(sorted_w)  # Compute cumulative sum of weights
            diff = np.abs(cumsum_w - p_j)  # Find the closest cumulative weight to the quantile
            j = np.argmin(diff)  # Get the index of the closest weight
            out[jt, jQ] = sorted_X[j]  # Store the corresponding value

    return out


def runKF_DK(y, A, C, Q, R, x_0, Sig_0, c1, c2):
    """
   Runs Kalman filter using the Durbin and Koopman simulation smoother.

   Args:
       y (numpy.ndarray): Matrix of observable variables of shape (n, T), where n is the number of variables and T
       is the time dimension.
       A (numpy.ndarray): Transition matrix of shape (m, m), where m is the dimension of the state vector.
       C (numpy.ndarray): Measurement matrix of shape (n, m).
       Q (numpy.ndarray): Covariance matrix Q of shape (m, m).
       R (numpy.ndarray): Covariance matrix R of shape (n, n).
       x_0 (numpy.ndarray): Initial state vector of shape (m,).
       Sig_0 (numpy.ndarray): Initial covariance matrix of shape (m, m).
       c1 (numpy.ndarray): Constant vector c1 of shape (n,).
       c2 (numpy.ndarray): Constant vector c2 of shape (m,).

   Returns:
       numpy.ndarray: Smoothed state vector of shape (m, T).
   """

    # Run the filter
    S = SKF(y, C, R, A, Q, x_0, Sig_0, c1, c2)
    # Run the smoother
    S = FIS(y, C, R, A, S)

    return S['AmT']


def rombextrap(StepRatio, der_init, rombexpon):
    """
    Do Romberg extrapolation for each estimate.

    Args:
        StepRatio (float): Ratio decrease in step.
        der_init (np.ndarray): Initial derivative estimates, shaped (n, 1).
        rombexpon (list): Higher order terms to cancel using the Romberg step.

    Returns:
        tuple: Contains the following elements -
            der_romb (np.ndarray): Derivative estimates returned, shaped (n, 1).
            errest (np.ndarray): Error estimates, shaped (n, 1).
    """

    srinv = 1 / StepRatio
    nexpon = len(rombexpon)
    rmat = np.ones((nexpon + 2, nexpon + 1))

    if nexpon == 0:
        # rmat is simple: ones(2,1)
        pass  # rmat is already initialized as ones, so we do nothing
    elif nexpon == 1:
        rmat[1, 1] = srinv ** rombexpon[0]
        rmat[2, 1] = srinv ** (2 * rombexpon[0])
    elif nexpon == 2:
        rmat[1, 1:3] = srinv ** np.array(rombexpon)
        rmat[2, 1:3] = srinv ** (2 * np.array(rombexpon))
        rmat[3, 1:3] = srinv ** (3 * np.array(rombexpon))
    elif nexpon == 3:
        rmat[1, 1:4] = srinv ** np.array(rombexpon)
        rmat[2, 1:4] = srinv ** (2 * np.array(rombexpon))
        rmat[3, 1:4] = srinv ** (3 * np.array(rombexpon))
        rmat[4, 1:4] = srinv ** (4 * np.array(rombexpon))

    qromb, rromb = np.linalg.qr(rmat)

    ne = len(der_init)
    rhs = vec2mat(der_init, nexpon + 2, max(1, ne - (nexpon + 2)))

    rombcoefs = np.linalg.solve(rromb, np.dot(qromb.T, rhs))
    der_romb = rombcoefs[0, :].reshape(-1, 1)

    s = np.sqrt(np.sum((rhs - np.dot(rmat, rombcoefs)) ** 2, axis=0))
    rinv = np.linalg.inv(rromb)
    cov1 = np.sum(rinv ** 2, axis=1).reshape(-1, 1)
    errest = (s * 12.7062047361747 * np.sqrt(cov1[0])).reshape(-1, 1)

    return der_romb, errest


def rosenbrock(x):
    """
    Rosenbrock function.

    Args:
        x (list or numpy.ndarray): A vector of length 2.

    Returns:
        float: The result of the Rosenbrock function evaluated at the given vector `x`.
    """
    y = (1 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 4
    return y


def set_priors(y, lags, **kwargs):
    """
    This function sets up the default choices for the priors of the BVAR of
    Giannone, Lenza and Primiceri (2015)

    Args:
        kwargs (dict): Keyword arguments for various options (see the script for details).
                       The optional keywords customize priors

    Returns:
        tuple: A tuple containing the following elements:
            - r (dict): Dictionary containing the set default choices for the priors.
            - mode (dict): Dictionary containing the mode values for hyperpriors.
            - sd (dict): Dictionary containing the standard deviations for hyperpriors.
            - priorcoef (dict): Dictionary containing coefficients for hyperpriors.
            - MIN (dict): Dictionary containing the minimum bounds for variables.
            - MAX (dict): Dictionary containing the maximum bounds for variables.
            - var_info (list): List containing information about variables in the function's scope.

    Examples:
        >>> some_kwargs = {'hyperpriors': 1, 'Vc': 10e6}
        >>> r, mode, sd, priorcoef, MIN, MAX, var_info = set_priors_covid(**some_kwargs)
        >>> # Now, r, mode, sd, priorcoef, MIN, MAX, and var_info can be used in the bvarGLP_covid function
    """

    # Initialize the r and mn dictionaries to represent r and mn structures
    r = {}
    mn = {}

    # Set hyperpriors
    hyperpriors = kwargs.get('hyperpriors', 1)
    r['setpriors'] = {'hyperpriors': hyperpriors}

    # Set Vc
    Vc = kwargs.get('Vc', 10e6)
    r['setpriors']['Vc'] = Vc

    # Set pos
    pos = kwargs.get('pos', [])
    r['setpriors']['pos'] = pos

    # Check if 'MNalpha' is in kwargs and set 'alpha' in 'mn' dictionary
    mn['alpha'] = kwargs.get('MNalpha', 0)
    # Set MNalpha
    MNalpha = kwargs.get('MNalpha', 0)
    r['setpriors']['MNalpha'] = MNalpha

    # Check if 'MNpsi' is in kwargs and set 'psi' in 'mn' dictionary
    mn['psi'] = kwargs.get('MNpsi', 1)
    # Add 'MNpsi' to 'setpriors' in 'r' dictionary
    r['setpriors']['MNpsi'] = mn['psi']

    # Set sur
    sur = kwargs.get('sur', 1)
    r['setpriors']['sur'] = sur

    # Set noc
    noc = kwargs.get('noc', 1)
    r['setpriors']['noc'] = noc

    # Set Fcast
    Fcast = kwargs.get('Fcast', 1)
    r['setpriors']['Fcast'] = Fcast

    # Set hz
    hz = kwargs.get('hz', list(range(1, 9)))
    r['setpriors']['hz'] = hz

    # Set mcmc
    mcmc = kwargs.get('mcmc', 1)
    r['setpriors']['mcmc'] = mcmc

    # Set Ndraws (M in MATLAB)
    M = kwargs.get('Ndraws', 20000)
    r['setpriors']['Ndraws'] = M

    # Set Ndrawsdiscard (N in MATLAB)
    N = kwargs.get('Ndrawsdiscard', round(M / 2))
    r['setpriors']['Ndrawsdiscard'] = N

    # Set MCMCconst (const in MATLAB)
    const = kwargs.get('MCMCconst', 1)
    r['setpriors']['MCMCconst'] = const

    # Set MCMCfcast
    MCMCfcast = kwargs.get('MCMCfcast', 1)
    r['setpriors']['MCMCfcast'] = MCMCfcast

    # Set MCMCstorecoeff
    MCMCstorecoeff = kwargs.get('MCMCstorecoeff', 1)
    r['setpriors']['MCMCstorecoeff'] = MCMCstorecoeff

    # Set MCMCMsur
    MCMCMsur = kwargs.get('MCMCMsur', 0)
    r['setpriors']['MCMCMsur'] = MCMCMsur

    # Check if 'long_run' is in kwargs and set it in 'r' dictionary
    long_run = kwargs.get('long_run', None)
    # If 'long_run' is not provided, calculate it as the mean of 'y' for the specified range
    if long_run is None:
        long_run = np.mean(y[0:lags, :], axis=0)

    # Set 'long_run' in 'r' dictionary
    r['setpriors']['long_run'] = long_run

    # Initialize to empty dictionaries
    mode = {}
    sd = {}

    # Other options
    if r['setpriors']['hyperpriors'] == 1:
        mode = {'lambda': 0.2, 'miu': 1, 'theta': 1}
        sd = {'lambda': 0.4, 'miu': 1, 'theta': 1}
        scalePSI = 0.02 ** 2

        # Coefficients of hyperpriors
        priorcoef = {'lambda': gamma_coef(mode['lambda'], sd['lambda'], 0),
                     'miu': gamma_coef(mode['miu'], sd['miu'], 0),
                     'theta': gamma_coef(mode['theta'], sd['theta'], 0),
                     'alpha.PSI': scalePSI,
                     'beta.PSI': scalePSI}

    else:
        priorcoef = {}

    # Bounds for maximization in dictionaries
    MIN = {'lambda': 0.0001, 'alpha': 0.1, 'theta': 0.0001, 'miu': 0.0001}
    MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5}

    return (r, mode, sd, priorcoef, MIN, MAX, hyperpriors, Vc, pos, mn, MNalpha, sur, noc, Fcast, hz, mcmc, M,
            N, const, MCMCfcast, MCMCstorecoeff, MCMCMsur, long_run)


def set_priors_covid(priors_params=None, **kwargs):
    """
    This function sets up the default choices for the priors of the BVAR of
    Giannone, Lenza and Primiceri (2015), augmented with a change in
    volatility at the time of Covid (March 2020).

    Args:
        kwargs (dict): Keyword arguments for various options (see the script for details).
                       The optional keywords customize priors

    Returns:
        tuple: A tuple containing the following elements:
            - r (dict): Dictionary containing the set default choices for the priors.
            - mode (dict): Dictionary containing the mode values for hyperpriors.
            - sd (dict): Dictionary containing the standard deviations for hyperpriors.
            - priorcoef (dict): Dictionary containing coefficients for hyperpriors.
            - MIN (dict): Dictionary containing the minimum bounds for variables.
            - MAX (dict): Dictionary containing the maximum bounds for variables.
            - var_info (list): List containing information about variables in the function's scope.

    Examples:
        >>> some_kwargs = {'hyperpriors': 1, 'Vc': 10e6}
        >>> r, mode, sd, priorcoef, MIN, MAX, var_info = set_priors_covid(**some_kwargs)
        >>> # Now, r, mode, sd, priorcoef, MIN, MAX, and var_info can be used in the bvarGLP_covid function
    """
    # Default priors_params if None is provided
    if priors_params is None:
        priors_params = {}

    # Initialize the r and mn dictionaries to represent r and mn structures
    r = {}
    mn = {}

    # Set hyperpriors
    hyperpriors = kwargs.get('hyperpriors', 1) # A flag to enable or disable hyperpriors (default: 1).
    r['setpriors'] = {'hyperpriors': hyperpriors}

    # Set Vc
    Vc = kwargs.get('Vc', 10e6)
    r['setpriors']['Vc'] = Vc

    # Set pos
    pos = kwargs.get('pos', []) #  The time period corresponding to the onset of COVID-19 (default: empty list).
    r['setpriors']['pos'] = pos

    # Check if 'MNalpha' is in kwargs and set 'alpha' in 'mn' dictionary
    mn['alpha'] = kwargs.get('MNalpha', 0)

    # Set MNalpha
    MNalpha = kwargs.get('MNalpha', 0)
    r['setpriors']['MNalpha'] = MNalpha

    # Set Tcovid: check if the key "Tcovid" exists in the r dictionary
    Tcovid = kwargs.get('Tcovid', [])
    r['Tcovid'] = Tcovid

    # Flags for including seemingly unrelated regressions (SUR)
    sur = kwargs.get('sur', 1)
    r['setpriors']['sur'] = sur

    # Set no constants
    noc = kwargs.get('noc', 1)
    r['setpriors']['noc'] = noc

    # Set forecasts
    Fcast = kwargs.get('Fcast', 1)
    r['setpriors']['Fcast'] = Fcast

    # Set time horizon for the forecasts
    hz = kwargs.get('hz', list(range(1, 9)))
    r['setpriors']['hz'] = hz

    # Set mcmc
    mcmc = kwargs.get('mcmc', 1)
    r['setpriors']['mcmc'] = mcmc

    # Set Ndraws (M in MATLAB)
    M = kwargs.get('Ndraws', 20000)
    r['setpriors']['Ndraws'] = M

    # Set Ndrawsdiscard (N in MATLAB)
    N = kwargs.get('Ndrawsdiscard', round(M / 2))
    r['setpriors']['Ndrawsdiscard'] = N

    # Set MCMCconst (const in MATLAB)
    const = kwargs.get('MCMCconst', 1)
    r['setpriors']['MCMCconst'] = const

    # Set MCMCfcast
    MCMCfcast = kwargs.get('MCMCfcast', 1)
    r['setpriors']['MCMCfcast'] = MCMCfcast

    # Set MCMCstorecoeff
    MCMCstorecoeff = kwargs.get('MCMCstorecoeff', 1)
    r['setpriors']['MCMCstorecoeff'] = MCMCstorecoeff

    # Initialize to empty dictionaries
    mode = {}
    sd = {}

    # Other options
    if r['setpriors']['hyperpriors'] == 1:
        scalePSI = 0.02 ** 2
        mode = {
            'lambda': priors_params.get('lambda_mode', 0.2),
            'miu': priors_params.get('miu_mode', 1),
            'theta': priors_params.get('theta_mode', 1)
        }
        sd = {'lambda': priors_params.get('lambda_sd', 0.4), 'miu': priors_params.get('miu_sd', 1),
              'theta': priors_params.get('theta_sd', 1)}

        mode['eta'] = np.array(priors_params.get('eta_mode', [0, 0, 0, 0.8]))
        sd['eta'] = np.array(priors_params.get('eta_sd', [0, 0, 0, 0.2]))

        # Coefficients of hyperpriors
        priorcoef = {
            'lambda': gamma_coef(mode['lambda'], sd['lambda'], 0),
            'miu': gamma_coef(mode['miu'], sd['miu'], 0),
            'theta': gamma_coef(mode['theta'], sd['theta'], 0)
        }

        # Solve for alpha and beta coefficients for eta4
        mosd = [mode['eta'][3], sd['eta'][3]]
        albet = (fsolve(beta_coef, [2, 2], args=(mosd,))).reshape(-1, 1)
        priorcoef['eta4'] = {'alpha': albet[0], 'beta': albet[1]}

    else:
        priorcoef = {}

    # Bounds for maximization in dictionaries

    MIN = {
        'lambda': priors_params.get('lambda_min', 0.0001),
        'alpha': priors_params.get('alpha_min', 0.1),
        'theta': priors_params.get('theta_min', 0.0001),
        'miu': priors_params.get('miu_min', 0.0001),
        'eta': np.array(priors_params.get('eta_min', [1, 1, 1, 0.005]))
    }
    MAX = {
        'lambda': priors_params.get('lambda_max', 5),
        'miu': priors_params.get('miu_max', 50),
        'theta': priors_params.get('theta_max', 50),
        'alpha': priors_params.get('alpha_max', 5),
        'eta': np.array(priors_params.get('eta_max', [500, 500, 500, 0.995]))
    }

    return (r, mode, sd, priorcoef, MIN, MAX, albet, mosd,  hyperpriors, Vc, pos, mn, MNalpha, Tcovid, sur, noc, Fcast,
            hz, mcmc, M, N, const, MCMCfcast, MCMCstorecoeff)


def SKF(Y, Z, R, T, Q, A_0, P_0, c1, c2):
    """
   Kalman filter for stationary systems with time-varying system matrices and missing data.

   The model is:
       \[
       y_t = Z \times a_t + \epsilon_t
       \]
       \[
       a_{t+1} = T \times a_t + u_t
       \]

   Args:
       Y (numpy.ndarray): Data with dimensions (n, nobs), where nobs is the number of observations and n is the number of observable variables.
       Z (numpy.ndarray): Measurement matrix with dimensions (n, m), where m is the number of state variables.
       R (numpy.ndarray): Covariance matrix R with dimensions (n, n).
       T (numpy.ndarray): Transition matrix T with dimensions (m, m).
       Q (numpy.ndarray): Covariance matrix Q with dimensions (m, m).
       A_0 (numpy.ndarray): Initial state vector with dimensions (m, 1).
       P_0 (numpy.ndarray): Initial covariance matrix with dimensions (m, m).
       c1 (numpy.ndarray): Constant vector c1 with dimensions (n, 1).
       c2 (numpy.ndarray): Constant vector c2 with dimensions (m, 1).

   Returns:
       dict: A dictionary containing:
           - 'Am': Predicted state vector \( A_t|t-1 \) with dimensions (m, nobs).
           - 'Pm': Predicted covariance of \( A_t|t-1 \) with dimensions (m, m, nobs).
           - 'AmU': Filtered state vector \( A_t|t \) with dimensions (m, nobs).
           - 'PmU': Filtered covariance of \( A_t|t \) with dimensions (m, m, nobs).
           - 'ZF': A list of arrays, each with dimensions depending on missing data. (nobs x 1, each cell m x n)
           - 'V': A list of arrays, each with dimensions depending on missing data. (nobs x 1, each cell n x 1)
   """

    # Output structure & dimensions
    n, m = Z.shape
    nobs = Y.shape[1]

    # Create a dictionary to represent the MATLAB struct S
    # S is a dictionary with numpy arrays filled with NaNs for Am, Pm, AmU, and PmU,
    # and lists of lists representing empty cells for ZF and V
    S = {
        'Am': np.full((m, nobs), np.nan),
        'Pm': np.full((m, m, nobs), np.nan),
        'AmU': np.full((m, nobs), np.nan),
        'PmU': np.full((m, m, nobs), np.nan),
        # Create a 6x6 list of lists, with each sub-list being empty, to represent the cell arrays
        'ZF': [[[] for _ in range(nobs)] for _ in range(nobs)],
        'V': [[[] for _ in range(nobs)] for _ in range(nobs)]
    }

    # The rest of S can be initialized as before

    Au = A_0  # A_0|0
    Pu = P_0  # P_0|0

    for t in range(nobs):
        # A = A_t|t-1 & P = P_t|t-1
        A = T.dot(Au) + c2
        P = T.dot(Pu).dot(T.T) + Q
        P = 0.5 * (P + P.T)

        # Handling the missing data
        y_t, Z_t, R_t, c1_t = MissData(Y[:, t].reshape(-1, 1), Z, R, c1)

        if y_t.size == 0:
            Au = A
            Pu = P
            ZF = np.zeros((m, 0))
            V = np.zeros((0, 1))
        else:
            PZ = P.dot(Z_t.T)
            F = (Z_t.dot(PZ) + R_t)
            ZF = Z_t.T.dot(np.linalg.inv(F))
            PZF = P.dot(ZF)

            V = y_t - Z_t.dot(A) - c1_t
            Au = A + PZF.dot(V)
            Pu = P - PZF.dot(PZ.T)
            Pu = 0.5 * (Pu + Pu.T)

        S['ZF'][t] = ZF  # set the t-th element of the list to ZF
        S['Am'][:, t] = A[:, 0]  # copy the values from column vector A to the t-th column of 2D array S['Am']
        S['Pm'][:, :, t] = P  # set the t-th matrix in 3D array S['Pm'] to the values in 2D array P
        S['V'][t] = V

        S['AmU'][:, t] = Au[:, 0]  # select the t-th column of S['AmU'] to the values in the column vector Au
        # select all rows and all columns of t-th matrix in 3D array S['PmU'] to the values in 2D array Pu
        S['PmU'][:, :, t] = Pu

    return S


def swapelement(vec, ind, val):
    """
    Replace the element at a specified index 'ind' with a new 'val' in the vector `vec`.

    Args:
        vec (list or numpy.ndarray): The original vector.
        ind (int): The index of the element to be swapped.
        val (float): The new value to be placed at index `ind`.

    Returns:
        list or numpy.ndarray: The vector after the swap.

    Example:
        >>> swapelement([1, 2, 3], 1, 4)
        [1, 4, 3]
    """
    vec[ind] = val
    return vec


def swap2(vec, ind1, val1, ind2, val2):
    """
    Swap the values at the specified indices in the input vector.

    Args:
        vec (list or ndarray): The input vector.
        ind1 (int): The index of the first element to swap.
        val1 (float or int): The value to insert at index `ind1`.
        ind2 (int): The index of the second element to swap.
        val2 (float or int): The value to insert at index `ind2`.

    Returns:
        list or ndarray: The modified vector with values swapped.

    Example:
        Input dimensions for ind1 = 2, ind2 = 1, val1 = 0.0089,
        val2 = 43.72, vec is a 7-element list or ndarray of zeros.
    """

    # Swap the values at the specified indices
    vec[ind1] = val1
    vec[ind2] = val2

    return vec


def transform_data(spec, data_raw):
    """
    Transforms the raw data based on the specified transformation.

    Args:
        spec (dict): A dictionary containing a field called 'Transformation' that specifies
            the transformation to apply to each column of data in data_raw. The
            'Transformation' field should be a list of strings, where each
            string is either 'log' (indicating a logarithmic transformation) or
            'lin' (indicating a linear transformation).
        data_raw (numpy.ndarray): A matrix containing the raw data to be transformed.

    Returns:
        numpy.ndarray: A matrix containing the transformed data.
    """
    # Initialize DataTrans to be a matrix of the same size as DataRaw but with
    # all elements set to NaN. This creates a matrix that will be filled with
    # transformed data.
    data_trans = np.full(data_raw.shape, np.nan)

    # Loop through each transformation specified in trans.
    for i, transformation in enumerate(spec['Transformation']):
        # Check if the ith transformation is a logarithmic transformation.
        if transformation == 'log':
            # If it is logarithmic, transform the ith column of DataRaw using a
            # logarithmic transformation and store the result in the ith column
            # of DataTrans.
            data_trans[:, i] = 100 * np.log(data_raw[:, i])
        # Check if the ith transformation is a linear transformation.
        elif transformation == 'lin':
            # If it is linear, simply copy the ith column of DataRaw into the
            # ith column of DataTrans.
            data_trans[:, i] = data_raw[:, i]
        # If the ith transformation is neither logarithmic nor linear, generate
        # an error.
        else:
            raise ValueError('Enter valid transformation')

    return data_trans


def trimr(x, n1, n2):
    """
    Return a matrix (or vector) x stripped of the specified rows.

    Parameters:
        x (numpy.array): Input matrix (or vector) (n x k)
        n1 (int): First n1 rows to strip
        n2 (int): Last n2 rows to strip

    Returns:
        z (numpy.array): x with the first n1 and last n2 rows removed
    """
    n, _ = x.shape
    if (n1 + n2) >= n:
        raise ValueError("Attempting to trim too much in trimr")

    h1 = n1
    h2 = n - n2

    z = x[h1:h2, :]

    return z


def TypecastToArray(variables):
    """
    Converts a list of variables to their array equivalents, if applicable.

    This function processes a list of variables (`variables`). For each variable, if it is
    a non-empty list, a float, or an int, it converts the variable to a numpy array. Empty
    lists are skipped. The function returns a list of the processed variables.

    Args:
        variables (list): A list of variables to be processed. Each variable can be of any
                          type (list, float, int, etc.).

    Returns:
        list: A list of processed variables, where each non-empty list, float, or int
              variable is converted to a numpy array. Empty lists are excluded.

    Example:
        >>> variables = [1.0, [], [3, 4, 5], 2]
        >>> TypecastToArray(variables)
        [array([1.]), array([3, 4, 5]), array([2])]
    """

    non_empty_variables = []

    for var in variables:
        # Skip empty lists
        if isinstance(var, list) and len(var) == 0:
            continue

        # Convert float or int to a numpy array
        elif isinstance(var, (float, int)):
            var = np.array([var])

        # Flatten the variable if it's neither a float nor an int
        else:
            var = var.flatten()

        # Add the processed variable to the list
        non_empty_variables.append(var)

    return non_empty_variables


def VARcf_DKcks(X, p, beta, Su, nDraws=0):
    """
    Computes conditional forecasts for the missing observations in X using a VAR, Kalman filter and
    the Durban and Koopman smoother.

    Args:
        X (numpy.ndarray): Matrix of observable variables of shape (T, N).
        p (int): Number of lags in VAR.
        beta (numpy.ndarray): Coefficients of the VAR of shape ((N * p + 1), N).
        Su (numpy.ndarray): Covariance matrix of the VAR of shape (N, N).
        nDraws (int, optional): Number of draws. If == 0 then we run a simple Kalman smoother,
                                otherwise we draw nDraws number of draws of the states. Default is 0.

    Returns:
        numpy.ndarray: Matrix where NaNs from X are replaced by the conditional forecasts. Shape is (T, N).
    """

    T, N = X.shape

    # Identify rows with missing observations: False: not missing, True: missing
    idxNaN = np.any(np.isnan(X), axis=1).reshape(-1, 1)
    idxNaNcs = (np.cumsum(idxNaN[::-1, 0])).reshape(-1, 1)
    nNaNs = np.sum(idxNaNcs == np.arange(1, T + 1).reshape(T, 1))

    # Split data into unbalanced and balanced parts
    Xub = X[-(nNaNs + 1):, :]
    X = X[:-nNaNs, :]
    Xinit = X

    # State-space representation: Transition equation
    AA = np.zeros((N * p, N * p))
    AA[:N, :N * p] = beta[:-1, :].T
    AA[N:N * p, :N * (p - 1)] = np.eye(N * (p - 1))
    c2 = np.concatenate([beta[-1, :], np.zeros(N * (p - 1))]).reshape(-1, 1)

    # State-space representation: Measurement equation
    CC = np.zeros((N, N * p))
    CC[:, :N] = np.eye(N)
    QQ = np.zeros((N * p, N * p))
    QQ[:N, :N] = Su
    c1 = np.zeros(N).reshape(-1, 1)

    # Initialize Kalman filter
    lags = list(range(0, p))  # Create a list of lags from 0 to p-1
    initx = lag_matrix(Xinit, lags)
    initx = initx[-1, :].reshape(-1, 1)  # Take the last row of the lag matrix

    initV = np.eye(len(initx)) * 1e-7

    # Conditional forecasts
    # Define yinput
    yinput = Xub[1:, :]
    Tub = yinput.shape[0]

    if nDraws == 0:
        # Point forecast: Kalman filter and smoother
        xsmooth = runKF_DK(yinput.T, AA, CC, QQ, np.diag(np.ones(N) * 1e-12), initx, initV, c1, c2)
        Xcond = np.vstack([Xinit, xsmooth[:N, :].T])
    else:
        # Durbin and Koopman simulation smoother
        Xcond = np.full((T, N, nDraws), np.nan)
        Xcond = Xcond.squeeze(axis=-1)

        for kg in range(nDraws):
            aplus = np.nan * np.empty((N * p, Tub))
            yplus = np.nan * np.empty((N, Tub))

            for t in range(Tub):
                # flatten() is used to convert the array to a 1D array
                aplus[:, t] = (AA @ initx).flatten() + np.concatenate([mvnrnd.rvs(np.zeros(N), Su),
                                                                       np.zeros(N * (p - 1))]) + c2.flatten()
                initx = aplus[:, t]
                yplus[:, t] = (CC @ aplus[:, t] + c1.flatten()).flatten()

            ystar = yinput.T - yplus
            ahatstar = runKF_DK(ystar, AA, CC, QQ, np.diag(np.ones(N) * 1e-12), np.zeros_like(initx).reshape(-1, 1),
                                initV, np.zeros(N).reshape(-1, 1), np.zeros_like(initx).reshape(-1, 1))
            atilda = ahatstar + aplus
            if Xcond.ndim == 3:
                Xcond[:, :, kg] = np.vstack([Xinit, atilda[:N, :].T])
            else:
                Xcond[:, :] = np.vstack([Xinit, atilda[:N, :].T])

    return Xcond


def vec2mat(vec, n, m):
    """
    Forms the matrix M, such that M[i, j] = vec[i + j - 1].

    Args:
        vec (numpy.ndarray): Input vector.
        n (int): Number of rows for the output matrix.
        m (int): Number of columns for the output matrix.

    Returns:
        numpy.ndarray: The resulting matrix.

    Example:
        >>> vec = np.array([1, 2, 3, 4, 5, 6])
        >>> vec2mat(vec, 2, 3)
        array([[1, 2],
               [2, 3],
               [3, 4]])
    """

    # Create grid indices i and j
    i, j = np.meshgrid(np.arange(1, n + 1), np.arange(0, m), indexing='ij')

    # Compute the indices for vec to form the matrix
    ind = i + j

    # Form the matrix using the indices
    mat = vec[ind - 1]
    # Remove singleton dimensions
    mat = np.squeeze(mat)
    # Check if mat is 1D
    if mat.ndim == 1:
        # Reshape to make it a column vector
        mat = mat.reshape(-1, 1)

    # Transpose mat if n == 1
    if n == 1:
        mat = mat.T

    return mat


def verify_bvar_results(bvar_results):
    """
    Check if the first beta and sigma matrices are all zeros.
    If so, replace them with the second beta and sigma matrices.

    Args:
        bvar_results (dict): The results from the BVAR estimation.

    Returns:
        dict: The corrected bvar_results.
    """
    beta = bvar_results['mcmc']['beta']
    sigma = bvar_results['mcmc']['sigma']

    if np.all(beta[:, :, 0] == 0) and np.all(sigma[:, :, 0] == 0):
        beta[:, :, 0] = beta[:, :, 1]
        sigma[:, :, 0] = sigma[:, :, 1]

    return bvar_results


def write_tex_sidewaystable(fid, header, style, table_body, above_tabular, below_tabular=None):
    """
    Writes a LaTeX sidewaystable to the given file identifier or prints it to the screen.

    Args:
        fid (file object): File identifier obtained by opening the file in write mode. Can be None to print to the screen.
        header (list of str): List of column titles/headers for the table.
        style (str): LaTeX style for the columns, such as 'r|cccc' or 'l|rr|rr|'.
        table_body (list of list): Nested list or cell matrix of table content to write. Can include text, numbers, NaN, or empty values.
        above_tabular (str or list of str): Text to write between '\\begin{table}' and '\\begin{tabular}{style}'. Can be a string or a list of strings.
        below_tabular (str or list of str, optional): Text to write between '\\end{tabular}' and '\\end{table}'. Can be a string or a list of strings.

    Example:
        >>> with open('file.tex', 'w') as fid:
                row_names = ['row1', 'row2', 'row3']
                data = [[1.23, 4.56, 7.89], [0.12, 3.45, 6.78], [9.01, 2.34, 5.67]]
                header = ['col1', 'col2', 'col3']
                table_data = [row_names, data]
                style = 'r|ccc'
                above_tabular = 'Random Numbers'
                write_tex_sidewaystable(fid, header, style, table_data, above_tabular)
    """

    def fmt(x):
        if isinstance(x, (int, float)):
            if abs(x) < 1:
                return f"{x:.3f}"
            elif abs(x) < 10:
                return f"{x:.2f}"
            else:
                return f"{x:.1f}"
        elif x == "":
            return ""
        else:
            return str(x).replace('&', '\\&')

    writeline = lambda s: print(s, file=fid)
    writerow = lambda rowcell: writeline(' & '.join(rowcell) + ' \\\\')
    write_cell = lambda to_write: [writeline(fmt(txt)) for txt in to_write]
    write_string_cell = lambda to_write: write_cell([to_write]) if isinstance(to_write, str) else write_cell(to_write)

    writeline('\\begin{sidewaystable}[htpb!]')
    if above_tabular:
        write_string_cell(above_tabular)
    writeline('\\centering')
    writeline(f'\\begin{{tabular}}{{{style}}}')  # Escaping the curly braces

    writerow(header)
    for row in table_body:
        writerow([fmt(cell) for cell in row])

    writeline('\\hline')
    writeline('\\end{tabular}')
    if below_tabular:
        write_string_cell(below_tabular)
    writeline('\\end{sidewaystable}')
    writeline('')
