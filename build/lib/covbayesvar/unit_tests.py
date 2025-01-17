# Importing necessary libraries
import os
import numpy as np
import pandas as pd
from numpy.random import gamma
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from scipy.stats import kde, gaussian_kde, beta, invgamma
import io
import contextlib
import unittest
from unittest.mock import patch
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import numpy.testing as npt

import sys

large_bvar_dir = '/Users/sudikshajoshi/Desktop/Fall 2022/ECON527 Macroeconometrics/BVAR of US Economy/Python Codes'
sys.path.append(large_bvar_dir)

import large_bvar as bvar


class TestBetaCoef(unittest.TestCase):

    def test_beta_coef(self):
        # Define the alpha, beta, mode, and standard deviation values
        x = [2, 5]
        mosd = [0.2, 0.1]

        # Expected results based on the given parameters
        expected_r1 = 0.0  # Expected result for r1
        expected_r2 = -0.05971914124998498  # Expected result for r2

        # Run the function to test
        result = bvar.beta_coef(x, mosd)

        # Check if the results are as expected
        self.assertAlmostEqual(result[0], expected_r1, places=8)
        self.assertAlmostEqual(result[1], expected_r2, places=8)


class TestBFGSI(unittest.TestCase):

    def test_bfgsi_specific_case(self):
        """Test that bfgsi produces the expected output for a specific input."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate a 7x7 diagonal matrix for H0
        H0 = np.diag(np.random.randint(1, 11, 7))

        # Generate 7x1 column vectors for dg and dx
        dg = np.random.rand(7, 1)
        dx = np.random.rand(7, 1)

        # Call the bfgsi function
        H_updated = bvar.bfgsi(H0, dg, dx)

        # Expected output matrix
        expected_H = np.array([
            [13.14000197, -1.85083459, 4.54069634, 3.7829455, -0.14228596, -0.51855766, -2.02075975],
            [-1.85083459, 3.99692431, -1.95826863, -1.21789215, -1.20962392, -0.01440345, -0.04728123],
            [4.54069634, -1.95826863, 10.73845063, 2.71607823, -1.42145471, -0.56288859, -2.18421487],
            [3.7829455, -1.21789215, 2.71607823, 7.32000363, -0.26084473, -0.34309535, -1.33577993],
            [-0.14228596, -1.20962392, -1.42145471, -0.26084473, 4.20682098, -0.36914089, -1.41874631],
            [-0.51855766, -0.01440345, -0.56288859, -0.34309535, -0.36914089, 9.99184358, -0.02909227],
            [-2.02075975, -0.04728123, -2.18421487, -1.33577993, -1.41874631, -0.02909227, 2.89698313]
        ])

        # Check if the output matches the expected output within a certain tolerance
        self.assertTrue(np.allclose(H_updated, expected_H, atol=1e-6))


class TestBvarFcst(unittest.TestCase):

    def test_bvarFcst(self):
        # Define the hypothetical data
        y = np.array([
            [1.2, 0.5, -0.3],
            [1.3, 0.7, -0.1],
            [1.1, 0.4, -0.2],
            [1.4, 0.8, 0.0],
            [1.2, 0.6, -0.1],
            [1.3, 0.7, -0.2],
            [1.2, 0.5, 0.1],
            [1.1, 0.6, -0.1],
            [1.0, 0.4, -0.2],
            [1.3, 0.7, 0.0]
        ])
        beta = np.array([
            [0.5, 0.3, -0.2],
            [0.1, -0.1, 0.2],
            [0.2, 0.0, 0.1],
            [-0.3, 0.2, -0.1],
            [0.0, -0.2, 0.3],
            [-0.1, 0.1, 0.0],
            [0.2, -0.3, 0.2]
        ])
        hz = [1, 2, 3]

        # Expected forecast values at specified horizons
        expected_forecast = np.array([
            [0.69, 0.07, 0.39],
            [0.396, 0.119, 0.296],
            [0.5456, 0.0716, 0.1465]
        ])

        # Run the function to test
        result = bvar.bvarFcst(y, beta, hz)

        # Check if the results are as expected
        self.assertTrue(np.allclose(result, expected_forecast, atol=1e-8))


class TestBvarIrfs(unittest.TestCase):

    def test_bvarIrfs(self):
        # Define the hypothetical data
        beta = np.array([
            [0.2, 0.3],
            [0.1, 0.4],
            [0.5, 0.1],
            [0.3, 0.2],
            [0.1, 0.3]
        ])
        sigma = np.array([
            [0.5, 0.1],
            [0.1, 0.6]
        ])
        nshock = 1
        hmax = 5

        # Expected Impulse Response Functions at different horizons
        expected_irf = np.array([
            [0.70710678, 0.0],
            [0.07071068, 0.28284271],
            [0.36062446, 0.1979899],
            [0.18455487, 0.26304372],
            [0.27796368, 0.23164818]
        ])

        # Run the function to test
        result_irf = bvar.bvarIrfs(beta, sigma, nshock, hmax)

        # Check if the results are as expected
        np.testing.assert_array_almost_equal(result_irf, expected_irf, decimal=7)


class TestCholredFunction(unittest.TestCase):

    def test_cholred(self):
        S = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        expected_result = np.array([
            [1.32114828e-01, - 3.63164917e-02, 5.63606210e-03],
            [8.37615653e-01, 3.34275536e+00, 1.90482279e+00],
            [-1.81133809e+00, - 5.08179769e+00, 9.71450557e+00]
        ])
        computed_result = bvar.cholred(S)
        # Assert that the returned Cholesky decomposition is almost equal to the expected one
        np.testing.assert_almost_equal(computed_result, expected_result, decimal=6)
        print("Unit test for cholred() was successful.")


class TestCsminit(unittest.TestCase):

    def test_csminit(self):
        # Initial parameters based on your example
        np.random.seed(0)
        x0 = np.random.rand(7, 1)
        f0 = 10.5
        g0 = np.random.rand(7, 1)
        badg = 0
        H0 = np.diag([0, 0, 0, 0, 0, 0, 0])

        # Initialize MIN and MAX dicts
        MIN = {'lambda': 0.2, 'alpha': 0.5, 'theta': 0.5, 'miu': 0.5, 'eta': np.array([1, 1, 1, 0.005])}
        MAX = {'lambda': 5, 'alpha': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995])}

        # Other parameters
        T = 50
        n = 4
        lags = 2
        k = n * lags + 1  # Total number of explanatory variables

        # Initialize b matrix with random 0s and 1s
        b = np.random.randint(0, 2, (k, n))
        SS = np.random.rand(n, 1)
        Vc = 10000
        pos = []
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, n)
        hyperpriors = 1
        y = np.random.rand(T, n)
        x = np.hstack([np.ones((T, 1)), np.random.rand(T, k - 1)])  # matrix with the first column as a vector of 1s

        priorcoef = {
            'lambda': {'k': 1.64, 'theta': 0.3123},
            'miu': {'k': 2.618, 'theta': 0.618},
            'theta': {'k': 2.618, 'theta': 0.618},
            'eta4': {'alpha': 3.0347, 'beta': 1.5089}
        }
        Tcovid = 40

        # Create varargin list
        varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]

        # Call csminit function
        fhat, xhat, fcount, retcode = bvar.csminit(bvar.logMLVAR_formin_covid, x0, f0, g0, badg, H0, *varargin)
        # Compare against expected results (replace these with your expected results)
        expected_fhat = 10.5  # Replace with your expected value
        expected_xhat = np.array([0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411, 0.43758721])
        expected_fcount = 25  # Replace with your expected value
        expected_retcode = 6  # Replace with your expected value

        np.testing.assert_allclose(fhat, expected_fhat, rtol=1e-5)
        np.testing.assert_allclose(xhat.reshape(-1), expected_xhat, rtol=1e-5)
        self.assertEqual(fcount, expected_fcount)
        self.assertEqual(retcode, expected_retcode)


class TestCsminwel(unittest.TestCase):

    def test_csminwel(self):
        # Initialize input parameters
        # Set random seed for reproducibility
        np.random.seed(42)

        # Initial parameters for the function
        x0 = np.random.rand(7, 1)  # 7x1 initial point
        H0 = np.diag([1, 1, 1, 1, 1, 1, 1])  # 7x7 initial Hessian
        crit = 0.0001  # Convergence criterion
        nit = 1000  # Number of iterations

        # Initialize MIN and MAX dicts based on your example
        MIN = {'lambda': 0.2, 'alpha': 0.5, 'theta': 0.5, 'miu': 0.5, 'eta': np.array([1, 1, 1, 0.005])}
        MAX = {'lambda': 5, 'alpha': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995])}

        # Simulation parameters
        T = 50  # Number of time periods
        n = 4  # Number of variables
        lags = 2  # Number of lags
        k = n * lags + 1  # Total number of explanatory variables
        Tcovid = 40  # Time of Covid

        # Initialize y and x matrices with random values
        y = np.random.rand(T, n)
        x = np.random.rand(T, k)

        # Initialize other parameters based on your example
        b = np.random.randint(0, 2, (k, n))  # Initialize b matrix with random 0s and 1s
        SS = np.random.rand(n, 1)  # Prior scale matrix
        Vc = 1000  # Prior variance for the constant
        pos = []  # Positions of variables without a constant
        mn = {'alpha': 0}  # Minnesota prior
        sur = 1  # Dummy for the sum-of-coefficients prior
        noc = 1  # Dummy for the no-cointegration prior
        y0 = np.random.rand(1, n)  # Initial values for the variables
        hyperpriors = 1  # Hyperpriors on the VAR coefficients

        priorcoef = {'lambda': {'k': 1.64, 'theta': 0.3123},
                     'miu': {'k': 2.618, 'theta': 0.618},
                     'theta': {'k': 2.618, 'theta': 0.618},
                     'eta4': {'alpha': 3.0347, 'beta': 1.5089}}

        # Assemble varargin list
        varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]

        # Call csminwel function
        fhat, xhat, grad, Hessian, itct, fcount, retcode = bvar.csminwel(bvar.logMLVAR_formin_covid, x0, H0, None, crit,
                                                                         nit, *varargin)

        # Assertions
        expected_fhat = 93.16644977366283
        # Expected best parameter estimates
        expected_xhat = np.array([[24.14819947],
                                  [-19.97180681],
                                  [-18.86958782],
                                  [-12.79727897],
                                  [-22.15735615],
                                  [-14.71176538],
                                  [-16.71914009]])

        # Tolerance for floating point comparison
        tolerance = 1e-6

        # Assertions
        self.assertAlmostEqual(fhat, expected_fhat, places=6, msg="Best function value does not match expected value")
        np.testing.assert_allclose(xhat, expected_xhat, atol=tolerance, rtol=0,
                                   err_msg="Best parameter estimates do not match expected values")

        self.assertIsInstance(fhat, float, "fhat should be a float")
        self.assertIsInstance(xhat, np.ndarray, "xhat should be a numpy array")
        self.assertIsInstance(grad, np.ndarray, "grad should be a numpy array")
        self.assertIsInstance(Hessian, np.ndarray, "Hessian should be a numpy array")
        self.assertIsInstance(itct, int, "itct should be an integer")
        self.assertIsInstance(fcount, int, "fcount should be an integer")
        self.assertIsInstance(retcode, int, "retcode should be an integer")


# Writing the unit test
class TestColsFunction(unittest.TestCase):

    def test_cols(self):
        # Test with a 3x2 matrix
        x1 = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(bvar.cols(x1), 2)

        # Test with a 4x4 matrix
        x2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        self.assertEqual(bvar.cols(x2), 4)

        # Test with a 2x5 matrix
        x3 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self.assertEqual(bvar.cols(x3), 5)

        print("The unit test for the cols() function was successful.")


class TestDrsnbrck(unittest.TestCase):

    def test_derivative(self):
        x = np.array([[1], [1]])
        expected_derivative = np.array([[0], [0]])
        expected_badg = 0

        derivative, badg = bvar.drsnbrck(x)

        # Check if the derivative is as expected
        np.testing.assert_array_almost_equal(derivative, expected_derivative, decimal=5)

        # Check if the bad gradient flag is as expected
        self.assertEqual(badg, expected_badg)


class TestFdamatFunction(unittest.TestCase):

    def test_fdamat(self):
        # Set parameters for fdamat function
        sr = 2  # Scalar for step ratio
        parity = 2  # Scalar for parity
        nterms = 4  # Number of terms

        # Call the fdamat function
        result_mat = bvar.fdamat(sr, parity, nterms)

        # Expected result matrix
        expected_mat = np.array([
            [5.00000000e-01, 4.16666667e-02, 1.38888889e-03, 2.48015873e-05],
            [1.25000000e-01, 2.60416667e-03, 2.17013889e-05, 9.68812004e-08],
            [3.12500000e-02, 1.62760417e-04, 3.39084201e-07, 3.78442189e-10],
            [7.81250000e-03, 1.01725260e-05, 5.29819065e-09, 1.47828980e-12]
        ])

        # Check if the result matches the expected value
        np.testing.assert_array_almost_equal(result_mat, expected_mat, decimal=10,
                                             err_msg="Resulting Matrix does not match expected values")


class TestGammaCoef(unittest.TestCase):

    def test_gamma_coef(self):
        # Define hypothetical data
        mode = 4.0
        sd = 1.5

        # Expected values (These are hypothetical and should be replaced by actual expected values)
        expected_k = 9.0  # Replace with actual expected value
        expected_theta = 0.5  # Replace with actual expected value

        # Call the function
        output = bvar.gamma_coef(mode, sd, 0)

        # Assertions to check if the output matches the expected output
        assert_almost_equal(output['k'], expected_k, decimal=6)
        assert_almost_equal(output['theta'], expected_theta, decimal=6)

    @patch('matplotlib.pyplot.show')
    def test_gamma_coef_plot(self, mock_show):
        # Define hypothetical data
        mode = 4.0
        sd = 1.5
        plotit = 1

        # Call the function
        bvar.gamma_coef(mode, sd, plotit)

        # Assertion to check if plt.show() is called
        mock_show.assert_called_once()


class TestGradestFunction(unittest.TestCase):

    def test_gradest(self):
        np.random.seed(42)  # Set the random seed for reproducibility

        # Initialize parameters
        y = np.random.rand(50, 4)  # Example values
        x = np.random.rand(50, 9)  # Example values
        lags = 2  # Example value
        T = 50  # Example value
        n = 4  # Example value
        b = np.random.rand(9, 4)  # Example values
        MIN = {'lambda': 1e-4, 'miu': 1e-4, 'theta': 1e-4, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
        MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
        SS = np.ones((4, 1)) * 0.5  # Example value
        Vc = 10000  # Example value
        pos = []  # Example value
        mn = {'alpha': 0}  # Example value
        sur = 1  # Example value
        noc = 1  # Example value
        y0 = np.random.rand(1, 4)  # Example values
        hyperpriors = 1  # Example value
        priorcoef = {
            'lambda': {'k': 1.6404, 'theta': 0.3123},
            'theta': {'k': 2.618, 'theta': 0.618},
            'miu': {'k': 2.618, 'theta': 0.618},
            'eta4': {'alpha': 3.0357, 'beta': 1.5089}
        }
        Tcovid = 40  # Example value

        # Function handle
        fun_handle = lambda par: bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur,
                                                             noc, y0, 0, hyperpriors, priorcoef, Tcovid)

        # Initial point (7x1 array)
        x0 = 0.5 + np.random.rand(7, 1)

        # Call gradest to compute the gradient, error, and final delta
        grad, err, finaldelta = bvar.gradest(fun_handle, x0)

        # Expected gradient, error, and final delta (based on your provided output)
        expected_grad = np.array(
            [[20.59197885, 23.13590865, 25.19025695, 18.56024145, 15.37091908, 27.76549828, 19.71576186]])
        expected_err = expected_grad  # Assuming error is equal to gradient in this case
        expected_finaldelta = expected_grad  # Assuming final delta is equal to gradient

        # Use numpy.testing to assert array equality
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=5,
                                             err_msg="Gradient does not match expected values")
        np.testing.assert_array_almost_equal(err, expected_err, decimal=5,
                                             err_msg="Error estimates do not match expected values")
        np.testing.assert_array_almost_equal(finaldelta, expected_finaldelta, decimal=5,
                                             err_msg="Final delta does not match expected values")


class TestHessianFunction(unittest.TestCase):

    def test_hessian(self):
        np.random.seed(42)  # For reproducibility

        # Define initial parameter estimates and other parameters
        par = 0.5 + np.random.rand(7, 1)
        T = 50
        n = 4
        k = 9
        b = (np.random.rand(k, n) > 0.5).astype(int)
        y = np.random.rand(T, n)
        x = np.hstack((np.ones((T, 1)), np.random.rand(T, k - 1)))
        lags = 2
        MIN = {'lambda': 1e-4, 'miu': 1e-4, 'theta': 1e-4, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
        MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
        SS = np.ones((n, 1)) * 0.5
        Vc = 10000
        pos = None
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, n)
        hyperpriors = 1
        Tcovid = 40
        priorcoef = {'lambda': {'k': 1.6404, 'theta': 0.3123},
                     'theta': {'k': 2.618, 'theta': 0.618},
                     'miu': {'k': 2.618, 'theta': 0.618},
                     'eta4': {'alpha': 3.0357, 'beta': 1.5089}}
        varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, 0, hyperpriors, priorcoef, Tcovid]
        fun = lambda params: bvar.logMLVAR_formcmc_covid(params, *varargin)

        # Expected Hessian matrix and error estimates (replace with actual expected values)
        # Expected Hessian matrix and error estimates (replace with actual expected values)
        expected_hessian = np.array([
            [-8.99912285e+01, 0., 0., 0., 0., 0., 0.],
            [0., 1.23692852e+00, 1.45424650e-01, 1.27051263e-02, 0., 1.51430926e-02, 9.74531909e-02],
            [0., 1.45424650e-01, -3.22214646e+00, 1.19026709e-01, 0., -4.37831511e-03, 1.27529524e-01],
            [0., 1.27051263e-02, 1.19026709e-01, -4.07661950e+00, 0., 3.89786664e-01, -8.79105396e-01],
            [0., 0., 0., 0., 7.10999322e+00, 0., 0.],
            [0., 1.51430926e-02, -4.37831511e-03, 3.89786664e-01, 0., 1.24603727e+01, 1.72966808e+00],
            [0., 9.74531909e-02, 1.27529524e-01, -8.79105396e-01, 0., 1.72966808e+00, 5.33168518e+01]
        ])

        expected_err = np.array([
            [1.23179077e-09, 0., 0., 0., 0., 0., 0.],
            [0., 1.02913332e-09, 1.16449056e-08, 1.44101361e-08, 0., 9.05304336e-09, 1.46563520e-08],
            [0., 1.16449056e-08, 4.40801920e-09, 2.64737552e-08, 0., 2.08347159e-08, 1.79511579e-08],
            [0., 1.44101361e-08, 2.64737552e-08, 6.32753484e-09, 0., 6.41493009e-09, 5.82344751e-08],
            [0., 0., 0., 0., 2.65218858e-08, 0., 0.],
            [0., 9.05304336e-09, 2.08347159e-08, 6.41493009e-09, 0., 5.59054262e-09, 3.48882302e-08],
            [0., 1.46563520e-08, 1.79511579e-08, 5.82344751e-08, 0., 3.48882302e-08, 6.74868419e-08]
        ])

        # Call the hessian function
        hess, err = bvar.hessian(fun, par)

        # Check if the results match the expected results
        np.testing.assert_array_almost_equal(hess, expected_hessian, decimal=4,
                                             err_msg="Hessian matrix does not match expected output.")
        np.testing.assert_array_almost_equal(err, expected_err, decimal=4,
                                             err_msg="Error estimates do not match expected output.")


class TestHessdiagFunction(unittest.TestCase):

    def test_hessdiag(self):
        np.random.seed(42)  # For reproducibility

        # Initialize parameters
        y = np.random.rand(50, 4)
        x = np.random.rand(50, 9)
        lags = 2
        T = 50
        n = 4
        b = np.random.rand(9, 4)
        MIN = {'lambda': 1e-4, 'miu': 1e-4, 'theta': 1e-4, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
        MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
        SS = np.ones((4, 1)) * 0.5
        Vc = 10000
        pos = []
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, 4)
        hyperpriors = 1
        priorcoef = {'lambda': {'k': 1.6404, 'theta': 0.3123},
                     'theta': {'k': 2.618, 'theta': 0.618},
                     'miu': {'k': 2.618, 'theta': 0.618},
                     'eta4': {'alpha': 3.0357, 'beta': 1.5089}}
        Tcovid = 40

        # Function handle
        fun_handle = lambda par: bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur,
                                                             noc, y0, 0, hyperpriors, priorcoef, Tcovid)

        # Initial point
        x0 = 0.5 + np.random.rand(7, 1)

        # Expected results (use actual expected values here)
        expected_HD = np.array([-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0])
        expected_err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        expected_finaldelta = np.array(
            [20.59197885, 23.13590865, 25.19025695, 18.56024145, 15.37091908, 27.76549828, 19.71576186])

        # Call hessdiag function
        HD, err, finaldelta = bvar.hessdiag(fun_handle, x0)

        # Check if the results match the expected results
        np.testing.assert_array_almost_equal(HD, expected_HD, decimal=4,
                                             err_msg="Hessian diagonal elements do not match expected output.")
        np.testing.assert_array_almost_equal(err, expected_err, decimal=4,
                                             err_msg="Error estimates do not match expected output.")
        np.testing.assert_array_almost_equal(finaldelta, expected_finaldelta, decimal=4,
                                             err_msg="Final delta does not match expected output.")


class TestKFilterConst(unittest.TestCase):

    def test_kfilter_const(self):
        np.random.seed(123)  # Seed for reproducibility

        # Define parameters
        n = 3  # Dimension of observation
        m = 2  # Dimension of state

        y = np.random.randn(n, 1)
        c = 0.5
        Z = np.random.randn(n, m)
        G = np.eye(n)
        C = 0.2
        T = np.random.randn(m, m)
        H = np.eye(m)
        shat = np.random.randn(m, 1)
        sig = np.eye(m)

        # Expected values (These should be calculated and filled in based on the known output)
        expected_shatnew = np.array([[0.85107116], [0.30028426]])
        expected_signew = np.array([[0.2398133, 0.10354184], [0.10354184, 0.16698419]])
        expected_v = np.array([[-0.22861471], [-2.219862], [0.74400761]])
        expected_k = np.array([[-0.42113884, 0.14477362, 0.02821841], [-0.25258161, -0.2342243, 0.16698094]])
        expected_sigmainv = np.array([[0.21949701, 0.0825495, 0.13912046], [0.0825495, 0.19252829, 0.35860827],
                                      [0.13912046, 0.35860827, 0.800716]])

        # Call the function
        shatnew, signew, v, k, sigmainv = bvar.kfilter_const(y, c, Z, G, C, T, H, shat, sig)

        # Assertions to check if the output matches the expected output
        assert_array_almost_equal(shatnew, expected_shatnew, decimal=6)
        assert_array_almost_equal(signew, expected_signew, decimal=6)
        assert_array_almost_equal(v, expected_v, decimal=6)
        assert_array_almost_equal(k, expected_k, decimal=6)
        assert_array_almost_equal(sigmainv, expected_sigmainv, decimal=6)


class TestLagFunction(unittest.TestCase):

    def test_default_lag(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(bvar.lag(x), expected_output)

    def test_custom_lag(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected_output = np.array([[0, 0], [0, 0], [1, 2]])
        np.testing.assert_array_equal(bvar.lag(x, n=2), expected_output)

    def test_custom_initial_values(self):
        x = np.array([[1], [2], [3]])
        expected_output = np.array([[999], [1], [2]])
        np.testing.assert_array_equal(bvar.lag(x, n=1, v=999), expected_output)

    def test_empty_output(self):
        x = np.array([[1, 2], [3, 4]])
        expected_output = np.array([])
        np.testing.assert_array_equal(bvar.lag(x, n=0), expected_output)


class TestLagMatrix(unittest.TestCase):

    def test_lagmatrix(self):
        # Define the input matrix and lags
        Y = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        lags = [-1, 0, 1]

        # Expected output matrix
        expected_YLag = np.array([
            [2., 7., 1., 6., np.nan, np.nan],
            [3., 8., 2., 7., 1., 6.],
            [4., 9., 3., 8., 2., 7.],
            [5., 10., 4., 9., 3., 8.],
            [np.nan, np.nan, 5., 10., 4., 9.]
        ])

        # Call the function
        output_YLag = bvar.lag_matrix(Y, lags)

        # Assert that the output matches the expected output
        np.testing.assert_array_almost_equal(output_YLag, expected_YLag, decimal=6)


class TestLogBetaPDF(unittest.TestCase):

    def test_log_beta_pdf(self):
        # Define parameters and sample values for testing
        alpha = 2
        beta_param = 5
        x_values = np.array([0.1, 0.3, 0.5])

        # Loop through each sample value and compare custom and scipy results
        for x_value in x_values:
            log_pdf_custom = bvar.log_beta_pdf(x_value, alpha, beta_param)
            log_pdf_scipy = np.log(beta.pdf(x_value, alpha, beta_param))

            # Use assertAlmostEqual because we're comparing floats
            self.assertAlmostEqual(log_pdf_custom, log_pdf_scipy, places=8)


class TestLogGammaPdf(unittest.TestCase):

    def test_scalar_input(self):
        # Test with scalar input
        self.assertAlmostEqual(bvar.log_gamma_pdf(0.2, 1.64, 0.3123), 0.34503764832403605, places=10)

    def test_array_input(self):
        # Test with array input
        x_values = np.array([0.1, 0.2, 0.3])
        expected_results = bvar.log_gamma_pdf(x_values, 1.64, 0.3123)
        calculated_results = np.array([bvar.log_gamma_pdf(0.1, 1.64, 0.3123),
                                       bvar.log_gamma_pdf(0.2, 1.64, 0.3123),
                                       bvar.log_gamma_pdf(0.3, 1.64, 0.3123)])
        np.testing.assert_almost_equal(expected_results, calculated_results, decimal=10)


class TestLogIG2PDF(unittest.TestCase):

    def test_log_ig2pdf(self):
        # Define some parameters and sample values for testing
        alpha = 3
        beta = 2
        x_values = np.array([1.0, 2.0, 3.0])

        # Expected log PDF values based on your earlier test output
        expected_values = np.array([-0.6137056388801095, -2.386294361119891, -3.6748214602192153])

        # Loop through each sample value and compare it to the expected log PDF
        for i, x_value in enumerate(x_values):
            log_pdf_custom = bvar.log_ig2pdf(x_value, alpha, beta)

            # Use the assertAlmostEqual method to check if the log PDF is almost equal to the expected value
            self.assertAlmostEqual(log_pdf_custom, expected_values[i], places=6)


class TestLogMLVARFormCMCCovid(unittest.TestCase):

    def test_output(self):
        # Initialize simulation parameters
        np.random.seed(42)  # For reproducibility

        T = 50  # Number of time periods
        n = 4  # Number of variables
        lags = 2  # Number of lags
        k = n * lags + 1  # Total number of explanatory variables
        Tcovid = 40  # Time of Covid

        # Initialize y and x matrices with random values
        y = np.random.rand(T, n)
        x = np.hstack([np.ones((T, 1)), np.random.rand(T, k - 1)])

        # Initialize other parameters
        b = np.eye(k, n)
        SS = np.random.rand(n, 1)
        Vc = 1000
        pos = []
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, n)
        draw = 1
        hyperpriors = 1

        # Initialize MIN and MAX dicts
        MIN = {'lambda': 0.000001, 'theta': 0.00001, 'miu': 0.00001, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
        MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}

        # Initialize priorcoef dict
        priorcoef = {
            'lambda': {'k': 1.64, 'theta': 0.3123},
            'miu': {'k': 2.618, 'theta': 0.618},
            'theta': {'k': 2.618, 'theta': 0.618},
            'eta4': {'alpha': 3.0347, 'beta': 1.5089}
        }

        # Initialization
        logML = -1e16
        while logML == -1e16:
            # Randomly generate initial parameters within bounds
            par = np.array([
                np.random.rand() * (MAX['lambda'] - MIN['lambda']) + MIN['lambda'],
                np.random.rand() * (MAX['theta'] - MIN['theta']) + MIN['theta'],
                np.random.rand() * (MAX['miu'] - MIN['miu']) + MIN['miu'],
                np.random.rand() * (MAX['alpha'] - MIN['alpha']) + MIN['alpha'],
                np.random.rand() * (MAX['eta'][3] - MIN['eta'][3]) + MIN['eta'][3],
                np.random.rand(),
                np.random.rand()
            ])  # Additional parameters, adjust as needed

            # Call the function
            logML, betadraw, drawSIGMA = bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                                     pos, mn, sur, noc, y0, draw, hyperpriors,
                                                                     priorcoef,
                                                                     Tcovid)

        expected_logML = np.array([-105.69820295])
        expected_betadraw = np.array([
            [-0.31123887, 0.24959518, 0.21967228, 0.2467134],
            [0.32409376, -0.12282054, 0.51212435, 0.14470938],
            [0.22814382, 0.17112802, -0.23152074, 0.16701049],
            [0.06380487, -0.06523445, 0.40805655, 0.35325742],
            [0.24355404, 0.03028042, 0.06121041, 0.21799587],
            [0.11082769, 0.04172674, -0.1082554, -0.10506035],
            [0.28891899, -0.06275433, -0.12859445, -0.11743278],
            [0.393452, 0.05508448, 0.03364207, 0.0300137],
            [-0.05522673, 0.3624359, -0.06193593, -0.26248391]
        ])
        expected_drawSIGMA = np.array([
            [0.07675985, 0.00410223, -0.00521122, 0.0270709],
            [0.00410223, 0.04625054, 0.00930697, 0.02281862],
            [-0.00521122, 0.00930697, 0.08293307, 0.01153445],
            [0.0270709, 0.02281862, 0.01153445, 0.06159708]
        ])

        # Check if the output matches the expected output
        np.testing.assert_almost_equal(logML, expected_logML, decimal=5)
        np.testing.assert_almost_equal(betadraw, expected_betadraw, decimal=5)
        np.testing.assert_almost_equal(drawSIGMA, expected_drawSIGMA, decimal=5)


class TestLogMLVARForminCovid(unittest.TestCase):

    def test_function(self):
        T = 50  # Number of time points
        n = 4  # Number of endogenous variables
        k = 9  # Number of lags * number of endogenous variables + 1
        Tcovid = 40  # The time of Covid change, just for this example
        np.random.seed(1234)
        y = np.random.randn(T, n)
        x = np.random.randn(T, k)
        lags = 2
        b = np.random.randn(k, n)
        MIN = {'lambda': 0.1, 'theta': 0.1, 'miu': 0.1, 'eta': np.array([0.1, 0.2, 0.3, 0.4]), 'alpha': 0.1}
        MAX = {'lambda': 1, 'theta': 1, 'miu': 1, 'eta': np.array([1, 1, 1, 1]), 'alpha': 1}
        SS = np.ones((n, 1)) * 0.5
        Vc = 10000
        pos = []
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.ones((1, n))
        hyperpriors = 1
        priorcoef = {'lambda': {'k': 1, 'theta': 1},
                     'theta': {'k': 1, 'theta': 1},
                     'miu': {'k': 1, 'theta': 1},
                     'eta4': {'alpha': 1, 'beta': 1}}
        par = np.ones((7, 1))

        # Call the function
        logML, betahat, sigmahat = bvar.logMLVAR_formin_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                              pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid)
        # Expected values based on your provided output
        expected_logML = np.array([349.46792507])
        expected_betahat = np.array([
            [-0.41670828, -0.18526284, 0.14186856, -0.04046313],
            [-0.02283031, 0.03584175, -0.11658471, -0.06921235],
            [0.20444668, -0.25867623, -0.07832122, 0.46818139],
            [0.04444082, -0.24771808, 0.00499505, 0.05005965],
            [0.08291467, 0.11964628, 0.1987281, 0.07635984],
            [0.17774874, 0.16530029, 0.15993353, -0.22262598],
            [-0.01239406, 0.43564166, 0.13072246, -0.06106638],
            [-0.22493754, 0.33691157, -0.07971811, -0.02512848],
            [-0.00719068, 0.10573903, 0.04674718, -0.1464284]])

        expected_sigmahat = np.array([
            [0.85964397, -0.08049621, 0.08914717, -0.10597384],
            [-0.08049621, 1.12915988, -0.10150925, -0.03443026],
            [0.08914717, -0.10150925, 0.90780937, -0.4315801],
            [-0.10597384, -0.03443026, -0.4315801, 1.14333825]
        ])

        # Check if the function output matches the expected output
        np.testing.assert_almost_equal(logML, expected_logML, decimal=8)
        np.testing.assert_almost_equal(betahat, expected_betahat, decimal=8)
        np.testing.assert_almost_equal(sigmahat, expected_sigmahat, decimal=8)


class TestMissData(unittest.TestCase):

    def test_miss_data(self):
        # Input data
        y = np.array([1, 2, np.nan, 4, 5])
        C = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        R = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        c1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Expected output
        expected_y = np.array([1., 2., 4., 5.])
        expected_C = np.array([[1, 2], [3, 4], [7, 8], [9, 10]])
        expected_R = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        expected_c1 = np.array([0.1, 0.2, 0.4, 0.5])

        # Call the function to test
        updated_y, updated_C, updated_R, updated_c1 = bvar.MissData(y, C, R, c1)

        # Check if the function output matches the expected output
        np.testing.assert_array_almost_equal(updated_y, expected_y, decimal=6)
        np.testing.assert_array_almost_equal(updated_C, expected_C, decimal=6)
        np.testing.assert_array_almost_equal(updated_R, expected_R, decimal=6)
        np.testing.assert_array_almost_equal(updated_c1, expected_c1, decimal=6)


class TestNumGrad(unittest.TestCase):

    def test_numgrad(self):
        # Your settings here (including the function logMLVAR_formin_covid and the actual parameters)

        # Initialize input parameters
        T = 50
        n = 4
        k = 9  # Number of lags * number of endogenous variables + 1
        lags = 2
        Tcovid = 40  # The time of Covid change, just for this example

        # Random data and initial parameter estimates
        np.random.seed(1234)
        par = np.random.rand(7, 1)
        y = np.random.randn(T, n)
        x = np.hstack([np.ones((T, 1)), np.random.randn(T, k - 1)])
        b = np.random.randint(0, 2, size=(k, n)).astype(float)

        # Initialize MIN and MAX dictionaries for hyperparameter bounds
        MIN = {'lambda': 0.1, 'theta': 0.1, 'miu': 0.1, 'eta': np.array([1, 1, 1, 0.005]), 'alpha': 0.1}
        MAX = {'lambda': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995]), 'alpha': 5}

        # Other parameters
        SS = np.ones((n, 1)) * 0.5
        Vc = 10000
        pos = None
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, n)
        hyperpriors = 1

        # Initialize priorcoef dictionary
        priorcoef = {
            'lambda': {'k': 1.64, 'theta': 0.3123},
            'theta': {'k': 2.618, 'theta': 0.618},
            'miu': {'k': 2.618, 'theta': 0.618},
            'eta4': {'alpha': 3.0347, 'beta': 1.5089}
        }

        # Package additional arguments into a tuple (analogous to varargin in MATLAB)
        varargin = (y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid)

        # Call numgrad function
        computed_grad, badg = bvar.numgrad(bvar.logMLVAR_formin_covid, par, *varargin)

        # Expected gradient and bad flag based on your previous Python execution
        expected_grad = np.array([
            [14.9289242],
            [2.08932659],
            [2.34590436],
            [11.64802791],
            [43.67792621],
            [20.84261996],
            [19.25043603]
        ])
        expected_badg = 0

        # Validate results
        np.testing.assert_array_almost_equal(computed_grad, expected_grad, decimal=6)
        self.assertEqual(badg, expected_badg)


class TestOls1(unittest.TestCase):

    def test_ols1(self):
        # Given data
        y = np.array([[2.5], [3.6], [4.2], [4.8], [6.1]])
        x = np.array([
            [1, 2.1, 1.5],
            [1, 2.8, 2.1],
            [1, 3.3, 2.9],
            [1, 3.7, 3.2],
            [1, 4.4, 3.8]
        ])

        # Expected output
        expected_nobs = 5
        expected_nvar = 3
        expected_bhatols = np.array([[-1.2767002], [2.34004024], [-0.78215962]])
        expected_yhatols = np.array([[2.46414487], [3.63287726], [4.17716968], [4.87853789], [6.04727029]])
        expected_resols = np.array([[0.03585513], [-0.03287726], [0.02283032], [-0.07853789], [0.05272971]])
        expected_sig2hatols = np.array([[0.00591818]])
        expected_sigbhatols = np.array([[0.07531206, -0.08942756, 0.08052053], [-0.08942756, 0.13098578, -0.12503188],
                                        [0.08052053, -0.12503188, 0.121142]])
        expected_R2 = 0.9983587976370011

        # Run function
        result = bvar.ols1(y, x)

        # Assertions
        self.assertEqual(result['nobs'], expected_nobs)
        self.assertEqual(result['nvar'], expected_nvar)
        np.testing.assert_array_almost_equal(result['bhatols'], expected_bhatols, decimal=6)
        np.testing.assert_array_almost_equal(result['yhatols'], expected_yhatols, decimal=6)
        np.testing.assert_array_almost_equal(result['resols'], expected_resols, decimal=6)
        np.testing.assert_array_almost_equal(result['sig2hatols'], expected_sig2hatols, decimal=6)
        np.testing.assert_array_almost_equal(result['sigbhatols'], expected_sigbhatols, decimal=6)
        self.assertAlmostEqual(result['R2'], expected_R2, places=6)


class TestPlotJointMarginal(unittest.TestCase):

    def test_plot_joint_marginal(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        N = 500
        Y1 = np.random.normal(0, 1, N)
        Y2 = 0.5 * Y1 + np.random.normal(0, 1, N)
        YY = np.column_stack((Y1, Y2))
        Y1CondLim = [-1, 1]
        xlab = 'Variable 1'
        ylab = 'Variable 2'

        # Call the function to test
        fig, ax = bvar.plot_joint_marginal(YY, Y1CondLim, xlab, ylab, vis=True)

        # Test if the function returns a figure and axes object
        self.assertIsInstance(fig, plt.Figure, "The function should return a matplotlib figure object")
        self.assertIsInstance(ax, plt.Axes, "The function should return a matplotlib axes object")

        # show the plot for visual inspection: manually close the figure to complete running the test
        plt.show()


class TestPrintPDF(unittest.TestCase):

    def setUp(self):
        # Initialize a figure for testing
        self.fig, self.ax = plt.subplots()
        self.ax.plot([1, 2, 3], [1, 2, 3])
        self.outfilename = "test_output.pdf"

    def tearDown(self):
        # Remove the test file if it exists
        if os.path.exists(self.outfilename):
            os.remove(self.outfilename)

    def test_printpdf(self):
        print("Current Working Directory:", os.getcwd())
        # Run the function
        bvar.printpdf(self.fig, self.outfilename)

        # Check if the file has been created
        self.assertTrue(os.path.exists(self.outfilename), "PDF file was not created")


class TestRombextrapFunction(unittest.TestCase):

    def test_rombextrap(self):
        np.random.seed(42)  # For reproducibility
        # Initialize parameters
        StepRatio = 2
        der_init = np.random.rand(23, 1)  # 23x1 numpy array
        rombexpon = [4, 6]

        # Expected derivative estimates and error estimates
        # (Use actual expected values here)
        expected_der = np.array([[0.65042152], [0.35344968], [0.13780844], [0.1031797],
                                 [0.49186654], [0.75596796], [0.64769791], [0.34014663],
                                 [0.50649885], [0.93497252], [0.49029567], [0.17028774],
                                 [0.18174954], [0.2489556], [0.42870272], [0.48350796],
                                 [0.35175832], [0.45918454], [0.36923581]])
        expected_errest = np.array([[0.7367285], [2.75392367], [0.21924387], [0.62627539],
                                    [5.19495949], [2.09067128], [0.81804874], [4.43305326],
                                    [6.38838258], [1.35203805], [3.8747612], [0.11333945],
                                    [0.02138434], [0.76856581], [1.3442091], [0.70001997],
                                    [0.84858528], [2.11128936], [3.16856029]])

        # Call the rombextrap function
        der_romb, errest = bvar.rombextrap(StepRatio, der_init, rombexpon)

        # Check if the results match the expected results
        np.testing.assert_array_almost_equal(der_romb, expected_der, decimal=4,
                                             err_msg="Derivative estimates do not match expected output.")
        np.testing.assert_array_almost_equal(errest, expected_errest, decimal=4,
                                             err_msg="Error estimates do not match expected output.")


class TestRsnbrckFunction(unittest.TestCase):

    def test_rsnbrck_at_origin(self):
        self.assertEqual(bvar.rosenbrock([0, 0]), 1)

    def test_rsnbrck_at_one_one(self):
        self.assertEqual(bvar.rosenbrock([1, 1]), 0)

    def test_rsnbrck_at_negative_one_two(self):
        self.assertEqual(bvar.rosenbrock([-1, 2]), 109)


class TestSwap2Function(unittest.TestCase):

    def test_swap2(self):
        # Initialize a 7x1 vector of zeros
        initial_vec = np.zeros((7, 1))

        # Define indices and values to be swapped
        ind1 = 1  # Python uses 0-based indexing, so ind1 = 1 corresponds to MATLAB's ind1 = 2
        val1 = 0.0089
        ind2 = 0  # Python uses 0-based indexing, so ind2 = 0 corresponds to MATLAB's ind2 = 1
        val2 = 43.72

        # Expected output after swap
        expected_vec = np.array([
            [43.72],
            [0.0089],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]
        ])

        # Call the swap2 function on a copy of initial_vec
        swapped_vec = bvar.swap2(initial_vec.copy(), ind1, val1, ind2, val2)

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(swapped_vec, expected_vec, decimal=7,
                                             err_msg="The resulting vector does not match the expected output.")


class TestSwapelementFunction(unittest.TestCase):

    def test_swapelement(self):
        # Create a 7x1 numpy array
        vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        # Choose an index and a value for the swap
        ind = 0  # Note that Python uses zero-based indexing
        val = 0.8745

        # Expected output after the swap
        expected_vec = np.array([0.8745, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        # Call the swapelement function
        vec_modified = bvar.swapelement(vec, ind, val)

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(vec_modified, expected_vec, decimal=4,
                                             err_msg="The resulting vector does not match the expected output.")


class TestTransformData(unittest.TestCase):

    def test_transform_data(self):
        # Sample raw data
        data_raw = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0]
        ])

        # Transformation specifications
        spec = {
            'Transformation': ['log', 'lin', 'log', 'lin']
        }

        # Expected result
        expected_result = np.array([
            [0., 2., 109.86122887, 4.],
            [69.31471806, 4., 179.17594692, 8.],
            [109.86122887, 6., 219.72245773, 12.]
        ])

        # Call the function
        result = bvar.transform_data(spec, data_raw)

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(result, expected_result, decimal=8)


class TestTrimrFunction(unittest.TestCase):

    def test_trimr(self):
        # Create a sample matrix
        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])

        # Specify the number of rows to trim from the top and bottom
        n1 = 1
        n2 = 1

        # Expected result
        expected_result = np.array([
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        # Call the function
        result = bvar.trimr(x, n1, n2)

        # Assert that the result matches the expected result
        np.testing.assert_array_equal(result, expected_result)


class TestWriteTexSidewaystable(unittest.TestCase):

    def test_write_tex_sidewaystable(self):
        # Prepare arguments for the function
        header = ['Header 1', 'Header 2', 'Header 3']
        style = 'l|c|r'
        table_body = [
            ['Row 1, Col 1', 1.23, 'Row 1, Col 3'],
            ['Row 2, Col 1', 4.56, 'Row 2, Col 3'],
            ['Row 3, Col 1', 7.89, 'Row 3, Col 3']
        ]
        above_tabular = 'This is a sample table.'
        below_tabular = 'Table notes go here.'

        # Redirect the output to a string
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            bvar.write_tex_sidewaystable(None, header, style, table_body, above_tabular, below_tabular)

            # Fetch the generated LaTeX code and strip leading/trailing white spaces from each line
            latex_output = '\n'.join(line.strip() for line in f.getvalue().split('\n')).strip()

            # Prepare the expected LaTeX code and strip leading/trailing white spaces from each line
            expected_output = '''\\begin{sidewaystable}[htpb!]
                                        This is a sample table.
                                        \\centering
                                        \\begin{tabular}{l|c|r}
                                        Header 1 & Header 2 & Header 3 \\\\
                                        Row 1, Col 1 & 1.23 & Row 1, Col 3 \\\\
                                        Row 2, Col 1 & 4.56 & Row 2, Col 3 \\\\
                                        Row 3, Col 1 & 7.89 & Row 3, Col 3 \\\\
                                        \\hline
                                        \\end{tabular}
                                        Table notes go here.
                                        \\end{sidewaystable}
                                        '''
            expected_output = '\n'.join(line.strip() for line in expected_output.split('\n')).strip()

            # Assert if the generated LaTeX code matches the expected code
            self.assertEqual(latex_output, expected_output)


class TestVec2Mat(unittest.TestCase):

    def test_vec2mat(self):
        np.random.seed(42)  # For reproducibility

        # Create a hypothetical vector vec of size 26x1
        vec = np.random.rand(26, 1)

        # Define n and m based on the comments
        n = 23
        m = 2

        # Expected result
        expected_result = np.array([
            [0.37454012, 0.95071431],
            [0.95071431, 0.73199394],
            [0.73199394, 0.59865848],
            [0.59865848, 0.15601864],
            [0.15601864, 0.15599452],
            [0.15599452, 0.05808361],
            [0.05808361, 0.86617615],
            [0.86617615, 0.60111501],
            [0.60111501, 0.70807258],
            [0.70807258, 0.02058449],
            [0.02058449, 0.96990985],
            [0.96990985, 0.83244264],
            [0.83244264, 0.21233911],
            [0.21233911, 0.18182497],
            [0.18182497, 0.18340451],
            [0.18340451, 0.30424224],
            [0.30424224, 0.52475643],
            [0.52475643, 0.43194502],
            [0.43194502, 0.29122914],
            [0.29122914, 0.61185289],
            [0.61185289, 0.13949386],
            [0.13949386, 0.29214465],
            [0.29214465, 0.36636184]
        ])

        # Call the vec2mat function
        result_mat = bvar.vec2mat(vec, n, m)

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(result_mat, expected_result, decimal=7,
                                             err_msg="The resulting matrix does not match the expected output.")

# Run the tests for all classes in the script
if __name__ == '__main__':
    unittest.main()


# Run the specific test
# if __name__ == '__main__':
#     # Load all tests from TestBetaCoef class
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestBetaCoef)
#
#     # Run the test suite
#     unittest.TextTestRunner().run(suite)


