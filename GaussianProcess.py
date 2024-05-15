from PyRT_Common import *
from math import exp
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod  # Abstract Base Class
from AppWorkbench import collect_samples, compute_estimate_cmc

class CovarianceFunction(ABC):

    @abstractmethod
    def eval(self, omega_i, omega_j):
        pass


# Sobolev Covariance function
# [Optimal Sample Weights paper (Marques et al. 2019, Computer Graphics Forum)]
class SobolevCov(CovarianceFunction):
    def __init__(self, s=1.4):
        self.s = s  # Smoothness parameter (controls the smoothness of the GP model)

    def eval(self, omega_i, omega_j):
        s = self.s
        r = Length(omega_i - omega_j)  # Euclidean distance between the samples
        return (2 ** (2 * s - 1)) / s - r ** (2 * s - 2)


# Squared Exponential (SE) Covariance function
# [Spherical Gaussian Framework paper (Marques et al. 2013, IEEE TVCG)]
class SECov(CovarianceFunction):

    def __init__(self, l, noise):
        super().__init__(noise)
        self.l = l  # Length-scale parameter (controls the smoothness of the GP model)

    def eval(self, omega_i, omega_j):
        r = Length(omega_i - omega_j)  # Euclidean distance between the samples
        return exp(-(r ** 2) / (2 * self.l ** 2))


# Gaussian Process class for the unit hemisphere
class GaussianProcess:

    # Initializer
    def __init__(self, cov_func, p_func, noise_=0.01):

        # Attribute containing the covariance function
        self.cov_func = cov_func

        # Analytically-known part of the integrand, i.e., the function p(x)
        self.p_func = p_func

        # Noise term to be added in the diagonal of the covariance matrix. This typically small value is used to avoid
        #  numerical instabilities when inverting the covariance matrix Q (preempts the matrix Q from being singular or
        #  close to singular, and thus not invertible).
        self.noise = noise_

        # Vector with the sample positions
        self.samples_pos = None

        # Vector with the sample values (Y)
        self.samples_val = None

        # Inverted coveriance matrix Q^{-1}
        self.invQ = None

        # Vector of z coefficients (see theory slides and Practice 3 text for more details)
        self.z = None

        # Sample weights. Contains the value by which each sample y_i must be multiplied to compute the BMC estimate.
        #  See text of Practice 3 for more details.
        self.weights = None

    # Method responsible for receiving the vector of sample positions and assigning it to the class attribute sample_pos
    #  Besides that, it also performs the following computations:
    #    - Compute and store the inverse of the covariance matrix (requires knowing the sample positions and the
    #       covariance function)
    #    - Compute and store the z vector (requires knowing the sample positions and the covariance function)
    #    - Compute and store the sample weights given by z^t Q^{-1}
    def add_sample_pos(self, samples_pos_):
        self.samples_pos = samples_pos_
        self.invQ = self.compute_inv_Q()
        self.z = self.compute_z()
        self.weights = self.z @ self.invQ

    # Method which receives and stores the sample values Y (i.e., the observations)
    def add_sample_val(self, samples_val_):
        self.samples_val = samples_val_

    # Method which computes and inverts the covariance matrix Q
    #  - IMPORTANT: requires that the samples positions are already known
    def compute_inv_Q(self):
        n = len(self.samples_pos)
        Q: ndarray = np.zeros((n, n))

        # ################## #
        # ADD YOUR CODE HERE #
        # ################## #
        assert self.samples_pos is not None, "Samples positions must be known to compute the covariance matrix Q"
        
        # Compute all pairwise covariances using broadcasting
        # Q[i, j] now contains self.cov_func.eval(self.samples_pos[i], self.samples_pos[j])
        # sample_pos_np = np.array(self.samples_pos)
        # Q = self.cov_func.eval(sample_pos_np[:, np.newaxis, :], sample_pos_np[np.newaxis, :, :])
        for i in range(n):
            for j in range(n):
                Q[i, j] = self.cov_func.eval(self.samples_pos[i], self.samples_pos[j])


        # Add a diagonal of a small amount of noise to avoid numerical instability problems
        Q = Q + np.eye(n, n) * self.noise ** 2
        return np.linalg.inv(Q)

    # Method in charge of computing the z vector.
    #  - IMPORTANT: requires that the samples positions are already known
    def compute_z(self):
        # The z vector z  = [z_1, z_2, ..., z_n] is a vector of integrals, where the value of each element is computed
        #  based on the position omega_n of the nth sample. In most of the cases, these integrals do not have an
        #  analytic solution. Therefore, we will use classic Monte Carlo to estimate the value of these integrals
        #  (that is, of each element z_i of z).
        
        assert self.samples_pos is not None, "Samples positions must be known to compute the z vector"
        
        # STEP 1: Set-up the pdf used to sample the integrals. We will use the same pdf for all integral estimates
        # (a uniform pdf). The number of samples used in the estimate is hardcoded (50.000). This is a rather
        # conservative figure which could perhaps be reduced without impairing the final result.
        uniform_pdf = UniformPDF()
        ns_z = 50000  # number of samples used to estimate z_i

        # STEP 2: Generate a samples set for the MC estimate
        sample_set_z, sample_prob_set_z = sample_set_hemisphere(ns_z, uniform_pdf)
        ns = len(self.samples_pos)
        z_vec = np.zeros(ns)

        # STEP 3: Compute each z_i element of z
        # for each sample in our GP model
        for i in range(ns):
            # STEP 3.1: Fetch the direction omega_i
            omega_i = self.samples_pos[i]

            # STEP 3.2: Use classic Monte Carlo Integration to compute z_i
            # ################## #
            # ADD YOUR CODE HERE #
            # ################## #
            
            # We want to compute the integral z = S k(omega_i, x)p(x)dx
            # We will use the classic Monte Carlo method to estimate this integral
            cov_func_omega_i = lambda x: self.cov_func.eval(omega_i, x)
            integrand_samples = collect_samples([self.p_func, cov_func_omega_i], sample_set_z)
            
            integral_estimate = compute_estimate_cmc(sample_prob_set_z, integrand_samples)
            
            z_vec[i] = integral_estimate

        return z_vec

    # Method in charge of computing the BMC integral estimate (assuming the the prior mean function has value 0)
    def compute_integral_BMC(self):
        assert self.samples_val is not None, "Samples values must be known to compute the integral estimate"
        res = BLACK

        # ################## #
        # ADD YOUR CODE HERE #
        # ################## #
        # Since prior mean function is 0, the integral estimate is given by z^t Q^{-1} Y
        
        res = np.sum(self.samples_val * self.weights)


        return res
    
    def initialize(self, ns, pdf):
        print("Initializing GP...")
        samples_dir, _ = sample_set_hemisphere(ns, pdf)
        self.add_sample_pos(samples_dir)
        print(f"Computed weights. Length: {len(self.weights)}")
