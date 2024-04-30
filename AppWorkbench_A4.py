from PyRT_Common import *
import matplotlib.pyplot as plt
import random
from GaussianProcess import *

# ############################################################################################## #
# Given a list of hemispherical functions (function_list) and a set of sample positions over the #
#  hemisphere (sample_pos_), return the corresponding sample values. Each sample value results   #
#  from evaluating the product of all the functions in function_list for a particular sample     #
#  position.                                                                                     #
# ############################################################################################## #
def collect_samples(function_list, sample_pos_):
    sample_values = []
    for i in range(len(sample_pos_)):
        val = 1
        for j in range(len(function_list)):
            if callable(function_list[j]):
                function = function_list[j]
            else:
                function = function_list[j].eval
            val *= function(sample_pos_[i])
        sample_values.append(RGBColor(val, 0, 0))  # for convenience, we'll only use the red channel
    return sample_values


# ########################################################################################### #
# Given a set of sample values of an integrand, as well as their corresponding probabilities, #
# this function returns the classic Monte Carlo (cmc) estimate of the integral.               #
# ########################################################################################### #
def compute_estimate_cmc(sample_prob_, sample_values_):
    num_samples = len(sample_prob_)
    estimate = 0
    for (sample_prob, sample_value) in zip(sample_prob_, sample_values_):
        estimate += sample_value.r/sample_prob
    return estimate/num_samples

# ----------------------------- #
# ---- Main Script Section ---- #
# ----------------------------- #

if __name__ == '__main__':

    # #################################################################### #
    # STEP 0                                                               #
    # Set-up the name of the used methods, and their marker (for plotting) #
    # #################################################################### #
    methods_label = [('MC', 'o'),('MC IS', 'v'), ('BMC', 'x')]
    # methods_label = [('MC', 'o'), ('MC IS', 'v'), ('BMC', 'x'), ('BMC IS', '1')] # for later practices
    n_methods = len(methods_label) # number of tested monte carlo methods

    # ######################################################## #
    #                   STEP 1                                 #
    # Set up the function we wish to integrate                 #
    # We will consider integrals of the form: L_i * brdf * cos #
    # ######################################################## #
    #l_i = ArchEnvMap()
    cosine_3 = CosineLobe(3)
    cosine_1 = CosineLobe(1)
    integrand = [cosine_1, cosine_3]  # l_i * brdf * cos

    # ############################################ #
    #                 STEP 2                       #
    # Set-up the pdf used to sample the hemisphere #
    # ############################################ #
    uniform_pdf = UniformPDF()
    exponent = 1
    cosine_pdf = CosinePDF(exponent)


    # ###################################################################### #
    # Compute/set the ground truth value of the integral we want to estimate #
    # NOTE: in practice, when computing an image, this value is unknown      #
    # ###################################################################### #
    cosine_4 = CosineLobe(4)
    ground_truth = cosine_4.get_integral()  # Assuming that L_i = 1 and BRDF = 1
    print('Ground truth: ' + str(ground_truth))


    # ################### #
    #     STEP 3          #
    # Experimental set-up #
    # ################### #
    ns_min = 20  # minimum number of samples (ns) used for the Monte Carlo estimate
    ns_max = 100  # maximum number of samples (ns) used for the Monte Carlo estimate
    ns_step = 20  # step for the number of samples
    ns_vector = np.arange(start=ns_min, stop=ns_max, step=ns_step)  # the number of samples to use per estimate
    n_estimates = 20  # the number of estimates to perform for each value in ns_vector
    n_samples_count = len(ns_vector)

    # Initialize a matrix of estimate error at zero
    results = np.zeros((n_samples_count, n_methods))  # Matrix of average error

    # ################################# #
    #      CLASSIC MONTE CARLO         #
    # ################################# #
    print('== Classic Monte Carlo ==')
    # for each sample count considered
    for k, ns in enumerate(ns_vector):

        print(f'Computing estimates using {ns} samples')
        avg_abs_error = 0

        # compute n_estimates for each sample count (ns)
        for _ in range(n_estimates):
            # sample the hemisphere using the uniform pdf
            #   samples_dir: List[Vector3D] - the sampled directions
            #   samples_prob: List[float] - the probability of each sample
            samples_dir, samples_prob = sample_set_hemisphere(ns, uniform_pdf) 
            
            # collect the sample values of the integrand
            #   integrand_samples: List[RGBColor] - the sample values of the integrand (red values only)
            integrand_samples = collect_samples(integrand, samples_dir)
            
            # compute the estimate of the integral using the classic Monte Carlo method
            #   integral_estimate: float - the estimate of the integral
            integral_estimate = compute_estimate_cmc(samples_prob, integrand_samples)
            abs_error_estimate = abs(ground_truth - integral_estimate)
            avg_abs_error += abs_error_estimate

        avg_abs_error /= n_estimates
        results[k, 0] = avg_abs_error
        
        print(f'Average error: {avg_abs_error}')
        
    # ################################# #
    #      CLASSIC MONTE CARLO IS        #
    # ################################# #

    print('== Classic Monte Carlo IS ==')
    # for each sample count considered
    for k, ns in enumerate(ns_vector):

        print(f'Computing estimates using {ns} samples')
        avg_abs_error = 0

        # compute n_estimates for each sample count (ns)
        for _ in range(n_estimates):
            # sample the hemisphere using the uniform pdf
            #   samples_dir: List[Vector3D] - the sampled directions
            #   samples_prob: List[float] - the probability of each sample
            samples_dir, samples_prob = sample_set_hemisphere(ns, cosine_pdf) 
            
            # collect the sample values of the integrand
            #   integrand_samples: List[RGBColor] - the sample values of the integrand (red values only)
            integrand_samples = collect_samples(integrand, samples_dir)
            
            # compute the estimate of the integral using the classic Monte Carlo method
            #   integral_estimate: float - the estimate of the integral
            integral_estimate = compute_estimate_cmc(samples_prob, integrand_samples)
            abs_error_estimate = abs(ground_truth - integral_estimate)
            avg_abs_error += abs_error_estimate

        avg_abs_error /= n_estimates
        results[k, 1] = avg_abs_error
        
        print(f'Average error: {avg_abs_error}')

    # ################################# #
    #       BAYESIAN MONTE CARLO        #
    # ################################# #
    print('== Bayesian Monte Carlo ==')
    
    conv_func = SobolevCov()
    p_func = Constant(1)
    
    gp = GaussianProcess(conv_func, p_func, noise_=0.01)    
    
    for k, ns in enumerate(ns_vector):
        print(f'Computing estimates using {ns} samples')
        avg_abs_error = 0
        
        samples_dir, _ = sample_set_hemisphere(ns, uniform_pdf)
        samples_val = collect_samples(integrand, samples_dir)
        gp.add_sample_pos(samples_dir)
        gp.add_sample_val(samples_val)
        for _ in range(n_estimates): 
            integral_estimate = gp.compute_integral_BMC().r
            abs_error_estimate = abs(ground_truth - integral_estimate)
            avg_abs_error += abs_error_estimate
            
        avg_abs_error /= n_estimates
        results[k, 2] = avg_abs_error
        
        print(f'Average error: {avg_abs_error}')


    # ################################################################################################# #
    # Create a plot with the average error for each method, as a function of the number of used samples #
    # ################################################################################################# #
    for k in range(len(methods_label)):
        method = methods_label[k]
        plt.plot(ns_vector, results[:, k], label=method[0], marker=method[1])

    plt.legend()
    plt.savefig('out/monte_carlo.png')
    plt.show()
