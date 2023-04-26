import numpy as np

# Define the likelihood function
def likelihood(x, mu, sigma):
    n = len(x)
    log_likelihood = -n/2*np.log(2*np.pi*sigma**2) - np.sum((x-mu)**2)/(2*sigma**2)
    return log_likelihood

# Define the partial derivatives of the likelihood function with respect to mu and sigma
def d_likelihood_mu(x, mu, sigma):
    n = len(x)
    d_log_likelihood_mu = np.sum((mu-x)/(sigma**2))
    return d_log_likelihood_mu

def d_likelihood_sigma(x, mu, sigma):
    n = len(x)
    d_log_likelihood_sigma = -n/(2*sigma**2) + np.sum((x-mu)**2)/(2*sigma**4)
    return d_log_likelihood_sigma

# Define a function to perform maximum likelihood estimation
def maximum_likelihood_estimation(x):
    # Initialize the values of mu and sigma
    mu = np.mean(x)
    sigma = np.std(x)

    # Set the learning rate and convergence threshold
    alpha = 0.1
    epsilon = 1e-5

    # Iterate until convergence
    while True:
        # Calculate the partial derivatives of the likelihood function
        d_mu = d_likelihood_mu(x, mu, sigma)
        d_sigma = d_likelihood_sigma(x, mu, sigma)

        # Update the values of mu and sigma
        mu -= alpha*d_mu
        sigma -= alpha*d_sigma

        # Check for convergence
        if np.abs(d_mu) < epsilon and np.abs(d_sigma) < epsilon:
            break

    return mu, sigma

# Generate some sample data
np.random.seed(123)
x = np.random.normal(loc=5, scale=2, size=100)

# Perform maximum likelihood estimation
mu, sigma = maximum_likelihood_estimation(x)

# Print the estimated values of mu and sigma
print('mu:', mu)
print('sigma:', sigma)
