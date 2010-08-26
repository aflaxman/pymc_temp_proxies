""" Script to replicate some of
http://probabilitynotes.wordpress.com/2010/08/22/global-temperature-proxy-reconstructions-bayesian-extrapolation-of-warming-w-rjags/
in pymc"""

from pylab import *
from pymc import *


# load data
data = csv2rec('BUGS_data.txt', delimiter='\t')


# define priors
beta = Normal('beta', mu=zeros(13), tau=.001, value=zeros(13))
sigma = Uniform('sigma', lower=0., upper=100., value=1.)


# define predictions
pc = array([data['pc%d'%(ii+1)] for ii in range(10)]) # copy pc data into an array for speed & convenience
@deterministic
def mu(beta=beta, temp1=data.lagy1, temp2=data.lagy2, pc=pc):
    return beta[0] + beta[1]*temp1 + beta[2]*temp2 + dot(beta[3:], pc)

@deterministic
def predicted(mu=mu, sigma=sigma):
    return rnormal(mu, sigma**-2.)

# define likelihood
@observed
def y(value=data.y, mu=mu, sigma=sigma):
    return normal_like(value, mu, sigma**-2.)


# generate MCMC samples
vars = [beta, sigma, mu, predicted, y]
mc = MCMC(vars)
mc.use_step_method(Metropolis, beta)
mc.sample(iter=20000, thin=10, burn=10000, verbose=1) # I like to burn-in for 50% of the total samples


# plotting setup
t = range(1999, 1850, -1)
quantiles = predicted.stats()['quantiles']

def smooth(x):
    from pymc import gp
    M = gp.Mean(lambda x: zeros(len(x)))
    C = gp.Covariance(gp.matern.euclidean, amp=1, scale=15, diff_degree=2)
    gp.observe(M, C, range(len(x)), x, .5)
    return M(range(len(x)))

# plot results
figure()
plot(t, smooth(quantiles[50]), color='black', label='Smoothed Estimate')
plot(t, smooth(quantiles[2.5]), color='grey', linestyle='dashed', label='Smoothed 90% Quantiles')
plot(t, smooth(quantiles[97.5]), color='grey', linestyle='dashed')
plot(t, quantiles[50], color='blue', alpha=.5, linewidth=3, label='Unsmoothed Estimate')
plot(t, data.y, color='red', label='HADCRU NH Data')
axis([1845, 2005, -1, 1])
legend(loc='upper left')

# scatter residuals
figure()
scatter(data.y, quantiles[50]-data.y)
xlabel('temp_t')
ylabel('residual_t')
