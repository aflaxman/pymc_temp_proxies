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
mc.sample(iter=500000, thin=100, burn=250000, verbose=1) # I like to burn-in for 50% of the total samples


# plot results
clf()
t = range(1999, 1850, -1)
quantiles = predicted.stats()['quantiles']
plot(t, quantiles[50], color='black', label='Smoothed Estimate')
plot(t, quantiles[2.5], color='grey', linestyle='dashed', label='Smoothed 90% Quantiles')
plot(t, quantiles[97.5], color='grey', linestyle='dashed')
plot(t, data.y, color='red', label='HADCRU NH Data')
axis([1850, 1999, -1, 1])
legend()
