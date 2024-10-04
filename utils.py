import numpy as np
import pandas as pd

def sanity_check(matrix):
    if not np.allclose(np.linalg.norm(matrix,axis=1),1):
        raise ValueError("At least one non unit vector")
    return True

def project_and_calc_dist(X,Y,theta,p):
    
    x_proj = np.dot(X, theta.T)
    y_proj = np.dot(Y, theta.T)
    #N,d = X.shape
    qs = np.linspace(0,1,100)
    xp_quantiles = np.quantile(x_proj, qs, axis=0, method="inverted_cdf")
    yp_quantiles = np.quantile(y_proj, qs, axis=0, method="inverted_cdf")

    
    dist_p = np.abs(xp_quantiles - yp_quantiles)**p

    #mu = np.mean(dist_p)
    #var  =np.var(dist_p)

    #print(mu*mu/var)
    #print(mu/var)
    return dist_p


def activation(vector):
    return np.exp(vector)/np.exp(vector).sum()


def sample_theta(X,num_smaples=10):
    _ , d = X.shape
    theta = np.random.randn(num_smaples,d)
    theta_norm = np.linalg.norm(theta, axis=1)
    theta_normed = theta / theta_norm[:, np.newaxis]
    return theta_normed



def get_mu_var(X):
    ### input array NXL
    N,d = X.shape
    mu_norm = np.linalg.norm(X.mean(axis=0),axis=0)**2/d
    cov = pd.DataFrame(X).cov().values
    trace = np.trace(cov)/d
    return mu_norm, trace