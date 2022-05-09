import numpy as np 
import random
import csv
import argparse


def UVM(data, lower, upper, var, rho, beta):
    n = len(data)
    sd = np.sqrt(var)
    
    clamped_data = np.copy(data)
    
    T_lower = lower - sd*np.sqrt(2*np.log(2*n/beta))
    T_upper = upper + sd*np.sqrt(2*np.log(2*n/beta))
    
    clamped_data[clamped_data>T_upper] = T_upper
    clamped_data[clamped_data<T_lower] = T_lower
    
    delta = (upper-lower+2*sd*np.sqrt(2*np.log(2*n/beta)))/n
    
    sd = (delta/np.sqrt(2*rho))
    Y = np.random.normal(0, sd)    
    Z = np.mean(clamped_data) + Y
    
    new_lower = Z - np.sqrt(2*((var/n) + (delta/np.sqrt(2*rho))**2)*np.log(2/beta))
    new_upper = Z + np.sqrt(2*((var/n) + (delta/np.sqrt(2*rho))**2)*np.log(2/beta))
    
    return new_lower, new_upper, sd



def UVMRec(data, bounds, true_var, t, rho, beta):

    lower = bounds[0]
    upper = bounds[1]
    
    for i in range(t-1):
        lower, upper, sd = UVM(data, lower, upper, true_var, rho/(4*(t-1)), (beta/(4*(t-1))))

    lower, upper, sd = UVM(data, lower, upper, true_var, 3*rho/4, beta/4)
    
    return ((lower+upper)/2), sd


def estimator_mean(data):
    return np.mean(data)


def blb(estimator_func, data, n, true_var, bounds_theta, bounds_sigma_sq):
    
    k = int(4 * np.sqrt(n)) # num partitions
    r = 500 #bootstrap iterations

    # tau: estimators
    tau = np.zeros([k,r]) 

    theta_vec = []
    sigma_sq_vec = []
    # randomly partition X into k subsets
    random.shuffle(data)
    partition_size = int(np.floor(n/k))
    partitions = [data[i:i + partition_size] for i in range(0, len(data), partition_size)]

    # range for randint
    low=0
    high=len(partitions[0]) #don't need to do b-1 because upper bound is exclusive

    # for each partition
    for i in range(k):
    
        b = len(partitions[i]) 
    
        # for a partition, create r subsets
        for c in range(r):
            I = np.random.randint(low, high, size=n)
            replicate = data[I]
        
            tau[i,c] = estimator_func(replicate)  
           
        low = low + b
        high = high + b
        
    # mean and var of estimators for each prtition, not the data        
    theta_vec = np.mean(tau,axis=1)
    sigma_sq_vec = np.var(tau,axis=1) #var

    #inputs for UVMRec
    t = 10
    rho = 20
    beta = 0.1
    
    est_sd_theta = np.sqrt(true_var * 1/len(data))
    
    theta, sd1 = UVMRec(theta_vec, bounds_theta, est_sd_theta, t, rho/2, beta)
    
    est_sd_sigma_sq = (2*(np.sqrt(true_var)**4))/(len(data)-1)
    
    sigma_sq, sd2 = UVMRec(sigma_sq_vec, bounds_sigma_sq, est_sd_sigma_sq, t, rho/2, beta)
    sigma_sq += np.square(sd1)

    return (theta, sigma_sq)


def run_experiment():
    header = ['n', 'T', 'true_parameter', 'estimator_mean', 'estimator_variance']

    bounds_theta = [180,200]
    bounds_sigma_sq = [0,5]

    T = 1000 #trials
    list_of_n = [250, 500, 750, 1000, 2500, 5000]
 

    true_mean = 190
    true_sd = 30
    true_var = np.square(true_sd)

    with open('blb_gaussian_mean_DP2.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)

        # write the header
        writer.writerow(header)

        for n in list_of_n:
            #print(" N is ", N)
            for t in range(T):
                #print(" t is ", t)
                data = np.random.normal(true_mean, true_sd, n)
                theta, sigma_sq = blb(estimator_mean, data, n, true_var,bounds_theta, bounds_sigma_sq) #returns mean and variance of estimator
                row = [n, t, true_mean, theta, sigma_sq]
                
                # write the data
                writer.writerow(row)

if __name__=="__main__":

    run_experiment()