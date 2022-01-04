import os
import argparse
import numpy as np

import GPy

from hdff import *
import hdtopology as hdt
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def load_eg(filename):
    collection = DataCollectionHandle()
    collection.attach(filename)
    dataset = collection.dataset(0)
    # load EG
    eg = hdt.ExtremumGraphExt()
    handle = dataset.getDataBlock(0)
    isIncludeFunctionIndexInfo = False
    # cube_dim = 2

    eg.load(handle, isIncludeFunctionIndexInfo)

    ##### test query ######
    # attrs = eg.getJoint().getAttr()
    # print("Attrs:", attrs)
    # hist = eg.getHist(attrs[:2])
    # print("Histogram Bin Value Range:", hist.min(), hist.max())

    return eg

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Compute hdff file from recarray.')
    parser.add_argument('--filename', type=str, help='Name of the input hdff datafile.', required=True)
    parser.add_argument('--datafile', type=str, help='Name of the of original dataset in npy format')
    # parser.add_argument('--datafile', type=str, help='Name of the of original dataset in npy format')

    parser.add_argument('--extremaCount', type=int, help='number of extrema to sample from', default=10)
    parser.add_argument('--samplePerExtrema', type=int, help='number of samples per extrema.', default=1000)

    parser.add_argument('--method', type=str, help='method for determine the samples', default='gaussian')
    #### option, gaussian, gaussian_process ####

    return parser.parse_args()

def gaussian_sampling(ex, coreSamples, sampleCount):
    sample_mean = np.mean(coreSamples, axis=0)
    sample_covariance = np.cov(coreSamples, rowvar=False)
    # print(sample_mean.shape, sample_covariance.shape)

    return np.random.default_rng().multivariate_normal(sample_mean, sample_covariance, size=sampleCount)

def gaussian_process_sampling(ex, coreSamples, sampleCount, f):

    sample_dim = coreSamples.shape[1]
    kernel = GPy.kern.RBF(input_dim=sample_dim, variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(coreSamples, f, kernel, noise_var=1e-10)

    # posteriorTestY = model.posterior_samples_f(testX, full_cov=True, size=3)
    # simY, simMse = model.predict(testX)

# def lookup_samples(samples, indices):
#     return samples[indices,:]

def normalize_domain(domain, domain_min, domain_max):
    return (domain-domain_min)/(domain_max-domain_min)

def rescale_domain(domain, domain_min, domain_max):
    return domain*(domain_max-domain_min)+domain_min

def main():
    """Main function."""
    args = parse_args()
    eg = load_eg(args.filename)

    data = np.load(args.datafile)#[:,:-1]

    domainNames = list(data.dtype.names[0:-1])
    print(domainNames)
    domain = data[domainNames]
    domain = pd.DataFrame(domain).to_numpy()
    frange = pd.DataFrame(data[data.dtype.names[-1]]).to_numpy()
    print("domain:", domain.shape)
    print("range:", frange.shape)

    print("function domain:", domain.shape)
    ##### get domain min max #####
    domain_min = np.min(domain, axis=0)
    domain_max = np.max(domain, axis=0)
    print("domain min:", domain_min)
    print("domain max:", domain_max)
    # exit()
    norm_domain = normalize_domain(domain, domain_min, domain_max)
    print("function range:", eg.minimum(), eg.maximum())

    all_core_seg = {}
    output_samples = []

    # all_core_samples = set()
    for i, ex in enumerate(eg.extrema()):
        ind = int(ex[2])
        seg = eg.segment(ind, args.extremaCount, eg.minimum())
        print("Seg:", len(seg))
        coreSeg = eg.coreSegment(ind, args.extremaCount)[1:]
        print("coreSeg:", ind, len(coreSeg))

        # print(ind in set(coreSeg))
        # all_core_seg[ind] = coreSeg
        # all_core_samples.add()
        sample_result = None
        if args.method == 'gaussian':
            sample_result = gaussian_sampling(norm_domain[ind,:], norm_domain[coreSeg,:], args.samplePerExtrema)
        elif args.method == 'nearest_neigbors':
            sample_result = nearest_neigbors(norm_domain[ind,:], norm_domain[coreSeg,:], args.samplePerExtrema)
        elif args.method == 'gaussian_process':
            sample_result = gaussian_process_sampling(norm_domain[ind,:], norm_domain[coreSeg,:], args.samplePerExtrema, frange[coreSeg])
        else:
            print("method "+args.method+" is not recognized")
            exit(0)

        ### map sample result back to the original space
        sample_rescaled = rescale_domain(sample_result, domain_min, domain_max)
        output_samples.append(sample_rescaled)

    output_samples = np.array(output_samples)
    # return output_samples
    print("output_samples size:", output_samples.shape)
    np.save('new_samples_'+args.filename.split(".")[0]+".npy", output_samples)

if __name__ == '__main__':
    main()
