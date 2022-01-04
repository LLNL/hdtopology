import os
import argparse
import numpy as np

#### topology computation ####
import ngl
from hdff import *
import hdtopology as hdt


#### compute sampling ####
import pandas as pd
# import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


class TDA_sampler:
    def __init__(self, graph_method = 'RelaxedGabriel', sampling_method='gaussian', data_cube_dim=1, max_neighbors=500, beta=1.0):
        self.beta = 1.0
        self.graph_method = graph_method
        self.sampling_method = sampling_method
        self.data_cube_dim = data_cube_dim
        self.max_neighbors = max_neighbors
        
    def gaussian_sampling(self, localExtrema, coreSamples, sampleCount):
        sample_mean = np.mean(coreSamples, axis=0)
        sample_covariance = np.cov(coreSamples, rowvar=False)
        # print(sample_mean.shape, sample_covariance.shape)

        return np.random.default_rng().multivariate_normal(sample_mean, sample_covariance, size=sampleCount)

    def gaussian_process_sampling(self, localExtrema, coreSamples, sampleCount):
        pass
        # sample_dim = coreSamples.shape[1]
        # kernel = GPy.kern.RBF(input_dim=sample_dim, variance=1., lengthscale=1.)
        # model = GPy.models.GPRegression(coreSamples, f, kernel, noise_var=1e-10)

        # posteriorTestY = model.posterior_samples_f(testX, full_cov=True, size=3)
        # simY, simMse = model.predict(testX)

    # def lookup_samples(samples, indices):
    #     return samples[indices,:]

    def normalize_domain(self, domain, domain_min, domain_max):
        return (domain-domain_min)/(domain_max-domain_min)

    def rescale_domain(self, domain, domain_min, domain_max):
        return domain*(domain_max-domain_min)+domain_min

    '''
        Data is a array that contains the domain and range of the function
    '''
    def compute_TDA(self, data, dimNames):
        data = data.astype('float32')
        self.domain = data[:, 0:-1]
        print("domain:", self.domain.shape, self.domain.dtype)
        self.frange = data[:,-1]
        print("frange:", self.frange.shape)

        ###### computing graph #######
        ### provide array of unint32 for the edges
        print(self.graph_method)
        edges = ngl.getSymmetricNeighborGraph(self.graph_method, self.domain, self.max_neighbors, self.beta)
        print(edges, type(edges), edges.dtype)

        ###### compute topology #######
        self.eg = hdt.ExtremumGraphExt()
        flag_array = np.array([0],dtype=np.uint8)
        mode = 0
        if self.data_cube_dim>1:
            mode = 1

        #### make data recarray ####
        types = ['f4']*len(dimNames)
        self.data = data.view(dtype=list(zip(dimNames,types)) ).view(np.recarray)

        self.eg.initialize(self.data, flag_array, edges, True ,10, mode, self.data_cube_dim)

        print("function range:", self.eg.minimum(), self.eg.maximum())

    def sampling(self, extremaCount, samplePerExtrema):
        ##### get domain min max #####
        domain_min = np.min(self.domain, axis=0)
        domain_max = np.max(self.domain, axis=0)
        print("domain min:", domain_min)
        print("domain max:", domain_max)
        # exit()
        norm_domain = self.normalize_domain(self.domain, domain_min, domain_max)

        all_core_seg = {}
        output_samples = []

        ##### per extrema resampling ####
        # all_core_samples = set()
        for i, ex in enumerate(self.eg.extrema()):
            if i >= extremaCount:
                break
            ind = int(ex[2])
            seg = self.eg.segment(ind, extremaCount, self.eg.minimum())
            print("Seg:", len(seg))
            coreSeg = self.eg.coreSegment(ind, extremaCount)[1:]
            print("coreSeg:", ind, len(coreSeg))

            # print(ind in set(coreSeg))
            # all_core_seg[ind] = coreSeg
            # all_core_samples.add()
            sample_result = None
            if self.sampling_method == 'gaussian':
                sample_result = self.gaussian_sampling(norm_domain[ind,:], norm_domain[coreSeg,:], samplePerExtrema)
            # elif self.sampling_method == 'nearest_neigbors':
            #     sample_result = self.nearest_neigbors(norm_domain[ind,:], norm_domain[coreSeg,:], self.samplePerExtrema)
            elif self.sampling_method == 'gaussian_process':
                sample_result = self.gaussian_process_sampling(norm_domain[ind,:], norm_domain[coreSeg,:], samplePerExtrema, frange[coreSeg])
            else:
                print("method "+self.sampling_method+" is not recognized")
                return None

            ### map sample result back to the original space
            sample_rescaled = self.rescale_domain(sample_result, domain_min, domain_max)
            output_samples.append(sample_rescaled)

        output_samples = np.array(output_samples)

        return output_samples
