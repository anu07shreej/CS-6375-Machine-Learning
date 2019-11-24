#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import sys
import numpy as np
from skimage import io, img_as_float
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

def k_means_clustering(image_vectors, k, num_iterations):
    lbs = np.full((image_vectors.shape[0],), -1)
    clstr_proto = np.random.rand(k, 3)
    for i in range(num_iterations):
        print('Iteration: ' + str(i + 1))
        points_label = [None for k_i in range(k)]


        for rgb_i, rgb in enumerate(image_vectors):
           
            rgb_row = np.repeat(rgb, k).reshape(3, k).T

          
            closest_label = np.argmin(np.linalg.norm(rgb_row - clstr_proto, axis=1))
            lbs[rgb_i] = closest_label

            if (points_label[closest_label] is None):
                points_label[closest_label] = []

            points_label[closest_label].append(rgb)

       
        for k_i in range(k):
            if (points_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_label[k_i]).sum(axis=0) / len(points_label[k_i])
                clstr_proto[k_i] = new_cluster_prototype

    return (lbs, clstr_proto)

def closest_centroids(X,c):
    K = np.size(c,0)
    idx = np.zeros((np.size(X,0),1))
    arr = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        temp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,temp),2)
        a = np.sum(b,axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0),1))
        #print(np.shape(a))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr,0,axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def compute_centroids(X,idx,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        ci = idx
        ci = ci.astype(int)
        total_number = sum(ci);
        ci.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(ci,1,n)
        ci = np.transpose(ci)
        total = np.multiply(X,total_matrix)
        centroids[i] = (1/total_number)*np.sum(total,axis=0)
    return centroids

def plot_image_colors_by_color(name, image_vectors):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb in image_vectors:
        ax.scatter(rgb[0], rgb[1], rgb[2], c=rgb, marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')


def plot_image_colors_by_label(name, image_vectors, lbs, clstr_proto):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb_i, rgb in enumerate(image_vectors):
        ax.scatter(rgb[0], rgb[1], rgb[2], c=clstr_proto[lbs[rgb_i]], marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')


if __name__ == '__main__':
    
    
    J =sys.argv[1]
    K=int(sys.argv[2])
    itr=int(sys.argv[3])
    image = io.imread(J)[:, :, :3] 
    image = img_as_float(image)

    image_dimensions = image.shape
    image_name = image

   
    image_vectors = image.reshape(-1, image.shape[-1])

    lbs, color_centroids = k_means_clustering(image_vectors, k=K, num_iterations=itr)

    output_image = np.zeros(image_vectors.shape)
    for i in range(output_image.shape[0]):
        output_image[i] = color_centroids[lbs[i]]

    output_image = output_image.reshape(image_dimensions)
    print('Saving the Compressed Image')
    io.imsave('Compressed.jpg' , output_image)
    print('Image Compression Completed')
    info = os.stat(J)
    print("Image size before : ",info.st_size/1024,"KB")
    info = os.stat('Compressed.jpg')
    print("Image size : ",info.st_size/1024,"KB")