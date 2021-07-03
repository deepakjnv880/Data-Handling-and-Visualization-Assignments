import numpy as np
import os
import random
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

dimention=2
data_folder_name="data/"
number_of_principal_component=1

def compute_mean_vector(data_vectors):
    mean_vector=[0,0]
    for data in data_vectors:
        mean_vector[0]+=float(data[0])
        mean_vector[1]+=float(data[1])
    mean_vector[0]/=(len(data_vectors))
    mean_vector[1]/=(len(data_vectors))
    return mean_vector

def compute_standard_deviation(data_vectors,mean_vector):
    sigma=[0,0]
    for data in data_vectors:
        sigma[0]+=((data[0]-mean_vector[0])**2)
        sigma[1]+=((data[1]-mean_vector[1])**2)
    sigma[0]/=(len(data_vectors)-1)
    sigma[1]/=(len(data_vectors)-1)
    return [math.sqrt(sigma[0]),math.sqrt(sigma[1])]


def normalize_data(data_vectors,mean_vector,sigma_vector):
    mini=[data_vectors[0][0],data_vectors[0][1]]
    maxi=[data_vectors[0][0],data_vectors[0][1]]

    for i in data_vectors:
        mini[0]=min(mini[0],i[0])
        mini[1]=min(mini[1],i[1])
        maxi[0]=max(maxi[0],i[0])
        maxi[1]=max(maxi[1],i[1])

    print(mini)
    for i in range(len(data_vectors)):
        print("===>",(data_vectors[i][0]-mini[0])/(maxi[0]-mini[0]))
        data_vectors[i][0]=(data_vectors[i][0]-mini[0])/(maxi[0]-mini[0])
        data_vectors[i][1]=(data_vectors[i][1]-mini[1])/(maxi[1]-mini[1])
    return data_vectors

def compute_covariance_matrix(data_vectors,mean_vector):
    covariance_matrix=[[0 for x in range(dimention)] for y in range(dimention)]
    for i in range(dimention):
        for j in range(dimention):
            for k in range(len(data_vectors)):
                covariance_matrix[i][j]+=(data_vectors[k][i]-mean_vector[i])*(data_vectors[k][j]-mean_vector[j])
            covariance_matrix[i][j]/=(len(data_vectors)-1);
    return covariance_matrix

def read_data(filename):
	temp = pd.read_excel(data_folder_name+filename , engine='openpyxl').values.tolist()
	data=[[],[]]
	for i in temp: data[int(i[2])-1].append([i[0],i[1]])
	return data

def perform_pca(filename):
    
    data=read_data(filename)
    data_vectors=data[0]+data[1]
    
    random.shuffle(data_vectors)
    number_of_data=len(data_vectors)
    
    data_vectors=np.asarray(data_vectors)
    
    mean_vector=compute_mean_vector(data_vectors)
    sigma_vector=compute_standard_deviation(data_vectors,mean_vector)
    # data_vectors=normalize_data(data_vectors,mean_vector,sigma_vector)
    
    covariance_matrix=compute_covariance_matrix(data_vectors,mean_vector)

    eig_vals, eig_vecs = np.linalg.eig(np.array(covariance_matrix))
    eig_vals=np.real(eig_vals)
    eig_vecs=np.real(eig_vecs)
    
    temp = []
    for i in range(len(eig_vals)):
        temp.append([eig_vals[i],eig_vecs[:,i]])

    temp.sort(key = lambda x:x[0], reverse=True)

    eig_vecs = []

    for i in range(number_of_principal_component):
        eig_vecs.append(temp[i][1])

    eig_vecs = np.asarray(eig_vecs)

    temp1 = np.dot(np.array(data[0]),temp[0][1].T)
    temp2 = np.dot(np.array(data[1]),temp[0][1].T)

    r=math.sqrt((eig_vecs[0][1]**2)+(eig_vecs[0][0]**2))
    ct=eig_vecs[0][0]/r
    st=eig_vecs[0][1]/r
    t1x = ct*np.array(temp1)
    t1y = st*np.array(temp1)

    t2x = ct*np.array(temp2)
    t2y = st*np.array(temp2)


    ################## plotting #########################

    plt.subplot(2, 2, 1)
    X=[i[0] for i in data[0]]
    Y=[i[1] for i in data[0]]
    plt.scatter(X,Y,color="red",s=1)
    X=[i[0] for i in data[1]]
    Y=[i[1] for i in data[1]]
    plt.scatter(X,Y,color="blue",s=1)
    plt.title("Data scatter plot")
    lx=plt.gca().get_xlim()
    ly=plt.gca().get_ylim()

    plt.subplot(2, 2, 2)
    X=[i[0] for i in data[0]]
    Y=[i[1] for i in data[0]]
    plt.scatter(X,Y,color="red",s=1)
    X=[i[0] for i in data[1]]
    Y=[i[1] for i in data[1]]
    plt.scatter(X,Y,color="blue",s=1)
    V = np.array([temp[0][1], temp[1][1]])
    origin = np.array([[0, 0],[0, 0]]) # origin point
    plt.quiver(*origin, V[:,0], V[:,1], color=['black','green'], scale=3)
    plt.title("Showing eigen vector")

    plt.subplot(2, 2, 3)
    plt.scatter(temp2,[0 for i in range(len(temp2))],color="blue",s=1)
    plt.scatter(temp1,[0 for i in range(len(temp1))],color="red",s=1)
    plt.title("Data in prominent eigen space")

    plt.subplot(2, 2, 4)
    plt.scatter(t2x,t2y,color="blue",s=1)
    plt.scatter(t1x,t1y,color="red",s=1)
    plt.xlim(lx)
    plt.ylim(ly)
    plt.title("1D data segregation")

    plt.savefig('pca_code_on_'+filename[:-5]+'.jpg')
    plt.show()


parser = argparse.ArgumentParser(description='Perform PCA (Principal component analysis on data file) on')
parser.add_argument('--input_file', type=str,  help='the csv file on which you want perform pca')
args = parser.parse_args()
fname = str(args.input_file)
assert fname != None , "Why you are not putting input file in argument?"
perform_pca(fname)
