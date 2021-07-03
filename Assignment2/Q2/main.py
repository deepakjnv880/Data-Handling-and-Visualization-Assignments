import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_hist(x,y,name):
    fig, ax = plt.subplots()
    ax.stem(x, y, markerfmt=' ')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xlabel('pixel value')
    ax.set_ylabel('frequency')
    a=name.split('_')
    ax.set_title(a[0].capitalize()+'_Plane_Histogram')
    plt.savefig(name)
    plt.clf()

def save_hist(hist_R,hist_G,hist_B,tag):
    x=[i for i in range(256)]
    plot_hist(x,hist_R,"red_hist_"+tag)
    plot_hist(x,hist_G,"green_hist_"+tag)
    plot_hist(x,hist_B,"blue_hist_"+tag)

def hist_equalization_mapping(hist,L,M,N):
    hist[0]*=(L-1)/(M*N);
    for i in range(1,L):
        hist[i]*=(L-1)/(M*N)
        hist[i]+=hist[i-1]
    hist=[round(i) for i in hist]
    return hist

def get_hist(img):
    hist_R=[0 for i in range(256)]
    hist_G=[0 for i in range(256)]
    hist_B=[0 for i in range(256)]

    for i in img:
        for j in i:
            hist_R[j[2]]+=1
            hist_G[j[1]]+=1
            hist_B[j[0]]+=1

    return hist_R,hist_G,hist_B


def hist_equalization(image_name):
    img = cv2.imread(image_name)

    hist_R,hist_G,hist_B=get_hist(img)

    height=np.shape(img)[0]
    width=np.shape(img)[1]
    channel=np.shape(img)[2]
    L=256

    print("image dimension => ",np.shape(img))

    save_hist(hist_R,hist_G,hist_B,"original")

    hist_R=hist_equalization_mapping(hist_R,L,height,width)
    hist_B=hist_equalization_mapping(hist_B,L,height,width)
    hist_G=hist_equalization_mapping(hist_G,L,height,width)

    for i in range(height):
        for j in range(width):
            img[i][j][0]=hist_B[img[i][j][0]]
            img[i][j][1]=hist_G[img[i][j][1]]
            img[i][j][2]=hist_R[img[i][j][2]]

    hist_R,hist_G,hist_B=get_hist(img)
    save_hist(hist_R,hist_G,hist_B,"updated")
    cv2.imwrite("Alter_"+image_name, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Histogram equalization of an Image')
    parser.add_argument('--input_image', type=str,  help='the image file you want equalize histogram')
    args = parser.parse_args()

    image_name = args.input_image
    assert image_name != None , "Why you are not putting input file in argument?"
    hist_equalization(image_name)
