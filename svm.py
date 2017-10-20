import numpy as np
import matplotlib.pyplot as plt
import random

def dataSet():
    dataSet = [map(float, line.strip().split('\t')) for line in open('testSet.txt').readlines()]
    dataSet = np.array(dataSet)
    return dataSet[:,:-1], dataSet[:,-1:].reshape([-1, 1])

def plot_point(points, labels, w, b, alphas):
    # class   
    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(points, labels):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    
    # plot line
    #w = get_w(alphas, dataset, labels)
    print '%s %s'%(w, b)
    x1, _ = np.max(points, axis=0)
    x2, _ = np.min(points, axis=0)
    a1, a2 = w[0,0], w[0,1]
    y1, y2 = (-b[0, 0] - a1*x1)/a2, (-b[0, 0] - a1*x2)/a2
    print y1.shape, y2.shape
    print [x1, x2], [y1, y2]
    ax.plot([x1, x2], [y1, y2])
    #ax.plot([y1[0, 0], y2[0, 0]], [x1, x2])
    
    # plot vector
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3 and alpha < 500.0:
            x, y = points[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')
    
    plt.show()


def get_w(alphas, dataset, labels):
    return np.sum(alphas * labels * dataset, axis=0).reshape([1, -1])

X, Y = dataSet()
Y[Y == 0] = -1.0
data_num, feature_num = X.shape
print 'X.shape:%s Y.shape:%s'%(X.shape, Y.shape)
#plot_point(X, Y)

#w = np.zeros((1, 2))
b = 0.0
alphas = np.zeros((data_num, 1))

w = get_w(alphas, X, Y)
C = 2.0
print w.shape
print w
k = 0 
for iter in range(10000):
    change_num = 0
    for i in range(data_num):
        alpha_1_index = i
        alpha_2_index = np.random.choice(range(data_num))
        while alpha_2_index == alpha_1_index:
            alpha_2_index = np.random.choice(range(data_num))
        #print 'index1: %d index2: %d'%(alpha_1_index, alpha_2_index)
        w_old = get_w(alphas, X, Y)
        b_old = b
        alpha_1_old, alpha_2_old = alphas[alpha_1_index, 0], alphas[alpha_2_index, 0]
        y_1, y_2 = Y[alpha_1_index, 0], Y[alpha_2_index, 0]

        f_1 = np.dot(w_old, X[alpha_1_index, :].reshape([1, -1]).T) + b_old
        f_2 = np.dot(w_old, X[alpha_2_index, :].reshape([1, -1]).T) + b_old
        #print 'f_1:%lf, f_2:%lf'%(f_1, f_2)
        K_11 = np.dot(X[alpha_1_index, :].reshape([1, -1]), X[alpha_1_index, :].reshape([1, -1]).T)[0, 0]
        K_12 = np.dot(X[alpha_1_index, :].reshape([1, -1]), X[alpha_2_index, :].reshape([1, -1]).T)[0, 0]
        K_22 = np.dot(X[alpha_2_index, :].reshape([1, -1]), X[alpha_2_index, :].reshape([1, -1]).T)[0, 0]

        E_1 = y_1 - f_1
        E_2 = y_2 - f_2
        eta = 2*K_12 - K_11 - K_22
        alpha_2_new = alpha_2_old + (E_1 - E_2) * y_2 / eta
        
        epsilon = alpha_1_old * y_1 + alpha_2_old * y_2
        if y_1 == y_2:
            H = min(C, alpha_1_old + alpha_2_old)
            L = max(0, alpha_1_old + alpha_2_old - C)
        else:
            H = min(C, C + alpha_2_old - alpha_1_old)
            L = max(0, alpha_2_old - alpha_1_old)
        #print 'L: %lf H: %lf'%(L,H)
        #clip
        if alpha_2_new > H:
            alpha_2_new = H
        elif alpha_2_new < L:
            alpha_2_new = L
        
        alpha_1_new = alpha_1_old + (alpha_2_old - alpha_2_new)*y_1*y_2
        if abs(alpha_2_old - alpha_2_new) < 1e-4:
           continue

        b_1_new = E_1 + y_1*K_11*(alpha_1_old - alpha_1_new) + y_2*K_12*(alpha_2_old - alpha_2_new) + b_old
        b_2_new = E_2 + y_1*K_12*(alpha_1_old - alpha_1_new) + y_2*K_22*(alpha_2_old - alpha_2_new) + b_old
        if alpha_1_new > 0 and alpha_1_new < C:
            b = b_1_new
        elif alpha_2_new > 0 and alpha_2_new < C:
            b = b_2_new
        else:
            b = (b_1_new + b_2_new) / 2.0
        
        alphas[alpha_1_index, 0] = alpha_1_new
        alphas[alpha_2_index, 0] = alpha_2_new
        change_num += 1
        ay = np.sum(alphas * Y)
        print'alpha_1: %lf -> %lf alpha_2: %lf -> %lf sum:%lf'%(alpha_1_old, alpha_1_new, alpha_2_old, alpha_2_new,ay)
    
    #if iter % 100 == 0:
    #    plot_point(X, Y, get_w(alphas, X, Y), b, alphas)
    if change_num == 0:
        k += 1
        if k > 40:
            break
    else:
        k = 0
plot_point(X, Y, get_w(alphas, X, Y), b, alphas)









