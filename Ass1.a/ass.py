import os                   
from scipy import misc
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle



def saveW(class_Eita_matrix):
    f = open('C:\\Users\\Samir\\nile uni big data\\Courses\\ML\\Assignments\\store.data', 'wb')
    pickle.dump(class_Eita_matrix, f)
    f.close()
    
def loadW():
    f = open('C:\\Users\\Samir\\nile uni big data\\Courses\\ML\\Assignments\\store.data', 'rb')
    data = pickle.load(f)
    f.close()
    return data


Train_Path='C:\\Users\\Samir\\nile uni big data\\Courses\\ML\\Assignments\\Train'
os.chdir(Train_Path)

#Training_Labels = np.loadtxt('Training Labels.txt')
Train_files=os.listdir(Train_Path)
#files.pop()
Train_files = sorted(Train_files,key=lambda x: int(os.path.splitext(x)[0]))

all_data=[]
for i in Train_files:    
    img=misc.imread(i)
    type(img)
    img.shape
    #change dimention to 1 dimentional array instead of (28x28)
    img=img.reshape(784,)
    img=np.append(img,1)
    all_data.append(img)

#print(np.shape(all_data[0]))

train =  np.loadtxt('C:\\Users\\Samir\\nile uni big data\\Courses\\ML\\Assignments\\Training Labels.txt')

t=[]
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]

for d in train:
    if d==0:
        t.append(1)
    else:
        t.append(-1)
    if d==1:
        t1.append(1)
    else:
        t1.append(-1)
    if d==2:
        t2.append(1)
    else:
        t2.append(-1)
    if d==3:
        t3.append(1)
    else:
        t3.append(-1)
    if d==4:
        t4.append(1)
    else:
        t4.append(-1)
    if d==5:
        t5.append(1)
    else:
        t5.append(-1)
    if d==6:
        t6.append(1)
    else:
        t6.append(-1)
    if d==7:
        t7.append(1)
    else:
        t7.append(-1)
    if d==8:
        t8.append(1)
    else:
        t8.append(-1)
    if d==9:
        t9.append(1)
    else:
        t9.append(-1)
        
#print((t[0]))

#w= np.zeros(785,)
#w[0]=1
#print(np.shape(w))
eta=[1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001]
def perceptron_sgd(X, Y,c):
    w= np.zeros(785,)
    w[0]=1
#    eta = []
    epochs = 600
    
#    for j in range(len(eta)):        
    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + c*X[i]*Y[i]
    return w


#WW=[]
#for n in eta:
#    w = perceptron_sgd(all_data,t,n)
#    w1 = perceptron_sgd(all_data,t1,n)
#    w2 = perceptron_sgd(all_data,t2,n)
#    w3 = perceptron_sgd(all_data,t3,n)
#    w4 = perceptron_sgd(all_data,t4,n)
#    w5 = perceptron_sgd(all_data,t5,n)
#    w6 = perceptron_sgd(all_data,t6,n)
#    w7 = perceptron_sgd(all_data,t7,n)
#    w8 = perceptron_sgd(all_data,t8,n)
#    w9 = perceptron_sgd(all_data,t9,n)
#    W=[w,w1,w2,w3,w4,w5,w6,w7,w8,w9]
#    for u in W:
#        WW.append(u)

#saveW(WW)
WW=loadW()
#    WW.append(w)
#    WW.append(w1)
#    WW.append(w2)
#    WW.append(w3)
#    WW.append(w4)
#    WW.append(w5)
#    WW.append(w6)
#    WW.append(w7)
#    WW.append(w8)
#    WW.append(w9)
print(np.shape(WW[0]))
print(len(WW))


##
##
##test zeros
##
##
Test_Path='C:\\Users\\Samir\\nile uni big data\\Courses\\ML\\Assignments\\Test'
os.chdir(Test_Path)

#Training_Labels = np.loadtxt('Training Labels.txt')
Test_files=os.listdir(Test_Path)
#files.pop()
Test_files = sorted(Test_files,key=lambda x: int(os.path.splitext(x)[0]))

tst_data=[]
for i in Test_files:    
    img=misc.imread(i)
    type(img)
    img.shape
    #change dimention to 1 dimentional array instead of (28x28)
    img=img.reshape(784,)
    img=np.append(img,1)
    tst_data.append(img)
#
#print(np.shape(tst_data))
#
test =  np.loadtxt('C:\\Users\\Samir\\nile uni big data\\Courses\\ML\\Assignments\\Test Labels.txt')
#t_tst=[]
#t_tst1=[]
#for d in train:
#    if d==0:
#        t_tst.append(1)
#    else:
#        t_tst.append(-1)
#    if d==1:
#        t_tst1.append(1)
#    else:
#        t_tst1.append(-1)
#print(np.shape(t_tst))

#E=[]
#for i in range(2400):
#   E.append(np.dot(np.dot(w,all_data[i]),t[i]))
#
#helw=0
#zeft=0
#z=0
#for z in range(2400):
#    if E[z]>=0:
#        helw=helw + 1
#    else:
#        zeft=zeft+1
#print(E)        
#print(helw)
#print(zeft)
#y=[]
f=np.zeros(10)
y_pred=[]
#y_pred=np.zeros(200)
#f0=np.zeros(200)
#f1=np.zeros(200)
#f2=np.zeros(200)
#f3=np.zeros(200)
#f4=np.zeros(200)
#f5=np.zeros(200)
#f6=np.zeros(200)
#f7=np.zeros(200)
#f8=np.zeros(200)
#f9=np.zeros(200)
for k in range(len(eta)):
    for i in range(200):
        for j in range(10):
            f[j]= np.dot(tst_data[i],WW[10*k+j])
        y_pred.append(np.argmax(f))
    
#    f0[i]=np.dot(tst_data[i],W[0])
#    f1[i]=np.dot(tst_data[i],W[1])
#    f2[i]=np.dot(tst_data[i],W[2])
#    f3[i]=np.dot(tst_data[i],W[3])
#    f4[i]=np.dot(tst_data[i],W[4])
#    f5[i]=np.dot(tst_data[i],W[5])
#    f6[i]=np.dot(tst_data[i],W[6])
#    f7[i]=np.dot(tst_data[i],W[7])
#    f8[i]=np.dot(tst_data[i],W[8])
#    f9[i]=np.dot(tst_data[i],W[9])
    
    
#    y=[f0,f1,f2,f3,f4,f5,f6,f7,f8,f9]
#
#    yp=np.argmax(f) 
#    ypred.append(yp)
    
print(np.shape(y_pred))
#f=np.dot(tst_data[0],W[0])
#f1=np.dot(tst_data[0],W[1])


#conf_arr = [[0, 0,0,0,0,0,0,0,0,0], [0, 0,0,0,0,0,0,0,0,0]]
y_true=[]

for d in test:
    if d==0:
        y_true.append(0)
    elif d==1:
        y_true.append(1)
    elif d==2:
        y_true.append(2)
    elif d==3:
        y_true.append(3)
    elif d==4:
        y_true.append(4)
    elif d==5:
        y_true.append(5)
    elif d==6:
        y_true.append(6)
    elif d==7:
        y_true.append(7)
    elif d==8:
        y_true.append(8)
    else:
        y_true.append(9)
print('d=')        
#print(exp)   
#print (ypred)
#        

#y_true =exp

#y_pred=ypred
print(y_true)   
print (y_pred)

cm0=confusion_matrix(y_true, y_pred[0:200], labels=[0,1,2,3,4,5,6,7,8,9])
cm1=confusion_matrix(y_true, y_pred[200:400], labels=[0,1,2,3,4,5,6,7,8,9])
cm2=confusion_matrix(y_true, y_pred[400:600], labels=[0,1,2,3,4,5,6,7,8,9])
cm3=confusion_matrix(y_true, y_pred[600:800], labels=[0,1,2,3,4,5,6,7,8,9])
cm4=confusion_matrix(y_true, y_pred[800:1000], labels=[0,1,2,3,4,5,6,7,8,9])
cm5=confusion_matrix(y_true, y_pred[1000:1200], labels=[0,1,2,3,4,5,6,7,8,9])
cm6=confusion_matrix(y_true, y_pred[1200:1400], labels=[0,1,2,3,4,5,6,7,8,9])
cm7=confusion_matrix(y_true, y_pred[1400:1600], labels=[0,1,2,3,4,5,6,7,8,9])
cm8=confusion_matrix(y_true, y_pred[1600:1800], labels=[0,1,2,3,4,5,6,7,8,9])
cm9=confusion_matrix(y_true, y_pred[1800:2000], labels=[0,1,2,3,4,5,6,7,8,9])
CM=[cm0,cm1,cm2,cm3,cm4,cm5,cm6,cm7,cm8,cm9]
for q in range(10):
    print("CM of eta number ",q,"is :")
    print (CM[q])
    






 


























