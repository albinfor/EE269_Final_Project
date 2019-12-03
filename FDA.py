import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA as PCA
from fileimport import FeatureFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class FDAPCA:
    def __init__(self, filename = "../Parkinson_Multiple_Sound_Recording/train_data.txt"):
        file = FeatureFile(filename)
        [self.X, self.y] = file.get_data_frame()
        self.newsplit()

    def newsplit(self):
        sc = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)#, random_state=0)
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def fda(self):
        self.lda = LDA()
        self.lda.fit_transform(self.X, self.y)
        return self.lda.transform(self.X)

    def pca(self,n_components):
        self.pca = PCA(n_components)
        self.pca.fit_transform(self.X)
        return self.pca.transform(self.X)


    def testKNN(self,X_test,n_neighbors):
        #clf = KNeighborsClassifier(n_neighbors)
        #clf = SVC(kernel='linear')
        #clf = RandomForestClassifier(n_neighbors)
        clf = LogisticRegression(random_state=0, solver='lbfgs')


        X_train, X_test, y_train, y_test = train_test_split(X_test, self.y, test_size=0.2)#, random_state=0)
        plotFDA(X_train,y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('Accuracy ' + str(accuracy_score(y_test, y_pred)))

def plotPCA(X,y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    Xtrue = X[y == 1]
    Xfalse = X[y == 0]
    xax = 10
    yax = 20
    Xxt = Xtrue[:,xax]
    Xyt = Xtrue[:,yax]
    Xxf = Xfalse[:,xax]
    Xyf = Xfalse[:,yax]
    ax1.scatter(Xxt, Xyt, s=10, c='b', marker="s", label='first')
    ax1.scatter(Xxf, Xyf, s=10, c='r', marker="o", label='second')
    plt.legend(loc='upper left');
    plt.show()

def plotFDA(X,y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    Xtrue = X[y == 1]
    Xfalse = X[y == 0]
    zerost = np.ones(Xtrue.shape)
    zerosf = np.zeros(Xfalse.shape)
    ax1.scatter(Xtrue,zerost , s=10, c='b', marker="s", label='Positive')
    ax1.scatter(Xfalse, zerosf, s=10, c='r', marker="o", label='Negative')
    plt.legend(loc='upper left');
    plt.show()





lda = LDA()

n_neighbors = 2
x = FDAPCA()
y = x.y
#xfda = x.fda()
xpca = x.pca(26)

#plotPCA(x.X.values,y)

xPCAFDA = lda.fit_transform(xpca, y)
x.testKNN(xPCAFDA, n_neighbors)
#plotFDA(xPCAFDA,y)
#plotFDA(xpca[0:1,:],y)
True == True
