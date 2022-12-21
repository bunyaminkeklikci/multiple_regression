import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt

data =pd.read_csv("C:/Users/kekli/Desktop/ML event/advertising.csv")
veri = data.copy()
#sütündan veri silme kalıcı olarak inplace kullanırız.
#veri.drop(columns=["Unnamed:0"],axis=1,inplace=True)
#print(veri)

y = veri["Sales"]
X = veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
model=lr.fit(X_train,y_train)


def skor(model,x_train,x_test,y_train,y_test):
    egitimtahmin=model.predict(x_train)
    testtahmin=model.predict(x_test)

    r2_egitim=mt.r2_score(y_train,egitimtahmin)
    r2_test=mt.r2_score(y_test,testtahmin)

    mse_egitim=mt.mean_squared_error(y_train,egitimtahmin)
    mse_test=mt.mean_squared_error(y_test,testtahmin)

    return[r2_egitim,r2_test,mse_egitim,mse_test]

sonuc1=skor(model=lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
#ilk r2 ve mse değeri
print("Egitim R2= {} Egitim MSE= {}".format(sonuc1[0],sonuc1[2]))
print("Test R2= {} Test MSE= {}".format(sonuc1[1],sonuc1[3]))

#Şimdi doğrulama işlemi yapıcaz.Egitim setini parçalama işlemi yapıyo
from sklearn.model_selection import KFold
#cross valudation
lr_cv =LinearRegression()
k=5
iterasyon=1
cv=KFold(n_splits=k)

for egitimindex,testindex in cv.split(X):
    X_train,X_test=X.loc[egitimindex],X.loc[testindex]
    y_train,y_test=y.loc[egitimindex],y.loc[testindex]

    lr_cv.fit(X_train,y_train)

    sonuc2=skor(model=lr_cv,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
    
    print("İterasyon:{}".format(iterasyon))
    print("Egitim R2= {} Egitim MSE= {}".format(sonuc2[0],sonuc2[2]))
    print("Test R2= {} Test MSE= {}".format(sonuc2[1],sonuc2[3]))
    iterasyon +=1



