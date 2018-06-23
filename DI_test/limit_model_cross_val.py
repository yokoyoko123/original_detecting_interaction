from sklearn.model_selection import KFold
import numpy as np

def score_limit_model(limited_forest,nonlimited_forest,X,y,cv,noi,noj):
    kf = KFold(n_splits=cv)
    score=0
    for train_index,test_index in kf.split(X):
        X_train, X_test = X[train_index],X[test_index]
        y_train, y_test = y[train_index],y[test_index]
 

        limited_forest.fit(X_train,y_train,noi,noj)
        nonlimited_forest.fit(X_train,y_train)
    
        RMSE_limited = 0.0
        RMSE_nonlimited = 0.0
        for i in range(y_test.shape[0]):
            RMSE_limited += (y_test[i]-limited_forest.predict(X_test[i]))**2
            RMSE_nonlimited += (y_test[i]-nonlimited_forest.predict(X_test[i]))**2
        
    RMSE_limited = np.sqrt(RMSE_limited/y_test.shape[0])
    RMSE_nonlimited = np.sqrt(RMSE_nonlimited/y_test.shape[0])
    print("limited RMSE     nonlimited_model       ,",RMSE_limited,RMSE_nonlimited)
    Iij=RMSE_nonlimited/np.std(y_train) - RMSE_limited/np.std(y_train)
    print("Iij      ",Iij)
