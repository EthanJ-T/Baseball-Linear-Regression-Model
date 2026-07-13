# Baseball-Linear-Regression-Model
Simple ML Linear Regression Model using Baseball data with Plot
Just a little coding/learning self project
Player Data from Baseball Refrence Updated 7-11-26
Data taken and coded on 7-12-26 

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

#Need Sklearn, pandas, and matplotlib, seaborn can be used but not here

BWar26 = pd.read_csv("BWarTop200py.csv")
#import Data
BWar26 = BWar26.drop(200)

#BWar26.dropna()
#Row 200 gave NaN value-> will update with not hard coded value


print(BWar26)
#check


Y = BWar26.WAR 
#y is value we are testing
print(Y)

X = BWar26[["TB"]]
print(X) #x is value used

X.shape
Y.shape
#check if data is ready

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
#using 60 samples out of 200
WARmodel = LinearRegression()

WARmodel.fit(pd.DataFrame(X_train), Y_train)
#making model fit
Y_pred = WARmodel.predict(X_test)
#what the value of (total bases hit) is used to predict WAR value
print (WARmodel.coef_)
print (WARmodel.intercept_)
#find slope and intercept
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
#mean error
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))
# r value

plt.scatter(X_test, Y_test,  color='gray')
plt.plot(X_test, Y_pred, color='red', linewidth=2)
plt.xlabel('Total bases recorded by batter')
plt.ylabel('Batter Wins Above Replacement (WAR) Value')
plt.title("Batter Player Value vs Bases Recorded")
plt.show()
#plot
