from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline

def model(X,y,title,size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

    # Create a pipeline for preprocessing and linear regression
    model = make_pipeline(
        StandardScaler(), # standardize the data
        PolynomialFeatures(degree=2), # create polynomial features
        LinearRegression() # apply linear regression
    )
    # Fit the model to the training data and predict the target values for the testing data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"-------------{title}--------------------")
    print("R-squared: ", r2)
    print("Mean Squared Error: ", mse)
    print()
    return y_pred, r2, mse

