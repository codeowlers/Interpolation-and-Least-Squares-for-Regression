import numpy as np
import pandas as pd

# Define a function to fill in missing data using the pseudo-inverse
def least_squares(df,colname):
    # Split the DataFrame into two parts - one with missing values, and one without
    df_missing = df[df.isna().any(axis=1)]
    df_complete = df.dropna()
    
    # Extract the values of the complete part of the DataFrame
    X = df_complete.index.values
    y = df_complete[colname].values
    
    # Compute the pseudo-inverse of X
    X_pinv = np.linalg.pinv(X.reshape(-1, 1))
    
    # Use the pseudo-inverse to estimate the missing values
    y_pred = X_pinv.dot(y)
    
    # Add the estimated values to the DataFrame
    df_missing[colname] = df_missing.index.values.reshape(-1, 1).dot(y_pred)
    
    # Combine the complete and estimated parts of the DataFrame
    df_result = pd.concat([df_complete, df_missing], axis=0)
    df_result = df_result.sort_index()
    return df_result


