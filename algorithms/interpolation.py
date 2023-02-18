import numpy as np
import pandas as pd




def linear_interp(x, x_known, y_known):
    """
    Perform linear interpolation of missing values in a dataset.
    
    Parameters:
    x (numpy.ndarray): The x values at which to interpolate the y values.
    x_known (numpy.ndarray): The known x values in the dataset.
    y_known (numpy.ndarray): The known y values in the dataset.
    
    Returns:
    numpy.ndarray: The interpolated y values at the specified x values.
    """
    # Sort the known x and y values by x value
    order = np.argsort(x_known)
    x_known = x_known[order]
    y_known = y_known[order]

    # Find the indices of the known x values that bracket each interpolation point
    indices = np.searchsorted(x_known, x)

    # Handle the case where x is less than the first known x value
    indices = np.where(indices == 0, 1, indices)

    # Handle the case where x is greater than the last known x value
    indices = np.where(indices == len(x_known), len(x_known) - 1, indices)

    # Compute the slope and y-intercept for each segment between known x values
    slope = (y_known[indices] - y_known[indices - 1]) / (x_known[indices] - x_known[indices - 1])
    y_intercept = y_known[indices - 1] - slope * x_known[indices - 1]

    # Interpolate the y values at the specified x values
    y = slope * x + y_intercept

    return y



import pandas as pd
import numpy as np

def global_interp(df, col):
    """
    Perform global interpolation to fill in missing values in a column of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to interpolate missing values.
    col (str): The name of the column to interpolate missing values.
    
    Returns:
    pandas.DataFrame: A copy of the input DataFrame with missing values interpolated using global interpolation.
    """
    # Copy the DataFrame to avoid modifying the input
    df_interp = df.copy()

    # Find the missing values
    mask = df_interp[col].isna()

    # Compute the x and y values for the known data points
    x = df_interp.loc[~mask, col].index
    y = df_interp.loc[~mask, col].values

    # Compute the x values for the missing data points
    x_new = df_interp.loc[mask, col].index

    # Compute the y values for the missing data points using linear interpolation
    y_new = np.interp(x_new, x, y)

    # Update the DataFrame with the interpolated values
    df_interp.loc[mask, col] = y_new

    return df_interp


def piecewise_interp(df, col, window_size=5):
    """
    Perform piecewise interpolation to fill in missing values in a column of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to interpolate missing values.
    col (str): The name of the column to interpolate missing values.
    window_size (int): The number of data points to consider in each direction when performing piecewise interpolation.
    
    Returns:
    pandas.DataFrame: A copy of the input DataFrame with missing values interpolated using piecewise interpolation.
    """
    # Copy the DataFrame to avoid modifying the input
    df_interp = df.copy()

    # Find the missing values
    mask = df_interp[col].isna()

    # Loop over the missing values and interpolate each one separately
    for i in np.where(mask)[0]:
        # Define the window of data points to use for interpolation
        start = max(0, i - window_size)
        stop = min(len(df_interp), i + window_size + 1)

        # Get the known x and y values in the window
        x = df_interp.loc[start:stop, col].dropna().index
        y = df_interp.loc[start:stop, col].dropna().values

        # Interpolate the missing y value using linear interpolation
        y_new = np.interp(i, x, y)

        # Update the DataFrame with the interpolated value
        df_interp.loc[i, col] = y_new

    return df_interp
