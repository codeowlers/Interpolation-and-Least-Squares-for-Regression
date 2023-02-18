from scipy import interpolate


def interpolate_data(x, y, method='global'):
    if method == 'global':
        f = interpolate.interp1d(x, y, kind='linear')
    elif method == 'piecewise':
        f = interpolate.PchipInterpolator(x, y)
    else:
        raise ValueError("Invalid interpolation method specified. Method must be either 'global' or 'piecewise'.")

    # Interpolate the data using the selected method
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = f(x_new)

    return x_new, y_new


# This function takes in two arrays x and y that contain the data points to be interpolated. The method argument is
# used to specify which interpolation method to use, with 'global' indicating global interpolation and 'piecewise'
# indicating piecewise interpolation.
#
# If the method is set to 'global', the function uses the interp1d function from scipy.interpolate to create a linear
# interpolation function f. The function then generates a new set of x values using np.linspace and evaluates the
# interpolation function at each of these points to get the interpolated y values.
#
# If the method is set to 'piecewise', the function uses the PchipInterpolator function from scipy.interpolate to
# create a piecewise cubic Hermite polynomial interpolation function f. Again, the function generates a new set of x
# values and evaluates the interpolation function at these points to get the interpolated y values.
#
# The function returns the new x and y values as a tuple. Note that in the case of piecewise interpolation,
# the interpolated function will only be defined within the range of the original data, so the x_new values may not
# extend beyond the bounds of x.