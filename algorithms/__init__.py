from .scipy_interpolation import interpolate_data
from .interpolation import global_interp, piecewise_interp
from .residual_plot import residual_plot
from .model import model
from .least_squares import least_squares


__all__ = ['interpolate_data','global_interp', 'piecewise_interp','least_squares', 'residual_plot', 'model']
