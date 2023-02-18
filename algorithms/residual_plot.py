import matplotlib.pyplot as plt

def residual_plot(y_test, y_pred):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, c=['r' if r > 0 else 'b' for r in residuals], alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()
