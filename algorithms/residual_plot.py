import matplotlib.pyplot as plt

def residual_plot(y_train, y_pred):
    residuals = y_train - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()