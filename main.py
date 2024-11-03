from regression_f import regression_function
from gradient_descent import gradient_descent
from feature_scaling import z_score_normalization, get_train_data, upload_history
from cost_f import cost_in_percent



if __name__ == '__main__':
    alpha = 0.1
    X, Y, w_last, b_last = get_train_data()
    X, Y = z_score_normalization(X, Y)
    print(X.shape)

    w, b = gradient_descent(X, Y,  w_last, b_last, alpha, 1001)
    print(w, b)
    upload_history(w, b)

    # x = list(map(int, input("Enter options separated by a space: ").split()))

    # print(f"Predicted salary: {regression_function(np.array(x, dtype=float), np.array([17210.92409854,   960.85178803]), 161319.35867139834)}")

    print(cost_in_percent(w_last, b_last, X, Y) * 100)


