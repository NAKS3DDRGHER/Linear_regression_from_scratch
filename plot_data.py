import matplotlib.pyplot as plt
import numpy as np
from feature_scaling import z_score_normalization, get_train_data
from regression_f import regression_function
from cost_f import cost_in_percent

X, Y, w, b = get_train_data()
X, Y = z_score_normalization(X, Y)
yp = np.array([regression_function(x, w, b) for x in X])

fig,ax=plt.subplots(3,3,figsize=(12, 8),sharey=True)
parameters = [
    ["YearsCode", "WorkExp", "YearsCodePro"],
    ["OrgSize", "Age", "PurchaseInfluence"],
    ["RemoteWork", "DevType", "EdLevel"]
]
idx = 0
for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i][j].scatter(X[:, idx], Y, label='target')
        ax[i][j].set_xlabel(parameters[i][j])
        ax[i][j].scatter(X[:, idx], yp, color="orange", label='predict')
        idx += 1
fig.tight_layout(pad=1)
plt.show()
print(cost_in_percent(w, b, X, Y) * 100)



