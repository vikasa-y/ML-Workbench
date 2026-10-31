
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn import neighbors, svm, tree
from sklearn import ensemble as ens
from sklearn import neural_network as nnet

# 1. Generate the dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.3,
    random_state=0
)

# 2. Split into training and testing data
Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    random_state=0
)

# 3. Create a neural network
mlp = nnet.MLPClassifier(
    (32, 32),
    max_iter=2000,
    random_state=0
)

# 4. Define the eight models
models = {
    "linear": lm.RidgeClassifier(),
    "logistic": lm.LogisticRegression(),
    "knn": neighbors.KNeighborsClassifier(15),
    "svm": svm.SVC(),
    "tree": tree.DecisionTreeClassifier(max_depth=5),
    "forest": ens.RandomForestClassifier(random_state=0),
    "boosting": ens.GradientBoostingClassifier(random_state=0),
    "neural": mlp
}

# 5. Train and evaluate each model
for name, model in models.items():
    model.fit(Xtr, ytr)
    accuracy = model.score(Xte, yte)
    print(f"{name:9} {accuracy:.1%}")
