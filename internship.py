import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings

warnings.filterwarnings("ignore")

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('🔍 Feature Correlation Heatmap')
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter search spaces
search_spaces = {
    'RandomForest': {
        'estimator': RandomForestClassifier(),
        'search_space': {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(2, 20),
            'min_samples_split': Integer(2, 20)
        }
    },
    'GradientBoosting': {
        'estimator': GradientBoostingClassifier(),
        'search_space': {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
            'max_depth': Integer(2, 10)
        }
    },
    'SVM': {
        'estimator': SVC(),
        'search_space': {
            'C': Real(1e-3, 1e3, prior='log-uniform'),
            'gamma': Real(1e-4, 1e1, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'poly'])
        }
    }
}

best_model = None
best_score = -np.inf
best_params = None
best_name = ""

# Perform Bayesian optimization
for name, config in search_spaces.items():
    print(f"\n🔍 Optimizing {name}...")
    opt = BayesSearchCV(
        estimator=config['estimator'],
        search_spaces=config['search_space'],
        n_iter=30,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    opt.fit(X_train, y_train)
    
    score = opt.best_score_
    print(f"✅ {name} best cross-validation score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = opt.best_estimator_
        best_params = opt.best_params_
        best_name = name

# Evaluate on test set
test_preds = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)

# Print results
print("\n🏆 Best Model:", best_name)
print("📈 Best Cross-Validation Score:", best_score)
print("🧪 Test Accuracy:", test_accuracy)
print("⚙ Best Hyperparameters:", best_params)

# Plot confusion matrix
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'📊 Confusion Matrix - {best_name}')
plt.show()