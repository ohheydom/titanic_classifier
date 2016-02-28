import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

def clean_data(data):
    """ Preprocesses data by doing the following:
        Fills in missing ages and fare values
        Removes NaN values from cabin information
        One hot encodes embarked and sex features

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing passenger information such as sex, class, fare, etc.

    Returns
    -------
    data : pandas DataFrame
        Returns processed DataFrame, ready for fitting
    """
    # Fill in missing ages with mean
    data.Age = data.Age.fillna(np.mean(data.Age))

    # Fill in missing fare values with means of each Pclass
    data.Fare = data.Fare.map(lambda x: np.nan if x == 0 else x)
    class_means = data.pivot_table('Fare', index='Pclass', aggfunc='mean')
    data.Fare = data[['Fare', 'Pclass']].apply(lambda x: class_means[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)

    # Remove Nan from Cabin information
    data.Cabin = data.Cabin.fillna('Unknown')

    # One-Hot Encoding of Embarked feature
    embarked = pd.get_dummies(data[['Embarked']])
    data = pd.concat([data.drop('Embarked', 1), embarked], axis=1)

    # One-Hot Encoding of Sex feature
    sex = pd.get_dummies(data[['Sex']])
    data = pd.concat([data.drop('Sex', 1), sex], axis=1)

    return data

# Load data
training_data = pd.read_csv('train.csv')
y = training_data.Survived
training_data.drop('Survived', axis=1, inplace=True)
testing_data = pd.read_csv('test.csv')

# Preprocess
training_data = clean_data(training_data)
testing_data = clean_data(testing_data)

# Feature Selection
features = ['Pclass', 'Age', 'Fare', 'Sex_male', 'Sex_female']
selected_training_data = training_data[features]
selected_testing_data = testing_data[features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(selected_training_data, y, test_size=0.2, random_state=0)

# Fit training data
params = {'max_features':(1, 2, 3, 4, 5)}
rf = RandomForestClassifier()
sss = StratifiedShuffleSplit(y_train, 10, test_size=0.1, random_state=0)
gs = GridSearchCV(estimator=rf, param_grid=params, cv=sss)
gs.fit(X_train, y_train)
clf = gs.best_estimator_

# Predict on test split
print("Accuracy score: %f" % accuracy_score(clf.predict(X_test), y_test))

# Predict unlabeled test data
pred = clf.predict(selected_testing_data)

# Save results to file
n = pd.DataFrame(data={'PassengerId': testing_data['PassengerId'], 'Survived': pred})
n.to_csv('rfpredictions.csv', index=False)
