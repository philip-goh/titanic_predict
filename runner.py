import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def extract_features(feature_names, data):
    fea = pd.DataFrame(index=data.index)
    for name in feature_names:
        if name in data:
            fea = fea.join(data[name])

    return fea

data = pd.read_csv("csv/train.csv", parse_dates=True)

#replace 'males' and 'females' with numeric values
data['sex'] = data['sex'].replace('male', 1)
data['sex'] = data['sex'].replace('female', 0)

#replace NaN age with mean age
mean_age = data['age'].mean()
data['age'] = data['age'].fillna(mean_age)

#replace S, C, Q with numeric equivalents
data['embarked'] = data['embarked'].replace('S', 0)
data['embarked'] = data['embarked'].replace('Q', 1)
data['embarked'] = data['embarked'].replace('C', 2)

feature_names = ["pclass", "sex", "age", "embarked", "fare", "sibsp", "parch"]
features = extract_features(feature_names, data)

print features

clf = RandomForestClassifier()
clf.fit(features, data["survived"])

score = clf.score(features, data["survived"])
print "Score: %.2f" % score
