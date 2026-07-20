
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('Suppy_Chain_Shipment_Data.csv')

##wanted columns
## removes the unecessary columns filtered through excel sheets and analyzing kaggle parameters

columns_used = [
    'country',
    'fulfill via',
    'shipment mode',
    'scheduled delivery date',
    'delivered to client date',
    'vendor',
    'dosage form',
    'manufacturing site'
]
## initial cleaning process

##uses wanted columns only, drops all others

df_filtered = df[columns_used].copy()
## converts to datetime for math processing

df_filtered["scheduled delivery date"] = pd.to_datetime(df_filtered['scheduled delivery date'])
df_filtered["delivered to client date"] = pd.to_datetime(df_filtered['delivered to client date'])


## converting string(country) to binary(one-hot) with getdummies! Still undecided... unsure how to
## go about this

c = pd.get_dummies(df_filtered, columns = ["country"], dtype = str)

## now making new column with boolean condition for the above ^

df_filtered["delivered late"] = df_filtered["delivered to client date"] > df['scheduled delivery date']

##new csv for filtered data
## syntax things: removing index prevents row index # from becoming an extra column within new dataset
df_filtered.to_csv('Filtered_Shipment_Date.csv', index = False)


# confirming changes
print(df_filtered.head())

print(df_filtered.isnull().sum())



#####E## new code outside filtering

## convert columns for processing -> one-hot encoding
##TESTING ALL THREE MODELS TIME!
##df_filtered['Late by metrics'] = df_filtered[] XX probably not going to use idk what this is


## first drop original cols now that I used them
#and you have to drop these because for the X, you are basically setting the features
##the y, or the target, is basically your answer key
#this is what it means by train test split with x and y
X = df_filtered.drop(columns = ["delivered late", "delivered to client date", "scheduled delivery date", ])
y = df_filtered["delivered late"]

## finally time for one hot encoding
# for drop first = true, it is necessary for logistic model. 
# Tree models would have actually been okay without it

new_X = pd.get_dummies(X, drop_first=True)

# set names of features

new_features = new_X.columns

## train test split time

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.2, random_state = 42)


## all for logistic regression cause its a fat baby and somehow needs this too(referenced notes)
## logistic regression is a giant calculator: if you don't convert scale columns down then the model will crash
## because it thinks that say a column that says 1000 means more than one with 2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# three models summoned

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

## most difficult piece of code 3: (ig jk)
## we need to do EVEN more for this goofy logistic regression

for model_name, model in models.items():
    print(f"{'='*50}")
    print(f"Model meep moop: {model_name}")
    print(f"{'='*50}")

    if model_name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Logistic regression uses coefficients to measure feature strength
        # take the absolute value because a strong negative weight is just as important as a positive one
        # ^ very important concept
        importances = np.abs(model.coef_[0])
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importance = model.feature_importances_


##print metrics

precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)

### done for the day so far, all three models are being run!!
## time to figure out the other stuff 