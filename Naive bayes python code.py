import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import operator

DATA_PATH = '20_newsgroups'
directory_names = sorted(os.listdir(DATA_PATH))

category_texts = {}
for folder in directory_names:
    category_texts[folder] = []
    for document in os.listdir(os.path.join(DATA_PATH, folder)):
        with open(os.path.join(DATA_PATH, folder, document), encoding='latin-1') as opened_file:
            category_texts[folder].append(opened_file.read())


punct_chars = list(punctuation)
stopWords = stopwords.words('english') + punct_chars + [
    'subject:', 'from:', 'date:', 'newsgroups:', 'message-id:', 'lines:', 'path:', 'organization:',
    'would', 'writes:', 'references:', 'article', 'sender:', 'nntp-posting-host:', 'people',
    'university', 'think', 'xref:', 'cantaloupe.srv.cs.cmu.edu', 'could', 'distribution:', 'first',
    'anyone', 'world', 'really', 'since', 'right', 'believe', 'still', 
    "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'"
]


word_bank = {}
for category in category_texts:
    for doc in category_texts[category]:
        for word in doc.split():
            word_lower = word.lower()
            if word_lower not in stopWords and len(word_lower) >= 5:
                word_bank[word_lower] = word_bank.get(word_lower, 0) + 1

sorted_wordlist = sorted(word_bank.items(), key=operator.itemgetter(1), reverse=True)
feature_list = [word for word, count in sorted_wordlist[:2000]]

Y = []
for category in category_texts:
    Y.extend([category] * len(category_texts[category]))
Y = np.array(Y)

df = pd.DataFrame(columns=feature_list)
for folder in directory_names:
    for document in os.listdir(os.path.join(DATA_PATH, folder)):
        df.loc[len(df)] = np.zeros(len(feature_list))
        with open(os.path.join(DATA_PATH, folder, document), encoding='latin-1') as opened_file:
            for word in opened_file.read().split():
                if word.lower() in feature_list:
                    df.at[len(df)-1, word.lower()] += 1

X = df.values


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)


print(f"Model accuracy on test set: {nb_model.score(x_test, y_test)}")

def fit(x_train, y_train):
    outcome = {"total_category_texts": len(y_train)}
    target_names = set(y_train)
    for current_label in target_names:
        outcome[current_label] = {"total_count": 0}
        current_rows = (y_train == current_label)
        x_train_current = x_train[current_rows]
        for i, feature in enumerate(feature_list):
            word_count = x_train_current[:, i].sum()
            outcome[current_label][feature] = word_count
            outcome[current_label]["total_count"] += word_count
    return outcome

def probability(x, dictionary, current_class):
    output = np.log(dictionary[current_class]["total_count"]) - np.log(dictionary["total_category_texts"])
    num_features = len(feature_list)
    for i in range(num_features):
        word_probability = np.log(dictionary[current_class][feature_list[i]] + 1) - np.log(dictionary[current_class]["total_count"] + num_features)
        output += x[i] * word_probability
    return output

def determine_class(x, dictionary):
    best_class = None
    best_prob = -np.inf
    for current_class in dictionary.keys():
        if current_class == "total_category_texts":
            continue
        class_prob = probability(x, dictionary, current_class)
        if class_prob > best_prob:
            best_prob = class_prob
            best_class = current_class
    return best_class

def predict(X_test, dictionary):
    class_predictions = []
    for x in X_test:
        class_predictions.append(determine_class(x, dictionary))
    return class_predictions

dictionary = fit(x_train, y_train)
y_pred = predict(x_test, dictionary)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))