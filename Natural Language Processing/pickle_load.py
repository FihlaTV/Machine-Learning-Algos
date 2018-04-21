from sklearn.externals import joblib
from nltk.stem import PorterStemmer
import numpy as np
from nltk.corpus import stopwords

pipeline = joblib.load('class.pkl')


def classify(array):
    X= pipeline.named_steps['cv'].transform(array).toarray()
    predict = pipeline.named_steps['classifier'].predict(X)
    print(predict)
    # predict = pipeline.predict(array)
    # print(predict)

# Testing on my own data

test_data = [
    "it was a wonderful story.. had fun with my wife",
    "the plot line sucked",
    "it has got nothing in it, dont waste your time and money",
    "what the fuck"
]

test_array = []

for test_review in test_data:
    test_review = test_review.lower()
    test_review = test_review.split()
    ps = PorterStemmer()
    test_review = [ps.stem(word) for word in test_review if word not in set(stopwords.words('english'))]
    test_review = ' '.join(test_review)
    test_array.append(test_review)

classify(test_data)

print("finish")