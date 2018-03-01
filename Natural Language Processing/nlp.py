#Natural language processing

#Importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Data Set
dataset= pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
corpus=[]

for n in range(0, 1000):
    review= re.sub('[^a-zA-Z]',' ', dataset['Review'][n])
    review= review.lower()
    review= review.split()
    #to convert all tense to present
    ps= PorterStemmer()
    #removing conjugation
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)

#Creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 1500)
X= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values

#Training the model

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("Finish")
