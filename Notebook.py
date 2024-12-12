from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

#PREPARING THE DATA
data = pd.read_csv('IMDB Dataset.csv')
X = data['review']
y = data['sentiment']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

#VECTORIZATION
vector = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3))
X_train_vectorized = vector.fit_transform(xtrain)
X_test_vectorized = vector.transform(xtest)

# TRAINING THE MODEL
model = LogisticRegression(max_iter=1000, C=1.8)
model.fit(X_train_vectorized, ytrain)

#Prediction 
ypred = model.predict(X_test_vectorized)

#SCORES
score = accuracy_score(ytest, ypred)
matrix = confusion_matrix(ytest, ypred)

print(f'Accuracy Score, {score*100}%')
print(f'Confusion Matrix, {matrix}')

#NEW DATA PREDICTION
user_input = input()
user_input_vec = vector.transform([user_input])
ypred_new = model.predict(user_input_vec)
print(ypred_new)
