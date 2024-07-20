import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure necessary resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')


def transform_text(text):
    # Initialize necessary resources
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Step 1: Convert text to lowercase
    text = text.lower()

    # Step 2: Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Step 3: Remove non-alphanumeric characters
    tokens = [token for token in tokens if token.isalnum()]

    # Step 4: Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Step 5: Stemming
    tokens = [ps.stem(token) for token in tokens]

    # Join the processed tokens back into a single string
    return " ".join(tokens)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam SMS Classifier')
input_sms = st.text_area("Enter the Message")

# 1 Preprocess
transform_SMS = transform_text(input_sms)

# 2 Vectorize
vector_input = tfidf.transform([transform_SMS])
# 3 Predict
result = model.predict(vector_input)[0]
# 4 Display
if st.button("Predict"):
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
