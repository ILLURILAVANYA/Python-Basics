#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()
input_string = "This is an example text.The sentence."
all_sentences = tokenizer.tokenize(input_string)
print(all_sentences)


# In[2]:


import re
text = "Hello, World ! 123"
text = re.sub(r'[^a-z\s]','',text)
print(text)


# In[ ]:


import re
from sklearn.feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = [
    ['spam', 'Had your mobile 11 months or more? U R entitled update to the latest model'],
    ['ham', 'Im gonna be home soon and I dont want to talk about this stuff'],
    ['spam', 'Congratulations! You have won a $1000 Walmart gift card'],
    ['ham', 'im here now'],
    ['ham', ' where r u'],
    ['spam', ' we have a $100 gift card for you claim it now']
]

labels=[row[0] for row in data]
colums = [

   
    text=text.lower()
    text = re.sub(r'[a-z\s]','',text)
    words= text.split()
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    stop_words=set(stopwords.words('english'))
    filtered_words=[word for word in words if word not in stop_words]
    return ' '.join(filtered_words)
preprocessed_images = [preprocess_text(msg) for msg in messages]
    


# In[3]:


import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = [
    ['spam','Had your mobie 11 months or more? U R entitled to Update to the latest'],
    ['ham',"I'm gone be home soon and I dont want to talk about this stuff"],
    ['spam','Congratulations! you have won a $1000 walmart gift card.call now'],
    ['ham','Hey,are we still meeting for lunch tomorrow at 12? Let me know.'],
    ['ham','Dont forget to bring the documents for the meeting'],
    ['spam','You have been selected for a free vaction!Reply YES to claim now'],
]
labels = [row[0] for row in data]
messages = [row[1] for row in data]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    words = text.split()
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)
preprocessed_messages = [preprocess_text(msg) for msg in messages]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_messages)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n",classification_report(y_test, y_pred))

new_sms = "Congatulations! you've won a free ticket to cghd"
new_sms_preprocessed = preprocess_text(new_sms)
new_sms_vectorized = vectorizer.transform([new_sms_preprocessed])
prediction = model.predict(new_sms_vectorized)
print(f"prediction for new SMS: {prediction[0]}")


# In[ ]:




