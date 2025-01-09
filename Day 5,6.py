#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install SpeechRecognition pyttsx3 pyaudio


# In[3]:


import pyttsx3
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()
def main():
    speak("Hello! I am your simple bot from Mallareddy university")
    speak("You can say hello, ask my name, or say goodbye.")
    while True:
        command=input("You: ").lower()
        if "hello" in command:
            speak("Hi there! welcome to Mallareddy college.")
        elif "What's your name" in command or "What is your name" in command:
            speak("My name is simple but from Mallareddy college.")
        elif "goodbye" in command:
            speak("Goodbye! Have a great day at mallareddy college")
            break
        else:
            speak("I didn't understand that. please try again.")
if __name__ == "__main__":
    main()


# In[4]:


import pandas as pd
df = pd.read_csv(r'C:\Users\lavan\Desktop\AD\Logistic_Regression.csv')
print(df.head())


# In[5]:


print(df.head(7))


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(5, 4))
# Using the 'c' parameter to specify color (0 = No, 1 = Yes)
plt.scatter(df['Age'], df['Purchased'], c=df['Purchased'], cmap='bwr', label='Purchased')
plt.xlabel('Age')
plt.ylabel('Purchased (0 = No, 1 = Yes)')
plt.title('Sony Project')
# Updating the legend to reflect the correct labels
plt.legend(['No purchase', 'Purchase'])
# Enabling the grid
plt.grid(True)
plt.show()


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x=df[['Age']]
y=df[['Purchased']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
new_age=[[29]]
prediction=model.predict(new_age)
probability=model.predict_proba(new_age)
print(f"Prediction for Age {new_age[0][0]}: {'will purchase' if prediction[0]==1 else 'will not purchase'}")


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Features and target variable
x = df[['Age']]
y = df[['Purchased']]
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)
# Make a prediction for a new age
new_age = [[59]]
prediction = model.predict(new_age)
probability = model.predict_proba(new_age)
# Output the result
print(f"Prediction for Age {new_age[0][0]}: {'will purchase' if prediction[0] == 1 else 'will not purchase'}")


# In[ ]:




