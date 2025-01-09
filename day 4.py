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


# In[ ]:




