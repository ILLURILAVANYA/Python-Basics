#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install SpeechRecognition pyttsx3 pyaudio


# In[ ]:


import pyttsx3
import speech_recognition as sr
engine=pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()
def listen():
    recognizer=sr.Recognizer()
    with sr.Microphone() as source:
        speak("I am Listening,please speak")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.UnknownValueError:
            speak("Sorry,i didnt catch that")
            return " "
def main():
    speak("Hello,i am basic voice bot for mallreddy college you can ")
    while True:
        command=listen()
        if "hello" in command:
            speak("hi there welcome to mallareddy college")
        elif "what's your name" in command or "what is your name" in command:
            speak("my name is basic bot from mallareddy college")
        elif "goodbye" in command:
            speak("goodbye have a great day at mallareddy college")
if __name__=="__main__":
    main()


# In[ ]:




