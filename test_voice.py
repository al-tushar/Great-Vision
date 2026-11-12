import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.say("Hello, I am your vision assistant")
engine.runAndWait()
print("Done speaking")