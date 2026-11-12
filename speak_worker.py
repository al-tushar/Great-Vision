import sys

import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.say(sys.argv[1])
engine.runAndWait()