from read_csv import Database
import pickle

'''training = Database("training")
print("loaded")
pickle.dump(training, open("save_training.p", "wb"))
print(1,"Done")'''

training_short = Database("training_short")
print("loaded")
pickle.dump(training_short, open("save_training_short.p", "wb"))
print(2,"Done")

training_lessshort = Database("training_lessshort")
print("loaded")
pickle.dump(training_lessshort, open("save_training_lessshort.p", "wb"))
print(3,"Done")

training_kindashort = Database("training_kindashort")
print("loaded")
pickle.dump(training_kindashort, open("save_training_kindashort.p", "wb"))
print(4,"Done")