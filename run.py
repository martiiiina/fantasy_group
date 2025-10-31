import Training
import preprocessing
import Prediction

print("Running Preprocessing...")
preprocessing.main()

print("Running Training.py...")
Training.main()

print("Running Prediction.py...")
Prediction.main()

print("All done!")
