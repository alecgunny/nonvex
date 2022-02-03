import keras_tuner as kt

hyperparameters = kt.HyperParameters()
hyperparameters.Float("learning_rate", 5e-6, 5e-4, sampling="log")
hyperparameters.Choice("batch_size", [32, 64, 128])
