import yaml
from configParser.globalWords import lossFunct, activationFunct, regularization

class Parameters:

    def extractParameters(self):
        with open("configFiles/5hiddenlayer.yaml", 'r') as stream:
            try:
                p = yaml.safe_load(stream)
                self.input_size = p["input_size"]
                self.verbose = p["verbose"]
                self.lrate = p["lrate"]
                self.mini_batch_size = p["mini_batch_size"]
                self.epochs = p["epochs"]
                self.size_layer = p["size_layer"]
                self.regularizationRate = p["regularization_rate"]
                self.wreg = p["weight"]
                self.br = p["bias"]
                self.softmax = p["softmax"]
                self.loss = lossFunct(p["loss"])
                functions = p["activation_function"]
                self.activation_functions = []
                for i, j in enumerate(functions):
                    self.activation_functions.append(activationFunct(j))
                self.regularization = regularization(p["regularization"])

            except yaml.YAMLError as exc:
                print(exc)



