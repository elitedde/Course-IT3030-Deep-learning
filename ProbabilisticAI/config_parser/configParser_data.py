import yaml
from configParser.globalWords import Model
class Data:

    def extractData(self):
        with open("configFiles/sample_data.yaml", 'r') as stream:
            try:
                p = yaml.safe_load(stream)

                self.mode = p["data_mode"]
                self.default_batch_size = p["default_batch_size"]
                self.batch_size = p["batch_size"]
                self.input_shape = p["input_shape"]

                self.force_learn= p["force_learn"]
                self.epochs = p["epochs"]

                self.model = Model(p["model"])
                self.latent_size= p["latent_size"]
                self.force_learn_AE= p["force_learn_AE"]
            except yaml.YAMLError as exc:
                print(exc)
