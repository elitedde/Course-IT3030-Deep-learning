import yaml

class Data:

    def extractData(self):
        with open("configFiles/sample_data.yaml", 'r') as stream:
            try:
                p = yaml.safe_load(stream)
                self.data_train_dim = p["data_train_dim"]
                self.data_val_dim = p["data_val_dim"]
                self.data_test_dim = p["data_test_dim"]
                self.n = p["n"]
                self.class_dimension = p["class_dimension"]
                self.noise_ratio = p["noise_ratio"]
                self.flatten = p["flatten"]
                self.centered = p["centered"]
                self.rect_range_height = p["rect_range_height"]
                self.rect_range_width = p["rect_range_width"]
                self.vertical_bar_width = p["vertical_bar_width"]
                self.horizontal_bar_width = p["horizontal_bar_width"]
                self.square_range = p["square_range"]

            except yaml.YAMLError as exc:
                print(exc)
