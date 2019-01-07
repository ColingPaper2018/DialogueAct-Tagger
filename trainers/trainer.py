class Trainer:
    def train_dimension(self, out_file):
        raise NotImplementedError()

    def train_task(self, out_file):
        raise NotImplementedError()

    def train_som(self, out_file):
        raise NotImplementedError()

    def train_all(self, out_file):
        raise NotImplementedError()

    def train(self, layer, out_file):
        if layer == 'all':
            self.train_all(out_file)
        elif layer == 'task':
            self.train_task(out_file)
        elif layer == 'som':
            self.train_som(out_file)
        elif layer == 'dim':
            self.train_dimension(out_file)
        else:
            raise NotImplementedError(f"Unknown taxonomy layer: {layer}")
