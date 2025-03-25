import pandas as pd


class ResultsMaker:

    def __init__(self, n_class:int, metrics: list[str], models: list[str]):
        # esce un dataframe che ha come colonne:
        # metrica1_modello1, metrica2_modello1 ...

        self.n_class = n_class
        self.metrics = metrics
        self.models = models
        self._column_names = []

        for metric in metrics:
            for model in models:
                self._column_names.append(metric + ' ' + model)

        self.results = []
