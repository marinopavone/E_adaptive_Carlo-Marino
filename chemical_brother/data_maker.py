from __future__ import annotations

import os
from enum import Enum
from random import randint

import pandas as pd


class ChemicalClass(Enum):
    ACETIC_ACID = "ACETIC_ACID"
    ACETONE = "ACETONE"
    AMMONIA = "AMMONIA"
    CALCIUM_NITRATE = "CALCIUM_NITRATE"
    ETHANOL = "ETHANOL"
    FORMIC_ACID = "FORMIC_ACID"
    HYDROCHLORIC_ACID = "HYDROCHLORIC_ACID"
    HYDROGEN_PEROXIDE = "HYDROGEN_PEROXIDE"
    NELSEN = "NELSEN"
    PHOSPHORIC_ACID = "PHOSPHORIC_ACID"
    POTABLE_WATER = "POTABLE_WATER"
    POTASSIUM_NITRATE = "POTASSIUM_NITRATE"
    SODIUM_CHLORIDE = "SODIUM_CHLORIDE"
    SODIUM_HYDROXIDE = "SODIUM_HYDROXIDE"
    SODIUM_HYPOCHLORITE = "SODIUM_HYPOCHLORITE"


class DataMaker:
    def __init__(self, folder_path: str):
        self.classes = None
        self.folder_path = os.path.join(os.getcwd(), folder_path)

    def make_full_dataset(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for file_name in os.listdir(self.folder_path):
            if file_name.split(".")[0][:-2] not in self.classes:
                continue
            df = pd.read_csv(os.path.join(self.folder_path, file_name))
            data = pd.concat([data, df])
        return data

    def make_steady_state_dataset(self, steady_state_index: int) -> pd.DataFrame:
        data = pd.DataFrame()
        for file_name in os.listdir(self.folder_path):
            if file_name.split(".")[0][:-2] not in self.classes:
                continue
            df = pd.read_csv(os.path.join(self.folder_path, file_name))
            df = df[steady_state_index:]
            data = pd.concat([data, df])
        return data

    def make_train_test_experiments(self, test_experiments: int, steady_state_index: int
    ):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        test_counter = {i: 0 for i in self.classes}
        for file_name in os.listdir(self.folder_path):
            if file_name.split(".")[0][:-2] not in self.classes:
                continue
            df = pd.read_csv(os.path.join(self.folder_path, file_name))
            df = df[steady_state_index:]
            r = randint(0, 10)
            if (test_counter[df["CLASS"].iloc[0]] < test_experiments) and (r < 5):
                test_counter[df["CLASS"].iloc[0]] += 1
                test_data = pd.concat([test_data, df])
            else:
                train_data = pd.concat([train_data, df])

        return train_data, test_data

    def split_train_test_by_experiment(self, test_experiments: list, stady_state_start, stady_state_end):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        test_experiments_list=[str(test) for test in test_experiments]

        for file_name in os.listdir(self.folder_path):
            if file_name.split(".")[0][:-2] not in self.classes:
                continue
            df = pd.read_csv(os.path.join(self.folder_path, file_name))
            df = df[stady_state_start:stady_state_end]
            if file_name.split(".")[0][-1] in test_experiments_list:
                test_data = pd.concat([test_data, df])
            else:
                train_data = pd.concat([train_data, df])

        return train_data, test_data
    def set_contamination_classes(self, classes: list[ChemicalClass] | None = None):
        self.classes = [c.value for c in classes]
