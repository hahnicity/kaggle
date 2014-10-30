from pandas import read_csv


def get_data(data_set):
    y = data_set["Survived"]
    del data_set["Survived"]
    del data_set["PassengerId"]
    del data_set["Fare"]
    del data_set["Ticket"]
    del data_set["Cabin"]
    del data_set["Embarked"]

def main():
    training_set = get_data(read_csv("train.csv"))
