import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("patents_data.csv")
    print(data.columns)

