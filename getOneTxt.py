import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("patents_data.csv")
    c = df.columns
    print(c)

    for index, row in df.iterrows():
        if row[c[1]] != "US11258057":
            continue
        print(1)
        text = row[c[4]]
        with open("US11258057B2.txt", 'w') as f:
            f.write(text)
        break
