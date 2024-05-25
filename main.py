import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    df.drop_duplicates(inplace=True)
    
    #pisahkan kelas dan feature
    X = df.drop(columns='NObeyesdad')
    y = df['NObeyesdad']

if __name__ == "__main__":
    main()