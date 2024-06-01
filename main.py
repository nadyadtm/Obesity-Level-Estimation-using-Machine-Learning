import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def main():
    df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    df.drop_duplicates(inplace=True)
    
    #pisahkan kelas dan feature
    X = df.drop(columns='NObeyesdad')
    y = df['NObeyesdad']

    #ubah class menjadi bentuk angka
    classes = {
        'Normal_Weight' : 0,
        'Insufficient_Weight' : 1,
        'Overweight_Level_I' : 2,
        'Overweight_Level_II' : 3,
        'Obesity_Type_I' : 4,
        'Obesity_Type_II' : 5,
        'Obesity_Type_III' : 6,
    }
    y = y.astype('category')
    y = y.cat.rename_categories(classes)

    #transform dengan label encoder
    categories = {
        'no' : 0,
        'Sometimes' : 1,
        'Frequently' : 2,
        'Always' : 3
    }

    X['CALC'] = X['CALC'].astype('category')
    X['CAEC'] = X['CAEC'].astype('category')

    X['CALC']=X['CALC'].cat.rename_categories(categories)
    X['CAEC']=X['CAEC'].cat.rename_categories(categories)

    #transform dengan one hot encoding
    onehot = pd.get_dummies(X[['Gender',
                  'MTRANS',
                  'family_history_with_overweight',
                 'SMOKE','SCC','FAVC']], dtype='float')

    X = X.drop(columns=['Gender',
                    'MTRANS',
                    'family_history_with_overweight',
                    'SMOKE','SCC','FAVC'])
    X_ = pd.concat([X,onehot], axis=1)

    X_.to_csv('data/data_processed.csv')

    #pembagian data
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y, test_size=0.2, random_state=42)
    # X_valid, X_test, y_valid, y_test = train_test_split(
    #     X_test, y_test, test_size=0.5, random_state=42)

    print("Data Latih : ",len(y_train))
    # print("Data Valid : ",len(y_valid))
    print("Data Test  : ",len(y_test))

    train_data = {"x":X_train, "y":y_train}
    test_data = {"x":X_test, "y":y_test}

    with open("data/train_data.pickle", "wb") as f:
        pickle.dump(train_data, f)
        
    with open("data/test_data.pickle", "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    main()