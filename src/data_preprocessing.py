import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Remove duplicates
    df = df.drop_duplicates()
    # Standardize 'Amount'
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    # Drop 'Time'
    df = df.drop(columns=['Time'])
    return df

if __name__ == "__main__":
    df = load_data('data/raw/creditcard.csv')
    df_clean = preprocess(df)
    df_clean.to_csv('data/processed/creditcard_clean.csv', index=False)