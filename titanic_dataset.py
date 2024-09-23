import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    df = pd.read_csv('train.csv')
    return df

def data_cleaning(df):

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df
def exploratory_data_analysis(df):

    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.countplot(df['Survived'])
    plt.title('Count Plot of Survived')
    plt.show()


    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Pclass vs Survived')
    plt.show()
def main():
    df = load_data()
    df = data_cleaning(df)
    exploratory_data_analysis(df)

if __name__ == "__main__":
    main()
