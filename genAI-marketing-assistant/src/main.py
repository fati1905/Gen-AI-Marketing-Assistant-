# This is the MAIN

# Import libraries
import pandas as pd

# Import local files
import model_training

if __name__ == '__main__':

    # ------ DATA -----
    df = pd.read_csv("/Users/mac-FBENKA22/Desktop/genAI-marketing-assistant/Data/data_cleaned.csv")

    # Drop unnecessary columns
    df = df.drop(columns=['age_category', 'duration']) # Duration affects directly the predictions

    # ------ Modelisation --------
    model = model_training.train_model(df=df, target="y")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
