# Import libraries
from datasets import load_dataset
import pandas as pd
from googletrans import Translator
import time
from tqdm import tqdm
# Function to convert labels to conventional label: 0 for Negative, 1 for Neutral and 2 for Positive
def convert_label(dataset_df: pd.DataFrame, current_negative, current_neutral, current_positive):
    dataset_df['Label'] = dataset_df['Label'].apply(lambda x: 0 if x == current_negative else 1 if x == current_neutral else 2 if x == current_positive else x)
    return dataset_df


# Function to call API from Google Translate VietnameseText to English and return EnglishText as a new column
def translate_vietnamese_to_english(dataset_df: pd.DataFrame):
    translator = Translator()

    # Translate every 10 rows to avoid Google Translate API's limit, then move to the next 10 rows
    for i in tqdm(range(0, len(dataset_df), 10)):
        # try to translate a batch of 10, if any element in the batch get error, return " " on EnglishText field, then continue to the next element of the batch
        try:
            dataset_df.loc[i:i+9, 'EnglishText'] = dataset_df.loc[i:i+9, 'VietnameseText'].apply(lambda x: translator.translate(x, src='vi').text if x != " " and x != "" else " ")
        except:
            print("Error at index i: ", i)
            for j in range(i, i+10):
                try:
                    dataset_df.loc[j, 'EnglishText'] = translator.translate(dataset_df.loc[j, 'VietnameseText'], src='vi').text if dataset_df.loc[j, 'VietnameseText'] != " " and dataset_df.loc[j, 'VietnameseText'] != "" else " "
                except:
                    print("Error at index j: ", j)
                    dataset_df.loc[j, 'EnglishText'] = " "
        time.sleep(2)


# Function to read dataset from txt file (each two lines is a record, the first line is VietnameseText, the second line is labels) and return a DataFrame
def read_dataset_from_txt(file_path: str, lines_per_record: int = 2):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        dataset = pd.DataFrame({'VietnameseText': lines[::lines_per_record], 'Label': lines[1::lines_per_record]})
        dataset['Label'] = dataset['Label'].apply(lambda x: x.strip())
        return dataset

if __name__ == '__main__':
    
    # Read dataset from txt file
    # aivivn_testset_path = "../data/non-translated-sentiment-dataset/aivivn/test_raw_ANS.txt"
    # aivivn_testset = read_dataset_from_txt(aivivn_testset_path)
    # aivivn_testset = convert_label(aivivn_testset, 'NEG', 'NEU', 'POS')
    # aivivn_testset.head(10)
    # aivivn_testset_backup = aivivn_testset.copy()``
    aivivn_testset_path = "../data/non-translated-sentiment-dataset/aivivn/test_aivivn.csv"
    aivivn_testset = pd.read_csv(aivivn_testset_path)
    aivivn_testset.drop(columns=['index'], inplace=True)
    aivivn_testset = aivivn_testset.rename(columns={'discriptions': 'VietnameseText', 'mapped_rating': 'Label'})
    aivivn_testset = convert_label(aivivn_testset, 1, None, 0)
    # Translate VietnameseText to English and save to a new column
    
    # Fill na in VietnameseText by " "
    aivivn_testset['VietnameseText'] = aivivn_testset['VietnameseText'].fillna(" ")

    # Get first 1000 rows 
    aivivn_testset = aivivn_testset.head(2000)

    translate_vietnamese_to_english(aivivn_testset)
    aivivn_testset.to_csv('test_aivivn.csv', index=False)
    aivivn_testset.to_csv('../data/translated-sentiment-dataset/test_aivivn_1.csv', index=False)
    