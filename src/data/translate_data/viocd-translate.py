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


if __name__ == '__main__':
    
    viocd_dataset = load_dataset("tarudesu/ViOCD")
    test_viocd = viocd_dataset['test']
    val_viocd = viocd_dataset['validation']

    test_viocd_df = pd.DataFrame(test_viocd)
    test_viocd_df = test_viocd_df[['review', 'label']].rename(columns={'review': 'VietnameseText', 'Label': 'label'})
    test_viocd_df = convert_label(test_viocd_df, 1, None, 0)
    test_viocd_df.head(10)

    translate_vietnamese_to_english(test_viocd_df)
    test_viocd_df.to_csv('../data/translated-sentiment-dataset/test_viocd.csv', index=False)