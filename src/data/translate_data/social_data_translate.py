import pandas as pd
import argparse
#from utils.utils import get_english_model, get_english_tokenizer, get_vietnamese_tokenizer, get_vietnamese_model, model_inference
# Import libraries
import pandas as pd
from googletrans import Translator
import time
from tqdm import tqdm


def translate_vietnamese_to_english(dataset_df: pd.DataFrame):
    translator = Translator()

    # Translate every 10 rows to avoid Google Translate API's limit, then move to the next 10 rows
    for i in tqdm(range(0, len(dataset_df), 20)):
        # try to translate a batch of 10, if any element in the batch get error, return " " on EnglishText field, then continue to the next element of the batch
        try:
            dataset_df.loc[i:i+19, 'EnglishText'] = dataset_df.loc[i:i+19, 'VietnameseText'].apply(lambda x: translator.translate(x, src='vi').text if x != " " and x != "" else " ")
        except:
            print("Error at index i: ", i)
            for j in range(i, i+20):
                try:
                    dataset_df.loc[j, 'EnglishText'] = translator.translate(dataset_df.loc[j, 'VietnameseText'], src='vi').text if dataset_df.loc[j, 'VietnameseText'] != " " and dataset_df.loc[j, 'VietnameseText'] != "" else " "
                except Exception as e:
                    print(e)
                    print("Error at index j: ", j)
                    dataset_df.loc[j, 'EnglishText'] = " "
        time.sleep(2)

def append_to_file(filename, text):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(text + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--is_negative', type=int, default=0)
    args = parser.parse_args()

    save_path = "/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/"
    # Load data
    if args.is_negative == 1:
        dataset_df = pd.read_csv('/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/social_data/negative_sample.csv')
    elif args.is_negative == 0:
        dataset_df = pd.read_csv('/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/social_data/non_negative_sample.csv')
    elif args.is_negative == 2:
        dataset_df = pd.read_csv('/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/social_data/UIT_VISD4SA.csv')

    if (args.batch_idx + 1)*args.batch_size < len(dataset_df):
        dataset_df = dataset_df.iloc[args.batch_idx*args.batch_size:(args.batch_idx+1)*args.batch_size].reset_index(drop=True)
    else:
        dataset_df = dataset_df.iloc[args.batch_idx*args.batch_size:].reset_index(drop=True)

    print(dataset_df)
    translate_vietnamese_to_english(dataset_df)

    # Save data
    if args.is_negative == 1:
        dataset_df.to_csv(f'/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/negative_sample_{args.batch_idx}.csv', index=False)
    elif args.is_negative == 0:
        dataset_df.to_csv(f'/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/non_negative_sample_{args.batch_idx}.csv', index=False)
    elif args.is_negative == 2:
        dataset_df.to_csv(f'/home/ldap-users-2/internship_2025/nguyenphuong-p/thesis/data/pretrain_data/social_uit_sample_{args.batch_idx}.csv', index=False)
    append_to_file(f'{save_path}log.txt', f'Batch {args.batch_idx} with {args.batch_size} of is negative ({args.is_negative}) at {time.strftime("%Y-%m-%d %H:%M:%S")} done')

if __name__ == "__main__":
    main()