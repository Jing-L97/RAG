# standard imports
"""
preprocess infant-directed speech transcripts as the train data
stratified sampling to ensure the fair proportion in each set
"""
################# Libraries #################
import os
import pandas as pd
from RAG.datasets.train_parser import clean_text

################# Functions #################
val_prop = 0.1
test_prop = 0.1
chunk_num = 11      # predefined chunk numbers, calculate through whole words

def count_words(data:list):
    """count word numbers in a list"""
    word_counts = 0
    for sentence in data:
        words = sentence.split()  # Split the sentence into words
        word_counts += len(words)
    return word_counts

def clean_data(text_path:str,CHILDES_path:str,clean_path:str):
    """clean and get stat of each file to prepare for the data chunking"""

    stat_dict = {}
    # read and clean txt files
    for filename in os.listdir(text_path):
        with open(text_path + filename, 'r', errors='ignore') as f:
            with open(clean_path + filename[:-6] + '.txt', 'w', errors='ignore') as file:
                print('Loading and cleaning ' + filename)
                text = f.readlines()
                # clean train data
                cleaned_text = clean_text(text)
                # get stat of the cleaned txt
                stat_dict[filename[:-6]] = count_words(cleaned_text)
                for sent in cleaned_text:
                    file.write(sent + '\n')

    # read and clean CHILDES csv
    CHILDES_trans = pd.read_csv(CHILDES_path)
    # filter parents' utterances
    CHILDES_parent = CHILDES_trans[CHILDES_trans['speaker'] != 'CHI']
    CHILDES_text = CHILDES_parent['content'].tolist()
    stat_dict['CHILDES'] = CHILDES_parent['num_tokens'].sum()
    with open(clean_path + 'CHILDES.txt', 'w', errors='ignore') as file:
        for sent in CHILDES_text:
            file.write(sent + '\n')
    # get the proportion fo the whole dataset as well
    token_all = sum(stat_dict.values())
    for dataset,token_num in stat_dict.items():
        # get prop of each part
        stat_dict[dataset] = [token_num,token_num/token_all]
    # print put a df to store the results
    data_list = [(key, *value) for key, value in stat_dict.items()]
    # Convert list of tuples to DataFrame
    stat = pd.DataFrame(data_list, columns=['dataset', 'token_num', 'prop'])
    stat.to_csv(clean_path + 'stat.csv')
    return stat_dict


def split_data(data:list):
    """split dataset into train,val and test sets"""
    size_train = int((1 - val_prop - test_prop) * len(data))
    size_val = int(val_prop * len(data))
    data_train, data_val, data_test = (data[:size_train],
                                       data[size_train:size_train + size_val], data[size_train + size_val:])
    # convert the result into string
    data_train = ' '.join(data_train)
    data_val = ' '.join(data_val)
    data_test = ' '.join(data_test)
    print("Done splitting input file into train/dev/test sets.")
    return data_train, data_val, data_test

def group_txt(all_train,out_path,mode):
    """group text into different chunks"""
        n = 0
        while n < chunk_num:
            txt_dir = out_path + '/' + str(n)
            if not os.path.exists(txt_dir):
                os.mkdir(txt_dir)
            with open(txt_dir + '/' + mode + '.txt', 'w', errors='ignore') as file:
                # loop over each type
                for sent in all_train:
                    # print put the results
                    file.write(sent[n])
            n += 1


def chunk_data(clean_path:str,out_path:str,chunk_num:int,stat_dict:dict):
    """
    chunk data based on each chunk size
    return the train,val,test for each chunk, aiming for a balanced prop
    """
    # loop over the clean_path
    all_train = []
    all_val = []
    all_test = []
    for file in os.listdir(clean_path):
        # divide by chunk number
        chunk_size = int(stat_dict[file[:-4]][0]/chunk_num)
        # chunk the data
        with open(clean_path + file, 'r', errors='ignore') as f:
            print('Loading ' + file)
            text_lst = f.readlines()
            # concatenate the results
            text = ' '.join(text_lst).split(' ')
            train = []
            val = []
            test = []
            n = 0
            while n < chunk_num -1:
                chunked_text = text[chunk_size*n:chunk_size*(n+1)]
                data_train, data_val, data_test = split_data(chunked_text)
                train.append(data_train)
                val.append(data_val)
                test.append(data_test)
                n += 1
            chunked_text = text[chunk_size*n:]     # the last chunk contains the rest of the results
            data_train, data_val, data_test = split_data(chunked_text)
            # convert back to string
            train.append(data_train)
            val.append(data_val)
            test.append(data_test)
        all_train.append(train)
        all_val.append(val)
        all_test.append(test)
    # group the results by order
    group_txt(all_train, out_path, 'train')
    group_txt(all_val, out_path, 'val')
    group_txt(all_test, out_path, 'test')


def get_all_stat(out_path):
    """get stat in the folder"""
    # get stat to verify the results
    stat_all = pd.DataFrame()
    for file_dir in os.listdir(out_path):
        # get train, val and test iteratively
        mode_lst = ['train', 'val', 'test']
        for mode in mode_lst:
            with open(out_path + file_dir + '/' + mode + '.txt', 'r', errors='ignore') as f:
                text = f.readlines()
                # clean train data
                stat = pd.DataFrame([file_dir, mode, count_words(text)]).T
                stat_all = pd.concat([stat_all, stat])
    stat_all.rename(columns={0: 'chunk', 1: 'set', 2: 'token_num'}, inplace=True)
    stat_all.to_csv(out_path + 'stat.csv')

################# Main #################
if __name__ == "__main__":

    # concatenate results into a txt format
    text_path = '/data/Generative_replay/knn-transformers/data/train/raw/babylm_100M/'
    CHILDES_path = '/data/Generative_replay/knn-transformers/data/train/raw/CHILDES_trans.csv'
    clean_path = '/data/Generative_replay/knn-transformers/data/train/cleaned/'
    out_path = '/data/Generative_replay/knn-transformers/data/train/chunked/'
    # clean and split text data
    stat_dict = clean_data(text_path, CHILDES_path, clean_path)
    # chunk the data
    chunk_data(clean_path,out_path,chunk_num,stat_dict)
    get_all_stat(out_path)