import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import Dataset
import text_normalization
from pickle import dump, load
from sklearn.model_selection import train_test_split

def loadTrainValData(batchsize=16, num_worker=2, pretraine_path="bert-base-uncased"):
    data = pd.read_csv('data/training_data.csv', delimiter=',')
    Train_data, Dev_data = train_test_split(data, test_size=0.2, stratify=data[['sarcasm', 'sentiment']], random_state=42, shuffle=True)
    Dev_data.to_csv('data/dev_set.csv')
    Train_data['tweet'] = Train_data['tweet'].apply(lambda x: text_normalization.clean(x))
    Dev_data['tweet'] = Dev_data['tweet'].apply(lambda x: text_normalization.clean(x))

    print(f'Training data size {Train_data.shape}')
    print(f'Validation data size {Dev_data.shape}')
    DF_train = Dataset.TrainDataset(Train_data, pretraine_path)
    DF_dev = Dataset.TrainDataset(Dev_data, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_dev_loader = DataLoader(dataset=DF_dev, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_dev_loader

def loadTestData(batchsize=16, num_worker=2, pretraine_path="bert-base-uncased"):
    Test_data = pd.read_csv('data/test_set.csv', delimiter=',')
    print(f'Test data size {Test_data.shape}')

    Test_data['tweet'] = Test_data['tweet'].apply(lambda x: text_normalization.clean(x))

    DF_test = Dataset.TestDataset(Test_data, pretraine_path)

    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                               num_workers=num_worker)
    return DF_test_loader


def loadTrainValData_v2(batchsize=16, num_worker=2, pretraine_path="bert-base-uncased"):
    Train_data = pd.read_csv('data/ArSarcasm_train.csv', delimiter=',')
    Train_data['tweet'] = Train_data['tweet'].apply(lambda x: text_normalization.clean(x))
    Dev_data = pd.read_csv('data/ArSarcasm_test.csv', delimiter=',')
    Dev_data['tweet'] = Dev_data['tweet'].apply(lambda x: text_normalization.clean(x))

    print(f'Training data size {Train_data.shape}')
    print(f'Validation data size {Dev_data.shape}')
    DF_train = Dataset.TrainDataset(Train_data, pretraine_path)
    DF_dev = Dataset.TrainDataset(Dev_data, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_dev_loader = DataLoader(dataset=DF_dev, batch_size=batchsize, shuffle=False,
                               num_workers=num_worker)
    return DF_train_loader, DF_dev_loader



