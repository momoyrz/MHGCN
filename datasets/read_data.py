import numpy as np
import pandas as pd
import torch
def get_sample_df(data_df, grade, n_samples, seed):
    return data_df[data_df['标签'] == grade].sample(n=n_samples, random_state=seed)


def get_data(args):
    data_df = pd.read_csv(args.csv_dir)

    # Load features as tensor and normalize
    features = np.load(args.features_path)
    features = torch.FloatTensor(features)
    attributes_H = np.load(args.attributes_H_path)
    H = torch.FloatTensor(attributes_H)

    labels = torch.LongTensor(data_df['标签'].tolist())

    # Create test and validation data frames
    test_df = pd.concat([get_sample_df(data_df, i, 50, args.seed) for i in range(7)])
    data_df = data_df.drop(test_df.index)
    val_df = pd.concat([get_sample_df(data_df, i, 30, args.seed) for i in range(7)])
    data_df = data_df.drop(val_df.index)

    # Convert DataFrame indices to tensors
    idx_train = torch.LongTensor(data_df.index.values)
    idx_val = torch.LongTensor(val_df.index.values)
    idx_test = torch.LongTensor(test_df.index.values)
    H = torch.FloatTensor(H)

    if not args.using_attributes:
        H = torch.zeros_like(H)

    return features, labels, idx_train, idx_val, idx_test, H