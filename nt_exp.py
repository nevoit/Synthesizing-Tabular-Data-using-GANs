import os

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from nt_gan import GAN
from nt_gg import GG

dataset_directory = 'datasets'
saved_models_path = 'outputs'


def prepare_architecture(arff_data_path):
    """
    This function create the architecture of the GAN network.
    The generator and the discriminator are created and then combined into the GAN model
    :param arff_data_path: data path for the arff file
    :return: a dictionary with all the relevant variables for the next stages
    """
    data, meta_data = arff.loadarff(arff_data_path)  # This function reads arff file into tuple of data and its meta.
    df = pd.DataFrame(data)
    columns = df.columns
    transformed_data, x, x_scaled, meta_data_rev, min_max_scaler = create_scaled_data(df, meta_data)

    number_of_features = len(transformed_data.columns)  # Define the GAN and training parameters

    return x_scaled, meta_data_rev, columns, min_max_scaler, number_of_features


def create_scaled_data(df, meta_data):
    """

    :param df:
    :param meta_data:
    :return:
    """
    meta_data_dict = {k: {a.replace(' ', ''): b + 1 for b, a in enumerate(v.values)} for k, v in
                      meta_data._attributes.items() if
                      v.type_name != 'numeric'}  # Starts from one and not zero because one is for Nan values
    meta_data_rev = {k: {b + 1: a.replace(' ', '') for b, a in enumerate(v.values)} for k, v in
                     meta_data._attributes.items() if
                     v.type_name != 'numeric'}  # Starts from one and not zero because one is for Nan values
    transformed_data = df.copy()
    for col in df.columns:
        if col in meta_data_dict:
            # Sometimes the values can not be found in the meta data, so we treat these values as Nan
            transformed_data[col] = transformed_data[col].apply(
                lambda x: meta_data_dict[col][str(x).split('\'')[1]] if str(x).split('\'')[1] in meta_data_dict[
                    col] else 0)
    x = transformed_data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return transformed_data, x, x_scaled, meta_data_rev, min_max_scaler


def re_scaled_data(data, columns, meta_data_rev, min_max_scaler):
    """
    This function re-scaled the fake data to the original format.
    :param data: the data we want to re scaled
    :param columns:
    :param meta_data_rev:
    :return:
    """
    data_inv = min_max_scaler.inverse_transform(data)
    df = pd.DataFrame(data_inv, columns=columns)
    transformed_data = df.copy()
    for col in transformed_data.columns:
        if col in meta_data_rev:
            # Sometimes the values can not be found in the meta data, so we treat these values as Nan
            transformed_data[col] = transformed_data[col].apply(
                lambda x: meta_data_rev[col][int(round(x))] if int(round(x)) in meta_data_rev[
                    col] else np.nan)
    return transformed_data


def first_question():
    """
    This function answers the first question
    :return:
    """
    to_plot_losses = True
    results_output = os.path.join(saved_models_path, f'question_one_results.csv')
    results = {'dataset': [], 'lr': [], 'ep': [], 'bs': [], 'alpha': [], 'dropout': [], 'gen_loss': [], 'dis_loss': [],
               'activation': [], 'fooled_len': [], 'not_fooled_len': [], 'mean_min_distance_fooled': [],
               'mean_min_distance_not_fooled': [], 'mean_min_distance_gap': []}
    # w1 * (MMDF + MMDNF) - w3 * (MMDG) + w2 * (NFL/ 100)
    # MMDG = MMDNF - MMDF
    # data_name = ["adult", "bank-full"]
    # learning_rate = [0.01, 0.001, 0.0001]
    # epochs = [5, 10, 15]
    # batch_size = [64, 128, 1024]
    # alpha_relu = [0.2, 0.5]
    # dropout = [0.3, 0.5]
    data_name = ["adult"]
    learning_rate = [0.001]
    epochs = [10]
    batch_size = [128]
    alpha_relu = [0.5]
    dropout = [0.5]
    loss = 'binary_crossentropy'
    activation = 'sigmoid'

    for data in data_name:
        for lr in learning_rate:
            for ep in epochs:
                for bs in batch_size:
                    for al in alpha_relu:
                        for dr in dropout:
                            arff_data_path = f'./datasets/{data}.arff'
                            model_name = f'data_{data}_ep_{ep}_bs_{bs}_lr_{lr}_al_{al}_dr_{dr}'
                            pca_output = os.path.join(saved_models_path, f'{model_name}_pca.png')
                            fooled_output = os.path.join(saved_models_path, f'{model_name}_fooled.csv')
                            not_fooled_output = os.path.join(saved_models_path, f'{model_name}_not_fooled.csv')

                            x_scaled, meta_data_rev, columns, min_max_scaler, number_of_features = prepare_architecture(
                                arff_data_path)
                            gan_obj = GAN(number_of_features=number_of_features, saved_models_path=saved_models_path,
                                          learning_rate=lr, alpha_relu=al, dropout=dr,
                                          loss=loss, activation=activation)
                            gen_loss, dis_loss = gan_obj.train(scaled_data=x_scaled, epochs=ep, batch_size=bs,
                                                               to_plot_losses=to_plot_losses, model_name=model_name)
                            dis_fooled_scaled, dis_not_fooled_scaled, mean_min_distance_fooled, mean_min_distance_not_fooled = gan_obj.test(
                                scaled_data=x_scaled, sample_num=100, pca_output=pca_output)
                            dis_fooled = re_scaled_data(data=dis_fooled_scaled, columns=columns,
                                                        meta_data_rev=meta_data_rev,
                                                        min_max_scaler=min_max_scaler)
                            dis_fooled.to_csv(fooled_output)
                            dis_not_fooled = re_scaled_data(data=dis_not_fooled_scaled, columns=columns,
                                                            meta_data_rev=meta_data_rev,
                                                            min_max_scaler=min_max_scaler)
                            dis_not_fooled.to_csv(not_fooled_output)
                            results['dataset'].append(data)
                            results['lr'].append(lr)
                            results['ep'].append(ep)
                            results['bs'].append(bs)
                            results['alpha'].append(al)
                            results['dropout'].append(dr)
                            results['gen_loss'].append(gen_loss)
                            results['dis_loss'].append(dis_loss)
                            results['activation'].append(activation)
                            results['fooled_len'].append(len(dis_fooled_scaled))
                            results['not_fooled_len'].append(len(dis_not_fooled_scaled))
                            results['mean_min_distance_fooled'].append(mean_min_distance_fooled)
                            results['mean_min_distance_not_fooled'].append(mean_min_distance_not_fooled)
                            results['mean_min_distance_gap'].append(mean_min_distance_not_fooled-mean_min_distance_fooled)
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(results_output, index=False)


def second_question():

    data_name = ["adult", "bank-full"]
    learning_rate = [0.001]
    epochs = [10]
    batch_size = [128]
    alpha_relu = [0.2]
    dropout = [0.3]
    results = {'dataset': [], 'lr': [], 'ep': [], 'bs': [], 'alpha': [], 'dropout': [], 'gen_loss': [], 'proba_error': []}
    combs = len(data_name) * len(learning_rate) * len(epochs) * len(batch_size) * len(alpha_relu) * len(dropout)
    i = 1
    for data in data_name:
        for lr in learning_rate:
            for ep in epochs:
                for bs in batch_size:
                    for al in alpha_relu:
                        for dr in dropout:
                            print(f'Running combination {i}/{combs}')
                            data_path = f'./datasets/{data}.arff'
                            model_name = f'data_{data}_ep_{ep}_bs_{bs}_lr_{lr}_part2'
                            x_scaled, meta_data_rev, cols, min_max_scaler, feature_num = prepare_architecture(data_path)
                            general_generator = GG(feature_num, saved_models_path, lr, dr, al)
                            x_train, x_test, y_train, y_test = train_test_split(x_scaled[:, :-1], x_scaled[:, -1], test_size=0.1)
                            general_generator.train_gg(x_train, y_train, ep, bs, model_name, data, saved_models_path, True)
                            error = general_generator.get_error()
                            results['dataset'].append(data)
                            results['lr'].append(lr)
                            results['ep'].append(ep)
                            results['bs'].append(bs)
                            results['alpha'].append(al)
                            results['dropout'].append(dr)
                            results['gen_loss'].append(general_generator.losses['gen_loss'][-1])
                            results['proba_error'].append(error.mean())
                            i += 1
                            # Test set performance
                            general_generator.plot_discriminator_results(x_test, y_test, data, saved_models_path)
                            general_generator.plot_generator_results(data, saved_models_path)

    results_output = os.path.join(saved_models_path, f'question_two_results.csv')
    results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv(results_output, index=False)



def main():
    # first_question()
    second_question()


if __name__ == '__main__':
    main()
