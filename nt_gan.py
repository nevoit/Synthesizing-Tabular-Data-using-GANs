import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from numpy.random import randn
from sklearn.decomposition import PCA
from tqdm import tqdm


class GAN(object):
    def __init__(self, number_of_features, saved_models_path, learning_rate, alpha_relu, dropout, loss, activation):
        """
        A constructor for the GAN class
        :param number_of_features: number of features
        :param saved_models_path: the output folder path
        """
        self.saved_models_path = saved_models_path
        self.number_of_features = number_of_features

        self.generator_model = None
        self.noise_dim = None
        self.discriminator_model = None
        self.learning_rate = learning_rate
        self.gan_model = None
        self.activation = activation
        self.alpha_relu = alpha_relu
        self.loss = loss
        self.dropout = dropout
        self.number_of_features = number_of_features

        self.build_generator()  # build the generator
        self.build_discriminator()  # build the discriminator
        self.build_gan()  # build the GAN

    def build_generator(self):
        """
        This function creates the generator model
        :return:
        """
        noise_size = int(self.number_of_features / 2)
        self.noise_dim = (noise_size,)  # size of the noise space

        self.generator_model = Sequential()
        self.generator_model.add(Dense(int(self.number_of_features * 2), input_shape=self.noise_dim))
        self.generator_model.add(LeakyReLU(alpha=self.alpha_relu))

        self.generator_model.add(Dense(int(self.number_of_features * 4)))
        self.generator_model.add(LeakyReLU(alpha=self.alpha_relu))
        self.generator_model.add(Dropout(self.dropout))

        self.generator_model.add(Dense(int(self.number_of_features * 2)))
        self.generator_model.add(LeakyReLU(alpha=self.alpha_relu))
        self.generator_model.add(Dropout(self.dropout))

        # Compile it
        self.generator_model.add(Dense(self.number_of_features, activation=self.activation))
        self.generator_model.summary()

    def build_discriminator(self):
        """
        Create discriminator model
        :return:
        """
        self.discriminator_model = Sequential()

        self.discriminator_model.add(Dense(self.number_of_features * 2, input_shape=(self.number_of_features,)))
        self.discriminator_model.add(LeakyReLU(alpha=self.alpha_relu))

        self.discriminator_model.add(Dense(self.number_of_features * 4))
        self.discriminator_model.add(LeakyReLU(alpha=self.alpha_relu))
        self.discriminator_model.add(Dropout(self.dropout))

        self.discriminator_model.add(Dense(self.number_of_features * 2))
        self.discriminator_model.add(LeakyReLU(alpha=self.alpha_relu))
        self.discriminator_model.add(Dropout(self.dropout))

        # Compile it
        self.discriminator_model.add(Dense(1, activation=self.activation))
        optimizer = Adam(lr=self.learning_rate)
        self.discriminator_model.compile(loss=self.loss, optimizer=optimizer)
        self.discriminator_model.summary()

    def build_gan(self):
        """
        Create the GAN network
        :return: the GAN model object
        """
        self.gan_model = Sequential()
        self.discriminator_model.trainable = False

        # The following lines connect the generator and discriminator models to the GAN.
        self.gan_model.add(self.generator_model)
        self.gan_model.add(self.discriminator_model)

        # Compile it
        optimizer = Adam(lr=self.learning_rate)
        self.gan_model.compile(loss=self.loss, optimizer=optimizer)

        return self.gan_model

    def train(self, scaled_data, epochs, batch_size, to_plot_losses, model_name):
        """
        This function trains the generator and discriminator outputs
        :param model_name:
        :param to_plot_losses: whether or not to plot history
        :param scaled_data: the data after min max scaling
        :param epochs: number of epochs
        :param batch_size: the batch size
        :return: losses_list: returns the losses dictionary the generator or discriminator outputs
        """
        dis_output, gen_output, prev_output = self.check_for_existed_output(model_name)
        if prev_output:
            return -1, -1

        losses_output = os.path.join(self.saved_models_path, f'{model_name}_losses.png')
        discriminator_loss = []
        generator_loss = []

        # We need to use half of the batch size for the fake data and half for the real one
        half_batch_size = int(batch_size / 2)
        iterations = int(len(scaled_data) / half_batch_size)
        iterations = iterations + 1 if len(scaled_data) % batch_size != 0 else iterations

        for epoch in range(1, epochs + 1):  # iterates over the epochs
            np.random.shuffle(scaled_data)
            p_bar = tqdm(range(iterations), ascii=True)
            for iteration in p_bar:
                dis_loss, gen_loss = self.train_models(batch_size=batch_size, half_batch_size=half_batch_size,
                                                       index=iteration, scaled_data=scaled_data)
                discriminator_loss.append(dis_loss)
                generator_loss.append(gen_loss)
                p_bar.set_description(
                    f"Epoch ({epoch}/{epochs}) | DISCRIMINATOR LOSS: {dis_loss:.2f} | GENERATOR LOSS: {gen_loss:.2f} |")

        # Save weights for future use
        self.discriminator_model.save_weights(dis_output)
        self.generator_model.save_weights(gen_output)

        # Plot losses
        if to_plot_losses:
            self.plot_losses(discriminator_loss=discriminator_loss, generator_loss=generator_loss,
                             losses_output=losses_output)

        return generator_loss[-1], discriminator_loss[-1]

    def check_for_existed_output(self, model_name) -> (str, str, bool):
        """
        This function checks for existed output
        :param model_name: model's name
        :return:
        """
        prev_output = False
        dis_output = os.path.join(self.saved_models_path, f'{model_name}_dis_weights.h5')
        gen_output = os.path.join(self.saved_models_path, f'{model_name}_gen_weights.h5')
        if os.path.exists(dis_output) and os.path.exists(gen_output):
            print("The model was trained in the past")
            self.discriminator_model.load_weights(dis_output)
            self.generator_model.load_weights(gen_output)
            prev_output = True
        return dis_output, gen_output, prev_output

    def train_models(self, batch_size, half_batch_size, index, scaled_data):
        """
        This function trains the discriminator and the generator
        :param batch_size: batch size
        :param half_batch_size: half of the batch size
        :param index:
        :param scaled_data:
        :return:
        """
        self.discriminator_model.trainable = True

        # Create a batch of real data and train the model
        x_real, y_real = self.get_real_samples(data=scaled_data, batch_size=half_batch_size, index=index)
        d_real_loss = self.discriminator_model.train_on_batch(x_real, y_real)

        # Create a batch of fake data and train the model
        x_fake, y_fake = self.create_fake_samples(batch_size=half_batch_size)
        d_fake_loss = self.discriminator_model.train_on_batch(x_fake, y_fake)

        avg_dis_loss = 0.5 * (d_real_loss + d_fake_loss)

        # Create noise for the generator model
        noise = randn(self.noise_dim[0] * batch_size).reshape((batch_size, self.noise_dim[0]))

        self.discriminator_model.trainable = False
        gen_loss = self.gan_model.train_on_batch(noise, np.ones((batch_size, 1)))

        return avg_dis_loss, gen_loss

    @staticmethod
    def get_real_samples(data, batch_size, index):
        """
        Generate batch_size of real samples with class labels
        :param data: the original data
        :param batch_size: batch size
        :param index: the index of the batch
        :return: x: real samples, y: labels
        """
        start_index = batch_size * index
        end_index = start_index + batch_size
        x = data[start_index: end_index]

        return x, np.ones((len(x), 1))

    def create_fake_samples(self, batch_size):
        """
        Use the generator to generate n fake examples, with class labels
        :param batch_size: batch size
        :return:
        """
        noise = randn(self.noise_dim[0] * batch_size).reshape((batch_size, self.noise_dim[0]))
        x = self.generator_model.predict(noise)  # create fake samples using the generator

        return x, np.zeros((len(x), 1))

    @staticmethod
    def plot_losses(discriminator_loss, generator_loss, losses_output):
        """
        Plot training loss values
        :param generator_loss:
        :param discriminator_loss:
        :param losses_output:
        :return:
        """
        plt.plot(discriminator_loss)
        plt.plot(generator_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Discriminator and Generator Losses')
        plt.legend(['Discriminator Loss', 'Generator Loss'])
        plt.savefig(losses_output)

    @staticmethod
    def return_minimum_euclidean_distance(scaled_data, x):
        """
        This function returns the
        :param scaled_data: the original data
        :param x: a record we want to compare with
        :return: the minimum distance and the index of the minimum value
        """
        s = np.power(np.power((scaled_data - np.array(x)), 2).sum(1), 0.5)
        return pd.Series([s[s.argmin()], s.argmin()])

    def test(self, scaled_data, sample_num, pca_output):
        """
        This function tests the model
        :param scaled_data: the original scaled data
        :param sample_num: number of samples to generate
        :param pca_output: the output of PCA
        :return:
        """
        x_fake, y_fake = self.create_fake_samples(batch_size=sample_num)
        fake_pred = self.discriminator_model.predict(x_fake)

        # Filter data to different matrices
        dis_fooled_scaled = np.asarray(list(compress(x_fake, fake_pred > 0.5)))
        dis_not_fooled_scaled = np.asarray(list(compress(x_fake, fake_pred <= 0.5)))

        # ------------- Euclidean -------------
        mean_min_distance_fooled, mean_min_distance_not_fooled = (-1, -1)
        if len(dis_fooled_scaled) > 0 and len(dis_not_fooled_scaled) > 0:
            mean_min_distance_fooled = self.get_mean_distance_score(scaled_data, dis_fooled_scaled)
            print(f'The mean minimum distance for fooled samples is {mean_min_distance_fooled}')
            mean_min_distance_not_fooled = self.get_mean_distance_score(scaled_data, dis_not_fooled_scaled)
            print(f'The mean minimum distance for not fooled samples is {mean_min_distance_not_fooled}')
        else:
            print(f'The fooled xor the not Fooled data frames is empty')

        # ------------- PCA --------------
        data_pca_df = self.get_pca_df(scaled_data, 'original')
        dis_fooled_pca_df = self.get_pca_df(dis_fooled_scaled, 'fooled')
        dis_not_fooled_pca_df = self.get_pca_df(dis_not_fooled_scaled, 'not fooled')
        pca_frames = [data_pca_df, dis_fooled_pca_df, dis_not_fooled_pca_df]
        pca_result = pd.concat(pca_frames)
        self.plot_pca(pca_result, pca_output)

        return dis_fooled_scaled, dis_not_fooled_scaled, mean_min_distance_fooled, mean_min_distance_not_fooled

    def get_mean_distance_score(self, scaled_data, dis_scaled):
        """
        This function returns the mean distance score for the given dataframe
        :param scaled_data: the original data
        :param dis_scaled: a dataframe
        :return:
        """
        dis_fooled_scaled_ecu = pd.DataFrame(dis_scaled)
        dis_fooled_scaled_ecu[['min_distance', 'similar_i']] = dis_fooled_scaled_ecu.apply(
            lambda x: self.return_minimum_euclidean_distance(scaled_data, x), axis=1)
        mean_min_distance_fooled = dis_fooled_scaled_ecu['min_distance'].mean()
        return mean_min_distance_fooled

    @staticmethod
    def plot_pca(pca_result, pca_output):
        """
        This function plots the PCA figure
        :param pca_result: dataframe with all the results
        :param pca_output: output path
        :return:
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('PCA With Two Components', fontsize=20)
        targets = ['original', 'fooled', 'not fooled']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets, colors):
            indices_to_keep = pca_result['name'] == target
            ax.scatter(pca_result.loc[indices_to_keep, 'comp1'], pca_result.loc[indices_to_keep, 'comp2'],
                       c=color, s=50)
        ax.legend(targets)
        ax.grid()
        plt.savefig(pca_output)

    @staticmethod
    def get_pca_df(scaled_data, data_name):
        """
        This function creates the PCA dataframe
        :param scaled_data: the original data
        :param data_name: the name of the column
        :return:
        """
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        principal_df = pd.DataFrame(data=principal_components, columns=['comp1', 'comp2'])
        principal_df['name'] = data_name
        return principal_df
