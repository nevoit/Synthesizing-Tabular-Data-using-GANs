import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from numpy.random import randn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from tqdm import tqdm


class GG(object):

    def __init__(self, number_of_features, saved_models_path, learning_rate, dropout, alpha):
        """
        The constructor for the General Generator class.
        :param number_of_features: Number of features in the data. Used to determine the noise dimensions
        :param saved_models_path: The folder where we save the models.
        """
        self.saved_models_path = saved_models_path
        self.number_of_features = number_of_features

        self.generator_model = None
        self.discriminator_model = RandomForestClassifier()
        self.dropout = dropout
        self.alpha = alpha
        self.noise_dim = int(number_of_features / 2)
        self.learning_rate = learning_rate
        self.number_of_features = number_of_features
        self.build_generator()  # build the generator.
        self.losses = {'gen_loss': [], 'dis_loss_pred': [], 'dis_loss_proba': []}
        # self.results = {}

    def build_generator(self):
        """
        This function creates the generator model for the GG.
        We used a fairly simple MLP architecture.
        :return:
        """

        self.generator_model = Sequential()
        self.generator_model.add(Dense(int(self.number_of_features * 2), input_shape=(self.noise_dim + 1, )))
        self.generator_model.add(LeakyReLU(alpha=self.alpha))

        self.generator_model.add(Dense(int(self.number_of_features * 4)))
        self.generator_model.add(LeakyReLU(alpha=self.alpha))
        self.generator_model.add(Dropout(self.dropout))

        self.generator_model.add(Dense(int(self.number_of_features * 2)))
        self.generator_model.add(LeakyReLU(alpha=self.alpha))
        self.generator_model.add(Dropout(self.dropout))

        self.generator_model.add(Dense(self.number_of_features, activation='sigmoid'))
        optimizer = Adam(lr=self.learning_rate)
        self.generator_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # self.generator_model.summary()

    def train_gg(self, x_train, y_train, epochs, batch_size, model_name, data, output_path, to_plot=False):
        """
        This function running the training stage manually.
        :param output_path: Path to save loss fig
        :param to_plot: Plots the losses if True
        :param x_train: the training set features
        :param y_train: the training set classes
        :param model_name: name of model to save (for generator)
        :param epochs: number of epochs
        :param batch_size: the batch size
        :return: trains the discriminator and generator.
        """

        losses_path = os.path.join(self.saved_models_path, f'{model_name}_losses')
        model_file = os.path.join(self.saved_models_path, f'{model_name}_part_2_gen_weights.h5')
        
        # First train the discriminator
        self.train_black_box_dis(x_train, y_train)
        self.train_generator(x_train, model_file, epochs, batch_size, losses_path)
        if to_plot:
            self.plot_losses(data, output_path)

    def train_black_box_dis(self, x_train, y_train):
        """
        Trains the discriminator and saves it.
        :param x_train: the training set features
        :param y_train: the training set classes
        :return:
        """
        dis_output = os.path.join(self.saved_models_path, 'black_box_dis_model')

        if os.path.exists(dis_output):
            # print('Blackbox discriminator already trained')
            with open(dis_output, 'rb') as rf_file:
                self.discriminator_model = pickle.load(rf_file)

        self.discriminator_model.fit(x_train, y_train)
        with open(dis_output, 'wb') as rf_file:
            pickle.dump(self.discriminator_model, rf_file)

    def train_generator(self, data, model_path, epochs, start_batch_size, losses_path):
        """
        Function for training the general generator.
        :param losses_path: The filepath for the loss results
        :param data: The normalized dataset
        :param model_path: The name of the model to save. includes epoch size, batches etc.
        :param epochs: Number of epochs
        :param start_batch_size: Size of batch to use.
        :return: trains the generator, saves it and the losses during training.
        """

        if os.path.exists(model_path):
            self.generator_model.load_weights(model_path)
            with open(losses_path, 'rb') as loss_file:
                self.losses = pickle.load(loss_file)
            return

        for epoch in range(epochs):  # iterates over the epochs
            np.random.shuffle(data)
            batch_size = start_batch_size
            for i in tqdm(range(0, data.shape[0], batch_size), ascii=True):  # Iterate over batches
                if data.shape[0] - i >= batch_size:
                    batch_input = data[i:i + batch_size]
                else:  # The last iteration
                    batch_input = data[i:]
                    batch_size = batch_input.shape[0]

                g_loss = self.train_generator_on_batch(batch_input)
                self.losses['gen_loss'].append(g_loss)

        self.save_generator_model(model_path, losses_path)

    def save_generator_model(self, generator_model_path, losses_path):
        """
        Saves the model and the loss data with pickle.

        :param generator_model_path: File path for the generator
        :param losses_path: File path for the losses
        :return:
        """
        self.generator_model.save_weights(generator_model_path)
        with open(losses_path, 'wb+') as loss_file:
            pickle.dump(self.losses, loss_file)

    def train_generator_on_batch(self, batch_input):
        """
        Trains the generator for a single batch. Creates the necessary input, comprised of noise and the real
        probabilities obtained from the black box. Compared to the target output, made of real samples and the
        probabilities made up by the generator.
        :param batch_input:
        :return:
        """
        batch_size = batch_input.shape[0]
        discriminator_probabilities = self.discriminator_model.predict_proba(batch_input)[:, -1:]
        # noise = randn(self.noise_dim * batch_size).reshape((batch_size, self.noise_dim))

        noise = randn(batch_size, self.noise_dim)
        gen_model_input = np.hstack([noise, discriminator_probabilities])
        generated_probabilities = self.generator_model.predict(gen_model_input)[:, -1:]  # Take only probabilities
        target_output = np.hstack([batch_input, generated_probabilities])
        g_loss = self.generator_model.train_on_batch(gen_model_input, target_output)  # The actual training

        return g_loss

    def plot_discriminator_results(self, x_test, y_test, data, path):
        """
        :param x_test: Test set
        :param y_test: Test classes
        :return: Prints the required plots.
        """

        blackbox_probs = self.discriminator_model.predict_proba(x_test)
        discriminator_predictions = self.discriminator_model.predict(x_test)
        count_1 = int(np.sum(y_test))
        count_0 = int(y_test.shape[0] - count_1)
        class_data = (['Class 0', 'Class 1'], [count_0, count_1])
        self.plot_data(class_data, path, mode='bar', x_title='Class', title=f'Distribution of classes - {data} dataset')
        self.plot_data(blackbox_probs[:, 0], path, title=f'Probabilities for test set - class 0 - {data} dataset')
        self.plot_data(blackbox_probs[:, 1], path, title=f'Probabilities for test set - class 1 - {data} dataset')

        min_confidence = blackbox_probs[:, 0].min(), blackbox_probs[:, 1].min()
        max_confidence = blackbox_probs[:, 0].max(), blackbox_probs[:, 1].max()
        mean_confidence = blackbox_probs[:, 0].mean(), blackbox_probs[:, 1].mean()

        print("Accuracy:", metrics.accuracy_score(y_test, discriminator_predictions))
        for c in [0, 1]:
            print(f'Class {c} - Min confidence: {min_confidence[c]} - Max Confidence: {max_confidence[c]} - '
                  f'Mean confidence: {mean_confidence[c]}')

    def plot_generator_results(self, data, path, num_of_instances=1000):
        """
        Creates plots for the generator results on 1000 instances.
        :param path:
        :param data: Name of dataset used.
        :param num_of_instances: Number of samples to generate.
        :return:
        """
        sampled_proba, generated_instances = self.generate_n_samples(num_of_instances)

        proba_fake = self.discriminator_model.predict_proba(generated_instances[:, :-1])
        for c in [0, 1]:
            title = f'Confidence Score for Class {c} of Fake Samples - {data} dataset'
            self.plot_data(proba_fake[:, c], path, x_title='Confidence Score', title=title)

        black_box_confidence = proba_fake[:, 1:]
        proba_error = np.abs(sampled_proba - black_box_confidence)
        generated_classes = np.array([int(round(c)) for c in generated_instances[:, -1].tolist()]).reshape(1000, 1)
        proba_stats = np.hstack([sampled_proba, generated_classes, proba_fake[:, :1], proba_fake[:, 1:], proba_error])

        for c in [0, 1]:
            class_data = proba_stats[proba_stats[:, 1] == c]
            class_data = class_data[class_data[:, 0].argsort()]  # Sort it for the plot
            title = f'Error rate for different probabilities, class {c} - {data} dataset'
            self.plot_data((class_data[:, 0], class_data[:, -1]), path, mode='plot', y_title='error rate', title=title)

    def generate_n_samples(self, n):
        """
        Functions for generating N samples with a uniformly distribution confidence level.
        :param n: Number of samples
        :return: a tuple of the confidence scores used and the samples created.
        """
        noise = randn(n, self.noise_dim)
        # confidences = np.sort(np.random.uniform(0, 1, (n, 1)), axis=0)
        confidences = np.random.uniform(0, 1, (n, 1))

        generator_input = np.hstack([noise, confidences])  # Stick them together
        generated_instances = self.generator_model.predict(generator_input)  # Create samples

        return confidences, generated_instances

    @staticmethod
    def plot_data(data, path, mode='hist', x_title='Probabilities', y_title='# of Instances', title='Distribution'):
        """
        :param path: Path to save
        :param mode: Mode to use
        :param y_title: Title of y axis
        :param x_title: Title of x axis
        :param data: Data to plot
        :param title: Title of plot
        :return: Prints a plot
        """
        plt.clf()

        if mode == 'hist':
            plt.hist(data)
        elif mode == 'bar':
            plt.bar(data[0], data[1])
        else:
            plt.plot(data[0], data[1])

        plt.title(title)
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        # plt.show()
        path = os.path.join(path, title)
        plt.savefig(path)

    def plot_losses(self, data, path):
        """
        Plot the losses while training
        :return:
        """
        plt.clf()
        plt.plot(self.losses['gen_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        # plt.show()
        plt.savefig(os.path.join(path, f'{data} dataset - general_generator_loss.png'))

    def get_error(self, num_of_instances=1000):
        """
        Calculates the error of the generator we created by measuring the difference between the probability that
        was given as input and the probability of the discriminator on the sample created.
        :param num_of_instances: Number of samples to generate.
        :return: An array of errors.
        """
        sampled_proba, generated_instances = self.generate_n_samples(num_of_instances)
        proba_fake = self.discriminator_model.predict_proba(generated_instances[:, :-1])
        black_box_confidence = proba_fake[:, 1:]
        return np.abs(sampled_proba - black_box_confidence)

