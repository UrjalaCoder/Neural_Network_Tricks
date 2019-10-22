import numpy as np
import random, math

# Network code
class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = np.array([np.random.randn(i, 1) for i in self.sizes[1:]])
        self.weights = np.array([np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])])

    # Returns the activations and z-values in two matrices -->
    def feed_forward(self, inputs):
        activations = [inputs]
        zs = []
        layer = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[layer]) + b
            zs.append(z)
            activations.append(self.sigmoid(z))
            layer = layer + 1
        return (np.array(activations), np.array(zs))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def cost_function(self, x, y):
        return 0.5 *math.pow(np.linalg.norm(x - y), 2)

    def cost_prime(self, a, y):
        return (a - y)

    def sigmoid_prime(self, z):
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))

    def input_gradient_helper(self, G):
        input_gradient = []

        for out_n in range(len(self.sizes[-1])):
            


    # 'x' refers to the input of the training sample.
    # 'y' refers to the correct output of the sample.
    def backpropagation(self, x, y):
        # Get the network guess.
        # print(x)
        # print(y)
        activations, zs = self.feed_forward(x)
        # Initializing the deltas list.
        deltas = [np.zeros((i, 1)) for i in self.sizes]
        partial_derivative_bias = [np.zeros(self.biases[i].shape) for i in range(len(self.biases))]
        partial_derivative_weights = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]

        partial_derivative_inputs = [np.zeros(self.sizes[i]) for i in range(len(self.sizes))]

        print(self.weights)

        # Calculate the partial derivative wrt the inputs.
        print(partial_derivative_inputs)
        gradients = []
        print("zs:")
        for l in range(len(zs) - 1, -1, -1):
            layer_gradients = []
            print("Layer: {}".format(l))
            for i in range(len(zs[l])):
                z = zs[l][i]
                weight = self.weights[l]
                activation_prime = self.sigmoid_prime(z)
                gradient = np.zeros(self.sizes[l])
                # print(gradient)

                # Wrt each input neuron
                for n in range(self.sizes[l]):
                    gradient[n] = activation_prime * self.weights[l][i][n]

                layer_gradients.append(gradient)
            gradients.append(layer_gradients)

        # gradients[l][i][n] is the partial derivative of the activation of a^l_i wrt a^(l-1)_n
        print(gradients)

        input_gradient = self.input_gradient_helper(gradients)


        # Final layer error -->
        deltas[-1] = self.cost_prime(activations[-1], y) * self.sigmoid_prime(zs[-1])
        partial_derivative_bias[-1] = deltas[-1]
        partial_derivative_weights[-1] = np.dot(deltas[-1], activations[-2].transpose())

        for l in range(2, len(self.sizes)):
            # Delta vector.
            d = np.dot(np.transpose(self.weights[-l+1]), deltas[-l+1]) * self.sigmoid_prime(zs[-l])
            deltas[-l] = d

            # Use delta to update 'partial_derivative_bias' and 'partial_derivative_weights'.
            partial_derivative_bias[-l] = d
            partial_derivative_weights[-l] = np.dot(d, activations[-l-1].transpose())

        return (partial_derivative_weights, partial_derivative_bias)

    def SGD(self, tr, mini_batch):
        total_cost = 0
        nabla_biases = np.array([np.zeros(self.biases[i].shape) for i in range(len(self.biases))])
        nabla_weights = np.array([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        for sample in mini_batch:
            x, y = sample
            guess = self.feed_forward(x)[0][-1]
            delta_nablas = self.backpropagation(x, y)
            nabla_weights = nabla_weights + delta_nablas[0]
            nabla_biases = nabla_biases + delta_nablas[1]
            total_cost = total_cost + self.cost_function(guess, y)
        self.weights = self.weights - (tr/len(mini_batch)) * nabla_weights
        self.biases = self.biases - (tr/len(mini_batch)) * nabla_biases

    # Main training function -->
    def train(self, data, cutoff, tr, batch_size):
        """
        data should be formatted like this:
            [0] = Pixel data 0-1 in a np column vector, shape(n, 1)
            [1] = Correct output vector in a column vector, shape(m, 1)
            n = 28 * 28
            m = 10
        """
        counter = 0
        while counter < cutoff:
            counter = counter + 1
            batch = self.get_mini_batch(data, batch_size)
            self.SGD(tr, batch)
            print("Iteration: " + str(counter))

    def get_mini_batch(self, data, size):
        mini_batch = []
        for i in range(size):
            index = random.randrange(0, len(data))
            mini_batch.append(data[index])
        return mini_batch

    def get_guess(self, input):
        output_vector = self.feed_forward(input)[0][-1]
        max = 0
        for index in range(len(output_vector)):
            if output_vector[index][0] > output_vector[max][0]:
                max = index
        return (max, output_vector[max])


    """
    Helper functions for storing and loading the network.
    """

    @staticmethod
    def store(path, net):
        """
        Stores the network in a file.
        'path' : Path of the file.
        'net': Tuple (w, b) where w is the network's weights and b the network's biases.
        """
        w, b = net
        np.save(path, np.array([w, b]))
    @staticmethod
    def load(filename):
        data = np.load(filename)
        return data
