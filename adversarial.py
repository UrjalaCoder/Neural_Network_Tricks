import numpy as np
from network import Network
from mnist import MNIST
from data_helper import *
from matplotlib import pyplot as plt



# Prepares the training data
def prepare_training_data(mndata):
    # Load using mndata
    images, labels = mndata.load_training()

    # Combine images and labels to one dataset.
    training_raw_data = combine_data(images, labels)

    # From combined data, form the actual input and output vectors. Store them in an array.
    training_data = preprocess_data(training_raw_data)

    # Return the result
    return training_data

def show(goal_fname, generated_fname, correct_label, wrong_label):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    goal_image = plt.imread(goal_fname, format="png")
    image_plot = plt.imshow(goal_image)
    a.set_title('Goal, correct label {}'.format(correct_label))

    a = fig.add_subplot(1, 2, 2)
    generated_image = plt.imread(generated_fname, format="png")
    image_plot = plt.imshow(generated_image)
    a.set_title('Generated, network thinks {}'.format(wrong_label))

    plt.show(a)

def main():
    mndata = MNIST("./data")
    training_set = prepare_training_data(mndata)

    x_goal, goal = training_set[1]

    # Just some false goal.
    false_goal = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape((10, 1))

    data_goal = x_goal.reshape((28, 28))
    plt.imsave("goal_test.png", data_goal, cmap=plt.cm.gray)


    net = Network([28*28, 16, 10])

    step_size = 100

    arc = Network.load("/".join(['nets', 'network20k.npy']))
    net.weights = arc[0]
    net.biases = arc[1]

    x = np.random.normal(.5, .3, size=(28*28, 1))
    # goal = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).transpose()
    # input_g = net.input_gradient(test_x, test_y)
    # print(input_g)
    eta = 1.7
    lam = 0.03
    for i in range(step_size):
        d = net.input_gradient(x, false_goal)
        d_vector = np.array(d).reshape((28*28, 1))
        x -= eta * (d_vector + lam * (x - x_goal))
    #
    data = x.reshape((28, 28))

    result_data = net.feed_forward(x)[0][-1]

    print(result_data)

    guess = net.get_guess(x)[0]

    print(goal)

    goal_label = -1
    for i in range(len(goal)):
        if goal[i][0] == 1:
            goal_label = i

    plt.imsave("generated_test.png", data, cmap=plt.cm.gray)

    show("goal_test.png", "generated_test.png", goal_label, guess)

if __name__ == '__main__':
    main()
