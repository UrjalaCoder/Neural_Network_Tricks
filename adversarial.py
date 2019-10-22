import numpy as np
from network import Network

# def generate_goal_image():
#

def main():
    net = Network([28*28, 16, 10])

    step_size = 200

    arc = Network.load("/".join(['nets', 'network20k.npy']))
    net.weights = arc[0]
    net.biases = arc[1]

    x = np.random.normal(.5, .3, size=(28*28, 1))
    goal = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).transpose()
    # input_g = net.input_gradient(test_x, test_y)
    # print(input_g)
    eta = 2.5
    for i in range(step_size):
        d = net.input_gradient(x, goal)
        guess = net.feed_forward(x)[0][-1]
        print(guess)
        d_vector = np.array(d).reshape((28*28, 1))
        x += eta * d_vector

    result = net.get_guess(x)
    print(result)

if __name__ == '__main__':
    main()
