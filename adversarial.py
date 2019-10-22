import numpy as np
from network import Network

def main():
    net = Network([3, 2, 1])

    test_x = np.array([[1, 0, 1]]).transpose()
    test_y = np.array([[1]]).transpose()

    # print(test_x)
    # print(test_y)

    net.backpropagation(test_x, test_y)

if __name__ == '__main__':
    main()
