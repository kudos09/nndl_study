import network
import network2
#import network3

import mnist_loader

training_data, validation_data, test_data =mnist_loader.load_data_wrapper()

#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)
net.SGD(training_data, 10, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, lmbda = 0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,monitor_training_cost=True, monitor_training_accuracy=True)

net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True)

net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True)

net.SGD(training_data, 30, 10, 0.1, lmbda=1000.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True)

net.SGD(training_data[:1000], 30, 10, 10.0, lmbda = 1000.0, evaluation_data=validation_data[:100], monitor_evaluation_accuracy=True)

net.SGD(training_data[:1000], 30, 10, 10.0, lmbda = 20.0, evaluation_data=validation_data[:100], monitor_evaluation_accuracy=True)

net.SGD(training_data[:1000], 30, 10, 100.0, lmbda = 20.0, evaluation_data=validation_data[:100], monitor_evaluation_accuracy=True)

net.SGD(training_data[:1000], 30, 10, 1.0, lmbda = 20.0, evaluation_data=validation_data[:100], monitor_evaluation_accuracy=True)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data,monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)



net = network2.Network([784, 30, 30, 10])
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

net = network2.Network([784, 30, 30, 30, 10])
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

net = network2.Network([784, 30, 30, 30, 30, 10])
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)



