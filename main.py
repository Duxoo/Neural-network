import neuralNetwork as nn
import numpy as np
import matplotlib.pyplot as mp

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2

n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# обучение
epochs = 2

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        # сначала приводим исходные значения к диапазону 0 - 1, деля их на 255
        # затем * на 0.99 для 0 - 0.99, затем прибавляем 0.01 для 0.01 - 1
        # asfarray - return an array converted to a float 
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создаем целевые значения 0.01, кроме желаемого - ему назначаем 0.99
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        #image_array = np.asfarray(all_values[1:]).reshape((28,28))
        n.train(inputs,targets)


test_data_file = open("dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# тестирование
scoreboard = []
for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    #print(correct_label, "Истинный маркер")
    # сначала приводим исходные значения к диапазону 0 - 1, деля их на 255
    # затем * на 0.99 для 0 - 0.99, затем прибавляем 0.01 для 0.01 - 1
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # опрос сети
    outputs = n.query(inputs)
    # выбор ответа сети
    label = np.argmax(outputs)
    #print(label, "Ответ сети")
    if (label == correct_label):
        scoreboard.append(1)
    else:
        scoreboard.append(0)

scoreboard = np.asarray(scoreboard)
print("Эффективность сети = ", scoreboard.sum() / scoreboard.size)

#mp.imshow(image_array, cmap="Greys", interpolation="None")
#mp.show()
