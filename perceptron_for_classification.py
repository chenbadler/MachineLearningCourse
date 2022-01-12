############################################################
# Homework 2
############################################################

student_name = "Chen, Shir"

############################################################
# Imports
############################################################

import homework2_data as data

class BinaryPerceptron(object):
    def __init__(self, examples, iterations):
        self.examples = examples
        self.iterations = iterations
        self.examples_list = []
        self.w = {}
        self.data_length = len(self.examples)

        for x, y in self.examples:
            x_dict = {}
            x_size = 0

            if type(x) is tuple:
                for i, x_i in enumerate(x):
                    x_size +=1
                    if x_i != 0:
                        x_dict[i] = x_i
                self.examples_list.append((x_dict, y))
            else:
                x_size = 1
                x_dict[0] = x
                self.examples_list.append((x_dict, y))
        BinaryPerceptron.w = dict.fromkeys(range(x_size), 0)

        for j in range(self.iterations):
            for x, y_true in self.examples_list:
                y_pred = self.predict(x)
                if y_true != y_pred:
                    if y_true > 0:
                        for key, value in x.items():
                            BinaryPerceptron.w[key] += value
                    else:
                        for key, value in x.items():
                            BinaryPerceptron.w[key] -= value
        pass

    def predict(self, x):
        product = 0
        for key, value in x.items():
            product += (value) * (BinaryPerceptron.w[key])
        if product > 0:
            return True
        else:
            return False
        pass


class MulticlassPerceptron(object):
    x_is_tuple = False
    class_list = []
    def __init__(self, examples, iterations):

        # define class members
        self.all_w = {}  # {class_1: w_1, class_2: w_2, ...}

        # compute w size
        if type(examples[0][0]) is not tuple:
            w_size = 1
        else:
            w_size = len(examples[0][0])
        MulticlassPerceptron.all_w = {y: w_size * [0] for (x, y) in examples}

        for i in range(iterations):
            for example in examples:
                y_true = example[1]
                products_dic = {}  # {class_1: product_1, class_2: product_2, ...}

                # convert w into dic with no '0'
                x_dic = {}
                for i, x_i in enumerate(example[0]):
                    if x_i != 0:
                        x_dic[i] = x_i
                        max_value = -1

                # return the y_max
                for clss, weight in MulticlassPerceptron.all_w.items():
                    products_dic[clss] = 0
                    for index, x_value in x_dic.items():
                        products_dic[clss] += x_value * weight[index]
                    if products_dic[clss] > max_value:
                        max_value = products_dic[clss]
                        y_pred = clss

                # y_pred = self.predict(example[0])
                # check and correct
                if y_true != y_pred:
                    for index, x_value in x_dic.items():
                        MulticlassPerceptron.all_w[y_pred][index] -= x_value
                        MulticlassPerceptron.all_w[y_true][index] += x_value
        #print(MulticlassPerceptron.all_w)
        pass

    def predict(self, x):
        products_dic = {}  # {class_1: product_1, class_2: product_2, ...}
        # convert w into dic with no '0'
        x_dic = {}
        for i, x_i in enumerate(x):
            if x_i != 0:
                x_dic[i] = x_i
                max_value = -1
        # predict
        for clss, weight in MulticlassPerceptron.all_w.items():
            products_dic[clss] = 0
            for index, x_value in x_dic.items():
                products_dic[clss] += x_value * weight[index]
            if products_dic[clss] > max_value:
                max_value = products_dic[clss]
                y_pred = clss
        return y_pred
        pass

############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):
    def __init__(self, data):
        self.Iris_hypothesis = MulticlassPerceptron(data, 47)
        pass

    def classify(self, instance):
        return MulticlassPerceptron.predict(self, instance)
        pass


class DigitClassifier(object):
    def __init__(self, data):
        self.Digit_hypothesis = MulticlassPerceptron(data, 293)
        pass

    def classify(self, instance):
        return MulticlassPerceptron.predict(self, instance)
        pass


class BiasClassifier(object):
    def __init__(self, data):
        self.Bias_X = []
        for example in data:
            self.Bias_X.append(((1, example[0]), example[1]))
        self.Bias_hypothesis = BinaryPerceptron(self.Bias_X, 9)
        pass

    def classify(self, instance):
        new_instance = {}
        new_instance[0] = 1
        new_instance[1] = instance
        return BinaryPerceptron.predict(self, new_instance)
        pass


class MysteryClassifier1(object):
    def __init__(self, data):
        self.Mystery_X = []
        for example in data:
            self.Mystery_X.append(((pow(example[0][0], 2) + pow(example[0][1], 2), 1), example[1]))
        self.Mystery_hypothesis = BinaryPerceptron(self.Mystery_X, 22)
        pass

    def classify(self, instance):
        new_instance = {}
        new_instance[0] = pow(instance[0], 2) + pow(instance[1], 2)
        new_instance[1] = 1
        return BinaryPerceptron.predict(self, new_instance)
        pass


class MysteryClassifier2(object):
    def __init__(self, data):
        #builds the new features vector
        self.Mystery_X = []
        for example in data:
            self.Mystery_X.append(((example[0][0] * example[0][1] * example[0][1]), example[1]))
        self.Mystery_hypothesis = BinaryPerceptron(self.Mystery_X, 1)
        pass

    def classify(self, instance):
        #builds the new instance
        new_instance = {}
        new_instance[0] = instance[0] * instance[1] * instance[2]
        return BinaryPerceptron.predict(self, new_instance)
        pass