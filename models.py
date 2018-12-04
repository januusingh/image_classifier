import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        product_node = self.run(x)
        product = nn.as_scalar(product_node)
        return 1 if product >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        cont = True
        while cont:
            cont = False
            for x, y in dataset.iterate_once(batch_size):
                prediction = self.get_prediction(x)
                y = nn.as_scalar(y)
                if prediction != y:
                    cont = True
                    self.w.update(x, y)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.random_size = 100
        self.w_1 = nn.Parameter(1, self.random_size)
        self.w_2 = nn.Parameter(self.random_size, 1)
        self.b_1 = nn.Parameter(1, self.random_size)
        self.b_2 = nn.Parameter(1, 1)
        self.learning_rate = -0.01


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        ax = nn.Linear(x, self.w_1)
        ax_b = nn.AddBias(ax, self.b_1)
        relu = nn.ReLU(ax_b)
        out_put = nn.Linear(relu, self.w_2)
        predicted_y = nn.AddBias(out_put, self.b_2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        cont = True
        steps = 0
        max_steps = 100000
        while cont:
            cont = False
            for x, y in dataset.iterate_once(self.random_size):
                loss = self.get_loss(x,y)
                prediction = nn.as_scalar(loss)
                if prediction > 0.02 and steps < max_steps:
                    cont = True
                    grad_wrt_w_1, grad_wrt_b_1, grad_wrt_w_2, grad_wrt_b_2 = nn.gradients(loss, [self.w_1, self.b_1, self.w_2, self.b_2])
                    self.w_1.update(grad_wrt_w_1, self.learning_rate)
                    self.b_1.update(grad_wrt_b_1, self.learning_rate)
                    self.w_2.update(grad_wrt_w_2, self.learning_rate)
                    self.b_2.update(grad_wrt_b_2, self.learning_rate)
                    steps += 1

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        self.random_size = 300
        self.random_size_2 = 400
        self.w_1 = nn.Parameter(784, self.random_size)
        self.b_1 = nn.Parameter(1, self.random_size)
        self.w_2 = nn.Parameter(self.random_size, self.random_size_2)
        self.b_2 = nn.Parameter(1, self.random_size_2)
        self.w_3 = nn.Parameter(self.random_size_2, 10)
        self.b_3 = nn.Parameter(1, 10)
        self.learning_rate = -0.23

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        ax = nn.Linear(x, self.w_1)
        ax_b = nn.AddBias(ax, self.b_1)
        relu = nn.ReLU(ax_b)
        out_put = nn.Linear(relu, self.w_2)
        out_put = nn.AddBias(out_put, self.b_2)
        relu2 = nn.ReLU(out_put)
        out_put = nn.Linear(relu2, self.w_3)
        predicted_y = nn.AddBias(out_put, self.b_3)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while dataset.get_validation_accuracy() <= 0.97:
            for x, y in dataset.iterate_once(self.random_size):
                loss = self.get_loss(x,y)
                prediction = nn.as_scalar(loss)
                grad_wrt_w_1, grad_wrt_b_1, grad_wrt_w_2, grad_wrt_b_2, grad_wrt_w_3, grad_wrt_b_3 = nn.gradients(loss, [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3])
                curr_rate = self.learning_rate #* (.9 ** i)
                self.w_1.update(grad_wrt_w_1, curr_rate)
                self.b_1.update(grad_wrt_b_1, curr_rate)
                self.w_2.update(grad_wrt_w_2, curr_rate)
                self.b_2.update(grad_wrt_b_2, curr_rate)
                self.w_3.update(grad_wrt_w_3, curr_rate)
                self.b_3.update(grad_wrt_b_3, curr_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # The dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.w_1 = nn.Parameter(self.num_chars, self.num_chars)
        self.b_1 = nn.Parameter(1, self.num_chars)
        self.w_2 = nn.Parameter(self.num_chars, self.num_chars)
        self.b_2 = nn.Parameter(1, self.num_chars)
        self.w_3 = nn.Parameter(self.num_chars, 5)
        self.b_3 = nn.Parameter(1, len(self.languages))
        self.learning_rate = -0.1

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        xa = nn.Linear(xs[0], self.w_1)
        xa_b = nn.AddBias(xa, self.b_1)
        out_put = nn.ReLU(xa_b)
        for i in range(len(xs)):
            ln = nn.Linear(xs[i], self.w_1)
            lx = nn.AddBias(ln, self.b_1)
            f_init = nn.ReLU(lx)
            out_init = nn.Linear(out_put, self.w_2)
            next = nn.AddBias(out_init, self.b_2)
            f_final = nn.Add(f_init, next)
            out_put = nn.ReLU(f_final)
        out_put = nn.Linear(out_put, self.w_3)
        out_put = nn.AddBias(out_put, self.b_3)
        return out_put

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while dataset.get_validation_accuracy() <= .87:
            for x, y in dataset.iterate_once(20):
                loss = self.get_loss(x, y)
                grad_wrt_w_1, grad_wrt_b_1, grad_wrt_w_2, grad_wrt_b_2, grad_wrt_w_3, grad_wrt_b_3 = nn.gradients(loss, [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3])
                self.w_1.update(grad_wrt_w_1, self.learning_rate)
                self.b_1.update(grad_wrt_b_1, self.learning_rate)
                self.w_2.update(grad_wrt_w_2, self.learning_rate)
                self.b_2.update(grad_wrt_b_2, self.learning_rate)
                self.w_3.update(grad_wrt_w_3, self.learning_rate)
                self.b_3.update(grad_wrt_b_3, self.learning_rate)
