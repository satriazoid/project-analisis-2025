import numpy as np

class SimpleLSTM:
    def __init__(self, input_size=1, hidden_size=16):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(1, hidden_size) * 0.1
        self.by = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        for x in seq:
            x = np.array([[x]])
            concat = np.vstack((h, x))

            f = self.sigmoid(self.Wf @ concat + self.bf)
            i = self.sigmoid(self.Wi @ concat + self.bi)
            o = self.sigmoid(self.Wo @ concat + self.bo)
            c_bar = np.tanh(self.Wc @ concat + self.bc)

            c = f * c + i * c_bar
            h = o * np.tanh(c)

        y = self.Wy @ h + self.by
        return y[0][0]

    def predict_next(self, seq):
        return self.forward(seq)
