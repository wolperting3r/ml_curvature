import tensorflow as tf


class Feedforward(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.unints),
            initializer='he_normal',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='he_normal',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias


class MLP(tf.keras.Model):
    def __init__(self, hidden_dimension):
        super().__init__()
        self.network = tf.keras.Sequential(
            [Feedforward(100), tf.keras.layers.ReLU(), Feedforward(1)]
        )

        def call(self, inputs):
            flattend = tf.keras.layers.Flatten()(inputs)
            return self.network(flattend)


    def call(self, input_features):
        hidden_output = self.hidden_layer(input_features)
        output = self.output(hidden_output)
        return output


def train(
    model: tf.keras.Model,
    path: str,
    train: tf.data.Dataset,
    epochs: int,
    steps_per_epoch: int,
    validation: tf.data.Dataset,
    steps_per_validation: int,
    stopping_epochs: int,
    optimizer=tf.optimizers.Adam()
):
    model.compile(
        optimizer=optimizer,
        loss=

def loss(model, input_values, input_labels):
    error = tf.reduce_mean(tf.square(tf.substract(model(input_values), input_labels)))
    return error


def train(loss, model, opt, input_values, input_labels):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, input_values, input_labels), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


def learn(train_data, train_labels, parameters):
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    hidden_dimension = parameters['layers']

    train_data = train_data.astype('float32')

    train_labels = train_labels.astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(train_data.shape[0])
    train_dataset = train_dataset.prefetch(batch_size * 4)

    mlp = MLP(hidden_dimension=hidden_dimension)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    writer = tf.summary.create_file_writer('tmp')

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                print(f'epoch:\n{epoch}')
                for step, batch_features in enumerate(training_dataset):
                    train(loss, model=mlp, opt, 
