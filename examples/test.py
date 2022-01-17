import gym

class UpdateRewardWrapper(gym.Wrapper):
    def __init__(self, env, descriminator):
        super().__init__(env)
    
    def step(self, action):
        next_state, reward, done, info =  self.env.step(action)
        reward = 1
        return next_state, reward, done, info

'''env = gym.make('Breakout-v0')
env = UpdateRewardWrapper(env)
state = env.reset()
done = False
while not done:
    action = 0
    next_state, reward, done, info = env.step(action)
    print(reward, done)
    state = next_state'''


from tensorflow.keras import datasets, layers, models, Model
import tensorflow as tf
import tensorflow.keras as keras
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
print(model.summary())

def custom_loss(y_val, y_pred):
    loss=tf.reduce_mean(tf.math.abs((y_val - y_pred)**3))
    return loss
optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(optimizer=optimizer,
              loss=custom_loss,
              metrics=['accuracy'])

class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = layers.Conv2D(chn=32, conv_kernel=(3,3), pool_kernel=(2,2))
        self.conv2 = layers.Conv2D(chn=64, conv_kernel=(3,3), pool_kernel=(2,2))
        self.conv3 = layers.Conv2D(chn=64, conv_kernel=(3,3), isPool=False)
        self.maxpool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dence1 = layers.Dense(64, activation='relu')
        self.dence2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dence1(x)
        x = self.dence2(x)
        return x

    def custom_loss(self, y_val, y_pred):
        loss=tf.reduce_mean(tf.math.abs((y_val - y_pred)**3))
        return loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.custom_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}