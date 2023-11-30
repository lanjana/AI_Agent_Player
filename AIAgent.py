import keyboard
import time
import copy
import tensorflow as tf
import numpy as np
import random


class agent:
    def __init__(self, game):
        self.game = game
        self.max_iterations = 1000000
        self.learning_rate = 0.2
        self.gamma = 0.95

        self.input_szie = 10

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.criterion = tf.keras.losses.MeanSquaredError()

        self.batch_size = 300

        self.memory = []
        self.distannces = []

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9

        self.build_model()
        # self.model = tf.keras.models.load_model("model.h5")

        while True:
            state_old = self.get_state()
            direction, action = self.act(state_old)
            score = self.game.score
            reward, Done = self.game_next_move(
                direction, self.game.score, self.game.head.distance(self.game.food))
            state_new = self.get_state()
            self.train_short_memory(state_old, action, reward, state_new, Done)
            self.remember(state_old, action, reward, state_new, Done)

            if Done:
                self.train_long_memory()
                if score >= self.game.high_score:
                    self.save_model()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                self.distannces = []
                print(score, self.epsilon)

    def save_model(self):
        self.model.save("model.h5")

    def game_next_move(self, action, score, distance):
        self.distannces.append(distance)
        self.game.set_direction(action)
        self.game.move()
        move = self.game.next_step()

        if not move:
            reward = -10
            return reward, True

        elif self.game.score > score:
            reward = 10
        elif self.game.head.distance(self.game.food) < min(self.distannces):
            reward = 1
        elif self.game.head.distance(self.game.food) == min(self.distannces):
            reward = 0
        else:
            reward = -1

        return reward, False

    def get_state(self):
        x_head, y_head = self.game.head.xcor(), self.game.head.ycor()
        x_food, y_food = self.game.food.xcor(), self.game.food.ycor()
        board = self.game.game_size
        state = np.array(
            [x_head/board+0.5, y_head/board+0.5, x_food/board+0.5, y_food/board+0.5, *self.game.direction, abs(x_head) > (board/2 - 40), abs(y_head) > (board/2 - 40)], dtype=np.float32).reshape(-1, self.input_szie)

        return state

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dense(3, activation="softmax"))

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 3)  # 0 left, 1 straight, 2 right
        else:
            action = self.model.predict(state, verbose=0)[0]
            action = np.argmax(action)

        model_out = [0, 0, 0]
        model_out[action] = 1

        direction_list = [[1, 0, 0, 0], [
            0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        indx = direction_list.index(self.game.direction)

        if action == 0:
            action = direction_list[(indx-1) % 4]
        elif action == 1:
            action = direction_list[(indx) % 4]
        elif action == 2:
            action = direction_list[(indx+1) % 4]

        return action, model_out

    def train(self, states, actions, rewards, next_states, dones):
        if len(states) == 1:
            states = states.reshape(-1, self.input_szie)
            next_states = [next_states.reshape(-1, self.input_szie)]
            actions = np.array(actions, dtype=np.float32).reshape(-1, 3)
            rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
            dones = dones
        else:
            states = states.reshape(-1, self.input_szie)
            next_states = next_states.reshape(-1, self.input_szie)
            actions = np.array(actions, dtype=np.float32).reshape(-1, 3)
            rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
            dones = dones

        # pred = self.model.predict(states, verbose=0)
        # target = tf.identity(pred).numpy()
        target = actions

        for ind in range(len(states)):
            Q_new = rewards[ind]
            if not dones[ind]:
                # Q_new = rewards[ind] + self.gamma * tf.reduce_max(self.model.predict(next_states[ind].reshape(-1, self.input_szie), verbose=0), axis=1)
                # Q_new += self.learning_rate * (
                #     rewards[ind] + self.gamma * tf.reduce_max(self.model.predict(next_states[ind].reshape(-1, self.input_szie), verbose=0), axis=1) - Q_new)
                Q_new = rewards[ind] + self.gamma * np.max(self.model.predict(
                    next_states[ind].reshape(-1, self.input_szie), verbose=0))

            action_idx = tf.argmax(actions[ind]).numpy()
            for i in range(3):
                target[ind, i] += -Q_new[0]/2
            target[ind, action_idx] = Q_new[0]

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)

        # for s, t in zip(states, target):
        #     print(s[:4], t)

        with tf.GradientTape() as tape:
            pred = self.model(states, training=True)
            loss = self.criterion(target, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train(np.array(states),
                   np.array(actions),
                   np.array(rewards),
                   np.array(next_states),
                   np.array(dones))

    def train_short_memory(self, state, action, reward, next_state, done):
        return 0
        self.train(state, action, reward, next_state, (done,))
