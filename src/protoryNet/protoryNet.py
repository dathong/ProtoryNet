import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from datetime import datetime

class ProtoryNet:

    def createModel(self,k_cents,k_protos=10,vect_size=512,alpha=0.0001,beta=0.01):
        loss_tracker = keras.metrics.Mean(name="loss")

        def make_variables(tf_name, k1, k2, initializer):
            return tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32), trainable=True, name=tf_name)

        class prototypeLayer(keras.layers.Layer):
            def __init__(self, k_protos, vect_size):
                super(prototypeLayer, self).__init__(name='proto_layer')
                self.n_protos = k_protos
                self.vect_size = vect_size
                self.prototypes = make_variables("prototypes", k_protos, vect_size,
                                                 initializer=tf_init)

            @tf.function
            def call(self, inputs):
                tmp1 = tf.expand_dims(inputs, 2)
                tmp1 = tf.broadcast_to(tmp1, [tf.shape(tmp1)[0], tf.shape(tmp1)[1], self.n_protos, self.vect_size])
                tmp2 = tf.broadcast_to(self.prototypes,
                                       [tf.shape(tmp1)[0], tf.shape(tmp1)[1], self.n_protos, self.vect_size])
                tmp3 = tmp1 - tmp2
                tmp4 = tmp3 * tmp3
                distances = tf.reduce_sum(tmp4, axis=3)
                return distances, self.prototypes

        class distanceLayer(keras.layers.Layer):
            def __init__(self):
                super(distanceLayer, self).__init__(name='distance_layer')
                self.a = 0.1
                self.beta = 1e6

            def e_func(self, x, e=2.7182818284590452353602874713527):
                return tf.math.pow(e, -(self.a * x))

            @tf.function
            def call(self, full_distances):
                min_dist_ind = tf.nn.softmax(-full_distances * self.beta)
                e_dist = self.e_func(full_distances) + 1e-8
                dist_hot_vect = min_dist_ind * e_dist
                return dist_hot_vect

        class CustomModel(keras.Model):

            @tf.function
            def train_step(self, data):
                x, y = data

                def pw_distance(A):
                    r = tf.reduce_sum(A * A, 1)
                    print('[db] r = ', r)
                    r = tf.reshape(r, [-1, 1])
                    D = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
                    return D

                def tight_pos_sigmoid_offset(x, offset, e=2.7182818284590452353602874713527):
                    return 1 / (1 + tf.math.pow(e, (1 * (offset * x - 0.5))))

                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)  # Forward pass
                    dist, prototypes = self.auxModel(x, training=True)
                    print("[db] pro sent dist = ", dist)
                    cost2 = tf.reduce_sum(tf.reduce_min(dist, axis=1))

                    d = pw_distance(prototypes)
                    diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
                    diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
                    d1 = d + diag_ones * tf.reduce_max(d)
                    d2 = tf.reduce_min(d1, axis=1)
                    min_d2_dist = tf.reduce_min(d2)
                    cost3 = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8


                    y_val = tf.expand_dims(y[-1], axis=0)
                    loss_object = tf.keras.losses.BinaryCrossentropy()
                    loss = loss_object(y_val, y_pred) + alpha * cost2 + beta * cost3

                # Compute gradients
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                # Update metrics (includes the metric that tracks the loss)
                self.compiled_metrics.update_state(y, y_pred)
                # Return a dict mapping metric names to current value
                loss_tracker.update_state(loss)

                return {"loss": loss_tracker.result()}

            @property
            def metrics(self):
                return [loss_tracker]

        inputLayer = tf.keras.layers.Input(shape=[], dtype=tf.string)

        l2 = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                            trainable=True)(inputLayer)
        seqEncoder = tf.expand_dims(l2, axis=0)

        tf_init = tf.constant_initializer(k_cents)
        proto_layer = prototypeLayer(k_protos, vect_size)
        distance_layer = distanceLayer()
        full_distances, protos = proto_layer(seqEncoder)
        dist_hot_vect = distance_layer(full_distances)

        RNN_CELL_SIZE = 128
        lstmop, forward_h, forward_c = LSTM(RNN_CELL_SIZE, return_sequences=True, return_state=True)(dist_hot_vect)

        z1 = tf.keras.layers.Dense(1, activation='sigmoid')(lstmop[:, -1, :])
        z = tf.squeeze(z1, axis=0)

        model = CustomModel(inputLayer, z1)

        print("db all layers: ", model.layers)
        for l in model.layers:
            print("[db] l = ", l.name)
            if "proto_layer" in l.name:
                protoLayerName = l.name
            if "distance_layer" in l.name:
                distanceLayerName = l.name

        protoLayer = model.get_layer(protoLayerName)
        distLayer = model.get_layer(distanceLayerName)

        print("[db] model.input = ", model.input)
        print("[db] protoLayerName = ", protoLayerName)
        print("[db] protoLayer = ", protoLayer)
        print("[db] protoLayer.output = ", protoLayer.output)
        print("[db] distanceLayer.output = ", distLayer.output)
        auxModel = keras.Model(inputs=model.input,
                               outputs=protoLayer.output)

        auxModel1 = keras.Model(inputs=model.input,
                                outputs=distLayer.output)

        # auxOutput = auxModel(l1)
        model.auxModel = auxModel
        model.auxModel1 = auxModel1

        model.summary()

        self.model = model
        return model

    def evaluate(self,x_valid, y):
        right, wrong = 0, 0
        count = 0
        y_preds = []
        for x, y in zip(x_valid, y):
            y_pred = self.model.predict(x)
            y_preds.append(y_pred)
            if count % 500 == 0:
                print('Evaluating y_pred, y ', y_pred, round(y_pred[0]), y)
            if round(y_pred[0]) == y:
                right += 1
            else:
                wrong += 1
            count += 1

        return y_preds, right / (right + wrong)

    def train(self,x_train,y_train,x_test,y_test):
        opt = tf.keras.optimizers.Adam(learning_rate=.0001)
        self.model.compile(optimizer=opt)

        i = 0

        maxEvalRes = 0

        for e in range(100):
            print("Epoch ", e)
            for i in range(len(x_train)):
                if i % 50 == 0:
                    print('i =  ', i)
                    self.model.fit(x_train[i],
                              len(x_train[i]) * [y_train[i]],
                              epochs=1, verbose=1,
                              validation_data=None)

                else:
                    self.model.fit(x_train[i],
                              len(x_train[i]) * [y_train[i]],
                              epochs=1, verbose=0,
                              validation_data=None)
                if i % 200 == 0:
                    y_preds, score = self.evaluate(x_test, y_test)
                    print("Evaluate on valid set: ", score)
                    if score > maxEvalRes:
                        maxEvalRes = score
                        print("This is the best eval res, saving the model...")
                        now = datetime.now()

                        print("saving model now =", now)

                        # dd/mm/YY H:M:S
                        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                        print("date and time =", dt_string)
                        # import os

                        # with open(saved_dir + dataset + '/y_preds', 'wb') as fp:
                        #     pickle.dump(y_preds, fp)
                        # model.save_weights(saved_dir + dataset + '/my_model_' + str(k_protos) + '.h5')
                        print("just saved")
