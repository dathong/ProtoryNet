import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from datetime import datetime
import operator

class ProtoryNet:

    #create the ProtoryNet:
    # inputs:
    ##+k_cents: the initialized values of prototypes. In the paper, we used KMedoids clustering
    #            to have these values
    ##vect_size: the dimension of the embedded sentence space, if using Google Universal Encoder,
    ##          this value is 512
    ## alpha and beta: the parameters used in the paper, default values are .0001 and .01
    def createModel(self,k_cents,k_protos=10,vect_size=512,alpha=0.0001,beta=0.01):
        loss_tracker = keras.metrics.Mean(name="loss")

        def make_variables(tf_name, k1, k2, initializer):
            return tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32), trainable=True, name=tf_name)

        #prototype layer
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

        #Distance layer: to convert the full distance matrix to sparse similarity matrix
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

        #Customized model
        class CustomModel(keras.Model):

            @tf.function
            def train_step(self, data):
                x, y = data
                def pw_distance(A):
                    r = tf.reduce_sum(A * A, 1)
                    r = tf.reshape(r, [-1, 1])
                    D = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
                    return D

                def tight_pos_sigmoid_offset(x, offset, e=2.7182818284590452353602874713527):
                    return 1 / (1 + tf.math.pow(e, (1 * (offset * x - 0.5))))

                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)  # Forward pass
                    dist, prototypes = self.auxModel(x, training=True)
                    #the second loss term
                    cost2 = tf.reduce_sum(tf.reduce_min(dist, axis=1))

                    d = pw_distance(prototypes)
                    diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
                    diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
                    d1 = d + diag_ones * tf.reduce_max(d)
                    d2 = tf.reduce_min(d1, axis=1)
                    min_d2_dist = tf.reduce_min(d2)
                    # the third loss term
                    cost3 = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8


                    y_val = tf.expand_dims(y[-1], axis=0)
                    loss_object = tf.keras.losses.BinaryCrossentropy()
                    #the final loss function
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

        #building the model
        inputLayer = tf.keras.layers.Input(shape=[], dtype=tf.string)

        l2 = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                            trainable=True)(inputLayer)
        seqEncoder = tf.expand_dims(l2, axis=0)

        tf_init = tf.constant_initializer(k_cents)
        self.proto_layer = prototypeLayer(k_protos, vect_size)
        self.distance_layer = distanceLayer()
        full_distances, protos = self.proto_layer(seqEncoder)
        dist_hot_vect = self.distance_layer(full_distances)

        RNN_CELL_SIZE = 128
        lstmop, forward_h, forward_c = LSTM(RNN_CELL_SIZE, return_sequences=True, return_state=True)(dist_hot_vect)

        z1 = tf.keras.layers.Dense(1, activation='sigmoid')(lstmop[:, -1, :])
        z = tf.squeeze(z1, axis=0)

        model = CustomModel(inputLayer, z)

        for l in model.layers:
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

    #Evalute the model performance on validation set
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

    #Method to train the model
    def train(self,x_train,y_train,x_test,y_test,saveModel = False):
        #We use Adam optimizer with default learning rate 0.0001.
        #Change this value based on your preference
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
                #Evaluate after every 200 iteration
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

                        # automatically save model after getting the best performance
                        if saveModel:
                            self.model.save_weights('my_model.h5')
                        print("just saved")

    #this method simple project prototypes to the closest sentences in
    #sample_sent_vects
    def projection(self,sample_sent_vects,data_size=10000):
        self.prototypes = self.proto_layer.prototypes.numpy()
        d_pos = {}
        #for each prototype
        for p_count, p in enumerate(self.prototypes):
            print('[db] p_count = ', p_count)
            s_count = 0
            d_pos[p_count] = {}
            #find its distances to all sample sentences
            for i, s in enumerate(sample_sent_vects[:data_size]):
                if len(sample_sent_vects[i]) < 5:
                    continue
                d_pos[p_count][i] = np.linalg.norm(sample_sent_vects[i] - p)
                s_count += 1
        #sort those distances, then assign the closest ones to new prototypes
        new_protos = []
        for p_count, p in enumerate(self.prototypes):
            sorted_d = sorted(d_pos[p_count].items(), key=operator.itemgetter(1))
            new_protos.append(sample_sent_vects[sorted_d[0][0]])
        #return these values
        self.prototypes = new_protos
        return new_protos

    #method to manually save the model
    def saveModel(self,name):
        self.model.save_weights(name + ".h5")

    #method to generate the number of closest sentences to each prototype
    def protoFreq(self,sample_sent_vect):
        d = {}
        for sent in sample_sent_vect:
            sent_dist = {}
            for i, p in enumerate(self.prototypes):
                sent_dist[i] = np.linalg.norm(sent - p)
                if i not in d:
                    d[i] = 0
            sorted_sent_d = sorted(sent_dist.items(), key=operator.itemgetter(1))
            # print(sorted_sent_d)
            picked_protos = sorted_sent_d[0][0]
            d[picked_protos] += 1
        print("Prototype freq = ", d)
        x = sorted(d.items(), key=lambda item: item[1], reverse=True)
        print("sorted :",x)

    #re-train the model with new pruned prototype
    def pruningTrain(self,new_k_protos,x_train,y_train,x_test,y_test):
        #print("[db] self prototypes: ",self.prototypes)
        k_cents = self.prototypes[:new_k_protos]
        k_cents = [p.numpy() for p in k_cents]
        #print("[db] k_cents = ",k_cents)
        self.createModel(k_cents=k_cents,k_protos=new_k_protos)
        self.train(x_train,y_train,x_test,y_test)

    # generate the sentence value for each prototype
    # and 10 closest sentences to it
    def prototypeInterpretation(self,sample_sentences,sample_sent_vects,k_protos=10):
        new_protos = self.projection(sample_sent_vects)
        data_size = 10000
        d_pos = {}
        for p_count, p in enumerate(new_protos):
            print('p_count = ', p_count)
            s_count = 0
            d_pos[p_count] = {}
            for i, s in enumerate(sample_sent_vects[:data_size]):
                if len(sample_sentences[i]) < 5:
                    continue
                d_pos[p_count][i] = np.linalg.norm(sample_sent_vects[i] - p)
                s_count += 1
            print('count = ', s_count)

        k_closest_sents = 10
        recorded_protos_score = {}
        for l in range(k_protos):
            print("prototype index = ", l)
            recorded_protos_score[l] = {}
            sorted_d = sorted(d_pos[l].items(), key=operator.itemgetter(1))
            for k in range(k_closest_sents):
                i = sorted_d[k][0]
                print(sorted_d[k], sample_sentences[i])


