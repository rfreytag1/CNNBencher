#!/usr/bin/env python3
from lasagne import layers as l
from lasagne import nonlinearities
from lasagne import init
from lasagne import objectives
from lasagne import updates
from lasagne import regularization

import numpy as np

import theano
import theano.tensor as T

IM_DIM = 1
IM_SIZE = [512,256]
INIT_GAIN = 1.0
DROPOUT = 0.5
NUM_CLASSES = 20

NONLINEARITY = nonlinearities.elu

MULTI_LABEL = False

L2_WEIGHT = 1e-4

def buildModel(mtype=1):


    #default settings (Model 1)
    filters = 64
    first_stride = 2
    last_filter_multiplier = 16

    #specific model type settings (see working notes for details)
    if mtype == 2:
        first_stride = 1
    elif mtype == 3:
        filters = 32
        last_filter_multiplier = 8

    #input layer
    net = l.InputLayer((None, IM_DIM, IM_SIZE[1], IM_SIZE[0]))

    #conv layers
    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters, filter_size=7, pad='same', stride=first_stride, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    if mtype == 2:
        net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters, filter_size=5, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
        net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * 2, filter_size=5, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * 4, filter_size=3, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * 8, filter_size=3, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * last_filter_multiplier, filter_size=3, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    print("\tFINAL POOL OUT SHAPE:", l.get_output_shape(net))

    #dense layers
    net = l.batch_norm(l.DenseLayer(net, 512, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.DropoutLayer(net, DROPOUT)
    net = l.batch_norm(l.DenseLayer(net, 512, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.DropoutLayer(net, DROPOUT)

    #Classification Layer
    if MULTI_LABEL:
        net = l.DenseLayer(net, NUM_CLASSES, nonlinearity=nonlinearities.sigmoid, W=init.HeNormal(gain=1))
    else:
        net = l.DenseLayer(net, NUM_CLASSES, nonlinearity=nonlinearities.softmax, W=init.HeNormal(gain=1))

    print("...DONE!")

    #model stats
    print("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS")
    print("MODEL HAS", l.count_params(net), "PARAMS")

    return net


#################### LOSS FUNCTION ######################
def calc_loss(prediction, targets):
    # categorical crossentropy is the best choice for a multi-class softmax output
    loss = T.mean(objectives.categorical_crossentropy(prediction, targets))
    return loss

def calc_loss_multi(prediction, targets):
    # we need to clip predictions when calculating the log-loss
    prediction = T.clip(prediction, 0.0000001, 0.9999999)
    # binary crossentropy is the best choice for a multi-class sigmoid output
    loss = T.mean(objectives.binary_crossentropy(prediction, targets))
    return loss

def calc_accuracy(prediction, targets):
    # we can use the lasagne objective categorical_accuracy to determine the top1 single label accuracy
    a = T.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
    return a

def calc_accuracy_multi(prediction, targets):
    # we can use the lasagne objective binary_accuracy to determine the multi label accuracy
    a = T.mean(objectives.binary_accuracy(prediction, targets))
    return a

if __name__ == '__main__':
    print("About to build mode...")
    NET = buildModel(1)

    targets = T.matrix('targets', dtype=theano.config.floatX)

    # we use L2 Norm for regularization
    l2_reg = regularization.regularize_layer_params(NET, regularization.l2) * L2_WEIGHT
    # get the network output
    prediction = l.get_output(NET)

    if MULTI_LABEL:
        loss = calc_loss_multi(prediction, targets) + l2_reg
    else:
        loss = calc_loss(prediction, targets) + l2_reg

    # we use dynamic learning rates which change after some epochs
    lr_dynamic = T.scalar(name='learning_rate')

    # get all trainable parameters (weights) of our net
    params = l.get_all_params(NET, trainable=True)

    param_updates = updates.nesterov_momentum(loss, params, learning_rate=lr_dynamic, momentum=0.9)

    #################### TRAIN FUNCTION ######################
    # the theano train functions takes images and class targets as input
    train_net = theano.function([l.get_all_layers(NET)[0].input_var, targets, lr_dynamic], loss, updates=param_updates)

    ################# PREDICTION FUNCTION ####################
    # we need the prediction function to calculate the validation accuracy
    # this way we can test the net during/after training
    net_output = l.get_output(NET, deterministic=True)

    # calculate accuracy
    if MULTI_LABEL:
        accuracy = calc_accuracy_multi(prediction, targets)
    else:
        accuracy = calc_accuracy(prediction, targets)

    test_net = theano.function([l.get_all_layers(NET)[0].input_var, targets], [net_output, loss, accuracy])