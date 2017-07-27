#!/usr/bin/env python3

from datetime import datetime
import time
import logging
import json
import queue
import threading
import os
import sys


import cv2
import numpy as np


from lasagne import layers as l

import CNNBenchUtils.DynamicValues.ValueTypes as dvt

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as cnnbp

import CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder as cnnb

import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTrainingFunctionBuilder as cnntrf
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTestFunctionBuilder as cnntef

BATCH_SIZE=128
MULTI_LABEL=True
TRAIN=[]
'''
def openImage(path, useCache=USE_CACHE):

    global CACHE

    #using a dict {path:image} cache saves some time after first epoch
    #but may consume a lot of RAM
    if path in CACHE:
        return CACHE[path]
    else:

        #open image
        img = cv2.imread(path)

        #DEBUG
        try:
            h, w = img.shape[:2]
        except:
            print "IMAGE NONE-TYPE:", path

        #original image dimensions
        try:
            h, w, d = img.shape

            #to gray?
            if IM_DIM == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        except:
            h, w = img.shape

            #to color?
            if IM_DIM == 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        #resize to conv input size
        img = cv2.resize(img, (IM_SIZE[0], IM_SIZE[1]))

        #convert to floats between 0 and 1
        img = np.asarray(img / 255., dtype='float32')  
        
        if useCache:
            CACHE[path] = img
        return img

def imageAugmentation(img):

    AUG = IM_AUGMENTATION

    #Random Crop (without padding)
    if 'crop' in AUG and RANDOM.choice([True, False], p=[AUG['crop'][0], 1 - AUG['crop'][0]]):
        h, w = img.shape[:2]
        cropw = RANDOM.randint(1, int(float(w) * AUG['crop'][1]))
        croph = RANDOM.randint(1, int(float(h) * AUG['crop'][1]))
        img = img[croph:-croph, cropw:-cropw]
        img = cv2.resize(img, (IM_SIZE[0], IM_SIZE[1]))

    #Flip - 1 = Horizontal, 0 = Vertical
    if 'flip' in AUG and RANDOM.choice([True, False], p=[AUG['flip'][0], 1 - AUG['flip'][0]]):    
        img = cv2.flip(img, AUG['flip'][1])

    #Wrap shift (roll up/down and left/right)
    if 'roll' in AUG and RANDOM.choice([True, False], p=[AUG['roll'][0], 1 - AUG['roll'][0]]):
        img = np.roll(img, int(img.shape[0] * (RANDOM.uniform(-AUG['roll'][1][1], AUG['roll'][1][1]))), axis=0)
        img = np.roll(img, int(img.shape[1] * (RANDOM.uniform(-AUG['roll'][1][0], AUG['roll'][1][0]))), axis=1)

    #substract/add mean
    if 'mean' in AUG and RANDOM.choice([True, False], p=[AUG['mean'][0], 1 - AUG['mean'][0]]):   
        img += np.mean(img) * AUG['mean'][1]

    #gaussian noise
    if 'noise' in AUG and RANDOM.choice([True, False], p=[AUG['noise'][0], 1 - AUG['noise'][0]]):
        img += RANDOM.normal(0.0, RANDOM.uniform(0, AUG['noise'][1]**0.5), img.shape)
        img = np.clip(img, 0.0, 1.0)

    #add noise samples
    if 'noise_samples' in AUG and RANDOM.choice([True, False], p=[AUG['noise_samples'][0], 1 - AUG['noise_samples'][0]]):
        img += openImage(NOISE[RANDOM.choice(range(0, len(NOISE)))], True) * AUG['noise_samples'][1]
        img -= img.min(axis=None)
        img /= img.max(axis=None)

    #adjust brightness
    if 'brightness' in AUG and RANDOM.choice([True, False], p=[AUG['brightness'][0], 1 - AUG['brightness'][0]]):
        img *= RANDOM.uniform(AUG['brightness'][1][0], AUG['brightness'][1][1])
        img = np.clip(img, 0.0, 1.0)
        
    #show
    #cv2.imshow("AUG", img)#.reshape(IM_SIZE[1], IM_SIZE[0], IM_DIM))
    #cv2.waitKey(-1)

    return img
    
def loadImageAndTarget(path, doAugmentation=True):

    #here we open the image
    img = openImage(path)

    #image augmentation?
    if IM_AUGMENTATION != None and doAugmentation:
        img = imageAugmentation(img)
    
    #we want to use subfolders as class labels
    label = path.split("/")[-2]

    #we need to get the index of our label from CLASSES
    index = CLASSES.index(label)

    #allocate array for target
    target = np.zeros((NUM_CLASSES), dtype='float32')

    #we set our target array = 1.0 at our label index, all other entries remain 0.0
    target[index] = 1.0
    
    #transpose image if dim=3
    try:
        img = np.transpose(img, (2, 0, 1))
    except:
        pass

    #we need a 4D-vector for our image and a 2D-vector for our targets
    img = img.reshape(-1, IM_DIM, IM_SIZE[1], IM_SIZE[0])
    target = target.reshape(-1, NUM_CLASSES)

    return img, target

def getSameClassAugmentation(x, y):

    #are there some samples with the same class label?
    scl = np.where(np.sum(y, axis=0) > 1)[0]
    acnt = 0
    while scl.shape[0] > 0 and acnt < MAX_SAME_CLASS_COMBINATIONS:

        #randomly chosen class
        c = RANDOM.choice(scl)

        #get all samples of this selected class
        s = []
        for i in range(0, y.shape[0]):
            if y[i][c] == 1:
              s.append(i)

        #combine first two samples
        x[s[0]] += x[s[1]]

        #re-normalize new image
        x[s[0]] -= x[s[0]].min(axis=None)
        x[s[0]] /= x[s[0]].max(axis=None)

        #remove augmented class
        scl = np.delete(scl, np.where(scl == c))
        acnt += 1

        #show
        #print scl, acnt, c
        #cv2.imshow("BA", x[s[0]].reshape(IM_SIZE[1], IM_SIZE[0], IM_DIM))
        #cv2.waitKey(-1)

    return x, y        

def getAugmentedBatches(x, y):

    #augment batch until desired number of target labels per image is reached
    while np.mean(np.sum(y, axis=1)) < MEAN_TARGETS_PER_IMAGE:

        #get two images to combine (we try to prevent i == j (which could result in infinite loops) with excluding ranges)
        i = RANDOM.choice(range(1, x.shape[0] - 1))
        j = RANDOM.choice(range(0, i) + range(i + 1, x.shape[0]))

        #add images
        x[i] += x[j]

        #re-normalize new image
        x[i] -= x[i].min(axis=None)
        x[i] /= x[i].max(axis=None)

        #combine targets (makes this task a multi-label classification!)
        y[i] = np.logical_or(y[i], y[j])

        #TODO: We still might end up in an infinite loop
        #and should add a break in case something is fishy

        #show
        #cv2.imshow("BA", x[i].reshape(IM_SIZE[1], IM_SIZE[0], IM_DIM))
        #cv2.waitKey(-1)
    
    return x, y
    
def getDatasetChunk(split):

    #get batch-sized chunks of image paths
    for i in xrange(0, len(split), BATCH_SIZE):
        yield split[i:i+BATCH_SIZE]

def getNextImageBatch(split=TRAIN, doAugmentation=True, batchAugmentation=MULTI_LABEL): 

    #fill batch
    for chunk in getDatasetChunk(split):

        #allocate numpy arrays for image data and targets
        x_b = np.zeros((BATCH_SIZE, IM_DIM, IM_SIZE[1], IM_SIZE[0]), dtype='float32')
        y_b = np.zeros((BATCH_SIZE, NUM_CLASSES), dtype='float32')
        
        ib = 0
        for path in chunk:

            try:
            
                #load image data and class label from path
                x, y = loadImageAndTarget(path, doAugmentation)

                #pack into batch array
                x_b[ib] = x
                y_b[ib] = y
                ib += 1

            except:
                continue

        #trim to actual size
        x_b = x_b[:ib]
        y_b = y_b[:ib]

        #same class augmentation?
        if doAugmentation and SAME_CLASS_AUGMENTATION and x_b.shape[0] > 2:
            x_b, y_b = getSameClassAugmentation(x_b, y_b)

        #batch augmentation?
        if batchAugmentation and x_b.shape[0] >= BATCH_SIZE // 2:
            x_b, y_b = getAugmentedBatches(x_b, y_b)

        #instead of return, we use yield
        yield x_b, y_b
'''


# Loading images with CPU background threads during GPU forward passes saves a lot of time
# Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)
def threaded_batch_generator(generator, num_cached=10):
    import queue
    bqueue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            bqueue.put(item)
        bqueue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = bqueue.get()
    while item is not sentinel:
        yield item
        bqueue.task_done()
        item = bqueue.get()


class FileLoader:
    def __init__(self):
        pass


class CachedFileLoader(FileLoader):
    file_cache = {}
    
    def __init__(self):
        super(CachedFileLoader, self).__init__()
    

class CachedImageLoader(CachedFileLoader):
    def __init__(self):
        super(CachedImageLoader, self).__init__()

    def load(self, filename):
        IM_SIZE = [512, 256]
        # using a dict {path:image} cache saves some time after first epoch
        # but may consume a lot of RAM
        if filename in CachedImageLoader.file_cache:
            return CachedImageLoader.file_cache[filename]

        # open image
        try:
            img = cv2.imread(filename)
        except:
            print("File \"", filename, "\" is not an image!")

        if img is None:
            return None

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize to conv input size
        img = cv2.resize(img, (512, 256))

        # convert to floats between 0 and 1
        img = np.asarray(img / 255., dtype='float32')

        CachedImageLoader.file_cache[filename] = img
        return img
        

class ThreadedBatchGenerator:
    queue_end = object()

    def __init__(self, dataset=None, batch_size=128):
        self.dataset = dataset
        self.file_queue = queue.Queue()
        self.batch_size = batch_size
        self.classes_maxnum = 20
        self.classes_count = 20
        self.classes = []
        self.image_dim = [512, 256, 3]
        self.cache_size = 20

    def getDatasetChunk(self, split):
        # get batch-sized chunks of image paths
        for i in range(0, len(split), self.batch_size):
            yield split[i:i + self.batch_size]

    def openImage(self, path):
        return np.ndarray([2,3], np.float32)

    def loadImageAndTarget(self, path):

        # here we open the image
        img = self.openImage(path)

        # image augmentation?

        # we want to use subfolders as class labels
        label = path.split("/")[-1]

        # we need to get the index of our label from CLASSES
        index = self.classes.index(label)

        # allocate array for target
        target = np.zeros(self.classes_maxnum, dtype='float32')

        # we set our target array = 1.0 at our label index, all other entries remain 0.0
        target[index] = 1.0

        # transpose image if dim=3
        try:
            img = np.transpose(img, (2, 0, 1))
        except:
            pass

        # we need a 4D-vector for our image and a 2D-vector for our targets
        img = img.reshape(-1, self.image_dim[2], self.image_dim[1], self.image_dim[0])
        target = target.reshape(-1, self.classes_maxnum)

        return img, target

    def getNextImageBatch(self, ):

        # fill batch
        for chunk in self.getDatasetChunk(split):

            # allocate numpy arrays for image data and targets
            image_batch_buffer = np.zeros((self.batch_size, self.image_dim[2], self.image_dim[1], self.image_dim[0]), dtype='float32')
            target_batch_buffer = np.zeros((self.batch_size, self.classes_maxnum), dtype='float32')

            loaded = 0
            for path in chunk:

                try:
                    # load image data and class label from path
                    x, y = self.loadImageAndTarget(path)

                    # pack into batch array
                    image_batch_buffer[loaded] = x
                    target_batch_buffer[loaded] = y
                    loaded += 1
                except:
                    continue

            # trim to actual size
            image_batch_buffer = image_batch_buffer[:loaded]
            target_batch_buffer = target_batch_buffer[:loaded]

            # instead of return, we use yield
            yield image_batch_buffer, target_batch_buffer

    def create_batch(self):
        pass
        '''
        for batched_paths in
        image_batch = np.zeros((self.batch_size, self.image_dim[2], self.image_dim[1], self.image_dim[0]), dtype='float32')
        targets_batch = np.zeros((self.batch_size, self.classes_count))
        '''

    def batch(self):
        import queue
        bqueue = queue.Queue(maxsize=self.cache_size)
        sentinel = object()  # guaranteed unique reference

        # define producer (putting items into queue)
        def producer():
            for item in self.getNextImageBatch():
                bqueue.put(item)
            bqueue.put(sentinel)

        # start producer (in a background thread)
        import threading
        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        # run as consumer (read items from queue, in current thread)
        item = bqueue.get()
        while item is not sentinel:
            yield item
            bqueue.task_done()
            item = bqueue.get()


class BaseDataset:
    def __init__(self, datasetpath=None):
        self.loaded = False
        self.datasetpath = datasetpath

    def is_loaded(self):
        return self.loaded

    def list(self, subpath=None):
        pass

    def load(self, datasetpath):
        self.datasetpath = datasetpath
        return True

    def open(self, filename):
        pass

    def close(self):
        pass


class DatasetDirectory(BaseDataset):
    def __init__(self, datasetpath=None):
        super(DatasetDirectory, self).__init__(datasetpath)
        self.loaded = True

    def list(self, subpath=None):
        dir_list = None
        if subpath is None:
            dir_list = os.listdir(self.datasetpath)
        else:
            dir_list = os.listdir(os.path.join(self.datasetpath, subpath))

        return dir_list

    def open(self, filename):
        fobj = open(os.path.join(self.datasetpath, filename))
        return fobj



imgloader = CachedImageLoader()
imgloader.load('test.jpg')

dsd = DatasetDirectory('./')

print(dsd.list('CNNBenchUtils'))

sys.exit()

stages = 10
runs = 5

default_log = logging.getLogger("CNNBencherDefault")
lfh = logging.FileHandler("./test_h.log")

# lfm = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%{name)s: %{module}s > %{funcName}s @ %{lineno}d', '%Y-%m-%dT%H:%M:%S%z')
lfm = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%(name)s: %(module)s > %(funcName)s @ %(lineno)d)', '%Y-%m-%dT%H:%M:%S%z')

lfh.setFormatter(lfm)
lfh.setLevel(logging.DEBUG)

default_log.addHandler(lfh)

default_log.warning("Tis' just a test!", {'lineno': 2})
bdp = cnnbp.BenchDescriptionJSONParser(True)
bench_desc = bdp.parse("./sample_cnn_bench1.json")


print("parsed", len(bench_desc['cnns']), "CNNs")
for cnn, cnnc in bench_desc['cnns'].items():
    print("Layers:", len(cnnc['layers']))
    print("Dynamic Values:", len(cnnc['selector'].dynamic_values))
    netbuilder = cnnb.LasagneCNNBuilder(cnnc)
    # net = netbuilder.build(cnnc)

    tensors = {}
    train_func_builder = cnntrf.LasagneTrainingFunctionBuilder(None, cnnc['training']['function'])
    test_func_builder = cnntef.LasagneTestFunctionBuilder(None, cnnc['training']['function'])

    for stage in range(0, bench_desc['stages']):
        print("Stage", stage+1, "of", bench_desc['stages'])
        cnnc['selector'].select_dvals(stage)

        epochs = cnnc['training']['params']['epochs'].value(stage)

        net = netbuilder.build(stage=stage)
        tensors.clear() # very important or else the functions will build with the wrong tensors(e.g. from previous builds)
        train_func = train_func_builder.build(net, tensors, stage=stage)
        test_func = test_func_builder.build(net, tensors, stage=stage)

        for run in range(0, bench_desc['runs']):
            print("Run", run+1, "of", bench_desc['runs'])
            lr_interp = cnnc['training']['function']['params']['learning_rate.interp'].value(stage)
            learning_rate = None
            if lr_interp != 'none':
                lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                lr_end = float(cnnc['training']['function']['params']['learning_rate.end'].value(stage))
                learning_rate = dvt.ValueLinear(lr_start, lr_end, epochs, True)
            else:
                lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                learning_rate = dvt.ValueStatic(lr_start, epochs, True)

            learning_rate.unlock()

            for epoch in range(0, epochs):
                print("Epoch", epoch+1, "of", epochs)
                # loss = train_func(image_batch, target_batch)
                # test_func...


#dataset = Dataset(bench_desc['datasets'][0]['filename'])
#batch_gen = ThreadedBatchGenerator(dataset)
'''
for image_batch, target_batch in batch_gen.batch():
    loss = train_func(image_batch, target_batch)
    # etc

lpd = open("layerparams.csv", 'w')
lpd.write("stage;")
lnums = {
}
for layerp in bench_desc['cnns']['TestCNN01']['layers']:
    ltype = layerp['type']
    if ltype not in lnums:
        lnums[ltype] = 0
    for lparmname, lparmval in layerp['params'].items():
        lpd.write(layerp['type'] + str(lnums[ltype]) + "." + lparmname + ";")
    lnums[ltype] += 1
lpd.write("\n")

for stage in range(0, 1):
    # make dvalue selection
    bench_desc['cnns']['TestCNN01']['selector'].select_dvals(stage)
    # build net
    net = netbuilder.build(stage=stage)
    train_func = train_func_builder.build(stage=stage)
    test_func = test_func_builder.build(stage=stage)

    # write current values to file
    lpd.write(str(stage) + ";")
    for layerp in bench_desc['cnns']['TestCNN01']['layers']:
        for lparmname, lparmval in layerp['params'].items():
            lpd.write(str(lparmval) + ";")
    lpd.write("\n")
    print("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS")
    print("MODEL HAS", l.count_params(net), "PARAMS")

    for run in range(0, 5):
        # do training and testing here
        pass
        
'''

'''
bdfp = open("./sample_cnn_bench1.json", "r")



bench_desc = json.load(bdfp, encoding='utf-8')

cnnvals = {}

valsel = RoundRobinValueSelector()

for net in bench_desc["cnn_configurations"]:
    cnnvals[net['cnn_name']] = {}
    lnum = 0
    for layer in net["layers"][1:]:
        cnnvals[net['cnn_name']][lnum] = {}
        for param in layer["params"]:
            ptype = param['type']
            dval = None
            if ptype == 'static':
                dval = cnndval.ValueStatic(param["value"])
            elif ptype == 'stepped':
                dval = cnndval.ValueStepped(param['start'], param['end'], param['step'])
            elif ptype == 'multi':
                dval = cnndval.ValueMulti(param['values'])

            if dval is not None:
                cnnvals[net['cnn_name']][lnum][param['key']] = dval
        lnum += 1

print(cnnvals)
            # create conv layer

for i in range(0, stages):
    print("Stage", i)
    # build model from json
    total_time = 0
    for j in range(0, runs):
        t0 = time.perf_counter()
        print("Run", j)
        # train and test net, calc losses etc. for quality measurement

        t1 = time.perf_counter()
        total_time += t1-t0

    default_log.info("Stage: " + str(i) + " Run: " + str(j) + " took " + str((total_time / runs) * 1000) + "ms")

    print("Time spent", (total_time/runs) * 1000, "ms")
'''