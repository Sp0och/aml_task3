import pickle
import gzip
import numpy as np
import os

import matplotlib.pyplot as plt
import cv2
import cv2 as cv
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from PIL import Image as im
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
from tensorflow.keras import backend as K
# import model checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
# import canny edge detector
from skimage.feature import canny
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening, center_of_mass, fourier_ellipsoid, generate_binary_structure


EPOCHS = 32
BATCH_SIZE = 8
N_AUGMENTATIONS = 10
INPUT_SHAPE = (360, 360)
SUBMISSION = True
TH = 0.5
NB_OF_AREAS = 3
EROSION_ITERATIONS = 5
MODEL_FILE = 'model.hdf5'
BEST_MODEL_FILE = 'best_model.hdf5'
# env2lmod
# module load cuda cudnn python
# pip3 install --user --upgrade pip
# pip3 install --user --upgrade h5py==3.6.0
# pip3 install --user --upgrade numpy==1.21.5
# pip3 install --user --upgrade opencv-python==4.6.0.66
# pip3 install --user --upgrade scikit-image==0.19.3
# pip3 install --user --upgrade scikit-learn==1.0.2
# pip3 install --user --upgrade pillow==9.0.0
# pip3 install --user --upgrade scipy==1.7.3
# pip3 install --user --upgrade tensorflow-gpu==2.8.0
# pip3 install --user --upgrade tensorflow==2.8.0

# Helper Functions
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)
# Model Building
def double_conv_block(x, n_filters):
   x = layers.Conv2D(n_filters, 3, padding="same")(x)
   x = layers.BatchNormalization()(x)
   x = layers.ReLU()(x)
   x = layers.Conv2D(n_filters, 3, padding="same")(x)
   x = layers.BatchNormalization()(x)
   x = layers.ReLU()(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   #p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   x = layers.Conv2DTranspose(n_filters, 2, 2, padding="valid")(x)
   x = layers.concatenate([x, conv_features])
   #x = layers.Dropout(0.3)(x)
   x = double_conv_block(x, n_filters)
   return x

def get_model(m = 4):
    inputs = layers.Input(shape=INPUT_SHAPE)
    f1, p1 = downsample_block(inputs, 8 * m)
    f2, p2 = downsample_block(p1, 16 * m)
    f3, p3 = downsample_block(p2, 32 * m)
    bottleneck = double_conv_block(p3, 64 * m)
    u7 = upsample_block(bottleneck, f3, 32 * m)
    u8 = upsample_block(u7, f2, 16 * m)
    u9 = upsample_block(u8, f1, 8 * m)
    outputs = layers.Conv2D(1, 1, padding="valid", activation = "sigmoid")(u9)
    return tf.keras.Model(inputs, outputs)

# Metrics
def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # https://github.com/karolzak/keras-unet/tree/master/keras_unet
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# Augmentation
def augmentation(
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    data_gen_args=dict(
        #horizontal_flip=True,
        #rotation_range=10,
        #zoom_range=0.1,
        #height_shift_range=0.1,
        #width_shift_range=0.1,
        fill_mode='constant'
        
        # vertical_flip=False,
        # shear_range=10,
        # zoom_range=[0.8, 1.1],
        # brightness_range=(0.3, 1)
    )
):
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=0)
    Y_datagen.fit(Y_train, augment=True, seed=0)
    X_train_augmented = X_datagen.flow(X_train, batch_size=BATCH_SIZE, shuffle=True, seed=0)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=BATCH_SIZE, shuffle=True, seed=0)

    if not (X_val is None) and not (Y_val is None):
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=False, seed=0)
        Y_datagen_val.fit(Y_val, augment=False, seed=0)
        X_val_augmented = X_datagen_val.flow(X_val, batch_size=BATCH_SIZE, shuffle=False, seed=0)
        Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=BATCH_SIZE, shuffle=False, seed=0)

        return zip(X_train_augmented, Y_train_augmented), zip(X_val_augmented, Y_val_augmented)
    else:
        return zip(X_train_augmented, Y_train_augmented), None

def create_datagen():
    datagen = ImageDataGenerator(horizontal_flip=True,
                    rotation_range=10,
                    shear_range=10,
                    zoom_range=[0.8, 1.1],
                    height_shift_range=0.1,
                    width_shift_range=0.1,
                    brightness_range=(0.3, 1))
    return datagen

def augment_data(image, mask):
    seed = 1
    data_gen_args = dict(horizontal_flip=True,
                         rotation_range=10,
                         shear_range=10,
                         zoom_range=[0.8, 1.1],
                         height_shift_range=0.1,
                         width_shift_range=0.1,
                         brightness_range=(0.3, 1))

    frame_augmentor = ImageDataGenerator(**data_gen_args)
    label_augmentor = ImageDataGenerator(**data_gen_args)

    myimg = image.reshape((1,) + image.shape + (1,))
    mask = mask.reshape((1,) + mask.shape + (1,))

    aug_frames = frame_augmentor.flow(myimg, seed=seed, batch_size=1)
    aug_labels = label_augmentor.flow(mask, seed=seed, batch_size=1)


    return aug_frames, aug_labels

def augmented_images_visualization():
    train_sample_id = 50
    labeled_frame_idx = 1

    video = np.copy(train_data[train_sample_id]['video'])
    labels = train_data[train_sample_id]['label']
    box = train_data[train_sample_id]['box']
    labeled_frames = train_data[train_sample_id]['frames']

    aug_frames, aug_labels = augment_data(video[:,:,labeled_frames[labeled_frame_idx]], 255*labels[:,:,labeled_frames[labeled_frame_idx]].astype(np.ubyte))

    nrow = 4
    ncol = 4
    # generate samples and plot
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 15 * nrow / ncol))

    for ox in ax.reshape(-1):
        # convert to unsigned integers
        image = next(aug_frames)[0].astype('uint8')
        mask = next(aug_labels)[0].astype('uint8')
        ox.imshow(image)
        ox.imshow(mask, alpha=0.5)
        ox.axis('off')

    plt.show()

# Data Set loading
def augmented_image_loading(n_augmentations=10):
    x_train = []
    y_train = []
    mask = np.zeros(INPUT_SHAPE)
    x_test = []
    for d in train_data:
        for i in d["frames"]:
            x = cv2.resize(d["video"][:,:,i], dsize=INPUT_SHAPE[:2])
            y = cv2.resize(255 * d["label"][:,:,i].astype(np.ubyte), dsize=INPUT_SHAPE)
            x_train.append(x)
            y_train.append(y)
            mask = np.logical_or(mask, y)

            aug_images, aug_masks = augment_data(x, y)
            for tt in range(n_augmentations):
                x_train.append(cv2.resize(next(aug_images)[0], dsize=(360, 360)))
                y_train.append(cv2.resize(next(aug_masks)[0], dsize=(360, 360)))

    mask = cv2.morphologyEx(255 * mask.astype(np.ubyte), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
    mask = binary_dilation(mask, iterations=10)

    for d in test_data:
        for i in range(d["video"].shape[2]):
            x_test.append(cv2.resize(d["video"][:, :, i], dsize=(360, 360)))

    x_train = np.expand_dims(np.array(x_train, dtype=np.single), 3)
    y_train = np.expand_dims(np.array(y_train, dtype=np.single), 3) / 255.0
    x_test = np.expand_dims(np.array(x_test, dtype=np.single), 3)

    return x_train, y_train, x_test, mask

def get_median_filtered(x_train,x_test, median_size=3):
    x_train_median_filtered = []
    x_test_median_filtered = []
    for i in range(x_train.shape[0]):
        x_train_median_filtered.append(median_filter(x_train[i,:,:,0], size=median_size))
    for i in range(x_test.shape[0]):
        x_test_median_filtered.append(median_filter(x_test[i,:,:,0], size=median_size))

    x_train_median_filtered = np.expand_dims(np.array(x_train_median_filtered, dtype=np.single), 3)
    x_test_median_filtered = np.expand_dims(np.array(x_test_median_filtered, dtype=np.single), 3)

    return x_train_median_filtered, x_test_median_filtered

def get_canny_edge_filtered(x_train,x_test):
    x_train_canny_edges = []
    x_test_canny_edges = []
    for i in range(x_train.shape[0]):
        x_train_canny_edges.append(canny(x_train[i,:,:,0], sigma=3))

    for i in range(x_test.shape[0]):
        x_test_canny_edges.append(canny(x_test[i,:,:,0], sigma=3))

    x_train_canny_edges = np.expand_dims(np.array(x_train_canny_edges, dtype=np.single), 3)
    x_test_canny_edges = np.expand_dims(np.array(x_test_canny_edges, dtype=np.single), 3)

    return x_train_canny_edges, x_test_canny_edges

def get_training_data(n_augmentations=10, median_size=3):
    x_train, y_train, x_test, mask = augmented_image_loading(n_augmentations=n_augmentations)
    x_train_median_filtered, x_test_median_filtered = get_median_filtered(x_train, x_test, median_size=median_size)
    x_train_canny_edges, x_test_canny_edges = get_canny_edge_filtered(x_train, x_test)

    x_train = np.concatenate([x_train, x_train_median_filtered, x_train_canny_edges], axis=3)
    x_test = np.concatenate([x_test, x_test_median_filtered, x_test_canny_edges], axis=3)

    return x_train, y_train, x_test, mask

    

if __name__ == "__main__":
    
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")

    x_train, y_train, x_test, mask = get_training_data(N_AUGMENTATIONS, median_size=3)
    
    amateur = KFold(n_splits=5, shuffle=True).split(range(46))
    expert = KFold(n_splits=5, shuffle=True).split(range(19))
    scores = []
    # TODO fix validation split
    
    for (train_idx_a, val_idx_a), (train_idx_e, val_idx_e) in zip(amateur, expert):
        # Get the flag indexes of the training and validation data
        flag_train_idx = np.concatenate((train_idx_a, [i + 46 for i in train_idx_e]))
        flag_val_idx = np.concatenate((val_idx_a, [i + 46 for i in val_idx_e]))
        # For each index add the three frames and their augmentations (11 images per entry => 33 * 65 total)
        train_idx = np.array([])
        val_idx = np.array([])
        for idx in flag_train_idx:
            for frame in range(3):
                for augm in range(N_AUGMENTATIONS):
                    train_idx = np.append(33 * idx + 11 * frame + N_AUGMENTATIONS)
        for idx in flag_val_idx:
            for frame in range(3):
                for augm in range(N_AUGMENTATIONS):
                    val_idx = np.append(33 * idx + 11 * frame + N_AUGMENTATIONS)

        
        print('------------------------------------------------------------------------------------')
        print('Training for fold')
        print('------------------------------------------------------------------------------------')
        
        keras.backend.clear_session()
        model = get_model()
        
        model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=1e-3),
          loss=keras.losses.BinaryCrossentropy(),
          metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=TH, name="IoU")]
        )
        
        checkpoint = ModelCheckpoint(
            filepath=MODEL_FILE, 
            monitor="val_IoU",
            verbose=0, 
            save_best_only=True,
            mode="max",
            save_weights_only=True
        )
        
        model.fit(
            x_train[train_idx],
            y_train[train_idx],
            validation_data=(x_train[val_idx], y_train[val_idx]),
            steps_per_epoch=len(x_train[train_idx]) // BATCH_SIZE,
            validation_steps=len(x_train[val_idx]) // BATCH_SIZE,
            callbacks=[checkpoint],
            verbose=1,
            epochs=EPOCHS
        )

        model.load_weights(MODEL_FILE)
        pred = np.squeeze(model.predict(x_train[val_idx]))
    
        m = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=TH)
        fehlt = 0
        for i in range(len(val_idx)):
            idx = val_idx[i]
            ff = train_data[idx//3]["frames"][idx%3]
            gt = train_data[idx//3]["label"][:,:,ff]
            im.fromarray(gt).save("pred/" + str(i) + "_0.jpg")
            
            pp = cv2.resize(255 * pred[i,:,:], dsize=gt.shape[::-1])
            pp = pp > (255 * TH)
            pp = np.logical_and(cv2.resize(mask.astype(np.ubyte), dsize=gt.shape[::-1]), pp)
            im.fromarray(pp).save("pred/" + str(i) + "_1.jpg")

            lab = label(pp)
            rps = regionprops(lab)
            area_idx = np.argsort([r.area for r in rps])[::-1]
            new_pp = np.zeros_like(pp)
            for j in area_idx[:NB_OF_AREAS]:
                new_pp[tuple(rps[j].coords.T)] = True
            #new_pp = binary_erosion(new_pp, iterations=EROSION_ITERATIONS)
            im.fromarray(new_pp).save("pred/" + str(i) + "_2.jpg")
            
            fehlt += np.count_nonzero(np.logical_and(gt, np.logical_not(new_pp)))
            m.update_state(gt.astype(np.ubyte), new_pp.astype(np.ubyte))
            
        print(fehlt)
        scores.append(m.result().numpy())
        print("score: " + str(scores[-1]))
        
        if m.result().numpy() == max(scores):
            model.save_weights(BEST_MODEL_FILE)
        
        
    
    print("total score: " + str(np.mean(scores)))
    
    if SUBMISSION:
        
        predictions = []
        model.load_weights(BEST_MODEL_FILE)
        
        for d in test_data:
            
            x_test = []
            for i in range(d["video"].shape[2]):
                x_test.append(cv2.resize(d["video"][:,:,i], dsize=INPUT_SHAPE[:2]))
            x_test = np.expand_dims(np.array(x_test, dtype=np.single), 3)
            
            pred = model.predict(x_test)
            pred = np.squeeze(pred)
            
            prediction = np.array(np.zeros_like(d['video']), dtype=bool)
            
            for i in range(pred.shape[0]):
                
                pp = cv2.resize(255 * pred[i,:,:], dsize=prediction.shape[::-1][1:])
                pp = pp > (255 * TH)
                pp = np.logical_and(cv2.resize(mask.astype(np.ubyte), dsize=prediction.shape[::-1][1:]), pp)
                pred_img = im.fromarray(pp)
                
                lab = label(pp)
                rps = regionprops(lab)
                area_idx = np.argsort([r.area for r in rps])[::-1]
                new_pp = np.zeros_like(pp)
                for j in area_idx[:NB_OF_AREAS]:
                    new_pp[tuple(rps[j].coords.T)] = True
                #new_pp = binary_erosion(new_pp, iterations=EROSION_ITERATIONS)
                new_pred_img = im.fromarray(new_pp)
                
                prediction[:,:,i] = new_pp
            
            predictions.append({'name': d['name'], 'prediction': prediction})
    
        save_zipped_pickle(predictions, 'my_predictions.pkl')



#Center of mass
#behalte nur die zwei die nahe beinander sind






# datagen = ImageDataGenerator(horizontal_flip=True, fill_mode='constant')
# i = 0
# for batch in datagen.flow(x_train, batch_size=16,
#                           save_to_dir='aug',
#                           save_prefix='aug',
#                           save_format='png'):   
#   x,y=batch
#   i += 1 
#   if i > 3:
#      break

#pred = im.fromarray((np.squeeze(model.predict(x_train[16:17]))>0.8))
#gt = im.fromarray(cv2.resize(255 * train_data[5]["label"][:,:,51].astype(np.ubyte), dsize=(360, 360)))
