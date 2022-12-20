import pickle
import gzip
import numpy as np
import cv2
#from torchmetrics.functional import jaccard_index
from PIL import Image as im
from skimage.measure import label, regionprops
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
#from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening, center_of_mass, fourier_ellipsoid, generate_binary_structure
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EPOCHS = 32
BATCH_SIZE = 8
INPUT_SHAPE = (360, 360, 1)
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

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

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


if __name__ == "__main__":
    
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")
    
    x_train = []
    y_train = []
    mask = np.zeros(INPUT_SHAPE[:2])
    for d in train_data:
        for i in d["frames"]:
            x = cv2.resize(d["video"][:,:,i], dsize=INPUT_SHAPE[:2])
            y = cv2.resize(255 * d["label"][:,:,i].astype(np.ubyte), dsize=INPUT_SHAPE[:2])
            x_train.append(x)
            y_train.append(y)
            mask = np.logical_or(mask, y)
    
    #im.fromarray(mask).save("aug/m1.jpg")
    mask = cv2.morphologyEx(255 * mask.astype(np.ubyte), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
    mask = binary_dilation(mask, iterations=10)
    #im.fromarray(mask).save("aug/m2.jpg")   
    
    x_train = np.expand_dims(np.array(x_train, dtype=np.single), 3)
    y_train = np.expand_dims(np.array(y_train, dtype=np.single), 3) / 255.0
    
    amateur = KFold(n_splits=5, shuffle=True).split(range(46))
    expert = KFold(n_splits=5, shuffle=True).split(range(19))
    scores = []
    
    for (train_idx_a, test_idx_a), (train_idx_e, test_idx_e) in zip(amateur, expert):

        train_idx = np.concatenate((train_idx_a, [i + 46 for i in train_idx_e]))
        test_idx = np.concatenate((test_idx_a, [i + 46 for i in test_idx_e]))
        train_idx = np.array([3*i for i in train_idx] + [3*i+1 for i in train_idx] + [3*i+2 for i in train_idx])
        test_idx = np.array([3*i for i in test_idx] + [3*i+1 for i in test_idx] + [3*i+2 for i in test_idx])
        
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
        a = augmentation(x_train[train_idx], y_train[train_idx], x_train[test_idx], y_train[test_idx])
        model.fit(
            a[0],
            validation_data=a[1],
            steps_per_epoch=len(x_train[train_idx]) // BATCH_SIZE,
            validation_steps=len(x_train[test_idx]) // BATCH_SIZE,
            callbacks=[checkpoint],
            verbose=1,
            epochs=EPOCHS
        )

        model.load_weights(MODEL_FILE)
        pred = np.squeeze(model.predict(x_train[test_idx]))
    
        m = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=TH)
        fehlt = 0
        for i in range(len(test_idx)):
            idx = test_idx[i]
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
