from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
import imagedatahelpers as idh
import modelsavecallback as msc
import resnet50generator as rngen

# useful constants
EPOCHS = 5
BATCH_SIZE = 128
IMAGE_SIZE = [320, 240]

if __name__ == "__main__":
    weights_path = './data/weights/resnet-blood-cells-weights'
    train_path = "./data/blood-cells/images/TRAIN"
    valid_path = "./data/blood-cells/images/TEST"

    # create a couple of arrays holding the paths of the training and test images
    image_files = glob(train_path + '/*/*.jp*g')
    valid_image_files = glob(valid_path + '/*/*.jp*g')

    # this will get us a list of category folders
    folders = glob(train_path + '/*')

    # generate the category names from the folders
    classes = idh.category_names_from_paths(folders)

    # create the model
    model = rngen.generate_resnet50(IMAGE_SIZE + [3], classes=len(classes))

    # view the model structure
    model.summary(line_length=150)

    # compile the model and provide the loss and optimization method
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics='accuracy')

    # define an image input generator
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )

    # create generators for the image data
    # NOTE: we made the loss function sparse_categorical_crossentropy, which means we needed to set the
    #   class_mode parameter to 'sparse' in these generators
    train_generator = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE,
                                              class_mode='sparse')
    valid_generator = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE,
                                              class_mode='sparse')

    # now do the fit using the generators
    r = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        steps_per_epoch=len(image_files) // BATCH_SIZE,
        validation_steps=len(valid_image_files) // BATCH_SIZE,
        callbacks=[msc.ModelSaveCallback(model, weights_path)]
    )

    # create confusion matrices for the training and validation data
    cm = idh.get_confusion_matrix(
        gen,
        train_path,
        len(image_files),
        batch_size=BATCH_SIZE * 2,
        model=model,
        image_size=IMAGE_SIZE
    )
    print(cm)
    valid_cm = idh.get_confusion_matrix(
        gen,
        valid_path,
        len(valid_image_files),
        batch_size=BATCH_SIZE * 2,
        model=model,
        image_size=IMAGE_SIZE
    )
    print(valid_cm)

    plt.plot(r.history['loss'], label='Train Loss')
    plt.plot(r.history['val_loss'], label='Val. Loss')
    plt.legend()
    plt.show()

    plt.plot(r.history['accuracy'], label='Train Acc.')
    plt.plot(r.history['val_accuracy'], label='Val. Acc.')
    plt.legend()
    plt.show()

    # display the confusion matrices
    # fig, axs = plt.subplots()
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # TODO: Figure out how to add a title cmd.title('Training Data Confusion Matrix')
    cmd.plot()
    plt.show()

    valid_cmd = ConfusionMatrixDisplay(confusion_matrix=valid_cm, display_labels=classes)
    # TODO: Figure out how to add a title valid_cmd.title('Validation Data Confusion Matrix')
    valid_cmd.plot()
    plt.show()
