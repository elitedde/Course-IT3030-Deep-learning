from matplotlib import pyplot as plt
from configParser.configParser_data import Data
from autoencoder import autoencoder
from verification_net import VerificationNet
from stacked_mnist import DataMode, StackedMNISTData
from configParser.globalWords import Model
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')


def get_mode(data_mode):
    data_modes = [
        DataMode.MONO_FLOAT_COMPLETE,
        DataMode.MONO_FLOAT_MISSING,
        DataMode.MONO_BINARY_COMPLETE,
        DataMode.MONO_BINARY_MISSING,
        DataMode.COLOR_FLOAT_COMPLETE,
        DataMode.COLOR_FLOAT_MISSING,
        DataMode.COLOR_BINARY_COMPLETE,
        DataMode.COLOR_BINARY_MISSING
    ]

    data_mode_name = [
        'mono_float_complete',
        'mono_float_missing',
        'mono_binary_complete',
        'mono_binary_missing',
        'color_float_complete',
        'color_float_missing',
        'color_binary_complete',
        'color_binary_missing'
    ]

    return data_modes[data_mode_name.index(data_mode)]
def cross_entropy(targets, predictions, epsilon=1 - 12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -(np.sum(targets * np.log(predictions + 1e-9))+np.sum((1-targets) * np.log(1-predictions + 1e-9))) / N
    return ce
def check_binary(images, mode):
    if 'binary' in mode:
        images[images >= .5] = 1.
        images[images < .5] = 0.
        images = images.astype(np.int)
        return images
    return images

def create_model(force_learn, batch_size, mode, model_name, input_shape, latent_size, color=False, binary=False):
    if model_name == Model.AE:
        ae = autoencoder(force_learn = force_learn, batch_size = batch_size, mode = mode, dim_input=input_shape, num_channels=3 if color else 1, latent_size=latent_size, variational=False, binary=binary)
        return ae, ae.encoder, ae.decoder
    elif model_name == Model.VAE:
        vae = autoencoder(force_learn = force_learn,batch_size = batch_size, mode = mode, dim_input=input_shape, num_channels=3 if color else 1, latent_size=latent_size, variational=True)
        return vae, vae.encoder, vae.decoder
    else:
        raise ValueError("not found")


def reconstruct_images(mode, model, auto_encoder, gen, verifier, num_channels):
    """
    autoencoder is able to learn how to decompose images into fairly small bits of data,
    and then using that representation,
    reconstruct the original data as closely as it can to the original
    The output is evaluated by comparing the reconstructed image by the original one

    """
    img, cls = gen.get_random_batch(training=False, batch_size=1000)
    im = img.reshape(img.shape[0], -1)
    rec_img = auto_encoder.predict(im)
    rec_img = rec_img.reshape(img.shape[0], 28, 28, num_channels)
    rec_img = check_binary(rec_img,mode)
    tolerance = 0.8 if num_channels == 1 else 0.5
    predictability, accuracy = verifier.check_predictability(rec_img, correct_labels=cls, tolerance=tolerance)
    print(f"Reconstruction predictability is {100*predictability:.3f}% and accuracy is {100*accuracy:.3f}%")
    #coverage: how many classes are in the dataset?
    coverage = verifier.check_class_coverage(rec_img, tolerance=tolerance)
    print(f"Coverage is {100*coverage:.3f}%")

    gen.plot_example(model, mode, "real", images=img[:9], labels=cls)
    gen.plot_example(model, mode, "reconstr", images=rec_img[:9], labels=cls)


def generate_images(mode, model, latent_dim, generator, gen, verifier, num_channels):
    """"
    It samples points from the distribution and feed them to the decoder to generate new input data samples
    The generator is the encoder
    """
    rand_latent_vec = np.random.normal(0, 1.1, 1000 * latent_dim).reshape(1000, latent_dim)  # features in latent layer
    gen_img = generator.predict(rand_latent_vec)
    gen_img = gen_img.reshape(1000, 28, 28, num_channels)
    gen_img = check_binary(gen_img,mode)
    _, rand_cls = gen.get_random_batch(batch_size=9)
    gen.plot_example(model, mode, "generated", images=gen_img[:9, :, :], labels=rand_cls)  #the labels ara randomly generated
    tolerance = 0.8 if num_channels == 1 else 0.5
    predictability, _ = verifier.check_predictability(gen_img, tolerance=tolerance)
    coverage = verifier.check_class_coverage(gen_img, tolerance=tolerance)
    print("Predictability is {0:.0f}% and coverage is {1:.0f}%".format(100*predictability, 100*coverage))


def display_digit_classes(gen, encoder):
    """
    Images are encoded into latent space and the distribution is shown using simple scatter plot
    (2D plot of the digit classes in the latent space)
    """
    img, cls = gen.get_random_batch(training=False, batch_size=1000)
    img = img.reshape(img.shape[0], -1)
    x_test_encoded = encoder.predict(img, batch_size=100)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=cls)
    plt.colorbar()
    plt.show()


def anomaly_detection(mode, ae, decoder, num_channels, latent_shape, model, gen):
    """
    I train the model without one class ( 8 ) and I expect the reconstruction loss to be higher
    """
    rec_loss = None
    x = None
    cls = None
    rec_img = None
    if model == Model.VAE:
        model = 'vae'
        num_samples = 50
        x, cls = gen.get_random_batch(training=False, batch_size=1000)
        rand_latent_vec = np.random.normal(0, 1.1, num_samples * latent_shape).reshape(num_samples, latent_shape)
        rand_latent_vec = rand_latent_vec.reshape(rand_latent_vec.shape[0], -1)
        noise_pred = decoder.predict(rand_latent_vec)
        noise_pred = noise_pred.reshape(num_samples, 28, 28, num_channels)
        noise_pred = check_binary(noise_pred, mode)
        x_norm, cls_norm = gen.get_random_batch(training=True, batch_size=1000)
        rec_loss = []
        rec_loss_norm = []
        for img_num in range(x.shape[0]):
            stacked_img = np.array([x[img_num] for _ in range(num_samples)])
            loss = cross_entropy(stacked_img.flatten(), noise_pred.flatten())
            rec_loss.append(loss)
            stacked_img_norm = np.array([x_norm[img_num] for _ in range(num_samples)])
            loss_norm = cross_entropy(stacked_img_norm.flatten(), noise_pred.flatten())
            rec_loss_norm.append(loss_norm)
        rec_loss = np.array(rec_loss)
        rec_loss_normal = np.array(rec_loss_norm)
        img = x.reshape(x.shape[0], -1)
        rec_img = ae.predict(img)
        rec_img = rec_img.reshape(rec_img.shape[0], 28, 28, num_channels)
        rec_img = check_binary(rec_img, mode)

    elif model == Model.AE:
        model = 'ae'
        x, cls = gen.get_full_data_set(training=False)
        img = x.reshape(x.shape[0], -1)
        rec_img = ae.predict(img)
        rec_img = rec_img.reshape(rec_img.shape[0], 28, 28, num_channels)
        rec_img = check_binary(rec_img, mode)
        rec_loss = np.mean((rec_img - x) ** 2, axis=(1, 2, 3))
        # ---------------------------------#
        x_normal, cls_normal = gen.get_random_batch(training=True, batch_size=5000)
        img_normal = x_normal.reshape(x_normal.shape[0], -1)
        rec_img_normal = ae.predict(img_normal)
        rec_img_normal = rec_img_normal.reshape(rec_img_normal.shape[0], 28, 28, num_channels)
        rec_loss_normal = np.mean((rec_img_normal - x_normal) ** 2, axis=(1, 2, 3))
        # --------------------------------------#
    avg_rec_loss_normal = rec_loss_normal.mean()
    print(f"Reconstruction loss normal: {avg_rec_loss_normal:.3f}")
    avg_rec_loss = rec_loss.mean()
    print(f"Reconstruction loss: {avg_rec_loss:.3f}")
    ind = rec_loss.argsort()[-25:]  # plot the most anomalous images
    gen.plot_example(model, mode, "anomaly_det_real", images=x[ind], labels=cls[ind])
    gen.plot_example(model, mode, "anomaly_det_recostr", images=rec_img[ind], labels=cls[ind])




def main():
    p = Data() #get parameters
    p.extractData()
    verifier = VerificationNet(force_learn= p.force_learn, file_name='./models/' + p.mode + '.h5')
    gen = StackedMNISTData(mode=get_mode(p.mode), default_batch_size=p.default_batch_size)
    print("-------database has been created-----")
    verifier.train(gen)
    # Initialize model, encoder and decoder.
    ae, encoder, decoder = create_model(p.force_learn_AE, p.batch_size, p.mode,  p.model, p.input_shape, p.latent_size, color='color' in p.mode, binary = "binary" in p.mode)
    print("-------AE has been created-----")
    ae.train(gen, p.epochs)

    # reconstruct_images(p.mode, 'ae'if p.model == Model.AE else 'vae', ae, gen, verifier, num_channels=3 if 'color' in p.mode else 1)
    # print("----------Images have been reconstruced-------------")
    #
    # generate_images(p.mode, 'ae'if p.model == Model.AE else 'vae', p.latent_size, decoder, gen, verifier, num_channels=3 if 'color' in p.mode else 1)
    # print("------------Image has been generated -------------")
    #
    # display_digit_classes(gen, encoder)
    # print("------------Digit classes have been displayed -------------")

    if 'missing' in p.mode:
        if p.model == Model.AE:
            anomaly_detection(p.mode, ae, decoder, 3 if 'color' in p.mode else 1, p.latent_size, p.model,  gen)
        else:
            anomaly_detection(p.mode, ae, decoder, 3 if 'color' in p.mode else 1, p.latent_size, p.model,  gen)
        print("----------Anomaly detection-----------")


if __name__ == "__main__":
    main()
