import numpy as np
import matplotlib.pyplot as plt

def encrypt_image(image_array, key):
    encrypted_array = np.bitwise_xor(np.round(image_array * 255).astype(np.uint8), key)
    return encrypted_array / 255.0

def decrypt_image(encrypted_array, key):
    decrypted_array = np.bitwise_xor(np.round(encrypted_array * 255).astype(np.uint8), key)
    return decrypted_array / 255.0

def plot_images(original_image, encrypted_image, decrypted_image):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(encrypted_image)
    plt.title('Encrypted Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(decrypted_image)
    plt.title('Decrypted Image')
    plt.axis('off')

    plt.show()