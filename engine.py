import streamlit as st
import numpy as np
from io import BytesIO
import librosa
import soundfile as sf
from PIL import Image
def xor_cipher(data, key, decrypt=False):
    np.random.seed(key)
    random_key = np.random.randint(0, 256, size=len(data), dtype=np.uint8)
    if decrypt:
        processed_data = np.bitwise_xor(data.astype(np.uint8), random_key.astype(np.uint8))
    else:
        processed_data = np.bitwise_xor(data.astype(np.uint8), random_key.astype(np.uint8))
    return processed_data.astype(np.float32)


def permutation_cipher(data, key, decrypt=False):
    np.random.seed(key)
    if not decrypt:
        permuted_indices = np.random.permutation(len(data))
    else:
        permuted_indices = np.argsort(np.random.permutation(len(data)))
    processed_data = data[permuted_indices]
    return processed_data


def encrypt_audio(audio_file, key, method='XOR Cipher'):
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    if method == 'XOR Cipher':
        encrypted_data = xor_cipher(audio_data, key)
    elif method == 'Permutation Cipher':
        encrypted_data = permutation_cipher(audio_data, key)
    return encrypted_data, sample_rate


def decrypt_audio(encrypted_data, key, sample_rate, method='XOR Cipher'):
    if method == 'XOR Cipher':
        decrypted_data = xor_cipher(encrypted_data, key, decrypt=True)
    elif method == 'Permutation Cipher':
        decrypted_data = permutation_cipher(encrypted_data, key, decrypt=True)
    return decrypted_data, sample_rate


def save_audio_button(data, sample_rate, caption):
    audio_bytes = BytesIO()
    sf.write(audio_bytes, data, sample_rate)
    st.download_button(label=f"Download {caption}", data=audio_bytes.getvalue(), file_name=f"{caption}.wav",
                       mime='audio/wav')


def xor_cipher_image(image, key, decrypt=False):
    pixels = np.array(image)
    np.random.seed(key)
    random_key = np.random.randint(0, 256, size=pixels.shape, dtype=np.uint8)
    if decrypt:
        processed_pixels = np.bitwise_xor(pixels, random_key)
    else:
        processed_pixels = np.bitwise_xor(pixels, random_key)
    return Image.fromarray(processed_pixels)


def henon_map_image(image, a=1.4, b=0.3, decrypt=False):
    pixels = np.array(image)
    x, y = 0.1, 0.1
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            x, y = 1 - a * x ** 2 + y, b * x
            pixels[i, j] = (pixels[i, j] + int(x * 255)) % 256 if not decrypt else (pixels[i, j] - int(x * 255)) % 256
    return Image.fromarray(pixels.astype('uint8'))


def logistic_map_image(image, r=3.99, x0=0.5, decrypt=False):
    pixels = np.array(image)
    x = x0
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            x = r * x * (1 - x)
            pixels[i, j] = (pixels[i, j] + int(x * 255)) % 256 if not decrypt else (pixels[i, j] - int(x * 255)) % 256
    return Image.fromarray(pixels.astype('uint8'))


def advanced_encryption_image(image, key):
    pixels = np.array(image)
    np.random.seed(key)
    random_sequence = np.random.permutation(pixels.size)
    pixels_flat = pixels.flatten()
    pixels_encrypted = pixels_flat[random_sequence].reshape(pixels.shape)
    return Image.fromarray(pixels_encrypted.astype('uint8'))


def advanced_decryption_image(image, key):
    pixels = np.array(image)
    np.random.seed(key)
    random_sequence = np.random.permutation(pixels.size)
    inverse_sequence = np.argsort(random_sequence)
    pixels_flat = pixels.flatten()
    pixels_decrypted = pixels_flat[inverse_sequence].reshape(pixels.shape)
    return Image.fromarray(pixels_decrypted.astype('uint8'))


def plot_rgb_distribution(image, title):
    pixels = np.array(image).reshape(-1, 3)
    fig, ax = plt.subplots()
    ax.scatter(pixels[:, 0], pixels[:, 1], c=pixels / 255.0, s=0.1)
    ax.set_title(title)
    st.pyplot(fig)


def save_image_button(image, caption):
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    st.download_button(label=f"Download {caption}", data=img_bytes.getvalue(), file_name=f"{caption}.png",
                       mime='image/png')


def save_audio_button(data, sample_rate, caption, format='wav'):
    audio_bytes = BytesIO()
    sf.write(audio_bytes, data, sample_rate, format=format)
    st.download_button(label=f"Download {caption}", data=audio_bytes.getvalue(), file_name=f"{caption}.{format}",
                       mime=f'audio/{format}')


def main():
    st.set_page_config(page_title="Encryption/Decryption Engine", layout="wide")

    st.title("Image and Audio Encryption and Decryption Engine")
    st.write("This application allows you to encrypt and decrypt images and audio using various methods.")
    st.write("")

    mode = st.radio("Choose Mode:", ("Encrypt", "Decrypt"), index=0, key='mode')

    if mode == "Encrypt":
        st.header("Image Encryption")
        image_encryption()

        st.header("Audio Encryption")
        audio_encryption()

    elif mode == "Decrypt":
        st.header("Image Decryption")
        image_decryption()

        st.header("Audio Decryption")
        audio_decryption()

def image_encryption():
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    key_image = st.number_input("Please Enter Key for Image Encryption:", min_value=0, step=1, value=42)

    if uploaded_image:
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption="Original Image", use_column_width=True)
        method_image = st.selectbox("Choose method for Image Encryption:", ["XOR Cipher", "Henon Map", "Logistic Map", "Advanced Encryption"])

        if method_image == "XOR Cipher":
            processed_image = xor_cipher_image(original_image, key_image)
        elif method_image == "Henon Map":
            processed_image = henon_map_image(original_image)
        elif method_image == "Logistic Map":
            processed_image = logistic_map_image(original_image)
        elif method_image == "Advanced Encryption":
            processed_image = advanced_encryption_image(original_image, key_image)

        st.image(processed_image, caption="Encrypted Image", use_column_width=True)
        save_image_button(processed_image, "Encrypted Image")

def audio_encryption():
    audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    key_audio = st.number_input("Please Enter Key for Audio Encryption:", min_value=0, step=1, value=42)

    if audio_file:
        encryption_method = st.selectbox("Choose encryption method for audio:", ["XOR Cipher", "Permutation Cipher"])
        encrypted_data, sample_rate = encrypt_audio(audio_file, key_audio, encryption_method)
        st.audio(encrypted_data, format='audio/wav', sample_rate=sample_rate)
        save_audio_button(encrypted_data, sample_rate, "Encrypted Audio")

def image_decryption():
    uploaded_encrypted_image = st.file_uploader("Choose an encrypted image...", type=["jpg", "jpeg", "png"])
    key_image_decrypt = st.number_input("Please Enter Key for Image Decryption:", min_value=0, step=1, value=42)

    if uploaded_encrypted_image:
        encrypted_image = Image.open(uploaded_encrypted_image)
        st.image(encrypted_image, caption="Encrypted Image", use_column_width=True)
        method_image = st.selectbox("Choose method for Image Decryption:", ["XOR Cipher", "Henon Map", "Logistic Map", "Advanced Encryption"])

        if method_image == "XOR Cipher":
            processed_image = xor_cipher_image(encrypted_image, key_image_decrypt, decrypt=True)
        elif method_image == "Henon Map":
            processed_image = henon_map_image(encrypted_image, decrypt=True)
        elif method_image == "Logistic Map":
            processed_image = logistic_map_image(encrypted_image, decrypt=True)
        elif method_image == "Advanced Encryption":
            processed_image = advanced_decryption_image(encrypted_image, key_image_decrypt)

        st.image(processed_image, caption="Decrypted Image", use_column_width=True)
        save_image_button(processed_image, "Decrypted Image")

def audio_decryption():
    encrypted_audio_file = st.file_uploader("Choose an encrypted audio file...", type=["wav", "mp3"])
    key_decryption = st.number_input("Please Enter Key for Audio Decryption:", min_value=0, step=1, value=42)

    if encrypted_audio_file:
        decryption_method = st.selectbox("Choose decryption method for audio:", ["XOR Cipher", "Permutation Cipher"])
        encrypted_data, sample_rate = librosa.load(encrypted_audio_file, sr=None)
        decrypted_data, _ = decrypt_audio(encrypted_data, key_decryption, sample_rate, decryption_method)
        st.audio(decrypted_data, format='audio/wav', sample_rate=sample_rate)
        save_audio_button(decrypted_data, sample_rate, "Decrypted Audio")

if __name__ == "__main__":
    main()
