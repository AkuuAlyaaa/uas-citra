import numpy as np
import cv2
import matplotlib.pyplot as plt

def inisialisasi_pusat(gambar, k):
    piksel = gambar.reshape(-1, 3)
    indeks = np.random.choice(piksel.shape[0], k, replace=False)
    return piksel[indeks]

def tetapkan_klaster(gambar, pusat):
    piksel = gambar.reshape(-1, 3)
    jarak = np.linalg.norm(piksel[:, np.newaxis] - pusat, axis=2)
    klaster = np.argmin(jarak, axis=1)
    return klaster

def perbarui_pusat(gambar, klaster, k):
    piksel = gambar.reshape(-1, 3)
    pusat_baru = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        piksel_klaster = piksel[klaster == i]
        if len(piksel_klaster) > 0:
            pusat_baru[i] = np.mean(piksel_klaster, axis=0)
    return pusat_baru

def kmeans(gambar, k, iterasi_maks=100):
    pusat = inisialisasi_pusat(gambar, k)
    for _ in range(iterasi_maks):
        klaster = tetapkan_klaster(gambar, pusat)
        pusat_baru = perbarui_pusat(gambar, klaster, k)
        if np.all(pusat == pusat_baru):
            break
        pusat = pusat_baru
    return klaster, pusat

def segmentasi_gambar(gambar, k, iterasi_maks=100):
    klaster, pusat = kmeans(gambar, k, iterasi_maks)
    gambar_tersegmentasi = pusat[klaster].reshape(gambar.shape)
    return gambar_tersegmentasi.astype(np.uint8)

def main():
    # Memuat gambar
    image_path = 'img/image2.jpg'  # Ganti dengan path gambar Anda
    gambar = cv2.imread(image_path)
    if gambar is None:
        raise FileNotFoundError(f"Tidak dapat memuat gambar dari path: {image_path}")
    gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

    # Parameter
    k = 6  # Jumlah cluster
    iterasi_maks = 100  # Jumlah iterasi maksimum

    # Segmentasikan gambar
    gambar_tersegmentasi = segmentasi_gambar(gambar, k, iterasi_maks)

    # Tampilkan gambar asli dan gambar yang tersegmentasi
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Gambar Asli')
    plt.imshow(gambar)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Gambar Tersegmentasi dengan {k} Klaster')
    plt.imshow(gambar_tersegmentasi)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
