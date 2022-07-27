# Preprocessing data
# Akses dataset handwriting dari google drive
# kemudian unzip di google colab
import zipfile
from google.colab import drive
drive.mount('/content/drive/')
lz = '/content/drive/MyDrive/Dataset/Handwritting.zip'
zip_ref = zipfile.ZipFile(lz,'r')
zip_ref.extractall('/tmp/Handwritting')
zip_ref.close()

# import library yang dibutuhkan
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# buat path dasar untuk mengakses semua folder
path_dasar = '/tmp/Handwritting'
path_training = os.path.join(path_dasar,'train_v2/train') # path folder data training
path_validasi = os.path.join(path_dasar,'validation_v2/validation') # path folder data validasi
path_testing = os.path.join(path_dasar,'test_v2/test') # path folder data testing

# buat list kosong untuk training, validasi, dan testing
training = []
validasi = []
testing = []

# membuat list yang berisi nama file image yang akan digunakan sebagai data
# training, validasi, dan testing. nama file diurutkan sesuai abjad
file_training = list(sorted(os.listdir(path_training)))
file_validasi = list(sorted(os.listdir(path_validasi)))
file_testing = list(sorted(os.listdir(path_testing)))

# melakukan looping ke list nama file image
# kemudian menggabungkan nama file dengan path folder masing-masing
# sehingga dihasilkan full path ke tiap file image
# full path kemudian ditampung ke list kosong yang sebelumnya dibuat
for i in file_training:
    training.append(os.path.join(path_training,i))
for j in file_validasi:
    validasi.append(os.path.join(path_validasi,j))
for k in file_testing:
    testing.append(os.path.join(path_testing,k))
print(training[0:10])

# print jumlah data training, validasi, dan testing
# ini merupakan jumlah data original yang belum kita olah
print('Jumlah data training sebelum cleaning: ',len(os.listdir(path_training)))
print('Jumlah data validasi sebelum cleaning: ',len(os.listdir(path_validasi)))
print('Jumlah data testing sebelum cleaning: ',len(os.listdir(path_testing)))

# Cleaning data untuk menghapus data tanpa label / None
'''
Cleaning dilakukan karena pada saat proses memasukan data label kedalam tensorflow
pipeline akan terjadi error, karena  tensorflow tidak bisa menggabungkan data kosong dengan
data yang ada nilainya menjadi tensor.
Solusinya adalah dengan melakukan filter None value pada label data, kemudian juga
menghapus data image yang berpasangan dengan label tersebut
Hasilnya akan didapatkan data image dan label dalam  file csv tanpa ada nilai kosong.
Untuk mencari nilai None ini kita akan menggunakan fungsi pandas.isna()
'''
#import package pandas
import pandas as pd

# Load data label training, validasi, dan testing dalam file csv
data_label_train = pd.read_csv('/tmp/Handwritting/written_name_train_v2.csv') # membuka file csv berisi label data training 
data_label_validasi = pd.read_csv('/tmp/Handwritting/written_name_validation_v2.csv') # membuka file csv berisi label data validasi
data_label_test = pd.read_csv('/tmp/Handwritting/written_name_test_v2.csv') # membuka file csv berisi label data testing

# akses label data dengan memilih kolom 'IDENTITY'
label_train = data_label_train['IDENTITY'] # mengambil label dan memasukannya ke dalam variabel label_train
label_validasi = data_label_validasi['IDENTITY'] # mengambil label dan memasukannya ke dalam variabel label_validasi
label_test = data_label_test['IDENTITY'] # mengambil label dan memasukannya ke dalam variabel label_test

# membuat fungsi untuk cleaning data
def cleaning(path_image, label_image):
    del_label = [] # list kosong untuk menampung indeks label yang akan dihapus
    del_image = [] # list kosong untuk menampung path image yang akan dihapus
    for i in range(len(label_image)):
        if pd.isna(label_image[i]) == True: # jika label pada indeks i bernilai NaN
            del_label.append(i) # masukan indeks i kedalam list del_label
            del_image.append(path_image[i]) # masukan path image pada indeks i ke dalam list del_image
    for j in del_label:
        label_image.pop(j) # menghapus element list pada indeks yang sudah ditampung tadi
    for k in del_image:
        os.remove(k) # menghapus data image sesuai element list path image yang sudah ditampung

# menjalankan fungsi cleaning untuk data training, testing, dan validasi
cleaning(training, label_train)
cleaning(testing, label_test)
cleaning(validasi, label_validasi)

# membuat list kosong untuk menampung full image path training, validasi, dan testing 
# untuk data yang sudah dilakukan cleaning 
training_new = []
validasi_new = []
testing_new = []
file_training_new = list(sorted(os.listdir(path_training)))
file_validasi_new = list(sorted(os.listdir(path_validasi)))
file_testing_new = list(sorted(os.listdir(path_testing)))

for i in file_training_new:
    training_new.append(os.path.join(path_training,i))
for j in file_validasi_new:
    validasi_new.append(os.path.join(path_validasi,j))
for k in file_testing_new:
    testing_new.append(os.path.join(path_testing,k))

# print jumlah data training, validasi, dan testing setelah cleaning
print('Jumlah data training setelah cleaning: ',len(training_new),'jumlah label: ',len(label_train))
print('Jumlah data validasi setelah cleaning: ',len(validasi_new),'jumlah label: ',len(label_validasi))
print('Jumlah data testing setelah cleaning: ',len(testing_new),'jumlah label: ',len(label_test))

# hitung panjang karakter maksimum dan jumlah vocabuary pada data training

karakter = set() # buat set kosong untuk menampung vocabulary berupa karakter unik (tidak ada duplikasi) pada label data training
max_karakter = 0

# Loop ke tiap kata pada list label_train 
for label in label_train:
    label = str(label)
    # label = label.replace(' ', '')
    for char in label: # loop ke tiap huruf dalam kata
        karakter.add(char) #menambahkan huruf ke dalam set "karakter"
    max_karakter = max(max_karakter,len(label)) # hitung jumlah maksimal karakter tiap label

#ubah set "karakter" menjadi list dan diurutkan berdasarkan abjad
karakter = sorted(list(karakter))

print("Maximum length: ", max_karakter) # print jumlah karakter terbanyak pada label data training
print("Vocab size: ", len(karakter)) # print jumlah vocabulary yang didapat dari label data training
print('characters',karakter) # print vocabulary yang didapat dari label data training

# Membuat vocabulary dari karakter
# pada step ini akan dibuat keras layer untuk melakukan translate dari karakter ke angka indeks dan sebaliknya
#autotune untuk meminta tf. runtime data untuk menyetel nilai secara dinamis saat runtime
AUTOTUNE = tf.data.AUTOTUNE

# berfungsi untuk melakukan 'translate' tiap karakter menjadi indeks integer
char2num = StringLookup(vocabulary=list(karakter), mask_token=None)

# berfungsi untuk melakukan translate dari integer kembali ke karakter asli
num2char = StringLookup(
    vocabulary=char2num.get_vocabulary(), mask_token=None, invert=True
)

# Melakukan resize image tanpa distorsi
'''
Resize dilakukan untuk menyamakan ukuran image. hal ini dikarenakan image untuk handwriting
seringkali tidak beraturan, sehingga harus kita samakan ukurannya tanpa mengubah 
informasi yang ada didalam image tersebut.
'''
def resize_data_image(image, ukuran_image):
    l, t = ukuran_image
    image = tf.image.resize(image, size=(t, l), preserve_aspect_ratio=True)

    # Cek ukuran padding yang dibutuhkan
    pad_tinggi = t - tf.shape(image)[0]
    pad_lebar = l - tf.shape(image)[1]

    # Digunakan agar ukuran padding sama di kedua sisi.
    if pad_tinggi % 2 != 0:
        tinggi = pad_tinggi // 2
        pad_tinggi_atas = tinggi + 1
        pad_tinggi_bawah = tinggi
    else:
        pad_tinggi_atas = pad_tinggi_bawah = pad_tinggi // 2

    if pad_lebar % 2 != 0:
        lebar = pad_lebar // 2
        pad_lebar_kiri = lebar + 1
        pad_lebar_kanan = lebar
    else:
        pad_lebar_kiri = pad_lebar_kanan = pad_lebar // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_tinggi_atas, pad_tinggi_bawah],
            [pad_lebar_kiri, pad_lebar_kanan],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

batch_size = 64
padding_token = 99
lebar_image = 370
tinggi_image = 72

# membuat preproses image, memangggil beberapa fungsi tensorflow untuk melakukan preproses
# fungsi resize_data_image yang telah dibuat akan dipanggil di dalam fungsi ini untuk melakukan preproses
def preprosess_image(image_path, img_size=(lebar_image, tinggi_image)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = resize_data_image(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# membuat fungsi untuk mengubah label data menjadi vektor
# pada fungsi ini juga dipanggil layer char2num yang sebelumnya sudah kita buat
def memvektorkan_label(label):
    label = char2num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    panjang = tf.shape(label)[0]
    jumlah_pad = max_karakter - panjang
    label = tf.pad(label, paddings=[[0, jumlah_pad]], constant_values=padding_token)
    return label

# fungsi ini akan memanggil fungsi preprosess_image dan memvektorkan_label yang sudah dibuat sebelumnya
# kemudian output kedua fungsi akan dibuat dictionary dengan key dari data image yang sudah di preprocess
# serta value dari data label yang sudah ditranslate kedalam vektor
def process_label_image(image_path, label):
    image = preprosess_image(image_path)
    label = memvektorkan_label(label)
    return {"image": image, "label": label}

# fungsi ini berguna untuk membuat tensorflow pipeline dengan input dari dictionary
# output daru fungsi process_label_image sebelumnya
def persiapan_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_label_image, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


# pada tahap ini dibuat tensorflow pipeline untuk data training, validasi, dan testing
# dengan cara memanggil fungsi persiapan_dataset
# pada tahap ini data yang digunakan hanya sebagian, karena sudah saya coba menggunaan semua data
# tapi RAM colab overload sehingga proses error dan berhenti
# saran agar lebih baik bisa berlangganan colab pro untuk menambah kapasitas RAM
train_dataset = persiapan_dataset(training_new[0:20000], label_train[0:20000])
validation_dataset = persiapan_dataset(validasi_new[0:1000], label_validasi[0:1000])
test_dataset = persiapan_dataset(testing_new[0:1000], label_test[0:1000])

# membuat object CTC layer, 
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        ukuran_batch = tf.cast(tf.shape(y_true)[0], dtype="int64")
        panjang_input = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        panjang_label = tf.cast(tf.shape(y_true)[1], dtype="int64")

        panjang_input = panjang_input * tf.ones(shape=(ukuran_batch, 1), dtype="int64")
        panjang_label = panjang_label * tf.ones(shape=(ukuran_batch, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, panjang_input, panjang_label)
        self.add_loss(loss)

        return y_pred


def buat_model():
    # membuat input data untuk image
    input_image = keras.Input(shape=(lebar_image, tinggi_image, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # membuat layer konvolusi pertama
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_image)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Membuat layer konvolusi kedua
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Output dari layer konvolusi selanjutnya dilakukan reshape
    # sebelum menjadi input bagi model RNN
    new_shape = ((lebar_image // 4), (tinggi_image // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # Membuat layer RNN
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 digunakan untuk memperhitungkan dua token khusus yang diperkenalkan CTC loss
    x = keras.layers.Dense(
        len(char2num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Menambahkan CTC layer untuk menghitung loss pada setiap step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # mendefinisikan model
    model = keras.models.Model(
        inputs=[input_image, labels], outputs=output, name="Pengenal_Tulisan_Tangan"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # compile model
    model.compile(optimizer=opt)
    return model


# Get the model.
model = buat_model()
model.summary()


## Training model
epochs = 60  

# memanggil fungsi untuk membuat model
model = buat_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)


# melakukan training model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

# fungsi untuk melakukan decode output yang dihasilkan.
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # menggunakan greedy search
    hasil = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_karakter
    ]
    # melakukan iterasi kedalam hasil dan mendapatkan text nya kembali
    output_text = []
    for res in hasil:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num2char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Cek hasil prediksi
for batch in test_dataset.take(1):
    batch_images = batch["image"]
    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    for i in range(16,32):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediksi: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

plt.show()



