# Klasifikasi Gambar menggunakan Transfer Learning (ResNet50)

## Deskripsi Proyek

Proyek ini bertujuan untuk melakukan klasifikasi gambar ke dalam 10 kelas berbeda dengan menggunakan model **ResNet50** sebagai backbone dan menambahkan beberapa layer fully connected. Dataset yang digunakan mengalami proses **undersampling** untuk mengatasi masalah data imbalance.

Model yang telah dilatih kemudian dikonversi ke format **TensorFlow Lite (TFLite)** untuk memungkinkan deployment pada perangkat dengan resource terbatas.

## Arsitektur Model
- Base Model: `ResNet50` (tanpa top layer)
- Layer Tambahan:
  - Global Average Pooling
  - Dense 512 unit (ReLU)
  - Batch Normalization
  - Dropout 0.3
  - Dense 10 unit (Softmax)

## Alur Data
1. **Data Preprocessing**:
   - Undersampling berdasarkan median jumlah gambar tiap label.
   - Membagi data menjadi:
     - **80%** data untuk training.
     - **10%** data untuk validation.
     - **10%** data untuk testing.

2. **Data Augmentation**:
   - Augmentasi dilakukan untuk data training menggunakan `ImageDataGenerator`.

3. **Training**:
   - Menggunakan `EarlyStopping` dan `ReduceLROnPlateau` untuk mencegah overfitting.
   - Hanya fine-tuning 30 layer terakhir dari ResNet50.

4. **Evaluasi**:
   - Menggunakan data validation saat training.
   - Menggunakan data testing untuk evaluasi akhir (confusion matrix, classification report).

5. **Export Model**:
   - Model disimpan dalam format `.h5`.
   - Model dikonversi ke format `.tflite` untuk inference.

## Cara Menjalankan Proyek

### 1. Setup Lingkungan
Install library yang diperlukan:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Proyek

#### Opsi 1: Melatih Model dari Awal
Jika ingin melakukan pelatihan model dari awal:
- Jalankan semua sel dari awal di file notebook.
- Model akan dilatih menggunakan data training dan validation.
- Model terbaik akan disimpan dan dapat dikonversi ke format TFLite.

#### Opsi 2: Langsung Menggunakan Model TFLite yang Sudah Ada
Jika ingin langsung melakukan prediksi menggunakan model TFLite (`model_resnet50.tflite`):

1. Pastikan file `model_resnet50.tflite` sudah tersedia.
2. Jalankan kode berikut di dalam notebook:

```python
# Load data test
x_test_batch, y_test_batch = next(iter(test_generator))

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_resnet50.tflite")
interpreter.allocate_tensors()

# Ambil detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lakukan prediksi
predictions = []
for img in x_test_batch:
    img = np.expand_dims(img, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions.append(output_data[0])

# Konversi hasil prediksi
predictions = np.array(predictions)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_batch, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Mapping label
class_indices = test_generator.class_indices
idx_to_label = {v: k for k, v in class_indices.items()}

# Visualisasi hasil prediksi
plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = np.clip(x_test_batch[i] * 255, 0, 255).astype("uint8")
    plt.imshow(img)
    plt.title(
        f"Pred: {idx_to_label[predicted_classes[i]]}\n"
        f"True: {idx_to_label[true_classes[i]]}\n"
        f"Conf: {confidence_scores[i] * 100:.2f}%"
    )
    plt.axis("off")
plt.tight_layout()
plt.show()
```

## Hasil Akhir
- Model mencapai **akurasi validasi sekitar 88-90%** setelah beberapa epoch.
- Model tidak mengalami overfitting berkat penggunaan **EarlyStopping** dan **ReduceLROnPlateau**.
- Evaluasi menggunakan **Confusion Matrix** dan **Classification Report** menunjukkan performa klasifikasi yang baik di sebagian besar kelas.

## Kesimpulan
- **Transfer Learning** menggunakan ResNet50 efektif untuk masalah klasifikasi gambar ini.
- **Undersampling** berhasil menyeimbangkan dataset tanpa membuat model underfit.
- **TFLite conversion** membuat model lebih ringan dan siap untuk deployment ke device edge.

## Saran untuk Pengembangan Selanjutnya
- Coba gunakan **oversampling** atau **SMOTE** untuk melihat perbandingan performa.
- Gunakan **learning rate scheduler** yang lebih adaptif seperti CosineAnnealing.
- Coba dengan arsitektur lain seperti **EfficientNet** untuk meningkatkan akurasi.
- Uji coba model pada data real-world di luar dataset training.
- Optimasi model TFLite lebih lanjut menggunakan **Post-Training Quantization** untuk memperkecil ukuran model.

---
