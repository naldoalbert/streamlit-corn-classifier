import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="centered"
)

st.title("Deteksi Penyakit Daun Jagung🌽")
st.write("Identifikasi dini penyakit pada daun jagung Anda dengan mudah\nmenggunakan teknologi AI. Dapatkan hasil prediksi akurat dalam\nhitungan detik.")

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/model_klasifikasi_jagung_DenseNet.h5')

model = load_model()

LABELS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
DESCRIPTIONS = {
    'Blight': "Penyakit Blight (Hawar Daun) disebabkan oleh jamur. Ciri-cirinya adalah lesi lonjong berwarna coklat keabu-abuan yang muncul di daun, seringkali dengan batas yang jelas.",
    'Common_Rust': "Penyakit Karat Umum ditandai dengan pustula kecil berwarna oranye-coklat yang timbul, terutama pada permukaan atas daun. Dapat menyebar dengan cepat dalam kondisi lembab.",
    'Gray_Leaf_Spot': "Gray Leaf Spot menyebabkan lesi panjang, sempit, berwarna abu-abu kehijauan pada daun. Lesi ini seringkali berbentuk persegi panjang dan dibatasi oleh urat daun.",
    'Healthy': "Daun jagung Anda terlihat sehat! Terus jaga kondisi tanaman Anda dengan baik."
}

def predict_image(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = LABELS[np.argmax(score)]
    confidence = np.max(score) * 100
    return predicted_class, confidence

st.markdown("---")
st.info("Silakan upload gambar atau aktifkan kamera untuk mengambil foto daun jagung.")

# --- Upload dari Device ---
uploaded_file = st.file_uploader("📁 Upload gambar dari perangkat", type=["jpg", "jpeg", "png"])

# --- Kontrol Kamera ---
with st.expander("📷 Gunakan Kamera"):
    camera_enabled = st.checkbox("Aktifkan Kamera")

camera_image = None
if camera_enabled:
    camera_image = st.camera_input("Ambil gambar menggunakan kamera")

# --- Validasi Gambar ---
image_to_process = None

if uploaded_file and camera_image:
    st.warning("Silakan pilih salah satu: Upload atau Kamera.")
elif uploaded_file:
    try:
        image_bytes = uploaded_file.read()
        image_to_process = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membaca gambar upload: {e}")
elif camera_image:
    try:
        image_bytes = camera_image.read()
        image_to_process = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membaca gambar kamera: {e}")

# --- Prediksi Otomatis ---
if image_to_process:
    st.image(image_to_process, caption="📷 Pratinjau Gambar", use_container_width=True)
    st.write("🔍 Melakukan prediksi...")

    try:
        predicted_class, confidence = predict_image(image_to_process)
        st.subheader("📊 Hasil Prediksi")
        if predicted_class == "Healthy":
            st.success(f"✅ Daun ini **Sehat** dengan akurasi **{confidence:.2f}%**.")
        else:
            st.warning(f"⚠️ Terkena penyakit **{predicted_class}** dengan akurasi **{confidence:.2f}%**.")
        st.markdown("---")
        st.subheader("🩺 Deskripsi Penyakit")
        st.info(DESCRIPTIONS.get(predicted_class, "Deskripsi tidak tersedia."))
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")

st.markdown("---")
st.caption("Aplikasi ini dibuat untuk tujuan edukasi. Hasil prediksi tidak menggantikan diagnosis ahli.")
