import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import joblib
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fungsi untuk membaca dataset (dengan cache)
@st.cache_data
def load_data():
    df = pd.read_csv("weatherHistory.csv")
    return df

# Fungsi untuk menambahkan CSS agar tampilan tidak plain
def add_custom_css():
    st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: Arial, sans-serif;
    }
    .main {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #333333;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        padding: 5px 20px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk menampilkan halaman Home
import streamlit as st

def show_home():
    # Judul Halaman
    st.title("🌤️ Selamat Datang di Aplikasi Prediksi Cuaca!")
    
    # Gambar Utama
    st.image("weather.jpg", caption="Prediksi Cuaca Berdasarkan Data", use_container_width=True)
    
    # Deskripsi Aplikasi dengan Paragraf Rata Kanan-Kiri
    st.markdown("""
    <div style="text-align: justify;">
    Aplikasi ini dirancang untuk membantu Anda dalam menganalisis dan memprediksi suhu nyata berdasarkan data cuaca yang tersedia. 
    Dengan memanfaatkan <strong>Data Science</strong> dan menggunakan metode <strong>Regresi Linear</strong>, aplikasi ini bertujuan untuk:
    </div>
    """, unsafe_allow_html=True)

    # Fitur Utama
    st.markdown("""
    <div style="text-align: justify;">
    <h3>🌟 Fitur Utama:</h3>
    <ul>
        <li><strong>Dataset:</strong> Akses dan eksplorasi dataset cuaca untuk analisis mendalam, termasuk suhu, kelembapan, dan kecepatan angin.</li>
        <li><strong>Visualisasi:</strong> Grafik interaktif yang membantu Anda memahami pola data cuaca, tren, dan pergerakannya dari waktu ke waktu.</li>
        <li><strong>Prediksi:</strong> Model berbasis <strong>Regresi Linear</strong> untuk memperkirakan suhu nyata, membantu perencanaan aktivitas luar ruangan.</li>
        <li><strong>Modeling:</strong> Penjelasan mendalam tentang metode, langkah-langkah, dan proses pengembangan model prediksi.</li>
        <li><strong>About:</strong> Informasi tentang pengembang aplikasi, tujuan pembuatan, dan manfaat aplikasi ini untuk pengguna.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Informasi Sidebar
    st.markdown("""
    <div style="text-align: justify;">
    ---
    Jika ingin menjelajahi lebih lanjut, silakan gunakan <strong>sidebar</strong> di sebelah kiri untuk memilih menu: 
    <strong>Home</strong>, <strong>Dataset</strong>, <strong>Visualisasi</strong>, <strong>Prediksi</strong>, <strong>Modeling</strong>, atau <strong>About</strong>. 
    Dengan fitur ini, Anda dapat dengan mudah mengeksplorasi semua fungsi yang tersedia dalam aplikasi.
    </div>
    """, unsafe_allow_html=True)

# Fungsi untuk menampilkan penjelasan modeling
def show_modeling():
    st.title("🧑‍🔬 **Modeling ML dan Metode yang Digunakan**")
    st.markdown(""" 
    <div style="text-align: justify;">
    Tujuan utama dari pemodelan ini adalah untuk memprediksi <strong>suhu yang nyaman bagi manusia</strong> berdasarkan data cuaca yang tersedia. Kami menggunakan metode <strong>regresi linier</strong>, yang merupakan teknik statistik untuk memodelkan hubungan antara variabel independen (seperti suhu, kelembapan, dan kecepatan angin) dengan variabel dependen (<strong>suhu nyata</strong> atau <em>apparent temperature</em>). Model ini diharapkan dapat memberikan prediksi yang cukup akurat, berguna dalam perencanaan aktivitas luar ruangan, pertanian, dan analisis cuaca secara lebih luas.

    ### 🔍 **Metode yang Digunakan:**
    Untuk membangun model prediksi suhu nyaman, kami menggunakan **regresi linier** sebagai algoritma utama. Regresi linier adalah metode yang sangat cocok untuk memodelkan hubungan linier antara variabel-variabel yang ada. Dalam hal ini, kami memodelkan hubungan antara tiga variabel input (suhu, kelembapan, dan kecepatan angin) dengan output berupa **Apparent Temperature (suhu nyata)**, yang menggambarkan suhu yang dirasakan oleh manusia.

    Proses ini dimulai dengan **analisis data** untuk memastikan dataset yang digunakan berkualitas. Kemudian, data dibagi menjadi dua bagian: **training set** (untuk melatih model) dan **testing set** (untuk menguji keakuratan model).

    ### 🛠️ **Langkah-langkah Pemodelan:**
    1. **Pra-pemrosesan Data:**
       - Data yang digunakan berasal dari sumber terpercaya seperti Kaggle, yang mencakup informasi penting terkait cuaca (suhu, kelembapan, kecepatan angin, dll).
       - Data yang kosong atau tidak relevan dihapus, dan tipe data disesuaikan. Statistik deskriptif digunakan untuk memahami pola dalam data.

    2. **Pemisahan Dataset:**
       - Dataset dibagi menjadi dua bagian menggunakan fungsi `train_test_split` dari pustaka `sklearn`. Data yang digunakan untuk melatih model terdiri dari **80%** dataset, sementara **20%** sisanya digunakan untuk menguji model.

    3. **Pembuatan Model Regresi Linier:**
       - Model **Linear Regression** dari pustaka `sklearn` digunakan untuk membangun model prediksi. Model ini mempelajari hubungan antara suhu, kelembapan, dan kecepatan angin dengan suhu nyata (Apparent Temperature).
       - Model dilatih menggunakan training set dan diuji dengan testing set untuk mengevaluasi kinerjanya.

    4. **Evaluasi Model:**
       - Setelah model dilatih, kami menggunakan beberapa metrik untuk mengevaluasi kinerjanya:
         - **Mean Absolute Error (MAE):** Mengukur rata-rata kesalahan absolut antara nilai yang diprediksi dan nilai aktual.
         - **Mean Squared Error (MSE):** Mengukur rata-rata kuadrat kesalahan, memberikan penalti lebih besar pada kesalahan besar.
         - **Root Mean Squared Error (RMSE):** Akar dari MSE, memberikan gambaran yang lebih jelas tentang seberapa besar kesalahan dalam satuan yang sama dengan data asli.
       
    5. **Pengujian Model dengan Data Baru:**
       - Setelah model diuji dengan testing set, kami menguji model dengan data cuaca yang belum pernah dilihat oleh model sebelumnya. Ini untuk memastikan model tidak mengalami **overfitting**.

    6. **Implementasi dan Prediksi:**
       - Setelah model dievaluasi dan diuji, aplikasi ini siap digunakan. Pengguna dapat memasukkan nilai suhu, kelembapan, dan kecepatan angin melalui antarmuka yang disediakan, dan model akan memberikan prediksi suhu yang nyaman atau suhu nyata berdasarkan input tersebut.

    ### 📊 **Evaluasi Model dan Hasil:**
    Model regresi linier ini menunjukkan hasil yang memuaskan dengan metrik evaluasi yang terkontrol. Pengujian pada data testing set menunjukkan bahwa model ini dapat memberikan prediksi yang akurat dengan kesalahan yang minim. Oleh karena itu, aplikasi ini dapat digunakan untuk memberikan perkiraan suhu yang nyaman dengan tingkat akurasi yang baik.
    
    **Kesimpulan:** Model ini diharapkan dapat membantu perencanaan aktivitas di luar ruangan, menentukan waktu terbaik untuk kegiatan pertanian, serta memprediksi kondisi cuaca yang lebih tepat. 

    </div>
    """, unsafe_allow_html=True)


# Fungsi untuk menampilkan informasi pembuat aplikasi

def show_about():
    st.title("🌍 Tentang Aplikasi")
    
    # Deskripsi Aplikasi
    st.markdown("""
    Aplikasi ini dirancang untuk memberikan kemudahan kepada pengguna dalam memprediksi suhu nyata berdasarkan data cuaca terkini.  
    Dengan memanfaatkan teknologi *data science* dan algoritma **regresi linear**, aplikasi ini menawarkan prediksi suhu yang akurat dan dapat diandalkan.
    
    ### 📋 Manfaat Aplikasi:
    - Membantu perencanaan aktivitas luar ruangan seperti olahraga, perjalanan, atau acara lainnya. 🏃
    - Mendukung pengambilan keputusan di bidang pertanian dengan memberikan prakiraan cuaca yang lebih tepat. 🌱
    - Memberikan peringatan cuaca yang relevan untuk keselamatan pengguna. 🚨
    
    Tujuan utama kami adalah menciptakan solusi praktis dan mudah diakses untuk memperkirakan kondisi cuaca berdasarkan data yang tersedia.
    """)
    
    # Informasi Pengembang
    st.markdown("""
    ### 👩‍💻 Pengembang:
    - Anindya Putri Nariswari (233307065)  
    - Nazala Syifa Julieta (233307085)  
    - Nindi Nurrahma Julitasari (233307086)  
    
    ### ✉️ Kontak:
    - **Email**: [kelompok8@gmail.com](mailto:kelompok8@gmail.com)  
    - **Telepon**: [0812-3456-7890](tel:+6281234567890)  
    
    ---
    <div style="text-align: center; color: grey;">
        <small>&copy; 2024 Aplikasi Prediksi Cuaca. Semua Hak Cipta Dilindungi.</small>
    </div>
    """, unsafe_allow_html=True)



# Fungsi untuk menampilkan dataset
def show_dataset(df):
    st.title("Dataset Cuaca")
    st.write("Data Cuaca")
    st.dataframe(df)

    st.write("Missing Data")
    st.write(df.isnull().sum())

    st.write("Statistik Deskriptif")
    st.write(df.describe())

    st.write("Tipe Data")
    st.write(df.dtypes)

# Fungsi untuk menampilkan visualisasi data
def show_visualizations(df):
    st.title("Visualisasi Data Cuaca")
    
    # Grafik distribusi suhu
    st.write("Grafik Distribusi Suhu")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Temperature (C)'], ax=ax, kde=True, color="skyblue")
    ax.set_title('Temperature Distribution Plot')
    st.pyplot(fig)

    # Grafik distribusi jenis cuaca
    st.write("Distribusi Jenis Cuaca")
    weather_counts = df['Summary'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    weather_counts.plot(kind="bar", ax=ax)
    ax.set_title("Weather Summary Distribution")
    ax.set_xlabel("Summary")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Word Cloud untuk jenis cuaca
    st.write("Word Cloud Jenis Cuaca")
    weather_summaries = ' '.join(df['Summary'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(weather_summaries)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Weather Summary Word Cloud')
    st.pyplot(fig)

    # Scatter plot antara suhu dan kelembapan
    st.write("Scatter Plot: Temperature vs Humidity")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['Temperature (C)'], df['Humidity'], color='orange')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Humidity')
    ax.set_title('Temperature vs Humidity')
    st.pyplot(fig)

# Fungsi untuk menampilkan prediksi suhu nyata
def show_predictions(df):
    st.title("🌡️ Prediksi Suhu Nyata Berdasarkan Data Cuaca")

    # Memisahkan fitur dan target
    X = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
    y = df['Apparent Temperature (C)']

    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cek jika model sudah ada, load model; jika tidak, train model baru
    model_path = 'weather_model.sav'
    try:
        model_regresi = joblib.load(model_path)
    except:
        st.warning("🔄 Melatih model baru karena model sebelumnya tidak ditemukan.")
        model_regresi = LinearRegression()
        model_regresi.fit(X_train, y_train)
        joblib.dump(model_regresi, model_path)
        st.success("📁 Model baru berhasil disimpan!")

    # Input dari pengguna
    st.subheader("📝 Masukkan Data untuk Prediksi:")
    temperature = st.slider('🌡️ Temperature (C)', min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
    humidity = st.slider('💧 Humidity', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    wind_speed = st.slider('🌬️ Wind Speed (km/h)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    # Tombol Prediksi
    if st.button('🔮 Prediksi'):
        try:
            prediction = model_regresi.predict([[temperature, humidity, wind_speed]])
            predicted_temperature = float(prediction[0])
            st.success(f"🎯 Suhu nyata yang diprediksi adalah: **{predicted_temperature:.2f}°C**")

            # Menambahkan penjelasan tentang hasil prediksi
            if predicted_temperature < 5:
                st.info("🥶 Cuaca sangat dingin diprediksi. Pastikan untuk mengenakan pakaian hangat yang tebal!")
            elif predicted_temperature >= 5 and predicted_temperature < 15:
                st.info("🌧️ Cuaca dingin diprediksi. Sebaiknya pakai pakaian hangat!")
            elif predicted_temperature >= 15 and predicted_temperature <= 25:
                st.info("☀️ Cuaca nyaman diprediksi. Cocok untuk kegiatan luar ruangan!")
            elif predicted_temperature > 25 and predicted_temperature <= 35:
                st.info("🔥 Cuaca panas diprediksi. Pastikan untuk banyak minum dan pakai pelindung matahari!")
            else:
                st.info("🌞 Cuaca sangat panas diprediksi. Hindari beraktivitas di luar ruangan pada siang hari!")

        except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memprediksi: {e}")

    # Evaluasi Model
    st.subheader("📊 Evaluasi Model:")
    model_regresi_pred = model_regresi.predict(X_test)
    mae = mean_absolute_error(y_test, model_regresi_pred)
    mse = mean_squared_error(y_test, model_regresi_pred)
    rmse = np.sqrt(mse)

    st.write(f"- **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"- **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.2f}")


# Main program
def main():
    add_custom_css()

    st.sidebar.title("Navigasi")
    menu = st.sidebar.selectbox(
        "Pilih Menu",
        ["Home", "Dataset", "Visualisasi", "Prediksi", "Modeling", "About"]
    )

    df = load_data()

    if menu == "Home":
        show_home()
    elif menu == "Dataset":
        show_dataset(df)
    elif menu == "Visualisasi":
        show_visualizations(df)
    elif menu == "Prediksi":
        show_predictions(df)
    elif menu == "Modeling":
        show_modeling()
    elif menu == "About":
        show_about()

if __name__ == "__main__":
    main()
