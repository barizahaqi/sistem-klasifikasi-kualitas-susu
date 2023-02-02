# Laporan Proyek Machine Learning - Bariza Haqi

## Domain Proyek

![alt text](http://asset-a.grid.id/crop/0x0:0x0/750x500/photo/intisarifoto/original/5741_benarkah-kualitas-susu-segar-lebih-baik-dari-susu-bubuk.jpg)

Domain yang dipilih untuk proyek machine learning ini adalah Kesehatan, dengan judul Klasifikasi Kualitas Susu

- Latar Belakang
Susu merupakan minuman sumber protein hewani yang sangat penting bagi kesehatan manusia. [Kualitas susu sapi merupakan hal yang sangat penting](https://www.nestle.co.id/kisah/susu-sapi-berkualitas-untuk-konsumsi-sehari-hari). Susu sapi berkualitas selain mengandung protein juga mengandung vitamin dan mineral yang dapat membantu menjaga kesehatan tubuh. Manfaat dari susu sapi berkualitas sendiri melalui nutrisi-nutrisi yang terdapat di dalamnya, antara lain membantu perkembangan dan pemeliharaan tulang dan otot, meningkatkan kesehatan otak seperti meningkatkan daya ingat dan ketajaman, serta meningkatkan metabolisme tubuh. Untuk itu diperlukan cara untuk menentukan kualiatas susu yang baik dengan cepat. Salah satu cara untuk menentukan kualitas susu adalah dengan menggunakan machine learning dengan metode klasifikasi dari data-data observasi yang didapat. Dengan ini, produsen susu dapat menjual produknya dengan kualiatas yang terbaik.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut merupakan rincian masalah dari proyek ini:
- Bagaimana membuat model machine learning yang dapat mengklasifikasikan kualitas susu dengan baik?
- Model machine learning apa yang memiliki akurasi paling baik?

### Goals

Tujuan dari proyek ini adalah:
- Membuat model machine learning yang dapat menentukan kualitas susu sehingga dapat dikembangkan dan digunakan oleh produsen susu
- Membandingkan beberapa model klasifikasi sehingga ditemukan akurasi model yang paling baik untuk mengklasifikasikan kualitas susu


### Solution statements
Untuk mencapai tujuan tersebut, dalam proyek ini akan dibuat beberapa model yang berbeda untuk dibandingkan yaitu:
- K-Nearest Neighbor adalah algoritma yang relatif sederhana dengan menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).
- Random forest adalah salah satu algoritma supervised learning dan sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. Algoritma ini juga merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama sehingga tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 
- Algoritma Boosting adalah Algoritma yang bekerja dengan membangun model dari data latih kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah sehingga membentuk suatu model yang kuat . Algoritma boosting yang digunakan pada proyek ini adalah AdaBoost yang menggunakan metode adaptive boosting.

## Data Understanding
Dataset yang diguanakan pada proyek ini adalah [data prediksi kualitas susu](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). Dataset ini berisi 1059 sampel yang tediri dari variabel-variabel pada tabel di bawah.
![alt text](/img/data.png)
Berikut adalah penjelasan untuk tiap variabelnya.
- pH : merupakan pH susu yang berkisar antara 3 sampai 9.5.
- Temprature : merupakan suhu susu yang berkisar antara 34'C sampai 90'C.
- Taste : merupakan rasa susu dengan nilai 0 jika rasanya tidak enak dan nilai 1 jika rasanya enak
- Odor : merupakan bau susu dengan nilai 0 jika tidak memiliki bau yang tidak enak dan nilai 1 jika memiliki bau yang tidak enak.
- Fat : merupakan lemak susu dengan nilai 0 jika memiliki lemak rendah dan nilai 1 jika memiliki lemak tinggi.
- Turbidity : merupakan kekentalan susu dengan nilai 0 jika kekentalannya rendah dan nilai 1 jika kekentalannya tinggi.
- Colour : merupakan warna susu yang berkisar antara 240 sampai 255.
- Grade : merupakan kualitas susu yang terdiri dari low(rendah), medium(sedang) dan high(tinggi).

Untuk distribusi data numeriknya adalah sebagai berikut.
![alt text](/img/distribution.png)
Dari informasi diatas didapat beberapa kesimpulan yaitu semua diagram tidak terdistribusi merata.

- Pada Diagram ph sampel terbagi menjadi 4 bagian dengan kenaikan mendekati 0.1 tiap bagian.
- Pada diagram Temperature, distribusi miring ke kiri (left-skewed) dengan nilai tertinggi berada di sekitar nilai 45.
- Pada diagram colour sebagian besar sampel memiliki nilai colour dikisaran 255.

## Data Preparation
Teknik yang digunakan dalam penyiapan adalah sebagai berikut.
- Data Split dengan menggunakan train-test-split untuk membagi data menjadi data latih (train) dan data uji(test). Pada proyek ini, 80% dataset digunakan untuk melatih model dan 20% sisanya digunakan untuk mengevaluasi model
- Standarisasi dengan menggunakan StandardScaler untuk membuat fitur data memiliki skala relatif sama sehingga mudah diolah oleh algoritma. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.

## Modeling
Pada proses modeling akan dibuat sebuah models yang tediri dari variabel KNN, RandomForest dan Boosting. 
- Pada KNN, nilai k tetangga yang dipilih adalah 5 tetangga. Kelebihan dari KNN adalah mudah diterapkan, mudah beradaptasi dan memiliki sedikit hyperparameter.
Kekurangan dari KNN adalah tidak berfungsi dengan baik pada dataset berukuran besar, Kurang cocok untuk dimensi tinggi, perlu penskalaan fitur dan sensitif terhadap noise data, missing values dan outliers.
- Pada Random Forest, n_estimator atau jumlah pohon yang dipilih sebanyak 20, max kedalamannya adalah 16, nilai random number generatornya adalah 43 dan n-jobs nya bernilai -1 yang berarti semua proses berjalan secara paralel. Kelebihan Random Forest yaitu dapat mengatasi noise dan missing value serta dapat mengatasi data dalam jumlah yang besar. Kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data.
- Pada Algoritma Boosting, nilai learning-rate yang dipilih adalah 0.05 dengan random number generator bernilai 43. Kelebihan algoritma boosting adalah Kemudahan implementasi, Pengurangan bias dan Efisiensi komputasional. Kekurangan algoritma boosting adalah kelemahan terhadap data outlier dan implementasi waktu nyata

Dari ketiga ketiga algoritma tersebut diperoleh rata-rata nilai akurasi masing masing sebagai berikut

![alt text](/img/mean.png)

Dari nilai di atas dipilihlah algoritma random forest karena memiliki rata-rata akurasi paling besar di antara semua algoritma.

## Evaluation
Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi yaitu cross-val-score. Cross-val-score berasal dari formula K-Fold Cross Validation yang digunakan untuk menangani masalah overfitting.

Langkah-langkah kerja cross-val-score adalah sebagai berikut.
- Jumlah lipatan ditentukan, nilai default-nya adalah 5.
- Kumpulan data dibagi berdasarkan lipatan dimana setiap lipatan memiliki kumpulan data pengujian yang unik.
- Model dilatih dan diuji untuk setiap lipatan.
- Setiap lipatan mengembalikan metrik untuk data pengujiannya.
- Deviasi rata-rata dan standar dari metrik ini kemudian dapat dihitung untuk memberikan satu metrik yang digunakan untuk proses tersebut.

Dari metrik tersebut kemudian diambil rata-ratanya sehingga menghasilkan rata-rata nilai akurasi sebesar 0.976923076923077.
Dari hasil evaluasi dapat disimpulkan bahwa algoritma random forest sangat cocok digunakan sebagai model dalam proyek ini dengan akurasi sekitar 0.977
