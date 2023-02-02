# Laporan Proyek Machine Learning - Bariza Haqi

## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Kesehatan**, dengan judul **Klasifikasi Kualitas Susu**

Latar Belakang
![alt text](http://asset-a.grid.id/crop/0x0:0x0/750x500/photo/intisarifoto/original/5741_benarkah-kualitas-susu-segar-lebih-baik-dari-susu-bubuk.jpg)

Susu merupakan minuman sumber protein hewani yang sangat penting bagi kesehatan manusia. Kualitas susu sapi merupakan hal yang sangat penting [[1](https://repository.ipb.ac.id/handle/123456789/26133)]. Susu sapi berkualitas selain mengandung protein juga mengandung vitamin dan mineral yang dapat membantu menjaga kesehatan tubuh. Manfaat dari susu sapi berkualitas sendiri melalui nutrisi-nutrisi yang terdapat di dalamnya, antara lain membantu perkembangan dan pemeliharaan tulang dan otot, meningkatkan kesehatan otak seperti meningkatkan daya ingat dan ketajaman, serta meningkatkan metabolisme tubuh. Untuk itu diperlukan cara untuk menentukan kualiatas susu yang baik dengan cepat. Salah satu cara untuk menentukan kualitas susu adalah dengan menggunakan machine learning dengan metode klasifikasi dari data-data observasi yang didapat. Dengan ini, produsen susu dapat menjual produknya dengan kualiatas yang terbaik.

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
- *K-Nearest Neighbor* adalah algoritma yang relatif sederhana dengan menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. *KNN* bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).
- *Random forest* adalah salah satu algoritma *supervised learning* dan sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. Algoritma ini juga merupakan salah satu model *machine learning* yang termasuk ke dalam kategori *ensemble (group) learning* yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama sehingga tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model *ensemble*, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model *ensemble* ini digabungkan untuk membuat prediksi akhir. 
- Algoritma *Boosting* adalah Algoritma yang bekerja dengan membangun model dari data latih kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah sehingga membentuk suatu model yang kuat . Algoritma *Boosting* yang digunakan pada proyek ini adalah AdaBoost yang menggunakan metode *Adaptive Boosting*.

## Data Understanding
Dataset yang diguanakan pada proyek ini adalah [data prediksi kualitas susu](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). Lima sampel teratas diperlihatkan pada Tabel 1. 

Tabel 1. Lima sampel teratas pada dataset *milkquality*

| PH  | Temprature | Taste | Odor | Fat | Turbidity | Colour | Grade  |
|-----|------------|-------|------|-----|-----------|--------|--------|
| 6.6 | 35         | 1     | 0    | 1   | 0         | 254    | high   |
| 6.6 | 36         | 0     | 1    | 0   | 1         | 253    | high   |
| 8.5 | 70         | 1     | 1    | 1   | 1         | 246    | low    |
| 9.5 | 34         | 1     | 1    | 0   | 1         | 255    | low    |
| 6.6 | 37         | 0     | 0    | 0   | 0         | 255    | medium |

Dataset ini berisi 1059 sampel yang tediri dari variabel-variabel berikut.
- *pH* : merupakan pH susu yang berkisar antara 3 sampai 9.5.
- *Temprature* : merupakan suhu susu yang berkisar antara 34'C sampai 90'C.
- *Taste* : merupakan rasa susu dengan nilai 0 jika rasanya tidak enak dan nilai 1 jika rasanya enak
- *Odor* : merupakan bau susu dengan nilai 0 jika tidak memiliki bau yang tidak enak dan nilai 1 jika memiliki bau yang tidak enak.
- *Fat* : merupakan lemak susu dengan nilai 0 jika memiliki lemak rendah dan nilai 1 jika memiliki lemak tinggi.
- *Turbidity* : merupakan kekentalan susu dengan nilai 0 jika kekentalannya rendah dan nilai 1 jika kekentalannya tinggi.
- *Colour* : merupakan warna susu yang berkisar antara 240 sampai 255.
- *Grade* : merupakan kualitas susu yang terdiri dari *low* (rendah), *medium* (sedang) dan *high* (tinggi).

Pada 1059 sampel di dataset *milkquality*, terdapat beberapa outlier pada data numeriknya yang diperlihatkan pada Gambar 2, 3 dan 4.

![alt text](/img/outlier_ph.png) 

Gambar 2. Outlier pada variabel pH

![alt text](/img/outlier_temperature.png) 

Gambar 3. Outlier pada variabel Temperature

![alt text](/img/outlier_colour.png) 

Gambar 4. Outlier pada variabel Colour

Pada diagram-diagram di atas terdapat outlier pada kolom pH, Temperature dan Colour sehingga harus dihilangkan terlebih dahulu dengan metode IQR yaitu menghilangkan batas bawah yang bernilai Q1-1.5\*IQR dan batas atas yang bernilai Q1+1.5\*IQR.
Berikut adalah source kodenya.

```sh
Q1 = milk.quantile(0.25)
Q3 = milk.quantile(0.75)
IQR = Q3-Q1
milk = milk[~((milk<(Q1-1.5*IQR))|(milk>(Q3+1.5*IQR))).any(axis=1)])
```
Setelah melukakan drop outliers, jumlah sampel yang ada pada dataset berkurang menjadi 648 sampel.

Distribusi numerik pada dataset diperlihatkan pada Gambar 5.
![alt text](/img/distribution.png) 
Gambar 5. Distribusi numerik pada variabel pH, Temperature dan Colour

Dari informasi diatas didapat beberapa kesimpulan yaitu:

- Semua diagram tidak terdistribusi merata.
- Pada Diagram *pH*, sampel terbagi menjadi 4 bagian dengan kenaikan mendekati 0.1 tiap bagian.
- Pada diagram *Temperature*, distribusi miring ke kiri (*left-skewed*) dengan nilai tertinggi berada di sekitar nilai 45.
- Pada diagram *colour* sebagian besar sampel memiliki nilai *colour* dikisaran 255.

## Data Preparation
Teknik yang digunakan dalam penyiapan adalah sebagai berikut.
- *Data Split* dengan menggunakan *train_test_split* untuk membagi data menjadi data latih (*train*) dan data uji(*test*). Pada proyek ini, 80% dataset digunakan untuk melatih model dan 20% sisanya digunakan untuk mengevaluasi model
- Standarisasi dengan menggunakan *StandardScaler* untuk membuat fitur data memiliki skala relatif sama sehingga mudah diolah oleh algoritma. *StandardScaler* melakukan proses standarisasi fitur dengan mengurangkan *mean* (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.

## Modeling
Pada proses modeling akan dibuat sebuah models yang tediri dari variabel *KNN*, *RandomForest* dan *Boosting*. 

Pada *KNN*, nilai *n_neighbors* yang dipilih adalah 5 tetangga. Berikut adalah *source* kodenya.

```sh
# Lakukan analisis menggunakan K-Nearest Nei
ghbor
knn = KNeighborsClassifier(n_neighbors=5)
# Latih model
knn.fit(X_train, y_train)
models.loc['score', 'KNN'] = cross_val_score(knn, X_train, y_train, scoring="accuracy", cv= 5).mean()
```

Kelebihan dari *KNN* adalah mudah diterapkan, mudah beradaptasi dan memiliki sedikit *hyperparameter*.
Kekurangan dari *KNN* adalah tidak berfungsi dengan baik pada dataset berukuran besar, Kurang cocok untuk dimensi tinggi, perlu penskalaan fitur dan sensitif terhadap noise data,* missing values* dan *outliers*.

Pada*Random Forest*, *n_estimator* atau jumlah pohon yang dipilih sebanyak 20, maks kedalamannya adalah 16, nilai *random number generator*-nya adalah 43 dan *n_jobs* nya bernilai -1 yang berarti semua proses berjalan secara paralel. Berikut adalah *source* kodenya.

```sh
# Lakukan analisis menggunakan Random Forest
rf = RandomForestClassifier(n_estimators=20, max_depth=16, random_state=43, n_jobs=-1)

# Latih model
rf.fit(X_train, y_train)
models.loc['score','RandomForest'] = cross_val_score(rf, X_train, y_train, scoring="accuracy", cv= 5).mean()
```

Kelebihan *Random Forest* yaitu dapat mengatasi *noise* dan *missing value* serta dapat mengatasi data dalam jumlah yang besar. Kekurangan pada algoritma *Random Forest* yaitu interpretasi yang sulit dan membutuhkan *tuning* model yang tepat untuk data.

Pada Algoritma *Boosting*, nilai *learning_rate* yang dipilih adalah 0.05 dengan *random number generator* bernilai 43. Berikut adalah *source* kodenya.

```sh
# Lakukan analisis menggunakan AdaBoost
boosting = AdaBoostClassifier(learning_rate=0.05, random_state=43)

# Latih model                            
boosting.fit(X_train, y_train)
models.loc['score','Boosting'] = cross_val_score(boosting, X_train, y_train, scoring="accuracy", cv= 5).mean()
```

Kelebihan Algoritma *Boosting* adalah kemudahan implementasi, pengurangan bias dan efisiensi komputasional. Kekurangan Algoritma *Boosting* adalah kelemahan terhadap data *outlier* dan implementasi waktu nyata

Pelatihan ketiga metode diatas menggunakan metrik *Cross validation score* dengan memasukan model, data *train* dan *cross validation* bernilai 5 untuk mendapatkan rata-rata akurasinya.
Dari ketiga algoritma tersebut diperoleh rata-rata nilai akurasi masing-masing yang ditunjukkan pada Tabel 2.

Tabel 2. Hasil rata-rata akurasi Algoritma KNN, Random Forest dan Boosting dengan cv = 5

|       | KNN      | RandomForest | Boosting |
|-------|----------|--------------|----------|
| score | 0.980657 | 0.994212     | 0.990246 |

Dari nilai pada tabel di atas dipilihlah Algoritma *Random Forest* karena memiliki rata-rata akurasi paling besar di antara semua algoritma.

## Evaluation
Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi yaitu *cross_val_score*. *Cross validation score* berasal dari formula K-Fold Cross Validation yang digunakan untuk menangani masalah overfitting [[2](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)].

Langkah-langkah kerja *cross validation score* adalah sebagai berikut.
- Menentukan jumlah lipatan, nilai *default*-nya adalah 5.
- Kumpulan data dibagi berdasarkan lipatan dimana setiap lipatan memiliki kumpulan data pengujian yang unik.
- Model dilatih dan diuji untuk setiap lipatan.
- Setiap lipatan mengembalikan metrik untuk data pengujiannya.
- Deviasi rata-rata dan standar dari metrik ini kemudian dapat dihitung untuk memberikan satu metrik yang digunakan untuk proses tersebut.

Berikut adalah *source* kode proses evaluasi model.

```sh
# Evaluasi akurasi model
result_accuracy = cross_val_score(rf, X_test, y_test, scoring="accuracy", cv= 5).mean()
```

Dari metrik tersebut kemudian diambil rata-ratanya sehingga menghasilkan rata-rata nilai akurasi sebesar 0.976923076923077.

## Kesimpulan
Dari hasil evaluasi dapat disimpulkan bahwa Algoritma *Random Forest* sangat cocok digunakan sebagai model dalam proyek ini. Hasil akurasi dan akurasi validasi yang didapatkan sangat besar. Dengan hasil ini, diharapkan model *machine learning* yang telah dibuat ini dapat bermanfaat untuk dikembangkan dan digunakan pada pengklasifikasian kualitas susu.

Referensi
[[1](https://repository.ipb.ac.id/handle/123456789/26133)] Sinaga, Kurniawan. (2000). "Kualitas Susu Sapi Berdasarkan Kepemilikan di Kawasan Usaha Peternakan Cibungbulang, Kabupaten Bogor". https://repository.ipb.ac.id/handle/123456789/26133

[[2](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)] Shaikh, Rahil. (2018). "Cross Validation Explained: Evaluating estimator performance". https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85