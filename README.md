# 👨‍🍳 NePisirsem_Chatbot: Akıllı Türk Mutfağı Asistanı (RAG Chatbot)

Bu proje, **Akbank GenAI Bootcamp** kapsamında, LangChain ve Google Gemini teknolojileri kullanılarak geliştirilmiş, **Retrieval Augmented Generation (RAG)** temelli bir chatbot uygulamasıdır. Chatbot, spesifik bir Türk Mutfağı veri seti üzerinden akıllı tarif önerileri sunar ve web arayüzü ile hizmet verir.

---

## 🎯 Projenin Amacı

Temel amaç, Büyük Dil Modellerinin (LLM) en büyük problemi olan **halüsinasyon (yanıltıcı bilgi üretme)** riskini ortadan kaldırmaktır. Proje, Gemini modelinin genel zekasını, kısıtlı bir Türk Mutfağı tarifleri veri setiyle birleştirerek, kullanıcının elindeki malzemeye, miktara ve mutfak kategorisine göre **doğru, güvenilir ve bağlamsal** tarif önerileri sunar.

## 📝 Veri Seti Hakkında Bilgi

### Veri Kaynağı

Projede, **50 popüler Türk Mutfağı yemeği** içeren özel olarak hazırlanmış yapısal bir PDF dosyası (`yemektarifleri.pdf`) kullanılmıştır.

### Hazırlanış Metodolojisi

Veri setindeki her tarif, RAG sisteminin gelişmiş filtreleme yapabilmesi için yapılandırılmıştır:
1.  **Kategorizasyon:** Her tarif, **Kahvaltılık, Ara Sıcak, Ana Yemek, Çorba** veya **Tatlı** olarak etiketlenmiştir.
2.  **Miktar Açıklığı:** Malzemeler listesinde **net adet/gramaj** (Örn: "2 yumurta") açıkça belirtilmiştir. Bu sayede chatbot, kullanıcının elindeki miktara göre öneri yapabilir.

---

## 🧪 Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, **LangChain** çatısı altında kurulan uçtan uca bir RAG zinciridir.

### Teknolojiler

| Bileşen | Teknoloji | Görev |
| :--- | :--- | :--- |
| **LLM (Generation)** | Google Gemini 2.5 Flash | Sorgulama ve Nihai Cevap Üretimi. |
| **Embedding Modeli** | Google `text-embedding-004` | Metinleri Vektörlere Dönüştürme. |
| **Vektör Veritabanı** | ChromaDB | Vektör Depolama ve Hızlı Arama (Retrieval). |
| **RAG Çatısı** | LangChain | Pipeline Yönetimi (`create_retrieval_chain`). |
| **Web Arayüzü** | Streamlit | Kullanıcı Dostu Arayüz Sunumu. |

### Elde Edilen Sonuçlar ve Gelişmiş Özellikler

* **Miktarsal Filtreleme:** Kullanıcının spesifik miktarlı sorularına, veri setindeki gereksinimleri aşmayan tarifleri önererek cevap verir.
* **Bağlamsal Takip:** Önceki konuşmayı hatırlayarak (Örn: "Köfte" $\rightarrow$ "Nasıl yaparım?") doğru tarifi sunar.
* **Güvenilir Yanıtlar:** `temperature=0.0` ayarı ve sıkı Prompt kuralları sayesinde, cevaplar **kesinlikle** veri setindeki bilgilerle sınırlı kalır.

---

## 🛠️ Kurulum

Projeyi lokal ortamınızda çalıştırmak için izlemeniz gereken adımlar:

### Ön Gereksinimler
* Python 3.10+
* Google Gemini API Anahtarı

### Adım 1: Kurulum ve API Anahtarı
1.  **Sanal Ortam Oluşturma ve Aktive Etme:**
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate  # Windows için
    # source venv/bin/activate # Linux/macOS için
    ```
2.  **Kütüphaneleri Yükleme:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **API Anahtarını Ayarlama:** Proje klasörüne `.env` adında bir dosya oluşturun  ve Gemini API key'inizi ekleyin
    ```env
    GEMINI_API_KEY="BURAYA_API_ANAHTARINIZI_YAZIN"
    ```

### Adım 2: Veritabanını Oluşturma 
Bu adım, `yemektarifleri.pdf` dosyasını okur, vektörlere dönüştürür ve **`chroma_db`** klasörünü oluşturur.
```bash
python data_loader.py
```
### Adım 3: Uygulamayı Başlatma

Sanal ortamınız aktifken aşağıdaki komutu kullanarak web arayüzünü başlatın:

```bash
python -m streamlit run app.py
```
Uygulama tarayıcınızda açılacaktır (http://localhost:8501)


---
## 🖥️ **Web Arayüzü & Kullanım Senaryoları**

| Senaryo | Test Sorgusu | Özellik |
|:--|:--|:--|
| **Miktara Göre Öneri** | `2 kabak ile ne yapabilirim?` | Miktarsal filtreleme |
| **Bağlamsal Sorgu** | (Önce “Köfte”, sonra “Nasıl yaparım?”) | Sohbet bağlamı takibi |

## 🌐 DEPLOY LİNKİ 
**[ ]**
