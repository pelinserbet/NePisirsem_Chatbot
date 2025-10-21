import os
from dotenv import load_dotenv

# Yeni ve doğru importlar
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader # Artık langchain_community'den alıyoruz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma # Artık langchain_community'den alıyoruz

# 1. API Anahtarını Yükle
# Bu, .env dosyanızdaki anahtarı kodun kullanabileceği ortama yükler.
load_dotenv()
# Eğer anahtar yüklenmezse hata verir.
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmadı. Lütfen .env dosyanızı kontrol edin.")

# 2. Sabit Değerler
PDF_PATH = "yemektarifleri.pdf"
CHROMA_DB_DIR = "./chroma_db"

def load_and_index_data():
    print(f"[{PDF_PATH}] dosyasını yüklüyor...")

    # A. Dokümanları Yükleme
    # pypdf kütüphanesi ile PDF dosyasını okur.
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # B. Metinleri Parçalama (Chunking)
    # Metni, bir tarifin malzemeleri ve adımları bir arada kalacak şekilde, anlamlı parçalara ayırıyoruz.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Bir parçadaki maksimum karakter sayısı
        chunk_overlap=150, # Parçalar arasında ne kadar metin örtüşsün (Bağlamı korumak için)
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Toplam {len(documents)} sayfa, {len(texts)} parçaya (chunks) ayrıldı.")

    # C. Embedding Modeli Tanımlama
    # Gemini'nin embedding modelini kullanarak metinleri sayısal vektörlere dönüştürüyoruz.
   # API anahtarını ortam değişkeninden çeker
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY # Anahtarı burada açıkça belirtiyoruz
    )
    # D. ChromaDB Veritabanını Oluşturma ve Kaydetme (Indexing)
    print("Vektörleştirme ve ChromaDB'ye kaydetme işlemi başlıyor...")
    
    # from_documents fonksiyonu, metinleri alır, vektörleştirir ve DB'ye yazar.
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR # Veritabanı dosyaları bu klasöre kaydedilecek
    )
    
    # DB'yi kalıcı olarak diske kaydetme komutu
    vectorstore.persist()
    print(f"Veritabanı başarıyla oluşturuldu ve [{CHROMA_DB_DIR}] klasörüne kaydedildi.")

if __name__ == "__main__":
    load_and_index_data()