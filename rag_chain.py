import os
from dotenv import load_dotenv

# LangChain ve Google AI bileşenleri
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# API Anahtarını Yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Sabit Değerler
CHROMA_DB_DIR = "./chroma_db"
GEMINI_MODEL = "gemini-1.5-flash"  # model ismini doğru sürümle eşleştirdik


def setup_rag_chain():
    """
    RAG Zincirini kurar (Retriever ve LLM'i birleştirir).
    """
    if not GEMINI_API_KEY:
        raise ValueError("API Anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")

    # 1️⃣ Embedding Modeli ve Vektör Veritabanını Yükleme
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2️⃣ Prompt Tanımı
    prompt_template = """
    Sen bir Türk Mutfağı Yemek Asistanısın. Görevin, SADECE sana verilen tarifler ({context}) üzerinden kullanıcının sorularını cevaplamaktır. Cevaplarını daima Türkçe vermelisin.

    Lütfen kullanıcının sorgusunu analiz et ve şu kurallara göre cevapla:

    1. **Tarif İsteği veya Detay Sorulursa (Yemek Adı Varsa):** Kullanıcı bir yemeğin adını sorduğunda, **DAİMA o yemeğin tarifini tamamen sun.**
    2. **Malzeme ve/veya Kategori Tabanlı Öneri (Filtreleme):** Kullanıcının elindeki malzemeleri ve kategori isteğini uygula, uygun 2-3 tarif öner.
    3. **Konuşma Dışı Geri Bildirim veya Selamlaşma:** Kullanıcı 'tamam', 'teşekkürler', 'merhaba' gibi ifadeler kullandığında, kibar ve kısa cevaplar ver.

    ---
    Verilen Tarifler (Context):
    {context}

    ---
    Kullanıcının Sorgusu: {input}
    ---
    Cevabın:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3️⃣ LLM Tanımlama
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0
    )

    # 4️⃣ RetrievalQA Zinciri Oluşturma
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain


# Test Bloğu (Streamlit deploy için değil, local test içindir)
if __name__ == "__main__":
    try:
        rag_executor = setup_rag_chain()
        print("\n--- Ne Pişirsem BAŞLATILDI ---")
        print("Sohbeti sonlandırmak için 'çıkış' yazın.")

        while True:
            user_input = input("\nSenin Sorun: ")

            if user_input.lower() in ["çıkış", "exit", "kapat"]:
                print("Ne Pişirsem sonlandırılıyor. Afiyet olsun!")
                break

            print("Ne Pişirsem Asistan Cevabı: ")
            response = rag_executor.run(user_input)
            print(response)

    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Lütfen API anahtarınızın doğru olduğundan ve 'data_loader.py' dosyasının başarıyla çalışmış olduğundan emin olun.")
