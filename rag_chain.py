import os
from dotenv import load_dotenv

# LangChain ve Google AI bileşenleri
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from langchain.chains.combine_documents import create_stuff_documents_chain

# API Anahtarını Yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Sabit Değerler
CHROMA_DB_DIR = "./chroma_db"
GEMINI_MODEL = "gemini-2.5-flash" 

def setup_rag_chain():
    """
    RAG Zincirini kurar (Retriever ve LLM'i birleştirir).
    """
    if not GEMINI_API_KEY:
        raise ValueError("API Anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")

    # 1. Embedding Modeli ve Vektör Veritabanını Yükleme (Indexing Aşaması Sonucu)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    # Daha önce oluşturduğumuz ChromaDB veritabanını disktenden (./chroma_db) yükler.
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    # Vektör veritabanını arama (retriever) aracına dönüştürür.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # En alakalı 5 parça (chunk) getir
    )

    # 2. Prompt (Yapay Zeka'ya Rolünü ve Kuralını Belirtme)
    # Bu Prompt, chatbot'un iki tür sorguya (Malzeme ve Yemek Adı) nasıl cevap vereceğini belirler.
    prompt_template = """
    Sen bir Türk Mutfağı Yemek Asistanısın. Görevin, SADECE sana verilen tarifler ({context}) üzerinden kullanıcının sorularını cevaplamaktır. Cevaplarını daima Türkçe vermelisin.

    Lütfen kullanıcının sorgusunu analiz et ve şu kurallara göre cevapla:

    1. **Tarif İsteği veya Detay Sorulursa (Yemek Adı Varsa):** Kullanıcı bir yemeğin adını (Örn: 'Menemen', 'Sulu Köfte', 'Nasıl yaparım?' gibi bağlamsal sorular) sorduğunda, **DAİMA o yemeğin tarifini tamamen sun.** Tarif, Yemek adını, HAZIRLIK SÜRESİNİ, PİŞİRME SÜRESİNİ, malzemeleri ve adım adım yapılışını listeler halinde, temiz bir formatta içermelidir.
    * **Önemli Kural:** Eğer isimle bir tarif soruluyorsa ve bağlamda o tarif bulunuyorsa, **asla** 'Üzgünüm' deme, tarifi sun!

    2. **Malzeme ve/veya Kategori Tabanlı Öneri (Filtreleme):**
    a) **Öncelik:** Eğer kullanıcı hem malzeme hem de kategori (Örn: '2 yumurta ile kahvaltı') belirtiyorsa, iki filtreyi birden uygula.
    b) **Miktar Kontrolü:** Kullanıcının elindeki miktarı **AŞAN** tarifleri ele. Elindeki miktarın YETERLİ olduğu 2 veya 3 uygun yemeğin adını listele.
    c) **Ek Malzeme:** Listelediğin her bir yemek için, elinde *olmayan* **ek olarak gereken temel malzemeleri** (tuz, su, karabiber, sıvı yağ gibi genel mutfak malzemeleri hariç) belirt.
    d) **Rastgele Öneri:** Eğer malzeme veya kategori belirtilmezse ('Bana bir şey öner'), rastgele 2-3 tarif öner.
    
    3. **Konuşma Dışı Geri Bildirim veya Selamlaşma:** Kullanıcı 'tamam', 'teşekkürler', 'merhaba' gibi ifadeler kullandığında veya alakasız bir soru sorduğunda, kibarca ve kısa cevaplar ver. **Asla** 'Üzgünüm, benim uzmanlık alanım...' kalıbını kullanma.
    * Örnek Cevaplar: "Rica ederim, başka nasıl yardımcı olabilirim?", "Merhaba! Hangi tarifi arıyorsunuz?", "Üzgünüm, sadece tarifler konusunda yardımcı olabilirim."
    ---
    Verilen Tarifler (Context):
    {context}

    ---
    Kullanıcının Sorgusu: {input}
    ---
    Cevabın:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3. LLM (Gemini) Tanımlama
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0 # Cevapların bilgiye dayalı ve tutarlı olması için yaratıcılığı kapatıyoruz.
    )

    # 4. RAG Zincirini Kurma
    # Document Chain: Retriever'dan gelen bağlamı alıp Prompt içine yerleştirir.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # LCEL (LangChain İfade Dili) ile RAG zincirini kurma:
    # 1. RunnablePassthrough: Zincire gelen input'u alır.
    # 2. assign: "context" anahtarını (retriever sonucu) ve "input" anahtarını (kullanıcı sorusu) korur.
    # 3. | (pipe operatörü): Çıktıyı bir sonraki bileşene (document_chain) iletir.
    rag_chain = RunnablePassthrough.assign(
        context=(lambda x: x["input"]) | retriever
    ) | document_chain

    return rag_chain

# Modülü test etmek için küçük bir örnek
# Yeni Test ve Sohbet Bloğu
if __name__ == "__main__":
    try:
        rag_executor = setup_rag_chain()
        print("\n--- Ne Pişirsem BAŞLATILDI ---")
        print("Sohbeti sonlandırmak için 'çıkış' yazın.")

        while True:
            # Kullanıcıdan giriş alma
            user_input = input("\nSenin Sorun: ")

            if user_input.lower() in ["çıkış", "exit", "kapat"]:
                print("Ne Pişirsem sonlandırılıyor. Afiyet olsun!")
                break
            
            # RAG zincirini çalıştırma ve cevabı yazdırma
            print("Ne Pişirsem Asistan Cevabı: ")
            response = rag_executor.invoke({"input": user_input})
            print(response['answer'])

    except Exception as e:
        print(f"Hata oluştu: {e}")

        print("Lütfen API anahtarınızın doğru olduğundan ve 'data_loader.py' dosyasının başarıyla çalışmış olduğundan emin olun.")








