import streamlit as st
from rag_chain import setup_rag_chain 
import os 
from dotenv import load_dotenv

# API Anahtarını Yükle 
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.error("API Anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")
    st.stop() # Anahtar yoksa uygulamayı durdur

# Sayfa Yapılandırması
st.set_page_config(page_title="Akıllı Yemek Asistanı ⚡", layout="wide")

# Başlık ve Açıklama
st.title("👨‍🍳 Ne Pişirsem")
st.markdown("Elindeki **50 Türk Mutfağı tarifinden** oluşan veri setine göre sana yardımcı olur. Sadece tarif içeriğindeki bilgilere cevap verir.")

# RAG Zincirini Bir Kez Kurma
# @st.cache_resource, RAG zincirinin (DB yükleme, model yükleme) sadece bir kez çalışmasını sağlar. 
# Bu, her kullanıcı etkileşiminde zincirin yeniden kurulmasını engeller.
@st.cache_resource
def get_rag_chain():
    try:
        # rag_chain.py dosyasındaki fonksiyonu çağırıyoruz
        return setup_rag_chain()
    except Exception as e:
        st.error(f"RAG Zinciri kurulurken hata oluştu: {e}")
        st.stop()
    
rag_executor = get_rag_chain()

# Sohbet Geçmişini Başlatma
# Streamlit'in session_state özelliği, sohbet geçmişini tutmamızı sağlar.
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Chatbot'tan ilk mesaj
    st.session_state.messages.append({"role": "assistant", "content": "Merhaba! Ben Akıllı Yemek Asistanıyım. Hangi tarifi arıyorsun ya da elinde hangi malzemeler var? Sana süresiyle birlikte yardımcı olabilirim."})

# Geçmiş Mesajları Gösterme
for message in st.session_state.messages:
    # Streamlit'in sohbet balonu formatını kullanır
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı Girişi
if prompt := st.chat_input("Hangi yemeği yapmak istiyorsun? (Örn: Menemen tarifi)"):
    
    # 1. Kullanıcı Mesajını Arayüze Ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Asistan Cevabını Hesaplama
    with st.chat_message("assistant"):
        # Cevap hazırlanırken kullanıcıya bekleme mesajı gösterilir
        with st.spinner("Tarifler taranıyor ve Gemini ile cevap hazırlanıyor..."):
            
            # RAG zincirini çalıştırma
            # Streamlit arayüzünde cevaplar akıcı gelsin diye Streamlit'in akış (streaming) özelliğini kullanıyoruz.
            response = rag_executor.invoke({"input": prompt})
            assistant_response = response['answer']
            
            # Cevabı arayüze yazdırma
            st.markdown(assistant_response)

    # 3. Asistan Cevabını Geçmişe Kaydetme

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
