# %%writefile komutu, bu hücrenin içeriğini 'app.py' adlı bir dosyaya kaydeder.

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
import pandas as pd
import warnings

# Gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

# --- 1. Gerekli Verilerin Kurulumu ---

# NLTK stopwords (etkisiz kelimeler) listesini indirme
# Streamlit sunucusu her başladığında NLTK verisini kontrol eder
# (Bunu daha sağlam hale getirmek için bir setup.sh dosyası da ekleyeceğiz)
@st.cache_resource
def load_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
load_nltk_data()
turkce_stopwords = stopwords.words('turkish')


# !!! DEĞİŞTİRİN !!!
# Modeli Hugging Face Hub'dan çekmek için yolu güncelleyin.
# 'KULLANICI_ADINIZ' kısmını Adım 4'teki Hugging Face kullanıcı adınızla değiştirin.
MODEL_YOLU = "Scaran/DijitalSessizlik-BERT-Modeli" 
# Örn: MODEL_YOLU = "ahmet_yılmaz/DijitalSessizlik-BERT-Modeli"

# Sınıf isimlerimizi tanımlıyoruz
SINIF_ADLARI = ['Açık Zorbalık', 'Örtük Zorbalık', 'Nötr']


# --- 2. Metin Ön İşleme Fonksiyonu (Adım 1'deki ile aynı) ---
def preprocess_text(text):
    text = str(text).lower() # Girdiyi string'e zorla ve küçük harfe çevir
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('#', '')
    text = emoji.demojize(text, language='tr')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text_tokens = text.split()
    text_without_stopwords = [word for word in text_tokens if not word in turkce_stopwords]
    text = ' '.join(text_without_stopwords)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. Model Yükleme (Cache ile) ---
@st.cache_resource
def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval() # Modeli değerlendirme (inference) moduna al
        st.success("Model başarıyla Hugging Face Hub'dan yüklendi.")
        return model, tokenizer
    except Exception as e:
        # Eğer kullanıcı adını girmeyi unuttuysa hata ver
        if "KULLANICI_ADINIZ" in model_path:
            st.error("HATA: app.py dosyasındaki 'MODEL_YOLU' değişkenini güncellemediniz.")
            st.code("MODEL_YOLU = 'KULLANICI_ADINIZ/DijitalSessizlik-BERT-Modeli'")
            st.stop()
        else:
            st.error(f"Model yüklenirken bir hata oluştu: {e}")
            st.info("Modelin Hugging Face Hub'da 'public' (herkese açık) olarak ayarlandığından emin olun.")
            return None, None

# --- 4. Streamlit Arayüz Kodu ---
st.title("Dijital Sessizlik: Siber Zorbalık Tespit Modeli (BERT)")
st.subheader("TÜBİTAK 2204-A Projesi Prototipi")

# Modeli ve tokenizer'ı yükle
model, tokenizer = load_model(MODEL_YOLU)

if model and tokenizer:
    # Kullanıcıdan metin girişi al
    user_input = st.text_area("Lütfen analiz edilecek metni girin:", 
                              "Bugün hava çok güzel ama bazı insanlar gerçekten can sıkıcı olabiliyor.", 
                              height=150)

    # "Sınıflandır" butonuna basıldığında...
    if st.button("Sınıflandır"):
        if user_input:
            with st.spinner("Metin işleniyor ve model tahmini yapılıyor..."):
                # 1. Girdiyi temizle
                processed_text = preprocess_text(user_input)
                
                # 2. Modeli hazırla (Tokenize et)
                inputs = tokenizer(processed_text, 
                                   return_tensors="pt", 
                                   truncation=True, 
                                   padding=True, 
                                   max_length=128)
                
                # 3. Tahmin yap
                with torch.no_grad(): # Gradient hesaplamalarını kapat (hızlandırır)
                    logits = model(**inputs).logits
                
                # 4. Sonuçları işle
                probabilities = torch.softmax(logits, dim=1)
                confidence = torch.max(probabilities).item()
                predicted_class_id = torch.argmax(probabilities, dim=1).item()
                predicted_class_name = SINIF_ADLARI[predicted_class_id]
                
                # 5. Sonuçları göster
                st.subheader("Sınıflandırma Sonucu:")
                
                if predicted_class_name == 'Açık Zorbalık':
                    st.error(f"🚨 TESPİT EDİLDİ: {predicted_class_name} ({confidence*100:.2f}%)")
                    st.write("Bu metin, belirli bir kimliği (yaş, ırk, cinsiyet vb.) hedef alan, bariz hakaret veya nefret söylemi içermektedir.")
                elif predicted_class_name == 'Örtük Zorbalık':
                    st.warning(f"⚠️ TESPİT EDİLDİ: {predicted_class_name} ({confidence*100:.2f}%)")
                    st.write("Bu metin, doğrudan hakaret içermese de alay, ima, dışlama veya genel taciz unsurları barındıran 'görünmeyen' zorbalık içermektedir.")
                else: # Nötr
                    st.success(f"✅ TESPİT EDİLDİ: {predicted_class_name} ({confidence*100:.2f}%)")
                    st.write("Bu metinde belirgin bir zorbalık içeriği tespit edilmemiştir.")
                
                # Detaylı skorları göster
                st.subheader("Model Güven Skorları:")
                prob_df = pd.DataFrame(probabilities.numpy(), columns=SINIF_ADLARI)
                st.dataframe(prob_df.style.format("{:.2%}"))

                st.subheader("İşlem Detayları:")
                st.text(f"Orijinal Metin: {user_input}")
                st.text(f"Temizlenmiş Metin: {processed_text}")
        else:
            st.warning("Lütfen analiz etmek için bir metin girin.")
