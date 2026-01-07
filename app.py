import streamlit as st
import threading
import time
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================== PARTIE 1 : BACKEND FASTAPI ====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
try:
    classifier = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    MODEL_READY = True
except Exception as e:
    print(f"Erreur modèle : {e}")
    MODEL_READY = False

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextData):
    if MODEL_READY:
        result = classifier(data.text)[0]
        star_value = int(result['label'].split()[0])
        label = "POSITIVE" if star_value >= 4 else "NEGATIVE"
        score = result['score']
    else:
        label = "POSITIVE"
        score = 0.0
    return {"label": label, "score": score}

# Fonction pour démarrer le serveur FastAPI en arrière-plan
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

# ==================== PARTIE 2 : FRONTEND STREAMLIT ====================

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiment IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'état de session
if 'history' not in st.session_state:
    st.session_state.history = []
if 'language' not in st.session_state:
    st.session_state.language = 'fr'
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'server_started' not in st.session_state:
    st.session_state.server_started = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Analyse'

# Démarrer le serveur FastAPI une seule fois
if not st.session_state.server_started:
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    st.session_state.server_started = True
    time.sleep(3)  # Attendre que le serveur démarre

# Traductions
translations = {
    'fr': {
        'title': '🎯 Analyseur de Sentiment IA',
        'subtitle': '✨ Analyse instantanée alimentée par l\'Intelligence Artificielle ✨',
        'input_label': '💬 Partagez votre texte :',
        'placeholder': 'Ex: Cette application est absolument géniale ! J\'adore l\'interface moderne et intuitive...',
        'analyze_btn': '🚀 Analyser maintenant',
        'warning_empty': '⚠️ Veuillez entrer du texte pour l\'analyser.',
        'analyzing': '🔮 L\'IA analyse votre texte en profondeur...',
        'positive': 'Sentiment Positif',
        'negative': 'Sentiment Négatif',
        'confidence': 'Confiance de l\'IA',
        'sentiment': 'Sentiment',
        'words_analyzed': 'Mots analysés',
        'error_server': '❌ Le serveur FastAPI a répondu avec une erreur. Veuillez réessayer.',
        'error_timeout': '⏱️ Délai d\'attente dépassé. Le serveur met trop de temps à répondre.',
        'error_connection': '🔌 Impossible de contacter l\'API. Patientez quelques secondes...',
        'how_it_works': 'ℹ️ Comment ça fonctionne ?',
        'examples': '📚 Exemples de phrases',
        'history': '📜 Historique',
        'clear_history': '🗑️ Effacer l\'historique',
        'no_history': 'Aucune analyse effectuée pour le moment.',
        'language': '🌍 Langue',
        'footer': 'Projet Master - Framework AI',
        'powered_by': 'Propulsé par ❤️ et Intelligence Artificielle',
        'dashboard': '📊 Tableau de Bord',
        'analysis': '🔍 Analyse',
        'total_analyses': 'Total Analyses',
        'positive_count': 'Analyses Positives',
        'negative_count': 'Analyses Négatives',
        'avg_confidence': 'Confiance Moyenne',
        'sentiment_distribution': 'Distribution des Sentiments',
        'confidence_evolution': 'Évolution de la Confiance',
        'word_count_distribution': 'Distribution du Nombre de Mots',
        'recent_analyses': 'Analyses Récentes',
        'sentiment_by_time': 'Sentiments par Heure',
        'no_data': 'Aucune donnée disponible. Effectuez des analyses pour voir les statistiques.',
    },
    'en': {
        'title': '🎯 AI Sentiment Analyzer',
        'subtitle': '✨ Instant Analysis Powered by Artificial Intelligence ✨',
        'input_label': '💬 Share your text:',
        'placeholder': 'Ex: This application is absolutely amazing! I love the modern and intuitive interface...',
        'analyze_btn': '🚀 Analyze now',
        'warning_empty': '⚠️ Please enter text to analyze.',
        'analyzing': '🔮 AI is analyzing your text in depth...',
        'positive': 'Positive Sentiment',
        'negative': 'Negative Sentiment',
        'confidence': 'AI Confidence',
        'sentiment': 'Sentiment',
        'words_analyzed': 'Words analyzed',
        'error_server': '❌ The FastAPI server responded with an error. Please try again.',
        'error_timeout': '⏱️ Timeout exceeded. The server is taking too long to respond.',
        'error_connection': '🔌 Unable to contact the API. Wait a few seconds...',
        'how_it_works': 'ℹ️ How does it work?',
        'examples': '📚 Sample sentences',
        'history': '📜 History',
        'clear_history': '🗑️ Clear history',
        'no_history': 'No analysis performed yet.',
        'language': '🌍 Language',
        'footer': 'Master Project - AI Framework',
        'powered_by': 'Powered by ❤️ and Artificial Intelligence',
        'dashboard': '📊 Dashboard',
        'analysis': '🔍 Analysis',
        'total_analyses': 'Total Analyses',
        'positive_count': 'Positive Analyses',
        'negative_count': 'Negative Analyses',
        'avg_confidence': 'Average Confidence',
        'sentiment_distribution': 'Sentiment Distribution',
        'confidence_evolution': 'Confidence Evolution',
        'word_count_distribution': 'Word Count Distribution',
        'recent_analyses': 'Recent Analyses',
        'sentiment_by_time': 'Sentiments by Hour',
        'no_data': 'No data available. Perform analyses to see statistics.',
    },
    'es': {
        'title': '🎯 Analizador de Sentimientos IA',
        'subtitle': '✨ Análisis instantáneo impulsado por Inteligencia Artificial ✨',
        'input_label': '💬 Comparte tu texto:',
        'placeholder': 'Ej: ¡Esta aplicación es absolutamente genial! Me encanta la interfaz moderna e intuitiva...',
        'analyze_btn': '🚀 Analizar ahora',
        'warning_empty': '⚠️ Por favor ingrese texto para analizar.',
        'analyzing': '🔮 La IA está analizando tu texto en profundidad...',
        'positive': 'Sentimiento Positivo',
        'negative': 'Sentimiento Negativo',
        'confidence': 'Confianza de la IA',
        'sentiment': 'Sentimiento',
        'words_analyzed': 'Palabras analizadas',
        'error_server': '❌ El servidor FastAPI respondió con un error. Por favor intente nuevamente.',
        'error_timeout': '⏱️ Tiempo de espera excedido. El servidor está tardando demasiado en responder.',
        'error_connection': '🔌 No se puede contactar con la API. Espere unos segundos...',
        'how_it_works': 'ℹ️ ¿Cómo funciona?',
        'examples': '📚 Frases de ejemplo',
        'history': '📜 Historial',
        'clear_history': '🗑️ Borrar historial',
        'no_history': 'No se ha realizado ningún análisis aún.',
        'language': '🌍 Idioma',
        'footer': 'Proyecto Máster - Framework IA',
        'powered_by': 'Impulsado por ❤️ e Inteligencia Artificial',
        'dashboard': '📊 Panel',
        'analysis': '🔍 Análisis',
        'total_analyses': 'Análisis Totales',
        'positive_count': 'Análisis Positivos',
        'negative_count': 'Análisis Negativos',
        'avg_confidence': 'Confianza Media',
        'sentiment_distribution': 'Distribución de Sentimientos',
        'confidence_evolution': 'Evolución de la Confianza',
        'word_count_distribution': 'Distribución de Palabras',
        'recent_analyses': 'Análisis Recientes',
        'sentiment_by_time': 'Sentimientos por Hora',
        'no_data': 'No hay datos disponibles. Realice análisis para ver estadísticas.',
    },
    'ar': {
        'title': '🎯 محلل المشاعر بالذكاء الاصطناعي',
        'subtitle': '✨ تحليل فوري مدعوم بالذكاء الاصطناعي ✨',
        'input_label': '💬 شارك نصك:',
        'placeholder': 'مثال: هذا التطبيق رائع للغاية! أحب الواجهة الحديثة والبديهية...',
        'analyze_btn': '🚀 تحليل الآن',
        'warning_empty': '⚠️ يرجى إدخال نص للتحليل.',
        'analyzing': '🔮 الذكاء الاصطناعي يحلل نصك بعمق...',
        'positive': 'مشاعر إيجابية',
        'negative': 'مشاعر سلبية',
        'confidence': 'ثقة الذكاء الاصطناعي',
        'sentiment': 'المشاعر',
        'words_analyzed': 'الكلمات المحللة',
        'error_server': '❌ استجاب خادم FastAPI بخطأ. يرجى المحاولة مرة أخرى.',
        'error_timeout': '⏱️ انتهت المهلة الزمنية. الخادم يستغرق وقتًا طويلاً للرد.',
        'error_connection': '🔌 تعذر الاتصال بواجهة برمجة التطبيقات. انتظر بضع ثوان...',
        'how_it_works': 'ℹ️ كيف يعمل؟',
        'examples': '📚 أمثلة على الجمل',
        'history': '📜 السجل',
        'clear_history': '🗑️ مسح السجل',
        'no_history': 'لم يتم إجراء أي تحليل حتى الآن.',
        'language': '🌍 اللغة',
        'footer': 'مشروع الماجستير - إطار الذكاء الاصطناعي',
        'powered_by': 'مدعوم بـ ❤️ والذكاء الاصطناعي',
        'dashboard': '📊 لوحة التحكم',
        'analysis': '🔍 تحليل',
        'total_analyses': 'إجمالي التحليلات',
        'positive_count': 'التحليلات الإيجابية',
        'negative_count': 'التحليلات السلبية',
        'avg_confidence': 'متوسط الثقة',
        'sentiment_distribution': 'توزيع المشاعر',
        'confidence_evolution': 'تطور الثقة',
        'word_count_distribution': 'توزيع عدد الكلمات',
        'recent_analyses': 'التحليلات الأخيرة',
        'sentiment_by_time': 'المشاعر حسب الساعة',
        'no_data': 'لا توجد بيانات متاحة. قم بإجراء تحليلات لرؤية الإحصائيات.',
    }
}

# Dataset d'exemples
example_datasets = {
    'fr': [
        "J'adore cette application, elle est incroyable et très intuitive !",
        "Le service client est excellent, j'ai reçu une aide rapide et efficace.",
        "Quelle déception ! Le produit ne correspond pas du tout à la description.",
        "Je suis très satisfait de mon achat, la qualité est au rendez-vous.",
        "C'est horrible, je ne recommande absolument pas cette expérience.",
        "Une expérience formidable ! Je reviendrai certainement.",
    ],
    'en': [
        "I love this application, it's amazing and very intuitive!",
        "The customer service is excellent, I received quick and efficient help.",
        "What a disappointment! The product doesn't match the description at all.",
        "I'm very satisfied with my purchase, the quality is there.",
        "It's horrible, I absolutely don't recommend this experience.",
        "A wonderful experience! I will definitely come back.",
    ],
    'es': [
        "¡Me encanta esta aplicación, es increíble y muy intuitiva!",
        "El servicio al cliente es excelente, recibí ayuda rápida y eficiente.",
        "¡Qué decepción! El producto no coincide en absoluto con la descripción.",
        "Estoy muy satisfecho con mi compra, la calidad está presente.",
        "Es horrible, no recomiendo absolutamente esta experiencia.",
        "¡Una experiencia maravillosa! Definitivamente volveré.",
    ],
    'ar': [
        "أحب هذا التطبيق، إنه مذهل وسهل الاستخدام للغاية!",
        "خدمة العملاء ممتازة، تلقيت مساعدة سريعة وفعالة.",
        "يا للخيبة! المنتج لا يتطابق مع الوصف على الإطلاق.",
        "أنا راضٍ جدًا عن عملية الشراء، الجودة موجودة.",
        "إنه فظيع، لا أوصي بهذه التجربة على الإطلاق.",
        "تجربة رائعة! سأعود بالتأكيد.",
    ]
}

def t(key):
    return translations[st.session_state.language].get(key, key)

# CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Poppins', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif; font-weight: 700; }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .positive-result { border-left: 6px solid #10b981; }
    .negative-result { border-left: 6px solid #ef4444; }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Options")
    
    # Navigation
    page = st.radio(
        "Navigation",
        [t('analysis'), t('dashboard')],
        key='navigation'
    )
    st.session_state.current_page = page
    
    st.markdown("---")
    
    lang_options = {
        'fr': '🇫🇷 Français',
        'en': '🇬🇧 English',
        'es': '🇪🇸 Español',
        'ar': '🇸🇦 العربية'
    }
    
    selected_lang = st.selectbox(
        t('language'),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.language)
    )
    
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {t('examples')}")
    st.info(f"📊 **Dataset** : {len(example_datasets[st.session_state.language])} phrases d'exemple")
    
    for i, example in enumerate(example_datasets[st.session_state.language]):
        if st.button(f"📝 Exemple {i+1}", key=f"example_{i}", use_container_width=True):
            st.session_state.current_text = example
            st.session_state.text_input = example
            st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {t('history')}")
    
    if st.session_state.history:
        if st.button(t('clear_history'), use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()
        
        st.markdown(f"**{len(st.session_state.history)} analyse(s)**")
    else:
        st.info(t('no_history'))

# PAGE: ANALYSE
if st.session_state.current_page == t('analysis'):
    # En-tête
    st.markdown(f'<h1 class="title">{t("title")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t("subtitle")}</p>', unsafe_allow_html=True)

    # Conteneur principal
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        user_text = st.text_area(
            t('input_label'),
            value=st.session_state.current_text,
            placeholder=t('placeholder'),
            height=200,
            key="text_input"
        )
        
        if user_text != st.session_state.current_text:
            st.session_state.current_text = user_text
        
        if st.button(t('analyze_btn'), use_container_width=True):
            if user_text.strip() == "":
                st.warning(t('warning_empty'))
            else:
                import requests
                with st.spinner(t('analyzing')):
                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/predict",
                            json={"text": user_text},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            label = data['label']
                            score = data['score']
                            
                            history_item = {
                                'text': user_text,
                                'label': label,
                                'score': score,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'emoji': '😊' if label == "POSITIVE" else '😔',
                                'word_count': len(user_text.split())
                            }
                            st.session_state.history.append(history_item)
                            
                            result_class = "positive-result" if label == "POSITIVE" else "negative-result"
                            sentiment_text = t('positive') if label == "POSITIVE" else t('negative')
                            
                            st.markdown(f"""
                            <div class="result-box {result_class}">
                                <h2 style="margin: 0; color: #1f2937;">
                                    {'😊' if label == "POSITIVE" else '😔'} {sentiment_text}
                                </h2>
                                <p style="font-size: 1.1rem; color: #6b7280; margin-top: 0.5rem;">
                                    {t('confidence')} : <strong>{score:.1%}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(score)
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric(label=f"📊 {t('sentiment')}", value=label)
                            
                            with metric_col2:
                                st.metric(label=f"🎯 {t('confidence')}", value=f"{score:.1%}")
                            
                            with metric_col3:
                                st.metric(label=f"📝 {t('words_analyzed')}", value=len(user_text.split()))
                            
                            st.balloons()
                            
                        else:
                            st.error(t('error_server'))
                            
                    except requests.exceptions.Timeout:
                        st.error(t('error_timeout'))
                        
                    except requests.exceptions.ConnectionError:
                        st.error(t('error_connection'))
                        
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

# PAGE: TABLEAU DE BORD
elif st.session_state.current_page == t('dashboard'):
    st.markdown(f'<h1 class="title">📊 {t("dashboard")}</h1>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info(t('no_data'))
    else:
        # Créer un DataFrame
        df = pd.DataFrame(st.session_state.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(df)
        positive = len(df[df['label'] == 'POSITIVE'])
        negative = len(df[df['label'] == 'NEGATIVE'])
        avg_conf = df['score'].mean()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📈 {total}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('total_analyses')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #10b981; margin: 0;">😊 {positive}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('positive_count')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ef4444; margin: 0;">😔 {negative}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('negative_count')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #f59e0b; margin: 0;">🎯 {avg_conf:.1%}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('avg_confidence')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des sentiments
            fig_pie = px.pie(
                df, 
                names='label', 
                title=t('sentiment_distribution'),
                color='label',
                color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'}
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Distribution du nombre de mots
            fig_hist = px.histogram(
                df, 
                x='word_count', 
                title=t('word_count_distribution'),
                color='label',
                color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
                nbins=20
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Évolution de la confiance
        df_sorted = df.sort_values('timestamp')
        df_sorted['index'] = range(1, len(df_sorted) + 1)
        
        fig_line = px.line(
            df_sorted, 
            x='index', 
            y='score',
            title=t('confidence_evolution'),
            color='label',
            color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
            markers=True
        )
        fig_line.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.95)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            xaxis_title="Analyse #",
            yaxis_title=t('confidence')
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Sentiments par heure
        if len(df['hour'].unique()) > 1:
            hourly_data = df.groupby(['hour', 'label']).size().reset_index(name='count')
            fig_bar = px.bar(
                hourly_data, 
                x='hour', 
                y='count',
                color='label',
                title=t('sentiment_by_time'),
                color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
                barmode='group'
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='rgba(255, 255, 255import streamlit as st
import threading
import time
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================== PARTIE 1 : BACKEND FASTAPI ====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
try:
    classifier = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    MODEL_READY = True
except Exception as e:
    print(f"Erreur modèle : {e}")
    MODEL_READY = False

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextData):
    if MODEL_READY:
        result = classifier(data.text)[0]
        star_value = int(result['label'].split()[0])
        label = "POSITIVE" if star_value >= 4 else "NEGATIVE"
        score = result['score']
    else:
        label = "POSITIVE"
        score = 0.0
    return {"label": label, "score": score}

# Fonction pour démarrer le serveur FastAPI en arrière-plan
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

# ==================== PARTIE 2 : FRONTEND STREAMLIT ====================

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiment IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'état de session
if 'history' not in st.session_state:
    st.session_state.history = []
if 'language' not in st.session_state:
    st.session_state.language = 'fr'
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'server_started' not in st.session_state:
    st.session_state.server_started = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Analyse'

# Démarrer le serveur FastAPI une seule fois
if not st.session_state.server_started:
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    st.session_state.server_started = True
    time.sleep(3)  # Attendre que le serveur démarre

# Traductions
translations = {
    'fr': {
        'title': '🎯 Analyseur de Sentiment IA',
        'subtitle': '✨ Analyse instantanée alimentée par l\'Intelligence Artificielle ✨',
        'input_label': '💬 Partagez votre texte :',
        'placeholder': 'Ex: Cette application est absolument géniale ! J\'adore l\'interface moderne et intuitive...',
        'analyze_btn': '🚀 Analyser maintenant',
        'warning_empty': '⚠️ Veuillez entrer du texte pour l\'analyser.',
        'analyzing': '🔮 L\'IA analyse votre texte en profondeur...',
        'positive': 'Sentiment Positif',
        'negative': 'Sentiment Négatif',
        'confidence': 'Confiance de l\'IA',
        'sentiment': 'Sentiment',
        'words_analyzed': 'Mots analysés',
        'error_server': '❌ Le serveur FastAPI a répondu avec une erreur. Veuillez réessayer.',
        'error_timeout': '⏱️ Délai d\'attente dépassé. Le serveur met trop de temps à répondre.',
        'error_connection': '🔌 Impossible de contacter l\'API. Patientez quelques secondes...',
        'how_it_works': 'ℹ️ Comment ça fonctionne ?',
        'examples': '📚 Exemples de phrases',
        'history': '📜 Historique',
        'clear_history': '🗑️ Effacer l\'historique',
        'no_history': 'Aucune analyse effectuée pour le moment.',
        'language': '🌍 Langue',
        'footer': 'Projet Master - Framework AI',
        'powered_by': 'Propulsé par ❤️ et Intelligence Artificielle',
        'dashboard': '📊 Tableau de Bord',
        'analysis': '🔍 Analyse',
        'total_analyses': 'Total Analyses',
        'positive_count': 'Analyses Positives',
        'negative_count': 'Analyses Négatives',
        'avg_confidence': 'Confiance Moyenne',
        'sentiment_distribution': 'Distribution des Sentiments',
        'confidence_evolution': 'Évolution de la Confiance',
        'word_count_distribution': 'Distribution du Nombre de Mots',
        'recent_analyses': 'Analyses Récentes',
        'sentiment_by_time': 'Sentiments par Heure',
        'no_data': 'Aucune donnée disponible. Effectuez des analyses pour voir les statistiques.',
    },
    'en': {
        'title': '🎯 AI Sentiment Analyzer',
        'subtitle': '✨ Instant Analysis Powered by Artificial Intelligence ✨',
        'input_label': '💬 Share your text:',
        'placeholder': 'Ex: This application is absolutely amazing! I love the modern and intuitive interface...',
        'analyze_btn': '🚀 Analyze now',
        'warning_empty': '⚠️ Please enter text to analyze.',
        'analyzing': '🔮 AI is analyzing your text in depth...',
        'positive': 'Positive Sentiment',
        'negative': 'Negative Sentiment',
        'confidence': 'AI Confidence',
        'sentiment': 'Sentiment',
        'words_analyzed': 'Words analyzed',
        'error_server': '❌ The FastAPI server responded with an error. Please try again.',
        'error_timeout': '⏱️ Timeout exceeded. The server is taking too long to respond.',
        'error_connection': '🔌 Unable to contact the API. Wait a few seconds...',
        'how_it_works': 'ℹ️ How does it work?',
        'examples': '📚 Sample sentences',
        'history': '📜 History',
        'clear_history': '🗑️ Clear history',
        'no_history': 'No analysis performed yet.',
        'language': '🌍 Language',
        'footer': 'Master Project - AI Framework',
        'powered_by': 'Powered by ❤️ and Artificial Intelligence',
        'dashboard': '📊 Dashboard',
        'analysis': '🔍 Analysis',
        'total_analyses': 'Total Analyses',
        'positive_count': 'Positive Analyses',
        'negative_count': 'Negative Analyses',
        'avg_confidence': 'Average Confidence',
        'sentiment_distribution': 'Sentiment Distribution',
        'confidence_evolution': 'Confidence Evolution',
        'word_count_distribution': 'Word Count Distribution',
        'recent_analyses': 'Recent Analyses',
        'sentiment_by_time': 'Sentiments by Hour',
        'no_data': 'No data available. Perform analyses to see statistics.',
    },
    'es': {
        'title': '🎯 Analizador de Sentimientos IA',
        'subtitle': '✨ Análisis instantáneo impulsado por Inteligencia Artificial ✨',
        'input_label': '💬 Comparte tu texto:',
        'placeholder': 'Ej: ¡Esta aplicación es absolutamente genial! Me encanta la interfaz moderna e intuitiva...',
        'analyze_btn': '🚀 Analizar ahora',
        'warning_empty': '⚠️ Por favor ingrese texto para analizar.',
        'analyzing': '🔮 La IA está analizando tu texto en profundidad...',
        'positive': 'Sentimiento Positivo',
        'negative': 'Sentimiento Negativo',
        'confidence': 'Confianza de la IA',
        'sentiment': 'Sentimiento',
        'words_analyzed': 'Palabras analizadas',
        'error_server': '❌ El servidor FastAPI respondió con un error. Por favor intente nuevamente.',
        'error_timeout': '⏱️ Tiempo de espera excedido. El servidor está tardando demasiado en responder.',
        'error_connection': '🔌 No se puede contactar con la API. Espere unos segundos...',
        'how_it_works': 'ℹ️ ¿Cómo funciona?',
        'examples': '📚 Frases de ejemplo',
        'history': '📜 Historial',
        'clear_history': '🗑️ Borrar historial',
        'no_history': 'No se ha realizado ningún análisis aún.',
        'language': '🌍 Idioma',
        'footer': 'Proyecto Máster - Framework IA',
        'powered_by': 'Impulsado por ❤️ e Inteligencia Artificial',
        'dashboard': '📊 Panel',
        'analysis': '🔍 Análisis',
        'total_analyses': 'Análisis Totales',
        'positive_count': 'Análisis Positivos',
        'negative_count': 'Análisis Negativos',
        'avg_confidence': 'Confianza Media',
        'sentiment_distribution': 'Distribución de Sentimientos',
        'confidence_evolution': 'Evolución de la Confianza',
        'word_count_distribution': 'Distribución de Palabras',
        'recent_analyses': 'Análisis Recientes',
        'sentiment_by_time': 'Sentimientos por Hora',
        'no_data': 'No hay datos disponibles. Realice análisis para ver estadísticas.',
    },
    'ar': {
        'title': '🎯 محلل المشاعر بالذكاء الاصطناعي',
        'subtitle': '✨ تحليل فوري مدعوم بالذكاء الاصطناعي ✨',
        'input_label': '💬 شارك نصك:',
        'placeholder': 'مثال: هذا التطبيق رائع للغاية! أحب الواجهة الحديثة والبديهية...',
        'analyze_btn': '🚀 تحليل الآن',
        'warning_empty': '⚠️ يرجى إدخال نص للتحليل.',
        'analyzing': '🔮 الذكاء الاصطناعي يحلل نصك بعمق...',
        'positive': 'مشاعر إيجابية',
        'negative': 'مشاعر سلبية',
        'confidence': 'ثقة الذكاء الاصطناعي',
        'sentiment': 'المشاعر',
        'words_analyzed': 'الكلمات المحللة',
        'error_server': '❌ استجاب خادم FastAPI بخطأ. يرجى المحاولة مرة أخرى.',
        'error_timeout': '⏱️ انتهت المهلة الزمنية. الخادم يستغرق وقتًا طويلاً للرد.',
        'error_connection': '🔌 تعذر الاتصال بواجهة برمجة التطبيقات. انتظر بضع ثوان...',
        'how_it_works': 'ℹ️ كيف يعمل؟',
        'examples': '📚 أمثلة على الجمل',
        'history': '📜 السجل',
        'clear_history': '🗑️ مسح السجل',
        'no_history': 'لم يتم إجراء أي تحليل حتى الآن.',
        'language': '🌍 اللغة',
        'footer': 'مشروع الماجستير - إطار الذكاء الاصطناعي',
        'powered_by': 'مدعوم بـ ❤️ والذكاء الاصطناعي',
        'dashboard': '📊 لوحة التحكم',
        'analysis': '🔍 تحليل',
        'total_analyses': 'إجمالي التحليلات',
        'positive_count': 'التحليلات الإيجابية',
        'negative_count': 'التحليلات السلبية',
        'avg_confidence': 'متوسط الثقة',
        'sentiment_distribution': 'توزيع المشاعر',
        'confidence_evolution': 'تطور الثقة',
        'word_count_distribution': 'توزيع عدد الكلمات',
        'recent_analyses': 'التحليلات الأخيرة',
        'sentiment_by_time': 'المشاعر حسب الساعة',
        'no_data': 'لا توجد بيانات متاحة. قم بإجراء تحليلات لرؤية الإحصائيات.',
    }
}

# Dataset d'exemples
example_datasets = {
    'fr': [
        "J'adore cette application, elle est incroyable et très intuitive !",
        "Le service client est excellent, j'ai reçu une aide rapide et efficace.",
        "Quelle déception ! Le produit ne correspond pas du tout à la description.",
        "Je suis très satisfait de mon achat, la qualité est au rendez-vous.",
        "C'est horrible, je ne recommande absolument pas cette expérience.",
        "Une expérience formidable ! Je reviendrai certainement.",
    ],
    'en': [
        "I love this application, it's amazing and very intuitive!",
        "The customer service is excellent, I received quick and efficient help.",
        "What a disappointment! The product doesn't match the description at all.",
        "I'm very satisfied with my purchase, the quality is there.",
        "It's horrible, I absolutely don't recommend this experience.",
        "A wonderful experience! I will definitely come back.",
    ],
    'es': [
        "¡Me encanta esta aplicación, es increíble y muy intuitiva!",
        "El servicio al cliente es excelente, recibí ayuda rápida y eficiente.",
        "¡Qué decepción! El producto no coincide en absoluto con la descripción.",
        "Estoy muy satisfecho con mi compra, la calidad está presente.",
        "Es horrible, no recomiendo absolutamente esta experiencia.",
        "¡Una experiencia maravillosa! Definitivamente volveré.",
    ],
    'ar': [
        "أحب هذا التطبيق، إنه مذهل وسهل الاستخدام للغاية!",
        "خدمة العملاء ممتازة، تلقيت مساعدة سريعة وفعالة.",
        "يا للخيبة! المنتج لا يتطابق مع الوصف على الإطلاق.",
        "أنا راضٍ جدًا عن عملية الشراء، الجودة موجودة.",
        "إنه فظيع، لا أوصي بهذه التجربة على الإطلاق.",
        "تجربة رائعة! سأعود بالتأكيد.",
    ]
}

def t(key):
    return translations[st.session_state.language].get(key, key)

# CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Poppins', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif; font-weight: 700; }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .positive-result { border-left: 6px solid #10b981; }
    .negative-result { border-left: 6px solid #ef4444; }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Options")
    
    # Navigation
    page = st.radio(
        "Navigation",
        [t('analysis'), t('dashboard')],
        key='navigation'
    )
    st.session_state.current_page = page
    
    st.markdown("---")
    
    lang_options = {
        'fr': '🇫🇷 Français',
        'en': '🇬🇧 English',
        'es': '🇪🇸 Español',
        'ar': '🇸🇦 العربية'
    }
    
    selected_lang = st.selectbox(
        t('language'),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.language)
    )
    
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {t('examples')}")
    st.info(f"📊 **Dataset** : {len(example_datasets[st.session_state.language])} phrases d'exemple")
    
    for i, example in enumerate(example_datasets[st.session_state.language]):
        if st.button(f"📝 Exemple {i+1}", key=f"example_{i}", use_container_width=True):
            st.session_state.current_text = example
            st.session_state.text_input = example
            st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {t('history')}")
    
    if st.session_state.history:
        if st.button(t('clear_history'), use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()
        
        st.markdown(f"**{len(st.session_state.history)} analyse(s)**")
    else:
        st.info(t('no_history'))

# PAGE: ANALYSE
if st.session_state.current_page == t('analysis'):
    # En-tête
    st.markdown(f'<h1 class="title">{t("title")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t("subtitle")}</p>', unsafe_allow_html=True)

    # Conteneur principal
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        user_text = st.text_area(
            t('input_label'),
            value=st.session_state.current_text,
            placeholder=t('placeholder'),
            height=200,
            key="text_input"
        )
        
        if user_text != st.session_state.current_text:
            st.session_state.current_text = user_text
        
        if st.button(t('analyze_btn'), use_container_width=True):
            if user_text.strip() == "":
                st.warning(t('warning_empty'))
            else:
                import requests
                with st.spinner(t('analyzing')):
                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/predict",
                            json={"text": user_text},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            label = data['label']
                            score = data['score']
                            
                            history_item = {
                                'text': user_text,
                                'label': label,
                                'score': score,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'emoji': '😊' if label == "POSITIVE" else '😔',
                                'word_count': len(user_text.split())
                            }
                            st.session_state.history.append(history_item)
                            
                            result_class = "positive-result" if label == "POSITIVE" else "negative-result"
                            sentiment_text = t('positive') if label == "POSITIVE" else t('negative')
                            
                            st.markdown(f"""
                            <div class="result-box {result_class}">
                                <h2 style="margin: 0; color: #1f2937;">
                                    {'😊' if label == "POSITIVE" else '😔'} {sentiment_text}
                                </h2>
                                <p style="font-size: 1.1rem; color: #6b7280; margin-top: 0.5rem;">
                                    {t('confidence')} : <strong>{score:.1%}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(score)
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric(label=f"📊 {t('sentiment')}", value=label)
                            
                            with metric_col2:
                                st.metric(label=f"🎯 {t('confidence')}", value=f"{score:.1%}")
                            
                            with metric_col3:
                                st.metric(label=f"📝 {t('words_analyzed')}", value=len(user_text.split()))
                            
                            st.balloons()
                            
                        else:
                            st.error(t('error_server'))
                            
                    except requests.exceptions.Timeout:
                        st.error(t('error_timeout'))
                        
                    except requests.exceptions.ConnectionError:
                        st.error(t('error_connection'))
                        
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

# PAGE: TABLEAU DE BORD
elif st.session_state.current_page == t('dashboard'):
    st.markdown(f'<h1 class="title">📊 {t("dashboard")}</h1>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info(t('no_data'))
    else:
        # Créer un DataFrame
        df = pd.DataFrame(st.session_state.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(df)
        positive = len(df[df['label'] == 'POSITIVE'])
        negative = len(df[df['label'] == 'NEGATIVE'])
        avg_conf = df['score'].mean()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📈 {total}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('total_analyses')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #10b981; margin: 0;">😊 {positive}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('positive_count')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ef4444; margin: 0;">😔 {negative}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('negative_count')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #f59e0b; margin: 0;">🎯 {avg_conf:.1%}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{t('avg_confidence')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des sentiments
            fig_pie = px.pie(
                df, 
                names='label', 
                title=t('sentiment_distribution'),
                color='label',
                color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'}
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Distribution du nombre de mots
            fig_hist = px.histogram(
                df, 
                x='word_count', 
                title=t('word_count_distribution'),
                color='label',
                color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
                nbins=20
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Évolution de la confiance
        df_sorted = df.sort_values('timestamp')
        df_sorted['index'] = range(1, len(df_sorted) + 1)
        
        fig_line = px.line(
            df_sorted, 
            x='index', 
            y='score',
            title=t('confidence_evolution'),
            color='label',
            color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
            markers=True
        )
        fig_line.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.95)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            xaxis_title="Analyse #",
            yaxis_title=t('confidence')
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Sentiments par heure
        if len(df['hour'].unique()) > 1:
            hourly_data = df.groupby(['hour', 'label']).size().reset_index(name='count')
            fig_bar = px.bar(
                hourly_data, 
                x='hour', 
                y='count',
                color='label',
                title=t('sentiment_by_time'),
                color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
                barmode='group'
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
                xaxis_title="Heure",
                yaxis_title="Nombre"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Tableau des analyses récentes
        st.markdown(f"### {t('recent_analyses')}")
        recent_df = df.sort_values('timestamp', ascending=False).head(10)
        display_df = recent_df[['emoji', 'text', 'label', 'score', 'word_count', 'timestamp']].copy()
        display_df.columns = ['', 'Texte', 'Sentiment', 'Confiance', 'Mots', 'Date/Heure']
        display_df['Confiance'] = display_df['Confiance'].apply(lambda x: f"{x:.1%}")
        display_df['Texte'] = display_df['Texte'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

# Footer
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p>🎓 <strong>{t('footer')}</strong> | {datetime.now().year}</p>
    <p>{t('powered_by')}</p>
</div>
""", unsafe_allow_html=True)
