import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2
from PIL import Image, ImageOps
from datetime import datetime
import pandas as pd

# Configuración de página
st.set_page_config(
    page_title="Detector de Neumonía",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_effnetv2.keras")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Cache del modelo
@st.cache_resource
def load_model():
    """Carga el modelo de manera eficiente con cache"""
    # Si el modelo no existe localmente, intentar descargarlo
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Modelo no encontrado localmente. Verificando repositorio...")
        # Aquí puedes agregar lógica para descargar desde GitHub Release o Drive
        st.error(f"❌ Modelo no encontrado en: {MODEL_PATH}")
        st.info("💡 Asegúrate de que 'best_effnetv2.keras' esté en la raíz del proyecto.")
        st.stop()

    with st.spinner("🧠 Cargando modelo..."):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success(f"✅ Modelo cargado | TensorFlow {tf.__version__}")
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            st.stop()

def preprocess_image(pil_img, target_size=IMG_SIZE):
    """Preprocesa la imagen para el modelo"""
    img = ImageOps.exif_transpose(pil_img).convert("RGB").resize(
        target_size, resample=Image.BICUBIC
    )
    arr = np.array(img).astype("float32")
    arr = efficientnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_image(model, pil_img):
    """Realiza la predicción"""
    x = preprocess_image(pil_img, IMG_SIZE)
    probs = model.predict(x, verbose=0)[0]

    # Manejo de salida sigmoide binaria
    if probs.shape[0] == 1:
        p = float(probs[0])
        probs = np.array([1.0 - p, p])

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

# Inicializar session_state para historial
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<p class="main-header">🩺 Detector de Neumonía por Rayos X</p>', unsafe_allow_html=True)
st.markdown("---")

# Cargar modelo
model = load_model()

# Sidebar
with st.sidebar:
    st.header("📋 Información del Paciente")
    patient_name = st.text_input("Nombre del Paciente", placeholder="Ej: Juan Pérez")
    document_id = st.text_input("Documento de Identidad", placeholder="Ej: 12345678")
    age = st.number_input("Edad", min_value=0, max_value=120, value=0, step=1)
    notes = st.text_area("Notas adicionales", placeholder="Observaciones médicas...")

    st.markdown("---")
    st.markdown("### ℹ️ Acerca del modelo")
    st.info("""
    **Arquitectura:** EfficientNetV2  
    **Clases:** Normal y Neumonía  
    **Input:** 224x224 RGB
    """)

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar Radiografía")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de rayos X del tórax",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        if st.button("🔍 Analizar Radiografía", type="primary", use_container_width=True):
        if st.button("🔍 Analizar Radiografía", type="primary"):
            with st.spinner("Analizando..."):
                label, prob, all_probs = predict_image(model, image)

                # Guardar en historial
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_name": patient_name or "Sin nombre",
                    "document_id": document_id or "N/A",
                    "age": age if age > 0 else "N/A",
                    "notes": notes or "Sin notas",
                    "prediction": label,
                    "confidence": prob
                }
                st.session_state.history.insert(0, record)

                # Mostrar resultado en col2
                with col2:
                    st.subheader("📊 Resultado del Análisis")

                    # Box de predicción
                    box_class = "normal-box" if label == "NORMAL" else "pneumonia-box"
                    emoji = "✅" if label == "NORMAL" else "⚠️"
                    st.markdown(
                        f'<div class="prediction-box {box_class}">{emoji} {label}</div>',
                        unsafe_allow_html=True
                    )

                    # Confianza
                    st.metric(
                        label="Nivel de Confianza",
                        value=f"{prob * 100:.2f}%",
                        delta=None
                    )

                    # Gráfico de probabilidades
                    st.markdown("#### Distribución de Probabilidades")
                    prob_df = pd.DataFrame({
                        'Clase': CLASS_NAMES,
                        'Probabilidad': all_probs * 100
                    })
                    st.bar_chart(prob_df.set_index('Clase'))

                    # Información del paciente
                    if patient_name or document_id:
                        st.markdown("#### 👤 Datos del Paciente")
                        info_cols = st.columns(2)
                        with info_cols[0]:
                            st.text(f"Nombre: {patient_name or 'N/A'}")
                            st.text(f"Edad: {age if age > 0 else 'N/A'}")
                        with info_cols[1]:
                            st.text(f"Doc: {document_id or 'N/A'}")

                        if notes:
                            st.text_area("Notas:", notes, disabled=True)
                            st.text_area("Notas:", notes, disabled=True, key="notes_display")

                    # Recomendación
                    if label == "PNEUMONIA":
                        st.error("⚠️ **Recomendación:** Se detectó posible neumonía. Consulte con un médico especialista de inmediato.")
                    else:
                        st.success("✅ **Resultado:** No se detectaron signos de neumonía en esta radiografía.")

with col2:
    if uploaded_file is None:
        st.subheader("📊 Resultado del Análisis")
        st.info("👈 Carga una radiografía para comenzar el análisis")

# Historial
st.markdown("---")
st.subheader("📜 Historial de Análisis")

if st.session_state.history:
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.history = []
        st.rerun()

    # Mostrar historial como tabla
    df = pd.DataFrame(st.session_state.history)
    
    # Formatear confianza como porcentaje
    df_display = df.copy()
    df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x*100:.2f}%")
    
    st.dataframe(
        df,
        use_container_width=True,
        df_display,
        hide_index=True,
        column_config={
            "timestamp": "Fecha/Hora",
            "patient_name": "Paciente",
            "document_id": "Documento",
            "age": "Edad",
            "prediction": st.column_config.TextColumn(
                "Diagnóstico",
                width="medium",
            ),
            "confidence": st.column_config.ProgressColumn(
                "Confianza",
                format="%.2f%%",
                min_value=0,
                max_value=1,
            ),
            "prediction": "Diagnóstico",
            "confidence": "Confianza",
            "notes": "Notas"
        }
    )
else:
    st.info("No hay análisis previos. Los resultados aparecerán aquí después de analizar radiografías.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🩺 Sistema de Detección de Neumonía | "
    "Desarrollado con Streamlit & TensorFlow</p>",
    unsafe_allow_html=True
)
