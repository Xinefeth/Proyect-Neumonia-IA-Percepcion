# 🫁 Sistema de Predicción de Neumonía  

Este proyecto implementa un sistema de **detección de neumonía** a partir de **radiografías de tórax** mediante *Deep Learning*.  
El modelo está basado en **EfficientNetV2** y fue entrenado en **Google Colab**.  

La aplicación incluye una **interfaz web con Flask**, que permite:  
- Subir imágenes médicas.  
- Obtener predicciones automáticas.  
- Registrar resultados en una **base de datos SQLite**.  

---

## 📂 Estructura del Proyecto  

    proyecto-neumonia/  
    ├── 📘 README.md              # Documentación principal  
    ├── 🐍 app.py                 # Aplicación Flask principal  
    ├── 🧠 best_effnetv2.keras    # Modelo entrenado con TensorFlow/Keras  
    ├── 📄 requirements.txt       # Dependencias de Python  
    ├── 🚫 .gitignore             # Archivos ignorados por Git  
    ├── 📂 templates/             # Templates HTML  
    │   ├── 🖥️ index.html         # Interfaz principal (subida y predicción)  
    │   └── 📑 records.html       # Listado de registros con resultados  
    ├── 🎨 static/                # Archivos estáticos (CSS, JS, Bootstrap)  
    ├── 🗂️ uploads/               # Imágenes subidas por usuarios  
    └── 🗄️ neumonia.db            # Base de datos SQLite (creada automáticamente)  

---

## ⚙️ Tecnologías utilizadas  

- **Python 3.10+**  
- **TensorFlow 2.17.0 / Keras 3.4.1**  
- **Flask 3.0.3**  
- **SQLAlchemy 2.0**  
- **Pillow** (procesamiento de imágenes)  
- **Bootstrap 5** (frontend responsivo)  

---

## 🧠 Modelo de Deep Learning  

El modelo fue entrenado en **Google Colab** usando **EfficientNetV2B0** con las siguientes configuraciones:  

- **Dataset:** Radiografías de tórax (*Normal* / *Neumonía*).  
- **Preprocesamiento:**  
  - Redimensionamiento a `224x224`.  
  - Normalización con `efficientnet_v2.preprocess_input`.  
- **Entrenamiento:**  
  - Fine-tuning sobre las capas superiores de EfficientNetV2.  
  - Métricas principales: *Accuracy* y *Matriz de Confusión*.  
- **Resultados:**  
  - Precisión superior al **97%** en imágenes de prueba.  

📌 **Fuente del dataset:**  
[Kaggle – Neumonía Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  


---

## 🖥️ Funcionalidades  

- ✅ Subida de radiografía de tórax.  
- ✅ Predicción automática (*Normal* / *Neumonía*).  
- ✅ Registro de resultados en la base de datos.  
- ✅ Consulta de registros anteriores con:  
  - Nombre del paciente.  
  - Edad.  
  - Resultado y probabilidad.  
  - Imagen asociada.  

---

## 🚀 Cómo ejecutar el proyecto  

1. Clonar el repositorio:  
   ```powershell
   git clone https://github.com/usuario/proyecto-neumonia.git
   cd  proyecto-neumonia
   code .

2. Crear un entorno:  
   ```powershell
    python -m venv .venv

3. Activar el entorno virtual:
   ```powershell
    .\.venv\Scripts\Activate

4. Instalar dependencias:
   ```powershell
    pip install --upgrade pip
    pip install -r requirements.txt

5. Ejecutar la aplicación:  
   ```powershell
    python app.py

