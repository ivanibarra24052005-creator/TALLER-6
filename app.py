
import streamlit as st
import numpy as np
import joblib

model  = joblib.load('knn_iris_model.joblib')
scaler = joblib.load('scaler_iris.joblib')

cluster_names = {
    0: "Grupo 0 - Iris compacta (baja longitud de pétalo)",
    1: "Grupo 1 - Iris intermedia",
    2: "Grupo 2 - Iris grande (alta longitud de pétalo)"
}

st.title("Clasificador de Iris - K-NN + K-Means")
st.markdown("Ingrese las características de una nueva muestra para clasificarla en el grupo correspondiente.")

sepal_length = st.slider("Longitud del sépalo (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Longitud del pétalo (cm)", 1.0, 7.0, 3.7)
petal_width  = st.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.2)

if st.button("Clasificar"):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_new_sc = scaler.transform(X_new)
    pred = model.predict(X_new_sc)[0]
    st.success(f"Grupo asignado: {pred} - {cluster_names.get(pred, 'Grupo desconocido')}")
    st.info("Este resultado indica el segmento al que pertenece la muestra según el agrupamiento K-Means.")
