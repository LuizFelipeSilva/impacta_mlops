import joblib
import pickle
import pandas as pd
import streamlit as st
import mlflow
import mlflow.pyfunc

#import os
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Configuracao da pagina
st.set_page_config(
    page_title="Diamond Price Predictor - Luiz Felipe",
    page_icon="💎",
    layout="centered"
)

# CSS personalizado com cor de fundo e estilo
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #e94560, #533483);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .author-name {
        color: #e94560;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(26, 26, 46, 0.9);
        color: #e94560;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_uri = "models:/diamonds_price_model@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

@st.cache_resource
def load_model_local():
    model_path = "models/model.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Cabecalho personalizado
    st.markdown('<p class="main-header">💎 Previsao de Preco de Diamantes</p>', unsafe_allow_html=True)
    st.markdown('<p class="author-name">Desenvolvido por: Luiz Felipe de Oliveira Lacerda e Silva</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("**Projeto MLOps - Impacta** | Modelo treinado com o dataset `diamonds` do seaborn.")

    model = load_model_local()

    st.subheader("Informe as características do diamante")

    # campos básicos do dataset diamonds
    carat = st.number_input("carat", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    depth = st.number_input("depth", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table = st.number_input("table", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
    x = st.number_input("x (comprimento)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    y = st.number_input("y (largura)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    z = st.number_input("z (altura)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)

    cut = st.selectbox("cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox(
        "clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    )

    if st.button("Prever preço"):
        data = pd.DataFrame(
            {
                "carat": [float(carat)],
                "depth": [float(depth)],
                "table": [float(table)],
                "x": [float(x)],
                "y": [float(y)],
                "z": [float(z)],
                "cut": [str(cut)],
                "color": [str(color)],
                "clarity": [str(clarity)],
            }
        )


        num_cols = ["carat", "depth", "table", "x", "y", "z"]
        data[num_cols] = data[num_cols].astype(float)

        cat_cols = ["cut", "color", "clarity"]
        data[cat_cols] = data[cat_cols].astype(str)

        EXPECTED_COLUMNS = [
            "carat",
            "depth",
            "table",
            "x",
            "y",
            "z",
            "cut",
            "color",
            "clarity",
        ]

        data = data[EXPECTED_COLUMNS]

        prediction = model.predict(data)[0]

        st.subheader("Resultado")
        st.success(f"💎 Preco estimado: **${prediction:,.2f}**")

    # Rodape com assinatura
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p><strong>Projeto MLOps - Impacta</strong></p>
        <p>Desenvolvido por: <span style="color: #e94560;">Luiz Felipe de Oliveira Lacerda e Silva</span></p>
        <p>📧 Disciplina: MLOps - Running ML in Production Environments</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
