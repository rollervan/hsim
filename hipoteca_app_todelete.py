import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="Simulador de Riesgo Hipotecario",
    layout="wide",
    page_icon="🏠"
)

# Estilos CSS para imitar la estética limpia de Fintech/Banca
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 { color: #1f2c3d; }
    h3 { color: #34495e; }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTOR MATEMÁTICO (MONTECARLO) ---

def generar_euribor_vasicek(n_sims, n_meses, r0, media_long_term, kappa, sigma):
    """
    Simula trayectorias del Euríbor usando el modelo de Vasicek.
    r0: Euribor inicial
    media_long_term: Hacia dónde tiende a ir el mercado
    kappa: Velocidad de reversión a la media
    sigma: Volatilidad (riesgo/oscilación)
    """
    dt = 1/12  # Pasos mensuales
    tasas = np.zeros((n_sims, n_meses))
    tasas[:, 0] = r0
    
    # Pre-calcular ruido aleatorio para velocidad
    shocks = np.random.normal(0, np.sqrt(dt), size=(n_sims, n_meses))
    
    for t in range(1, n_meses):
        # Fórmula: dr = a(b-r)dt + sigma*dW
        drift = kappa * (media_long_term - tasas[:, t-1]) * dt
        diffusion = sigma * shocks[:, t]
        tasas[:, t] = tasas[:, t-1] + drift + diffusion
    
    return tasas

def calcular_hipoteca_vectorizada(capital, plazo_anos, tipo_producto, params_producto, matriz_euribor):
    """
    Calcula la evolución de la cuota para 1.000 escenarios simultáneamente.
    """
    n_sims, n_meses = matriz_euribor.shape
    
    # 1. Construir matriz de Tipos de Interés Aplicables según producto
    matriz_tasas_aplicables = np.zeros_like(matriz_euribor)
    
    diferencial = params_producto.get('diferencial', 0.0)
    
    if tipo_producto == "Variable":
        matriz_tasas_aplicables = matriz_euribor + diferencial
        
    elif tipo_producto == "Mixta":
        meses_fijos = params_producto.get('meses_fijos', 0)
        tasa_fija = params_producto.get('tasa_fija', 0.0)
        
        # Llenar todo con variable primero
        matriz_tasas_aplicables = matriz_euribor + diferencial
        # Sobreescribir la parte fija
        if meses_fijos > 0:
            matriz_tasas_aplicables[:, :meses_fijos] = tasa_fija

    elif tipo_producto == "Fija":
        tasa_fija = params_producto.get('tasa_fija', 0.0)
        matriz_tasas_aplicables[:] = tasa_fija

    # Evitar tipos negativos extremos (suelo bancario implícito suele ser 0% en el índice, pero aquí permitimos matemáticas puras)
    # matriz_tasas_aplicables = np.maximum(matriz_tasas_aplicables, 0.0)

    # 2. Bucle de Amortización (Mes a Mes)
    # Usamos float64 para máxima precisión y evitar el error de casting
    saldo = np.full(n_sims, capital, dtype=np.float64) 
    matriz_cuotas = np.zeros((n_sims, n_meses), dtype=np.float64)
    
    for m in range(n_meses):
        meses_pendientes = n_meses - m
        if meses_pendientes <= 0: break
        
        tasas_m = matriz_tasas_aplicables[:, m] / 100 / 12  # Mensual decimal
        
        # Fórmula de cuota francesa vectorizada
        # Si tasa es 0 o muy cercana, división simple
        # Usamos np.where para evitar división por cero si tasa es 0
        tasas_m = np.where(tasas_m == 0, 1e-10, tasas_m)
        
        factor = (1 + tasas_m) ** meses_pendientes
        cuotas_m = saldo * (tasas_m * factor) / (factor - 1)
        
        # Calcular intereses y capital
        intereses_m = saldo * tasas_m
        amort_m = cuotas_m - intereses_m
        
        # Guardar resultado
        matriz_cuotas[:, m] = cuotas_m
        
        # Actualizar saldo (Vectorizado)
        saldo -= amort_m
        
        # Corrección de precisión (evitar -0.0001)
        saldo = np.maximum(saldo, 0)
        
    return matriz_cuotas, matriz_tasas_aplicables

# --- 3. INTERFAZ DE USUARIO ---

st.title("🛡️ Simulador de Incertidumbre Hipotecaria")
st.markdown("""
Esta herramienta utiliza el **Método Montecarlo (1.000 simulaciones)** para proyectar cómo podría comportarse tu cuota en el futuro.
Analiza no solo lo que pagarás hoy, sino el **riesgo** de subidas mañana.
""")

# --- SIDEBAR: DATOS ---
with st.sidebar:
    st.header("1. Datos del Préstamo")
    capital = st.number_input("Capital (€)", value=200000, step=1000)
    plazo = st.slider("Plazo (Años)", 10, 40, 30)
    
    st.header("2. Tipo de Hipoteca")
    tipo = st.selectbox("Producto", ["Mixta", "Variable", "Fija"])
    
    params = {}
    if tipo == "Mixta":
        anos_fijos = st.slider("Años a Tipo Fijo", 1, 15, 5)
        tasa_fija = st.number_input("Tipo Fijo Inicial (%)", value=2.25, step=0.05)
        diferencial = st.number_input("Diferencial posterior (%)", value=0.79, step=0.05)
        params = {'meses_fijos': anos_fijos*12, 'tasa_fija': tasa_fija, 'diferencial': diferencial}
        
    elif tipo == "Variable":
        diferencial = st.number_input("Diferencial + Euríbor (%)", value=0.79, step=0.05)
        params = {'diferencial': diferencial}
        
    elif tipo == "Fija":
        tasa_fija = st.number_input("Tipo Fijo Total (%)", value=2.95, step=0.05)
        params = {'tasa_fija': tasa_fija}

    st.markdown("---")
    st.header("3. Calibración Mercado")
    with st.expander("Ajustes Avanzados (Montecarlo)"):
        euribor_hoy = st.number_input("Euríbor Actual (%)", value=2.80)
        media_largo = st.number_input("Tendencia LP (%)", value=2.50, help="Media histórica esperada")
        volatilidad = st.slider("Volatilidad", 0.1, 2.0, 0.7, help="Incertidumbre del mercado")

# --- 4. EJECUCIÓN ---

if st.button("🚀 Ejecutar Análisis de Riesgo"):
    with st.spinner("Calculando 1.000 futuros posibles..."):
        
        # 1. Generar Escenarios Económicos
        meses_totales = plazo * 12
        n_sims = 1000
        
        matriz_euribor = generar_euribor_vasicek(n_sims, meses_totales, euribor_hoy, media_largo, 0.15, volatilidad)
        
        # 2. Calcular Hipoteca
        matriz_cuotas, matriz_tipos = calcular_hipoteca_vectorizada(capital, plazo, tipo, params, matriz_euribor)
        
        # 3. Estadísticas (Percentiles)
        # P50 = Escenario Central (Lo más probable)
        # P90 = Escenario Pesimista (Risk Management)
        # P10 = Escenario Optimista
        
        p10 = np.percentile(matriz_cuotas, 10, axis=0)
        p50 = np.percentile(matriz_cuotas, 50, axis=0)
        p90 = np.percentile(matriz_cuotas, 90, axis=0)
        
        cuota_inicial = p50[0]
        max_riesgo = np.max(p90)
        
        # --- 5. RESULTADOS ---
        
        st.markdown("### 📊 Resultados del Análisis")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cuota Inicial", f"{cuota_inicial:,.2f} €")
        
        # Lógica de colores para riesgo
        delta_riesgo = max_riesgo - cuota_inicial
        color_riesgo = "normal" if delta_riesgo < 100 else "inverse"
        
        col2.metric("Pico Máximo (Escenario Adverso)", f"{max_riesgo:,.2f} €", 
                   delta=f"+{delta_riesgo:,.0f} €", delta_color=color_riesgo)
        
        media_intereses = np.mean(np.sum(matriz_cuotas, axis=1) - capital)
        col3.metric("Coste Total Intereses (Medio)", f"{media_intereses:,.0f} €")
        
        prob_subida = np.mean(np.max(matriz_cuotas, axis=1) > (cuota_inicial + 200)) * 100
        col4.metric("Prob. subida > 200€", f"{prob_subida:.1f}%")

        # --- GRÁFICO PRINCIPAL (CONO DE INCERTIDUMBRE) ---
        
        st.subheader("Evolución Probabilística de la Cuota")
        
        eje_x_anos = np.arange(meses_totales) / 12
        
        fig = go.Figure()
        
        # Relleno del área de incertidumbre (P10 a P90)
        fig.add_trace(go.Scatter(
            x=np.concatenate([eje_x_anos, eje_x_anos[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill='toself',
            fillcolor='rgba(60, 150, 240, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Rango 80% Probabilidad'
        ))
        
        # Línea Mediana
        fig.add_trace(go.Scatter(
            x=eje_x_anos, y=p50,
            line=dict(color='#2980b9', width=3),
            name='Escenario Central (Mediana)'
        ))
        
        # Línea Pesimista (Risk)
        fig.add_trace(go.Scatter(
            x=eje_x_anos, y=p90,
            line=dict(color='#c0392b', width=2, dash='dot'),
            name='Escenario Pesimista (P90)'
        ))
        
        # Línea Optimista
        fig.add_trace(go.Scatter(
            x=eje_x_anos, y=p10,
            line=dict(color='#27ae60', width=1, dash='dot'),
            name='Escenario Optimista (P10)'
        ))

        fig.update_layout(
            title="Proyección Montecarlo (1.000 Escenarios)",
            xaxis_title="Años",
            yaxis_title="Cuota Mensual (€)",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            height=500
        )
        
        if tipo == "Mixta":
            fig.add_vline(x=anos_fijos, line_dash="dash", annotation_text="Fin Tipo Fijo")
            
        st.plotly_chart(fig, use_container_width=True)
        
        # --- EXPLICACIÓN EXPERTA ---
        st.info(f"""
        **Interpretación del Gráfico:**
        * **Línea Azul (Central):** Es lo que el mercado espera hoy que ocurra.
        * **Línea Roja (Punteada):** Representa un escenario adverso (Top 10% peores casos). Si puedes pagar esta cuota ({max_riesgo:,.0f}€), tu perfil de riesgo es adecuado.
        * **Área Azul Sombreada:** El 80% de los futuros posibles caen dentro de esta zona. Cuanto más ancha sea la zona, más incertidumbre tiene el producto (Variable pura es más ancha que Mixta).
        """)

else:
    # Pantalla de bienvenida / Estado inicial
    st.info("👈 Configura tu hipoteca en el menú lateral y pulsa 'Ejecutar Análisis' para ver la simulación Montecarlo.")
