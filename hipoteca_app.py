import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulador Montecarlo Winvest", layout="wide", page_icon="📈")

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .big-stat { font-size: 1.5rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE CÁLCULO MONTECARLO ---

def generar_trayectorias_euribor(n_simulaciones, n_meses, euribor_actual, media_largo_plazo, volatilidad, velocidad_reversion=0.15):
    """
    Genera trayectorias de Euribor usando el modelo de Vasicek (Reversión a la media).
    dr_t = a(b - r_t)dt + sigma * dW_t
    """
    dt = 1/12  # Pasos mensuales
    tasas = np.zeros((n_simulaciones, n_meses))
    tasas[:, 0] = euribor_actual
    
    # Generamos los componentes aleatorios (Ruido Browniano)
    shocks = np.random.normal(0, np.sqrt(dt), size=(n_simulaciones, n_meses))
    
    for t in range(1, n_meses):
        # Ecuación diferencial estocástica discretizada
        drift = velocidad_reversion * (media_largo_plazo - tasas[:, t-1]) * dt
        diffusion = volatilidad * shocks[:, t]
        tasas[:, t] = tasas[:, t-1] + drift + diffusion
        
        # Suelo del Euribor (opcional, bancos suelen poner 0% si es negativo en variable pura, pero aquí dejamos flotar)
        # tasas[:, t] = np.maximum(tasas[:, t], -0.5) 
        
    return tasas

def calcular_cuota_vectorizada(principal, tasa_anual_vector, meses_restantes):
    """Calcula la cuota mensual para un vector de tasas (numpy array)."""
    r = tasa_anual_vector / 100 / 12
    # Evitar división por cero
    r = np.where(r == 0, 1e-9, r)
    
    numerador = principal * r * (1 + r)**meses_restantes
    denominador = (1 + r)**meses_restantes - 1
    return numerador / denominador

# --- INTERFAZ ---

st.title("🎲 Simulador de Riesgo Hipotecario (Montecarlo)")
st.markdown("""
Esta herramienta simula **1.000 futuros posibles** del Euríbor para analizar el riesgo real de una Hipoteca Mixta/Variable.
Replica la metodología de consultoras como *Winvest* para medir la probabilidad de escenarios adversos.
""")

# --- INPUTS ---
col_conf1, col_conf2 = st.columns([1, 2])

with col_conf1:
    st.subheader("1. Datos del Préstamo")
    capital = st.number_input("Capital (€)", value=200000, step=5000)
    plazo_anos = st.slider("Plazo (Años)", 10, 40, 30)
    
    st.subheader("2. Producto Hipotecario")
    tipo_prod = st.radio("Tipo de Hipoteca", ["Mixta", "Variable"])
    
    dif_variable = st.number_input("Diferencial (%)", value=0.79, step=0.05)
    
    periodo_fijo = 0
    tipo_fijo = 0.0
    if tipo_prod == "Mixta":
        periodo_fijo = st.slider("Años Fijos (Mixta)", 1, 20, 5)
        tipo_fijo = st.number_input("Tipo Fijo Inicial (%)", value=2.50, step=0.05)

with col_conf2:
    st.subheader("3. Calibración Montecarlo (Mercado)")
    st.info("Configura cómo se comportará el Euríbor matemáticamente.")
    
    c1, c2, c3 = st.columns(3)
    euribor_hoy = c1.number_input("Euríbor Actual (%)", value=3.7)
    media_objetivo = c2.number_input("Tendencia a largo plazo (%)", value=2.5, help="Hacia dónde tiende a ir el Euribor en 10-20 años")
    volatilidad = c3.slider("Volatilidad del Mercado", 0.1, 2.0, 0.8, help="Qué tanto oscila el precio. Más alto = más riesgo/incertidumbre.")
    
    if st.button("🔄 Ejecutar 1.000 Simulaciones", type="primary"):
        with st.spinner('Calculando escenarios estocásticos...'):
            # 1. Generar Escenarios de Tipos de Interés
            meses_totales = plazo_anos * 12
            n_sims = 1000
            
            # Matriz: Filas=Simulaciones, Columnas=Meses
            escenarios_euribor = generar_trayectorias_euribor(n_sims, meses_totales, euribor_hoy, media_objetivo, volatilidad)
            
            # 2. Aplicar Diferencial
            escenarios_tipos = escenarios_euribor + dif_variable
            
            # Si es Mixta, sobreescribir los primeros meses con el Tipo Fijo
            if tipo_prod == "Mixta":
                meses_fijos = periodo_fijo * 12
                escenarios_tipos[:, :meses_fijos] = tipo_fijo  # Los primeros meses son fijos y seguros
            
            # 3. Calcular Cuotas Mes a Mes
            # Nota: Para hacerlo vectorizado y rápido, simplificamos asumiendo revisión anual o recálculo continuo
            # Aquí hacemos recálculo continuo para máxima precisión en la simulación
            
            matriz_cuotas = np.zeros((n_sims, meses_totales))
            saldo = np.full(n_sims, capital) # Vector de saldos iniciales
            
            # Loop mensual (necesario porque el saldo depende del mes anterior)
            for m in range(meses_totales):
                meses_pendientes = meses_totales - m
                
                # Tasa para este mes (Simulaciones x 1)
                tasa_mes = escenarios_tipos[:, m]
                
                # Calcular cuota
                cuota_m = calcular_cuota_vectorizada(saldo, tasa_mes, meses_pendientes)
                
                # Calcular intereses y amortización
                interes_m = saldo * (tasa_mes / 100 / 12)
                amort_m = cuota_m - interes_m
                
                # Guardar y actualizar saldo
                matriz_cuotas[:, m] = cuota_m
                saldo -= amort_m
                saldo = np.maximum(saldo, 0) # No saldos negativos

            # 4. Análisis Estadístico (Percentiles)
            # Calculamos el percentil 50 (Mediana), 90 (Pesimista) y 10 (Optimista) para cada mes
            p10_cuota = np.percentile(matriz_cuotas, 10, axis=0)
            p50_cuota = np.percentile(matriz_cuotas, 50, axis=0)
            p90_cuota = np.percentile(matriz_cuotas, 90, axis=0)
            
            # --- VISUALIZACIÓN ---
            st.markdown("---")
            
            # KPI Cards
            kpi1, kpi2, kpi3 = st.columns(3)
            
            cuota_inicial = matriz_cuotas[0,0]
            max_p90 = np.max(p90_cuota)
            prob_subida_brutal = np.mean(np.max(matriz_cuotas, axis=1) > (cuota_inicial * 1.5)) * 100
            
            kpi1.metric("Tu Cuota Inicial", f"{cuota_inicial:,.2f} €")
            kpi2.metric("Techo Riesgo (Escenario Adverso)", f"{max_p90:,.2f} €", 
                        delta=f"+{max_p90-cuota_inicial:,.0f} € posibles", delta_color="inverse")
            kpi3.metric("Probabilidad de Cuota x 1.5", f"{prob_subida_brutal:.1f} %", 
                        help="Probabilidad de que tu cuota llegue a aumentar un 50% en algún momento")

            # Gráfico de Cuotas (Fan Chart)
            st.subheader("Evolución Probabilística de tu Cuota")
            
            eje_x = np.arange(1, meses_totales + 1) / 12 # Años
            
            fig = go.Figure()
            
            # Área de incertidumbre (entre optimista y pesimista)
            fig.add_trace(go.Scatter(
                x=np.concatenate([eje_x, eje_x[::-1]]),
                y=np.concatenate([p90_cuota, p10_cuota[::-1]]),
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Rango de Incertidumbre (80% prob.)'
            ))
            
            # Línea Mediana (Escenario Base)
            fig.add_trace(go.Scatter(
                x=eje_x, y=p50_cuota,
                line=dict(color='rgb(31, 119, 180)', width=3),
                name='Escenario Central (Mediana)'
            ))
            
            # Línea Pesimista (Risk)
            fig.add_trace(go.Scatter(
                x=eje_x, y=p90_cuota,
                line=dict(color='rgb(231, 76, 60)', width=2, dash='dot'),
                name='Escenario Adverso (P90)'
            ))

            fig.update_layout(
                title="Proyección de Cuotas (Montecarlo 1k iteraciones)",
                xaxis_title="Años transcurridos",
                yaxis_title="Cuota Mensual (€)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            
            # Añadir línea vertical donde termina la hipoteca mixta (si aplica)
            if tipo_prod == "Mixta":
                fig.add_vline(x=periodo_fijo, line_dash="dash", line_color="green", annotation_text="Fin Tipo Fijo")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicación del gráfico
            st.info("""
            **¿Cómo leer este gráfico?**
            * La **línea azul** es lo más probable que ocurra.
            * La **línea roja punteada** es un escenario "malo" (el Euribor sube más de lo esperado). Tienes un 90% de probabilidad de pagar MENOS que esa línea.
            * El **área sombreada** representa la incertidumbre del mercado.
            """)

            # Gráfico de Tipos (Euribor simulado)
            with st.expander("Ver Proyecciones del Euribor (Materia Prima)"):
                fig_eur = go.Figure()
                # Mostramos solo 50 trazas aleatorias para no saturar el gráfico
                for i in range(50):
                    fig_eur.add_trace(go.Scatter(
                        x=eje_x, y=escenarios_euribor[i, :],
                        mode='lines',
                        line=dict(color='grey', width=1, check_on_open=True),
                        opacity=0.1,
                        showlegend=False
                    ))
                fig_eur.add_trace(go.Scatter(x=eje_x, y=np.median(escenarios_euribor, axis=0), name="Euribor Mediano", line=dict(color="black", width=2)))
                
                fig_eur.update_layout(title="50 Trayectorias Aleatorias del Euribor", xaxis_title="Año", yaxis_title="Euribor %")
                st.plotly_chart(fig_eur, use_container_width=True)

    else:
        st.info("👈 Ajusta los parámetros y pulsa 'Ejecutar' para iniciar la simulación estocástica.")
