import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO PRO
# ==========================================
st.set_page_config(
    page_title="Simulador Hipotecario Pro 4.3", 
    page_icon="🏢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO PARA LOOK "SAAS" ---
st.markdown("""
<style>
    /* Fuente principal más limpia */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Ajustes del encabezado para ganar espacio */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Estilo de Tarjetas para Métricas */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-color: #ced4da;
    }
    
    /* Títulos y Subtítulos con color corporativo */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Sidebar más elegante */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f5;
        border-right: 1px solid #dee2e6;
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        color: #0d47a1;
        border-color: #90caf9;
    }
    
    /* Ocultar elementos default de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE - INTACTO)
# ==========================================
def calcular_hipoteca_core(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc, es_autopromotor, meses_carencia):
    n_meses_total = int(anios * 12)
    
    if es_autopromotor:
        saldo_real = 0.0
        disposicion_mensual = capital / max(1, meses_carencia)
    else:
        saldo_real = round(float(capital), 2)
        disposicion_mensual = 0
        meses_carencia = 0 
        
    saldo_teorico = round(float(capital), 2) 
    
    data = []
    mes_global = 1
    
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    
    for anio in range(anios):
        if modo == 'FIJA':
            tasa_anual = tipo_fijo
        elif modo == 'VARIABLE':
            tasa_anual = puntos_eur[anio] + diferencial
        else: # MIXTA
            if anio < anios_fijos:
                tasa_anual = tipo_fijo
            else:
                val_eur = puntos_eur[idx_var] if idx_var < len(puntos_eur) else puntos_eur[-1]
                tasa_anual = val_eur + diferencial
                idx_var += 1
        
        tasa_mensual = (max(0, tasa_anual) / 100) / 12
        
        for m in range(12):
            meses_restantes = n_meses_total - (mes_global - 1)
            
            en_periodo_carencia = es_autopromotor and (mes_global <= meses_carencia)
            
            if en_periodo_carencia:
                saldo_real += disposicion_mensual
                if saldo_real > capital: saldo_real = capital
                cuota = saldo_real * tasa_mensual
                interes_m = cuota
                capital_m = 0 
                
            else:
                if es_autopromotor and mes_global == meses_carencia + 1:
                      saldo_real = round(float(capital), 2)
                
                if saldo_real <= 0.01:
                    saldo_real = 0
                    cuota = 0
                    interes_m = 0
                    capital_m = 0
                else:
                    base_calc = saldo_teorico if tipo_reduc == 'PLAZO' else saldo_real
                    if base_calc < saldo_real: base_calc = saldo_real
                    
                    if tasa_mensual > 0:
                        try:
                            cuota = base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1)
                        except:
                            cuota = base_calc / meses_restantes
                    else:
                        cuota = base_calc / meses_restantes
                    
                    cuota = round(cuota, 2)
                    interes_m = round(saldo_real * tasa_mensual, 2)
                    capital_m = round(cuota - interes_m, 2)
                    
                    if capital_m > saldo_real:
                        capital_m = saldo_real
                        cuota = round(capital_m + interes_m, 2)

                    saldo_real = round(saldo_real - capital_m, 2)
                    
                    int_teorico = round(saldo_teorico * tasa_mensual, 2)
                    amort_teorica = round(cuota - int_teorico, 2)
                    saldo_teorico = round(saldo_teorico - amort_teorica, 2)
                    if saldo_teorico < 0: saldo_teorico = 0

            data.append({
                'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual if saldo_real > 0 else 0, 
                'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m, 
                'Saldo': saldo_real, 'Amort_Extra': 0,
                'Fase': 'Carencia' if en_periodo_carencia else 'Amortización'
            })
            
            if not en_periodo_carencia:
                if m == 11 and saldo_real > 0 and puntos_amort[anio] > 0:
                    ejec = round(min(puntos_amort[anio], saldo_real), 2)
                    saldo_real = round(saldo_real - ejec, 2)
                    if tipo_reduc == 'CUOTA': saldo_teorico = saldo_real
                    
                    data[-1]['Amort_Extra'] = ejec
                    data[-1]['Capital'] = round(data[-1]['Capital'] + ejec, 2)
            
            mes_global += 1

    return pd.DataFrame(data)

def simular_vasicek(r0, theta, kappa, sigma, anios, n_sims=100):
    dt = 1 
    sims = []
    for _ in range(n_sims):
        camino = [r0]
        for t in range(anios - 1):
            dr = kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()
            camino.append(max(-1.0, camino[-1] + dr))
        sims.append(camino)
    return np.array(sims)

# ==========================================
# 2. INTERFAZ: SIDEBAR MEJORADO
# ==========================================
# Cabecera simulada con HTML
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-size: 2.2rem; color: #1a237e;">Simulador Hipotecario <span style="font-weight:lighter; font-size: 1.5rem;">Pro 4.3</span></h1>
        <p style="margin: 0; color: #666;">Análisis financiero avanzado & Stress Test Euríbor</p>
    </div>
    <div style="background-color: #e8eaf6; padding: 10px 20px; border-radius: 8px; font-weight: bold; color: #1a237e;">
        BETA STABLE
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=60)
    st.title("Parámetros")
    
    with st.expander("👤 Perfil Financiero", expanded=True):
        ingresos = st.number_input("Ingresos Mensuales (€)", value=2500, step=100)
        ahorro_inicial = st.number_input("Ahorro Inicial (€)", value=30000, step=1000)
        precio_vivienda = st.number_input("Valor Vivienda (€)", value=220000, step=5000)
    
    with st.expander("⚙️ Préstamo & Modalidad", expanded=True):
        modo_h = st.selectbox("Tipo de Hipoteca", ["MIXTA", "VARIABLE", "FIJA"])
        
        # --- OPCIÓN AUTOPROMOTOR ---
        es_autopromotor = st.checkbox("🏗️ Autopromoción (Obra)", value=False)
        meses_carencia = 0
        if es_autopromotor:
            meses_carencia = st.number_input("Meses Carencia", value=18, min_value=1, max_value=36)
            st.caption(f"ℹ️ Solo intereses durante {meses_carencia} meses.")
        # ---------------------------
        
        capital_init = st.number_input("Capital a Solicitar (€)", value=180000, step=1000)
        anios_p = st.number_input("Plazo (Años)", value=30, min_value=1)
        tipo_reduc = st.radio("Amortización Extra reduce:", ["PLAZO", "CUOTA"], horizontal=True)
    
    st.markdown("### 🏦 Condiciones")
    
    tipo_fijo = 0.0
    diferencial = 0.0
    anios_fijos = 0
    
    if modo_h == "FIJA":
        tipo_fijo = st.number_input("TIN Fijo (%)", value=2.50, step=0.05)
    elif modo_h == "VARIABLE":
        diferencial = st.number_input("Diferencial + Euríbor (%)", value=0.60, step=0.05)
    elif modo_h == "MIXTA":
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            tipo_fijo = st.number_input("TIN Fijo (%)", value=2.25, step=0.05)
        with col_m2:
            anios_fijos = st.number_input("Años Fijos", value=5, min_value=1, max_value=anios_p-1)
        diferencial = st.number_input("Resto: Dif. + Euríbor (%)", value=0.60, step=0.05)

    with st.expander("🖇️ Gastos & Vinculaciones"):
        col_s1, col_s2 = st.columns(2)
        with col_s1: s_hogar = st.number_input("Seg. Hogar (€/año)", value=250)
        with col_s2: s_vida = st.number_input("Seg. Vida (€/año)", value=180)
        
        st.markdown("**Gastos Mensuales (Vida)**")
        g_comida = st.number_input("Supermercado (€)", value=300, step=50)
        g_suministros = st.number_input("Luz/Agua/Net (€)", value=150, step=10)
        g_gasolina = st.number_input("Transporte (€)", value=100, step=10)
        g_otros = st.number_input("Ocio/Otros (€)", value=200, step=10)


# ==========================================
# 3. CONFIGURACIÓN EURÍBOR
# ==========================================
caminos_eur = []
modo_prev = "N/A"
n_sims = 1

if modo_h != "FIJA":
    with st.sidebar:
        st.markdown("### 📈 Proyección Euríbor")
        modo_prev = st.radio("Modelo de Predicción", ["Estocástico (Monte Carlo)", "Manual (Curva Propia)"])
        
        if modo_prev == "Estocástico (Monte Carlo)":
            n_sims = st.select_slider("Precisión / Sims", [10, 50, 100, 200], value=50)
            theta = st.slider("Media Largo Plazo (θ)", 0.0, 5.0, 2.50, help="Hacia dónde tiende el Euribor")
            sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.50, help="Cuánto oscila el mercado")
            kappa = st.slider("Velocidad de ajuste (κ)", 0.0, 1.0, 0.20)
            r0 = st.number_input("Euríbor Actual (%)", value=2.50)
        else:
            n_sims = 1

    n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
    
    if n_años_var > 0:
        if modo_prev == "Manual (Curva Propia)":
            with st.expander("🛠️ Ajustar Curva de Tipos", expanded=True):
                st.info("Define el valor del Euríbor para cada año variable:")
                eur_list = []
                cols_man = st.columns(3)
                for i in range(n_años_var):
                    val = cols_man[i%3].number_input(f"Año {i + 1 + anios_fijos}", -1.0, 10.0, 2.5, step=0.1, key=f"e{i}")
                    eur_list.append(val)
            caminos_eur = [eur_list]
        else:
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)
else:
    caminos_eur = [[0.0] * anios_p]
    n_sims = 1

# Área de Amortización más limpia
st.markdown("### 💰 Plan de Aportaciones Extraordinarias")
with st.expander("Desplegar Planificador Anual", expanded=False):
    st.caption("Introduce la cantidad que esperas amortizar cada año (además de la cuota).")
    cols_a = st.columns(5)
    amort_list = [cols_a[i%5].number_input(f"Año {i+1}", 0, 50000, 0, step=1000, key=f"a{i}") for i in range(anios_p)]

# ==========================================
# 4. CÁLCULO Y PROCESAMIENTO
# ==========================================
kpis_int, kpis_ahorro, kpis_pat, kpis_seguros = [], [], [], []
cuotas_matrix, eur_matrix = [], []
df_median, df_base_median = None, None

if n_sims > 50: 
    prog_text = st.empty()
    bar = st.progress(0)

total_gastos_vida_mensual = g_comida + g_suministros + g_gasolina + g_otros
coste_mensual_seguros = (s_hogar + s_vida) / 12

for i, camino in enumerate(caminos_eur):
    # Escenario Actual
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    # Escenario Base (sin amortizar)
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    
    # --- CÁLCULOS PATRIMONIALES ---
    df['Seguros_Pagados'] = np.where(df['Saldo'] > 0, coste_mensual_seguros, 0)
    gasto_tot = df['Cuota'] + df['Seguros_Pagados'] + total_gastos_vida_mensual
    
    df['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df['Amort_Extra'].cumsum()
    df['Equity'] = precio_vivienda - df['Saldo']
    df['Patrimonio'] = df['Ahorro_Liquido'] + df['Equity']
    
    kpis_int.append(df['Intereses'].sum())
    kpis_seguros.append(df['Seguros_Pagados'].sum())
    kpis_ahorro.append(df_base['Intereses'].sum() - df['Intereses'].sum())
    kpis_pat.append(df['Patrimonio'].iloc[-1])
    
    cuotas_matrix.append(df['Cuota'].values)
    eur_matrix.append(camino)
    
    if i == 0: 
        df_median = df
        df_base_median = df_base 

    if n_sims > 50: bar.progress((i+1)/n_sims)

if n_sims > 50: 
    bar.empty()
    prog_text.empty()

# Seleccionar escenario mediana
idx_med = np.argsort(kpis_int)[len(kpis_int)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    df_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    df_base_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    
    df_median['Seguros_Pagados'] = np.where(df_median['Saldo'] > 0, coste_mensual_seguros, 0)
    gasto_tot = df_median['Cuota'] + df_median['Seguros_Pagados'] + total_gastos_vida_mensual
    df_median['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df_median['Amort_Extra'].cumsum()
    df_median['Equity'] = precio_vivienda - df_median['Saldo']
    df_median['Patrimonio'] = df_median['Ahorro_Liquido'] + df_median['Equity']

# ==========================================
# 5. DASHBOARD VISUAL (PRO)
# ==========================================
intereses_totales = np.median(kpis_int)
seguros_totales = np.median(kpis_seguros)
coste_total_operacion = intereses_totales + seguros_totales
ahorro_intereses = np.median(kpis_ahorro)
meses_con_hipoteca = len(df_median[df_median['Saldo'] > 0])
meses_ahorrados = (anios_p * 12) - meses_con_hipoteca

st.markdown("## 📊 Resultados del Análisis")

# Contenedor principal de KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Intereses Totales", f"{intereses_totales:,.0f} €", delta="Coste Financiero", delta_color="inverse")
kpi2.metric("Coste Seguros", f"{seguros_totales:,.0f} €", delta="Vinculaciones", delta_color="inverse")
kpi3.metric("Ahorro por Amortizar", f"{ahorro_intereses:,.0f} €", delta="Beneficio Neto", delta_color="normal")
kpi4.metric("Tiempo Reducido", f"{meses_ahorrados // 12} años y {meses_ahorrados % 12} m", delta="Libertad financiera")

st.markdown("---")

# --- PANEL DE RIESGO ---
if n_sims > 1 and modo_h != "FIJA":
    p5_int = np.percentile(kpis_int, 5)
    p95_int = np.percentile(kpis_int, 95)
    
    st.info(f"🎲 **Análisis de Riesgo (Monte Carlo):** Con un 90% de probabilidad, el coste total en intereses oscilará entre **{p5_int:,.0f} €** (optimista) y **{p95_int:,.0f} €** (pesimista).")
elif modo_h == "FIJA":
    st.success("🔒 **Riesgo Cero:** Al ser tipo FIJO, conoces el coste exacto desde el primer día.")

# TABS DE GRÁFICOS
tabs = ["📉 Cuotas & Euríbor", "🛡️ Estrategia Deuda", "💰 Evolución Patrimonio", "📋 Datos Tabla"]
t1, t2, t3, t4 = st.tabs(tabs)

# Estilo común para gráficos
layout_common = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#2c3e50'),
    margin=dict(t=20, b=20, l=40, r=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#e9ecef')
)

with t1:
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown("##### 💶 Evolución Cuota Mensual")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=df_median['Mes'], y=df_median['Cuota'], 
            line=dict(color='#e74c3c', width=3), 
            fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)',
            name='Cuota'
        ))
        if es_autopromotor:
            fig_q.add_vline(x=meses_carencia, line_dash="dash", line_color="#27ae60", annotation_text="Fin Carencia")
        
        fig_q.update_layout(height=300, **layout_common)
        st.plotly_chart(fig_q, use_container_width=True)

    with c_g2:
        if modo_h == "FIJA":
            st.info("Visualización de Euríbor desactivada (Hipoteca Fija).")
        else:
            st.markdown("##### 📈 Proyección Euríbor (Mediana)")
            mat_eur = np.array(eur_matrix)
            p50_eur = np.percentile(mat_eur, 50, axis=0)
            anios_x = np.arange(1, len(p50_eur)+1)
            
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=anios_x, y=p50_eur, line=dict(color='#3498db', width=3), name='Euríbor'))
            fig_e.update_layout(height=300, **layout_common)
            st.plotly_chart(fig_e, use_container_width=True)

with t2:
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### 📉 Ritmo de Amortización")
        fig_race = go.Figure()
        fig_race.add_trace(go.Scatter(x=df_base_median['Saldo'], name='Sin Amortizar', line=dict(color='#95a5a6', dash='dash'))) 
        fig_race.add_trace(go.Scatter(
            x=df_median['Saldo'], 
            name='Con Estrategia', 
            fill='tozeroy', 
            fillcolor='rgba(39, 174, 96, 0.1)',
            line=dict(color='#27ae60', width=3)
        ))
        fig_race.update_layout(height=350, legend=dict(orientation="h", y=1.1), **layout_common)
        st.plotly_chart(fig_race, use_container_width=True)
    
    with g2:
        st.markdown("##### 💸 Interés Acumulado Pagado")
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df_base_median['Mes'], y=df_base_median['Intereses'].cumsum(), name='Base', line=dict(color='#95a5a6', dash='dash')))
        fig_r.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Intereses'].cumsum(), name='Real', line=dict(color='#c0392b', width=3)))
        fig_r.update_layout(height=350, legend=dict(orientation="h", y=1.1), **layout_common)
        st.plotly_chart(fig_r, use_container_width=True)

with t3:
    st.markdown("##### 🏛️ Patrimonio Neto (Ahorro + Valor Casa - Deuda)")
    fig_nw = go.Figure()
    
    gasto_base = df_base_median['Cuota'] + coste_mensual_seguros + total_gastos_vida_mensual
    ahorro_base = ahorro_inicial + (ingresos - gasto_base).cumsum()
    pat_base = ahorro_base + (precio_vivienda - df_base_median['Saldo'])
    
    fig_nw.add_trace(go.Scatter(
        x=df_median['Mes'], y=df_median['Patrimonio'], 
        name='Patrimonio Optimizado', 
        line=dict(color='#8e44ad', width=4)
    ))
    
    fig_nw.add_trace(go.Scatter(
        x=df_base_median['Mes'], y=pat_base, 
        name='Patrimonio Base', 
        line=dict(color='#bdc3c7', dash='dot')
    ))
    
    fig_nw.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", y=1.05), **layout_common)
    st.plotly_chart(fig_nw, use_container_width=True)

with t4:
    st.markdown("##### 📥 Detalle Numérico (Escenario Mediana)")
    numeric_cols = df_median.select_dtypes(include=[np.number]).columns
    st.dataframe(
        df_median.style.format("{:.2f}", subset=numeric_cols).background_gradient(cmap='Blues', subset=['Cuota']), 
        use_container_width=True, 
        height=400
    )
