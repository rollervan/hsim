import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Simulador Financiero Pro - Comparador",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif; color: #2c3e50;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
    div[data-testid="stExpander"] {border: 1px solid #e0e0e0; border-radius: 8px;}
    .css-16idsys p {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE)
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
    
    # Ajustar longitud de puntos euribor y amortizaciones al plazo específico de este escenario
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, int(anios) - len(euribor_puntos)))
    # Recortamos o extendemos la lista de amortizaciones según los años de ESTE escenario
    amort_len = len(amortizaciones)
    if anios > amort_len:
        puntos_amort = list(amortizaciones) + [0] * (int(anios) - amort_len)
    else:
        puntos_amort = list(amortizaciones[:int(anios)])

    idx_var = 0 
    
    for anio in range(int(anios)):
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
                    saldo_real = 0; cuota = 0; interes_m = 0; capital_m = 0
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
# 2. INTERFAZ: SIDEBAR CON MODO COMPARACIÓN
# ==========================================
with st.sidebar:
    st.title("Parámetros")
    
    # --- MODO COMPARACIÓN ---
    st.markdown("### ⚖️ Configuración")
    comparar = st.checkbox("🆚 Comparar dos escenarios", value=False)
    
    with st.expander("Perfil Económico (Común)", expanded=not comparar):
        ingresos = st.number_input("Ingresos netos (€)", value=2500, step=100)
        ahorro_inicial = st.number_input("Ahorro inicial (€)", value=10000, step=1000)
        precio_vivienda = st.number_input("Valor Vivienda (€)", value=200000, step=5000)
        capital_init_global = st.number_input("Capital Préstamo (€)", value=180000, step=1000)

    st.markdown("---")
    
    # Lógica para mostrar inputs simples o dobles
    if comparar:
        st.subheader("Escenario A vs B")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("#### 🅰️ Opción A")
            modo_A = st.selectbox("Modo A", ["MIXTA", "VARIABLE", "FIJA"], key="mA")
            anios_A = st.number_input("Años A", value=25, key="yA")
            
            tipo_fijo_A = 0.0
            diferencial_A = 0.0
            anios_fijos_A = 0
            
            if modo_A == "FIJA":
                tipo_fijo_A = st.number_input("TIN A (%)", value=2.50, step=0.05, key="tfA")
            elif modo_A == "VARIABLE":
                diferencial_A = st.number_input("Dif. A (%)", value=0.55, step=0.05, key="dfA")
            elif modo_A == "MIXTA":
                tipo_fijo_A = st.number_input("Fijo A (%)", value=2.25, step=0.05, key="mfaA")
                anios_fijos_A = st.number_input("Años Fijos A", value=5, key="myaA")
                diferencial_A = st.number_input("Dif. Var A", value=0.55, step=0.05, key="mdaA")

        with colB:
            st.markdown("#### 🅱️ Opción B")
            modo_B = st.selectbox("Modo B", ["MIXTA", "VARIABLE", "FIJA"], index=2, key="mB")
            anios_B = st.number_input("Años B", value=20, key="yB")
            
            tipo_fijo_B = 0.0
            diferencial_B = 0.0
            anios_fijos_B = 0
            
            if modo_B == "FIJA":
                tipo_fijo_B = st.number_input("TIN B (%)", value=2.90, step=0.05, key="tfB")
            elif modo_B == "VARIABLE":
                diferencial_B = st.number_input("Dif. B (%)", value=0.45, step=0.05, key="dfB")
            elif modo_B == "MIXTA":
                tipo_fijo_B = st.number_input("Fijo B (%)", value=2.15, step=0.05, key="mfaB")
                anios_fijos_B = st.number_input("Años Fijos B", value=3, key="myaB")
                diferencial_B = st.number_input("Dif. Var B", value=0.45, step=0.05, key="mdaB")
                
    else:
        st.subheader("Condiciones del Préstamo")
        modo_A = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
        anios_A = st.number_input("Duración (Años)", value=25, min_value=1)
        
        tipo_fijo_A = 0.0
        diferencial_A = 0.0
        anios_fijos_A = 0
        
        c1, c2 = st.columns(2)
        if modo_A == "FIJA":
            tipo_fijo_A = c1.number_input("TIN Fijo (%)", value=2.50, step=0.05)
        elif modo_A == "VARIABLE":
            diferencial_A = c1.number_input("Diferencial (%)", value=0.55, step=0.05)
        elif modo_A == "MIXTA":
            tipo_fijo_A = c1.number_input("Fijo (%)", value=2.25, step=0.05)
            anios_fijos_A = c2.number_input("Años Fijos", value=5)
            diferencial_A = st.number_input("Dif. Variable (%)", value=0.55, step=0.05)
            
        # En modo simple, B es igual a A para no romper el código
        modo_B, anios_B = modo_A, anios_A
        tipo_fijo_B, diferencial_B, anios_fijos_B = tipo_fijo_A, diferencial_A, anios_fijos_A

    st.markdown("---")
    tipo_reduc = st.radio("Amortizar reduciendo:", ["PLAZO", "CUOTA"])

    with st.expander("Gastos y Vinculaciones", expanded=False):
        s_hogar = st.number_input("Seguro Hogar (€/año)", value=300)
        s_vida = st.number_input("Seguro Vida (€/año)", value=300)
        st.markdown("**Gastos Mensuales**")
        g_comida = st.number_input("Comida", value=300)
        g_suministros = st.number_input("Suministros", value=150)
        g_gasolina = st.number_input("Transporte", value=100)
        g_otros = st.number_input("Otros", value=200)

    # Configuración Euríbor
    caminos_eur = []
    n_sims = 1
    
    # Si alguno de los dos escenarios usa variable/mixta, necesitamos simulación
    necesita_euribor = (modo_A != "FIJA") or (modo_B != "FIJA")
    
    if necesita_euribor:
        st.markdown("---")
        with st.expander("Simulación Euríbor", expanded=True):
            modo_prev = st.selectbox("Método", ["Monte Carlo (Vasicek)", "Manual"])
            
            if modo_prev == "Monte Carlo (Vasicek)":
                n_sims = st.select_slider("Iteraciones", [50, 100, 500, 1000], value=100)
                st.caption("Parámetros Estocásticos")
                theta = st.slider("Media (Long Term)", 0.0, 5.0, 2.25)
                sigma = st.slider("Volatilidad", 0.0, 2.0, 0.60)
                kappa = st.slider("Reversión", 0.0, 1.0, 0.30)
                r0 = st.number_input("Euríbor Actual", value=2.24)
            else:
                n_sims = 1

        # Calculamos el plazo máximo para generar suficientes puntos de euribor
        max_anios = max(anios_A, anios_B)
        
        if modo_prev == "Manual":
            eur_list = []
            for i in range(max_anios):
                 eur_list.append(st.slider(f"Año {i + 1}", -1.0, 7.0, 3.2, key=f"e{i}"))
            caminos_eur = [eur_list]
        else:
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, max_anios, n_sims)
    else:
        # Si ambos son fijos, generamos ceros, da igual
        caminos_eur = [[0.0] * max(anios_A, anios_B)]
        n_sims = 1

# ==========================================
# 3. ÁREA PRINCIPAL
# ==========================================

st.title("Simulador Financiero Pro 4.4")
if comparar:
    st.caption("🅰️ Escenario A vs 🅱️ Escenario B")
else:
    st.markdown("Herramienta de análisis hipotecario y proyección de riesgo.")

# AMORTIZACIONES CON SLIDERS (Hasta 10.000€)
# Nota: Aplicamos la misma estrategia de amortización a ambos escenarios para ver cuál responde mejor
with st.expander("Estrategia de Amortización Anticipada (Se aplica a ambos escenarios)"):
    st.info("Ajusta el capital extra que quieres aportar cada año (Máx. 10.000€)")
    cols_a = st.columns(4) 
    amort_list = []
    max_anios_slider = max(anios_A, anios_B)
    for i in range(max_anios_slider):
        val = cols_a[i % 4].slider(f"Año {i+1}", 0, 10000, 0, step=500, key=f"s_a{i}")
        amort_list.append(val)

# ==========================================
# CÁLCULOS (DOBLE BUCLE SI ES NECESARIO)
# ==========================================
# Variables para almacenar resultados de A y B
kpis_int_A, kpis_int_B = [], []
df_median_A, df_median_B = None, None

total_gastos = g_comida + g_suministros + g_gasolina + g_otros
coste_mes_seguros = (s_hogar + s_vida) / 12

if n_sims > 100: prog_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # --- ESCENARIO A ---
    df_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino, amort_list, tipo_reduc, False, 0)
    df_A['Seguros'] = np.where(df_A['Saldo'] > 0, coste_mes_seguros, 0)
    kpis_int_A.append(df_A['Intereses'].sum() + df_A['Seguros'].sum()) # Guardamos Coste Total (Int+Seg)
    
    # --- ESCENARIO B (Solo si comparar es True, o si es False B=A implícitamente por inputs) ---
    df_B = calcular_hipoteca_core(capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B, modo_B, camino, amort_list, tipo_reduc, False, 0)
    df_B['Seguros'] = np.where(df_B['Saldo'] > 0, coste_mes_seguros, 0)
    kpis_int_B.append(df_B['Intereses'].sum() + df_B['Seguros'].sum())

    # Guardamos los DF de la primera iteración (o mediana más tarde) para gráficos
    if i == 0: 
        df_median_A = df_A
        df_median_B = df_B
        
    if n_sims > 100: prog_bar.progress((i+1)/n_sims)

if n_sims > 100: prog_bar.empty()

# Selección de escenario Mediana basada en A
idx_med = np.argsort(kpis_int_A)[len(kpis_int_A)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    # Recalculamos A con camino mediano
    df_median_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino_med, amort_list, tipo_reduc, False, 0)
    df_median_A['Seguros'] = np.where(df_median_A['Saldo'] > 0, coste_mes_seguros, 0)
    # Recalculamos B con el MISMO camino mediano (para comparar peras con peras)
    df_median_B = calcular_hipoteca_core(capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B, modo_B, camino_med, amort_list, tipo_reduc, False, 0)
    df_median_B['Seguros'] = np.where(df_median_B['Saldo'] > 0, coste_mes_seguros, 0)

# --- KPIs FINALES ---
# Coste Total (Intereses + Seguros)
coste_A = df_median_A['Intereses'].sum() + df_median_A['Seguros'].sum()
coste_B = df_median_B['Intereses'].sum() + df_median_B['Seguros'].sum()

# Tiempo Total (Meses hasta saldo 0)
meses_A = len(df_median_A[df_median_A['Saldo'] > 0])
meses_B = len(df_median_B[df_median_B['Saldo'] > 0])

# Cuota Inicial
cuota_ini_A = df_median_A.iloc[0]['Cuota']
cuota_ini_B = df_median_B.iloc[0]['Cuota']

# ==========================================
# DASHBOARD COMPARATIVO O INDIVIDUAL
# ==========================================

if comparar:
    st.markdown("### ⚖️ Comparativa Directa")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    # 1. COSTE TOTAL
    dif_coste = coste_B - coste_A
    col_c1.metric(
        "Coste Total (Intereses + Seguros)", 
        f"{coste_A:,.0f} € vs {coste_B:,.0f} €",
        f"{dif_coste:,.0f} € (Diferencia)",
        delta_color="inverse" # Rojo si B es más caro, Verde si B es más barato
    )
    
    # 2. TIEMPO TOTAL
    dif_meses = meses_B - meses_A
    
    def formato_tiempo(m):
        a = m // 12
        r = m % 12
        return f"{a}a {r}m"
        
    col_c2.metric(
        "Tiempo Real de Pago",
        f"{formato_tiempo(meses_A)} vs {formato_tiempo(meses_B)}",
        f"{dif_meses} meses",
        delta_color="inverse"
    )
    
    # 3. CUOTA INICIAL
    dif_cuota = cuota_ini_B - cuota_ini_A
    col_c3.metric(
        "Cuota Inicial",
        f"{cuota_ini_A:,.0f} € vs {cuota_ini_B:,.0f} €",
        f"{dif_cuota:,.0f} €",
        delta_color="inverse"
    )
    
    st.markdown("---")

else:
    # MODO INDIVIDUAL (Original + Ahorro Tiempo)
    # Lógica de Ahorro de Tiempo (Años y Meses) vs Base teórica
    meses_ahorrados_total = (anios_A * 12) - meses_A
    ahorro_anios = meses_ahorrados_total // 12
    ahorro_meses = meses_ahorrados_total % 12

    if ahorro_anios > 0 and ahorro_meses > 0:
        texto_tiempo = f"-{ahorro_anios} años y {ahorro_meses} meses"
    elif ahorro_anios > 0:
        texto_tiempo = f"-{ahorro_anios} años"
    elif ahorro_meses > 0:
        texto_tiempo = f"-{ahorro_meses} meses"
    else:
        texto_tiempo = "Sin reducción"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cuota Inicial", f"{cuota_ini_A:,.2f} €", f"{df_median_A.iloc[0]['Tasa']:.2f}% TIN")
    k2.metric("Total Intereses", f"{df_median_A['Intereses'].sum():,.0f} €", delta_color="inverse")
    k3.metric("Coste Operación (Int+Seg)", f"{coste_A:,.0f} €", delta_color="inverse")
    k4.metric("Ahorro por Amortizar", f"{(coste_A - coste_A):,.0f} € (Ref)", texto_tiempo) # Aquí habría que calcular vs base sin amortizar si queremos métrica exacta
    
    st.markdown("---")


# GRÁFICOS
tabs = st.tabs(["📊 Evolución Saldo", "💰 Comparativa Acumulada", "📑 Datos Detallados"])

with tabs[0]:
    st.subheader("Reducción de Deuda (Saldo Vivo)")
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'], fill='tozeroy', name='Escenario A', line=dict(color='#0055aa')))
    if comparar:
        fig_s.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Saldo'], name='Escenario B', line=dict(color='#ff7f0e', dash='dash', width=3)))
    fig_s.update_layout(template='plotly_white', height=400, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_s, use_container_width=True)

with tabs[1]:
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        st.subheader("Intereses Acumulados")
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Intereses'].cumsum(), name='Intereses A', line=dict(color='#0055aa')))
        if comparar:
             fig_i.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Intereses'].cumsum(), name='Intereses B', line=dict(color='#ff7f0e', dash='dash')))
        fig_i.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_i, use_container_width=True)
        
    with c_a2:
        st.subheader("Evolución Cuota")
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Cuota'], name='Cuota A', line=dict(color='#0055aa')))
        if comparar:
            fig_c.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Cuota'], name='Cuota B', line=dict(color='#ff7f0e', dash='dash')))
        fig_c.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_c, use_container_width=True)

with tabs[2]:
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown("#### Datos Escenario A")
        num_cols = df_median_A.select_dtypes(include=[np.number]).columns
        st.dataframe(df_median_A.style.format("{:.2f}", subset=num_cols), use_container_width=True, height=300)
    
    if comparar:
        with c_d2:
            st.markdown("#### Datos Escenario B")
            st.dataframe(df_median_B.style.format("{:.2f}", subset=num_cols), use_container_width=True, height=300)
