import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Simulador Financiero Pro - Integrado",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif; color: #2c3e50;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
    div[data-testid="stExpander"] {border: 1px solid #e0e0e0; border-radius: 8px;}
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
    
    # Ajustamos longitudes para evitar errores de índice
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, int(anios) - len(euribor_puntos)))
    
    # Ajustamos amortizaciones al plazo actual
    len_amort = len(amortizaciones)
    if int(anios) > len_amort:
        puntos_amort = list(amortizaciones) + [0] * (int(anios) - len_amort)
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
# 2. INTERFAZ: SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Parámetros")
    
    # TOGGLE DE COMPARACIÓN
    st.markdown("### ⚙️ Modo de Análisis")
    comparar = st.checkbox("🆚 Comparar dos escenarios", value=False)
    
    # INICIALIZACIÓN SEGURA DE VARIABLES PARA EVITAR NAMEERROR
    es_autopromotor = False
    meses_carencia = 0
    
    with st.expander("Perfil Económico (Común)", expanded=not comparar):
        # TUS VALORES ORIGINALES
        ingresos = st.number_input("Ingresos netos (€)", value=2500, step=100)
        ahorro_inicial = st.number_input("Ahorro inicial (€)", value=0, step=1000)
        precio_vivienda = st.number_input("Valor Vivienda (€)", value=0, step=5000)
        capital_init_global = st.number_input("Capital Préstamo (€)", value=180000, step=1000)

    st.markdown("---")
    
    if comparar:
        # --- MODO COMPARACIÓN (A vs B) ---
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
                anios_fijos_A = st.number_input("Años Fijos A", value=7, key="myaA") # DEFAULT: 7
                diferencial_A = st.number_input("Dif. Var A", value=0.55, step=0.05, key="mdaA")

        with colB:
            st.markdown("#### 🅱️ Opción B")
            modo_B = st.selectbox("Modo B", ["MIXTA", "VARIABLE", "FIJA"], index=2, key="mB")
            anios_B = st.number_input("Años B", value=25, key="yB")
            
            tipo_fijo_B = 0.0
            diferencial_B = 0.0
            anios_fijos_B = 0
            
            if modo_B == "FIJA":
                tipo_fijo_B = st.number_input("TIN B (%)", value=2.75, step=0.05, key="tfB")
            elif modo_B == "VARIABLE":
                diferencial_B = st.number_input("Dif. B (%)", value=0.55, step=0.05, key="dfB")
            elif modo_B == "MIXTA":
                tipo_fijo_B = st.number_input("Fijo B (%)", value=2.25, step=0.05, key="mfaB")
                anios_fijos_B = st.number_input("Años Fijos B", value=7, key="myaB") # DEFAULT: 7
                diferencial_B = st.number_input("Dif. Var B", value=0.55, step=0.05, key="mdaB")
                
    else:
        # --- MODO INDIVIDUAL (CLÁSICO) ---
        st.subheader("Préstamo")
        modo_A = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
        
        # AQUÍ SOBRESCRIBIMOS LA VARIABLE CON EL INPUT (DEFAULT TRUE)
        es_autopromotor = st.checkbox("Es Autopromoción", value=True)
        
        if es_autopromotor:
            meses_carencia = st.number_input("Meses carencia", value=11, min_value=1, max_value=36)
            
        anios_A = st.number_input("Duración (Años)", value=25, min_value=1)
        
        st.subheader("Condiciones")
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
            # DEFAULT: 7 AÑOS
            anios_fijos_A = c2.number_input("Años Fijos", value=7)
            diferencial_A = st.number_input("Dif. Variable (%)", value=0.55, step=0.05)
            
        # Replicamos variables para que el código B no rompa
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
    
    necesita_euribor = (modo_A != "FIJA") or (modo_B != "FIJA" and comparar)
    
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

        max_anios = max(anios_A, anios_B)
        
        if modo_prev == "Manual":
            eur_list = []
            for i in range(max_anios):
                 eur_list.append(st.slider(f"Año {i + 1}", -1.0, 7.0, 3.2, key=f"e{i}"))
            caminos_eur = [eur_list]
        else:
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, max_anios, n_sims)
    else:
        caminos_eur = [[0.0] * max(anios_A, anios_B)]
        n_sims = 1

# ==========================================
# 3. ÁREA PRINCIPAL
# ==========================================

st.title("Simulador Financiero Pro 4.5")
if comparar:
    st.caption("🅰️ Escenario A vs 🅱️ Escenario B")
else:
    st.markdown("Herramienta de análisis hipotecario y proyección de riesgo.")

# AMORTIZACIONES CON SLIDERS (Hasta 10.000€)
with st.expander("Estrategia de Amortización Anticipada"):
    st.info("Ajusta el capital extra que quieres aportar cada año (Máx. 10.000€)")
    cols_a = st.columns(4) 
    amort_list = []
    max_anios_slider = max(anios_A, anios_B)
    for i in range(max_anios_slider):
        val = cols_a[i % 4].slider(f"Año {i+1}", 0, 10000, 0, step=500, key=f"s_a{i}")
        amort_list.append(val)

# ==========================================
# CÁLCULOS
# ==========================================
kpis_int_A, kpis_int_B = [], []
kpis_pat_A = [] # Necesario para patrimonio individual
kpis_ahorro_A = [] # Necesario para ahorro individual
eur_matrix = [] # Necesario para gráfico euribor

df_median_A, df_median_B = None, None
df_base_median_A = None # Para comparativa 'sin amortizar' en modo individual

total_gastos = g_comida + g_suministros + g_gasolina + g_otros
coste_mes_seguros = (s_hogar + s_vida) / 12

if n_sims > 100: prog_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # --- ESCENARIO A ---
    # En modo Comparar forzamos Autopromotor a False para evitar complejidad, 
    # o usamos el valor del checkbox si es modo Individual.
    ap_flag = es_autopromotor if not comparar else False
    carencia_val = meses_carencia if not comparar else 0
    
    df_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino, amort_list, tipo_reduc, ap_flag, carencia_val)
    
    # Cálculos adicionales para Modo Individual (Patrimonio, Ahorro vs Base)
    if not comparar:
        df_base_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino, [0]*anios_A, 'PLAZO', ap_flag, carencia_val)
        kpis_ahorro_A.append(df_base_A['Intereses'].sum() - df_A['Intereses'].sum())
    
    df_A['Seguros'] = np.where(df_A['Saldo'] > 0, coste_mes_seguros, 0)
    gasto_tot_A = df_A['Cuota'] + df_A['Seguros'] + total_gastos
    
    # Patrimonio
    df_A['Ahorro_Liq'] = ahorro_inicial + (ingresos - gasto_tot_A).cumsum() - df_A['Amort_Extra'].cumsum()
    df_A['Patrimonio'] = df_A['Ahorro_Liq'] + (precio_vivienda - df_A['Saldo'])
    
    kpis_int_A.append(df_A['Intereses'].sum() + df_A['Seguros'].sum()) 
    if not comparar:
        kpis_pat_A.append(df_A['Patrimonio'].iloc[-1])
        eur_matrix.append(camino)

    # --- ESCENARIO B ---
    if comparar:
        df_B = calcular_hipoteca_core(capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B, modo_B, camino, amort_list, tipo_reduc, False, 0)
        df_B['Seguros'] = np.where(df_B['Saldo'] > 0, coste_mes_seguros, 0)
        kpis_int_B.append(df_B['Intereses'].sum() + df_B['Seguros'].sum())
        if i == 0: df_median_B = df_B

    # Guardar primera iteración
    if i == 0: 
        df_median_A = df_A
        if not comparar: df_base_median_A = df_base_A
        
    if n_sims > 100: prog_bar.progress((i+1)/n_sims)

if n_sims > 100: prog_bar.empty()

# MEDIANA
idx_med = np.argsort(kpis_int_A)[len(kpis_int_A)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    
    ap_flag = es_autopromotor if not comparar else False
    carencia_val = meses_carencia if not comparar else 0

    # Recalcular A
    df_median_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino_med, amort_list, tipo_reduc, ap_flag, carencia_val)
    df_median_A['Seguros'] = np.where(df_median_A['Saldo'] > 0, coste_mes_seguros, 0)
    df_median_A['Ahorro_Liq'] = ahorro_inicial + (ingresos - (df_median_A['Cuota'] + df_median_A['Seguros'] + total_gastos)).cumsum() - df_median_A['Amort_Extra'].cumsum()
    df_median_A['Patrimonio'] = df_median_A['Ahorro_Liq'] + (precio_vivienda - df_median_A['Saldo'])
    
    if not comparar:
        df_base_median_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino_med, [0]*anios_A, 'PLAZO', ap_flag, carencia_val)

    if comparar:
        # Recalcular B con el mismo camino
        df_median_B = calcular_hipoteca_core(capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B, modo_B, camino_med, amort_list, tipo_reduc, False, 0)
        df_median_B['Seguros'] = np.where(df_median_B['Saldo'] > 0, coste_mes_seguros, 0)

# KPIs Generales
coste_A = df_median_A['Intereses'].sum() + df_median_A['Seguros'].sum()
meses_A = len(df_median_A[df_median_A['Saldo'] > 0])

# Aquí estaba el problema previo del NameError. Ahora es_autopromotor y comparar están siempre definidos.
idx_ref = 0 if not (es_autopromotor and not comparar) else meses_carencia
# Protección extra por si idx_ref se sale de rango (raro pero posible en carencias largas)
if idx_ref >= len(df_median_A): idx_ref = 0

cuota_ini_A = df_median_A.iloc[idx_ref]['Cuota']

# ==========================================
# DASHBOARD: LÓGICA DE VISUALIZACIÓN
# ==========================================

if comparar:
    # --- MODO COMPARACIÓN (SIMPLIFICADO PARA ENFRENTAMIENTO) ---
    coste_B = df_median_B['Intereses'].sum() + df_median_B['Seguros'].sum()
    meses_B = len(df_median_B[df_median_B['Saldo'] > 0])
    cuota_ini_B = df_median_B.iloc[0]['Cuota']

    st.markdown("### ⚖️ Comparativa Directa")
    col_c1, col_c2, col_c3 = st.columns(3)
    
    dif_coste = coste_B - coste_A
    col_c1.metric("Coste Total", f"{coste_A:,.0f} € vs {coste_B:,.0f} €", f"{dif_coste:,.0f} € (Dif)", delta_color="inverse")
    
    dif_meses = meses_B - meses_A
    def fmt_t(m): return f"{m//12}a {m%12}m"
    col_c2.metric("Tiempo Pago", f"{fmt_t(meses_A)} vs {fmt_t(meses_B)}", f"{dif_meses} meses", delta_color="inverse")
    
    dif_cuota = cuota_ini_B - cuota_ini_A
    col_c3.metric("Cuota Inicial", f"{cuota_ini_A:,.0f} € vs {cuota_ini_B:,.0f} €", f"{dif_cuota:,.0f} €", delta_color="inverse")
    
    st.markdown("---")
    
    tabs = st.tabs(["📊 Evolución Saldo", "💰 Comparativa Acumulada", "📑 Datos"])
    
    with tabs[0]:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'], fill='tozeroy', name='Escenario A', line=dict(color='#0055aa')))
        fig_s.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Saldo'], name='Escenario B', line=dict(color='#ff7f0e', dash='dash', width=3)))
        st.plotly_chart(fig_s, use_container_width=True)
    
    with tabs[1]:
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            fig_i = go.Figure()
            fig_i.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Intereses'].cumsum(), name='Intereses A', line=dict(color='#0055aa')))
            fig_i.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Intereses'].cumsum(), name='Intereses B', line=dict(color='#ff7f0e', dash='dash')))
            st.plotly_chart(fig_i, use_container_width=True)
        with c_a2:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Cuota'], name='Cuota A', line=dict(color='#0055aa')))
            fig_c.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Cuota'], name='Cuota B', line=dict(color='#ff7f0e', dash='dash')))
            st.plotly_chart(fig_c, use_container_width=True)
            
    with tabs[2]:
        st.dataframe(df_median_A, use_container_width=True, height=200)
        st.dataframe(df_median_B, use_container_width=True, height=200)

else:
    # --- MODO INDIVIDUAL (COMPLETO - RESTAURADO) ---
    # Recuperamos la lógica de ahorro de tiempo detallada
    meses_ahorrados = (anios_A * 12) - meses_A
    a_save = meses_ahorrados // 12
    m_save = meses_ahorrados % 12
    
    if a_save > 0 and m_save > 0: txt_tiempo = f"-{a_save} años y {m_save} meses"
    elif a_save > 0: txt_tiempo = f"-{a_save} años"
    elif m_save > 0: txt_tiempo = f"-{m_save} meses"
    else: txt_tiempo = "0 meses"

    ahorro_int = np.median(kpis_ahorro_A)

    st.markdown("### Resumen Ejecutivo")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cuota Inicial", f"{cuota_ini_A:,.2f} €", f"{df_median_A.iloc[idx_ref]['Tasa']:.2f}% TIN")
    k2.metric("Total Intereses", f"{df_median_A['Intereses'].sum():,.0f} €", delta_color="inverse")
    k3.metric("Coste Operación (Int+Seg)", f"{coste_A:,.0f} €", delta_color="inverse")
    k4.metric("Ahorro por Amortizar", f"{ahorro_int:,.0f} €", txt_tiempo)
    
    st.markdown("---")

    # PESTAÑAS COMPLETAS RESTAURADAS
    tabs = st.tabs(["📊 Evolución", "📉 Comparativa Amortización", "💰 Patrimonio", "🎲 Riesgo (Monte Carlo)"])

    # TAB 1: EVOLUCIÓN (CON BANDAS)
    with tabs[0]:
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            st.subheader("Euríbor: Proyección e Incertidumbre")
            if modo_A == "FIJA":
                st.info("Tipo Fijo: Sin incertidumbre de mercado.")
            else:
                mat = np.array(eur_matrix)
                p10, p50, p90 = np.percentile(mat, [10, 50, 90], axis=0)
                x_ax = np.arange(1, len(p50)+1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_ax, y=p90, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x_ax, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,250,0.15)', name='Rango 80%'))
                fig.add_trace(go.Scatter(x=x_ax, y=p50, mode='lines', line=dict(color='#0055aa', width=3), name='Mediana'))
                fig.update_layout(template='plotly_white', height=350, margin=dict(t=30), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)
        with c_e2:
            st.subheader("Cuota Mensual")
            fig2 = px.line(df_median_A, x='Mes', y='Cuota')
            fig2.update_traces(line_color='#d9534f', line_width=2.5)
            fig2.update_layout(template='plotly_white', height=350)
            if es_autopromotor:
                fig2.add_vline(x=meses_carencia, line_dash="dot", annotation_text="Fin Carencia")
            st.plotly_chart(fig2, use_container_width=True)

    # TAB 2: AMORTIZACIÓN (DETALLE)
    with tabs[1]:
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.subheader("Pago de Intereses Acumulado")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=df_base_median_A['Intereses'].cumsum(), name='Sin Amortizar', line=dict(color='gray', dash='dash')))
            fig3.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Intereses'].cumsum(), name='Con Amortización', line=dict(color='#d9534f', width=3)))
            fig3.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig3, use_container_width=True)
        with c_a2:
            st.subheader("Reducción de Deuda")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=df_base_median_A['Saldo'], name='Saldo Base', line=dict(color='gray', dash='dash')))
            fig4.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'], fill='tozeroy', name='Saldo Real', line=dict(color='#5cb85c')))
            fig4.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig4, use_container_width=True)

    # TAB 3: PATRIMONIO (RESTAURADO)
    with tabs[2]:
        st.subheader("Evolución del Patrimonio Neto")
        fig5 = go.Figure()
        # Recalcular base para patrimonio
        g_base = df_base_median_A['Cuota'] + coste_mes_seguros + total_gastos
        ah_base = ahorro_inicial + (ingresos - g_base).cumsum()
        pat_base = ah_base + (precio_vivienda - df_base_median_A['Saldo'])
        
        fig5.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=pat_base, name='Escenario Base', line=dict(color='gray', dash='dot')))
        fig5.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Patrimonio'], name='Escenario Actual', line=dict(color='#6f42c1', width=3)))
        fig5.update_layout(template='plotly_white', height=400, hovermode="x unified")
        st.plotly_chart(fig5, use_container_width=True)

    # TAB 4: RIESGO (RESTAURADO)
    with tabs[3]:
        if modo_A == "FIJA":
            st.success("✅ Hipoteca FIJA: Coste determinista.")
        elif n_sims < 10:
            st.warning("⚠️ Aumenta iteraciones para ver riesgo.")
        else:
            st.subheader("Probabilidad de Coste de Intereses")
            p5, p50, p95 = np.percentile(kpis_int_A, [5, 50, 95])
            
            c_r1, c_r2 = st.columns([2, 1])
            with c_r1:
                fig_h = px.histogram(x=kpis_int_A, nbins=30, labels={'x': 'Intereses Totales (€)'}, color_discrete_sequence=['#8884d8'])
                fig_h.add_vline(x=p5, line_dash="dash", line_color="green", annotation_text="P5")
                fig_h.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text="P95")
                fig_h.update_layout(template='plotly_white', height=400, showlegend=False)
                st.plotly_chart(fig_h, use_container_width=True)
            with c_r2:
                st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:15px; border-radius:5px;">
                <b>Mejor Caso (P10):</b> {np.percentile(kpis_int_A, 10):,.0f}€<br>
                <b>Mediana:</b> {p50:,.0f}€<br>
                <b>Peor Caso (P90):</b> {np.percentile(kpis_int_A, 90):,.0f}€<br>
                <hr>
                <b>Riesgo (P95):</b> {p95:,.0f}€
                </div>
                """, unsafe_allow_html=True)
