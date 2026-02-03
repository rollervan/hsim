import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Simulador de Hipoteca",
    page_icon="🏠",
    layout="wide"
)

# Estilos
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: sans-serif; color: #333;}
    .stMetric {background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px;}
    div[data-testid="stExpander"] {border: 1px solid #ddd; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR DE CÁLCULO
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
    
    # Ajustamos longitudes
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, int(anios) - len(euribor_puntos)))
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
                
                # CORRECCIÓN DE PRECISIÓN: Si saldo es menor a 1 euro, liquidamos
                if saldo_real <= 1.0:
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
            
            # APLICACIÓN DE AMORTIZACIÓN ANTICIPADA
            if not en_periodo_carencia:
                # Solo aplicamos si hay saldo vivo
                if m == 11 and saldo_real > 1.0 and puntos_amort[anio] > 0:
                    ejec = round(min(puntos_amort[anio], saldo_real), 2)
                    saldo_real = round(saldo_real - ejec, 2)
                    
                    # Si reducimos plazo, el saldo teórico no cambia
                    # Si reducimos cuota, el saldo teórico baja para recalcular cuota siguiente
                    if tipo_reduc == 'CUOTA': 
                        saldo_teorico = saldo_real
                        
                    data[-1]['Amort_Extra'] = ejec
                    data[-1]['Capital'] = round(data[-1]['Capital'] + ejec, 2)
                    data[-1]['Saldo'] = saldo_real
                    
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
    st.header("Configuración")
    
    comparar = st.checkbox("Comparar dos opciones", value=False)
    
    # Inicialización
    es_autopromotor = False
    meses_carencia = 0
    s_hogar_A, s_vida_A = 0, 0
    s_hogar_B, s_vida_B = 0, 0
    
    with st.expander("Datos Económicos", expanded=not comparar):
        ingresos = st.number_input("Ingresos netos (€)", value=2500, step=100)
        ahorro_inicial = st.number_input("Ahorro inicial (€)", value=0, step=1000)
        precio_vivienda = st.number_input("Valor Vivienda (€)", value=0, step=5000)
        capital_init_global = st.number_input("Importe Hipoteca (€)", value=180000, step=1000)

    st.markdown("---")
    
    if comparar:
        st.subheader("Opción A vs Opción B")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("#### Opción A")
            modo_A = st.selectbox("Tipo A", ["MIXTA", "VARIABLE", "FIJA"], key="mA")
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
                anios_fijos_A = st.number_input("Años Fijos A", value=7, key="myaA")
                diferencial_A = st.number_input("Dif. Var A", value=0.55, step=0.05, key="mdaA")
            
            st.caption("Seguros A (€/año)")
            s_hogar_A = st.number_input("Hogar A", value=300, key="shA")
            s_vida_A = st.number_input("Vida A", value=300, key="svA")

        with colB:
            st.markdown("#### Opción B")
            modo_B = st.selectbox("Tipo B", ["MIXTA", "VARIABLE", "FIJA"], index=2, key="mB")
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
                anios_fijos_B = st.number_input("Años Fijos B", value=7, key="myaB")
                diferencial_B = st.number_input("Dif. Var B", value=0.55, step=0.05, key="mdaB")

            st.caption("Seguros B (€/año)")
            s_hogar_B = st.number_input("Hogar B", value=300, key="shB")
            s_vida_B = st.number_input("Vida B", value=300, key="svB")
                
    else:
        st.subheader("Datos Préstamo")
        modo_A = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
        
        es_autopromotor = st.checkbox("Autopromoción", value=True)
        if es_autopromotor:
            meses_carencia = st.number_input("Meses carencia", value=11, min_value=1, max_value=36)
            
        anios_A = st.number_input("Plazo (Años)", value=25, min_value=1)
        
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
            anios_fijos_A = c2.number_input("Años Fijos", value=7)
            diferencial_A = st.number_input("Dif. Variable (%)", value=0.55, step=0.05)
        
        st.markdown("**Seguros Vinculados**")
        s_hogar_A = st.number_input("Seguro Hogar (€/año)", value=300)
        s_vida_A = st.number_input("Seguro Vida (€/año)", value=300)

        modo_B, anios_B = modo_A, anios_A
        tipo_fijo_B, diferencial_B, anios_fijos_B = tipo_fijo_A, diferencial_A, anios_fijos_A
        s_hogar_B, s_vida_B = s_hogar_A, s_vida_A

    st.markdown("---")
    tipo_reduc = st.radio("Amortización anticipada:", ["Reducir PLAZO", "Reducir CUOTA"])

    with st.expander("Otros Gastos Mensuales", expanded=False):
        g_comida = st.number_input("Comida", value=300)
        g_suministros = st.number_input("Suministros", value=150)
        g_gasolina = st.number_input("Transporte", value=100)
        g_otros = st.number_input("Otros", value=200)

    caminos_eur = []
    n_sims = 1
    
    necesita_euribor = (modo_A != "FIJA") or (modo_B != "FIJA" and comparar)
    
    if necesita_euribor:
        st.markdown("---")
        with st.expander("Previsión Euríbor", expanded=True):
            modo_prev = st.selectbox("Método", ["Monte Carlo (Simulación)", "Manual"])
            
            if modo_prev == "Monte Carlo (Simulación)":
                n_sims = st.select_slider("Simulaciones", [50, 100, 500, 1000], value=100)
                st.caption("Ajustes Mercado")
                theta = st.slider("Media L/P", 0.0, 5.0, 2.25)
                sigma = st.slider("Volatilidad", 0.0, 2.0, 0.60)
                kappa = st.slider("Reversión", 0.0, 1.0, 0.30)
                r0 = st.number_input("Euríbor Hoy", value=2.24)
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
# 3. VISUALIZACIÓN
# ==========================================

st.title("Simulador de Hipoteca")

with st.expander("Amortización Anticipada"):
    st.info("Capital extra anual (Máx. 10.000€)")
    cols_a = st.columns(4) 
    amort_list = []
    max_anios_slider = max(anios_A, anios_B)
    for i in range(max_anios_slider):
        val = cols_a[i % 4].slider(f"Año {i+1}", 0, 10000, 0, step=500, key=f"s_a{i}")
        amort_list.append(val)

hay_amortizacion = sum(amort_list) > 0

kpis_int_A, kpis_int_B = [], []
kpis_pat_A = []
kpis_ahorro_A = []
eur_matrix = [] 

df_median_A, df_median_B = None, None
df_base_median_A = None 

total_gastos = g_comida + g_suministros + g_gasolina + g_otros

coste_mes_seguros_A = (s_hogar_A + s_vida_A) / 12
coste_mes_seguros_B = (s_hogar_B + s_vida_B) / 12

if n_sims > 100: prog_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # ESCENARIO A
    ap_flag = es_autopromotor if not comparar else False
    carencia_val = meses_carencia if not comparar else 0
    
    # 1. Calculamos con amortización (REAL)
    df_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino, amort_list, tipo_reduc, ap_flag, carencia_val)
    
    # 2. Calculamos SIN amortización (BASE) para comparar siempre manzanas con manzanas
    if not comparar:
        df_base_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino, [0]*anios_A, 'PLAZO', ap_flag, carencia_val)
        kpis_ahorro_A.append(df_base_A['Intereses'].sum() - df_A['Intereses'].sum())
    
    df_A['Seguros'] = np.where(df_A['Saldo'] > 0, coste_mes_seguros_A, 0)
    gasto_tot_A = df_A['Cuota'] + df_A['Seguros'] + total_gastos
    
    df_A['Ahorro_Liq'] = ahorro_inicial + (ingresos - gasto_tot_A).cumsum() - df_A['Amort_Extra'].cumsum()
    df_A['Patrimonio'] = df_A['Ahorro_Liq'] + (precio_vivienda - df_A['Saldo'])
    
    kpis_int_A.append(df_A['Intereses'].sum() + df_A['Seguros'].sum()) 
    if not comparar:
        kpis_pat_A.append(df_A['Patrimonio'].iloc[-1])
        eur_matrix.append(camino)

    # ESCENARIO B
    if comparar:
        df_B = calcular_hipoteca_core(capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B, modo_B, camino, amort_list, tipo_reduc, False, 0)
        df_B['Seguros'] = np.where(df_B['Saldo'] > 0, coste_mes_seguros_B, 0)
        kpis_int_B.append(df_B['Intereses'].sum() + df_B['Seguros'].sum())
        if i == 0: df_median_B = df_B

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

    df_median_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino_med, amort_list, tipo_reduc, ap_flag, carencia_val)
    df_median_A['Seguros'] = np.where(df_median_A['Saldo'] > 0, coste_mes_seguros_A, 0)
    df_median_A['Ahorro_Liq'] = ahorro_inicial + (ingresos - (df_median_A['Cuota'] + df_median_A['Seguros'] + total_gastos)).cumsum() - df_median_A['Amort_Extra'].cumsum()
    df_median_A['Patrimonio'] = df_median_A['Ahorro_Liq'] + (precio_vivienda - df_median_A['Saldo'])
    
    if not comparar:
        df_base_median_A = calcular_hipoteca_core(capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A, modo_A, camino_med, [0]*anios_A, 'PLAZO', ap_flag, carencia_val)

    if comparar:
        df_median_B = calcular_hipoteca_core(capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B, modo_B, camino_med, amort_list, tipo_reduc, False, 0)
        df_median_B['Seguros'] = np.where(df_median_B['Saldo'] > 0, coste_mes_seguros_B, 0)

# KPIs Generales
coste_A = df_median_A['Intereses'].sum() + df_median_A['Seguros'].sum()

# ==========================================
# CÁLCULO DE DURACIÓN EXACTA
# ==========================================
# Contamos cuántos meses hay con saldo > 1.0 (para evitar errores de decimales)
meses_reales_A = len(df_median_A[df_median_A['Saldo'] > 1.0])

idx_ref = 0 if not (es_autopromotor and not comparar) else meses_carencia
if idx_ref >= len(df_median_A): idx_ref = 0
cuota_ini_A = df_median_A.iloc[idx_ref]['Cuota']

if comparar:
    coste_B = df_median_B['Intereses'].sum() + df_median_B['Seguros'].sum()
    meses_reales_B = len(df_median_B[df_median_B['Saldo'] > 1.0])
    cuota_ini_B = df_median_B.iloc[0]['Cuota']

    st.markdown("### Resultados Comparativa")
    col_c1, col_c2, col_c3 = st.columns(3)
    
    dif_coste = coste_B - coste_A
    col_c1.metric("Coste Total (Int + Seguros)", f"{coste_A:,.0f} € vs {coste_B:,.0f} €", f"{dif_coste:,.0f} € (Diferencia)", delta_color="inverse")
    
    dif_meses = meses_reales_B - meses_reales_A
    def fmt_t(m): 
        a = m // 12
        r = m % 12
        if a > 0 and r > 0: return f"{a}a {r}m"
        elif a > 0: return f"{a} años"
        else: return f"{r} meses"

    col_c2.metric("Duración Real", f"{fmt_t(meses_reales_A)} vs {fmt_t(meses_reales_B)}", f"{dif_meses} meses", delta_color="inverse")
    
    dif_cuota = cuota_ini_B - cuota_ini_A
    col_c3.metric("Cuota Inicial", f"{cuota_ini_A:,.0f} € vs {cuota_ini_B:,.0f} €", f"{dif_cuota:,.0f} €", delta_color="inverse")
    
    st.markdown("---")
    
    tabs = st.tabs(["Evolución Deuda", "Costes Acumulados", "Tabla de Datos"])
    
    with tabs[0]:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'], fill='tozeroy', name='Opción A', line=dict(color='#0055aa')))
        fig_s.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Saldo'], name='Opción B', line=dict(color='#ff7f0e', dash='dash', width=3)))
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
    # --- VISTA INDIVIDUAL ---
    
    if hay_amortizacion:
        # 1. Calculamos duración del escenario BASE (sin amortizar)
        meses_base = len(df_base_median_A[df_base_median_A['Saldo'] > 1.0])
        
        # 2. Calculamos duración del escenario ACTUAL (con amortización)
        meses_actual = len(df_median_A[df_median_A['Saldo'] > 1.0])
        
        # 3. La diferencia es el ahorro REAL
        meses_ahorrados = max(0, meses_base - meses_actual)
        
        # Formateo de la duración final
        a_fin = meses_actual // 12
        m_fin = meses_actual % 12
        if m_fin > 0: txt_duracion = f"{a_fin} años y {m_fin} meses"
        else: txt_duracion = f"{a_fin} años"

        # Formateo del ahorro
        a_save = meses_ahorrados // 12
        m_save = meses_ahorrados % 12
        
        if tipo_reduc == 'Reducir PLAZO':
            if a_save > 0 and m_save > 0: txt_tiempo = f"-{a_save} años y {m_save} meses"
            elif a_save > 0: txt_tiempo = f"-{a_save} años"
            elif m_save > 0: txt_tiempo = f"-{m_save} meses"
            else: txt_tiempo = "0 meses"
        else:
            txt_tiempo = "Baja cuota (mismo plazo)"
            txt_duracion = f"{meses_base // 12} años" # Si bajamos cuota, el plazo se mantiene
        
        ahorro_int = np.median(kpis_ahorro_A)
    else:
        # Sin amortización
        meses_base = len(df_median_A[df_median_A['Saldo'] > 1.0])
        txt_duracion = f"{meses_base // 12} años"
        if meses_base % 12 > 0: txt_duracion += f" y {meses_base % 12} meses"
        
        txt_tiempo = "Sin cambios"
        ahorro_int = 0

    st.markdown("### Resumen")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cuota Inicial", f"{cuota_ini_A:,.2f} €", f"{df_median_A.iloc[idx_ref]['Tasa']:.2f}% TIN")
    k2.metric("Total Intereses", f"{df_median_A['Intereses'].sum():,.0f} €", delta_color="inverse")
    
    # NUEVA MÉTRICA DE TIEMPO CLARA
    k3.metric("Plazo Final", txt_duracion, delta_color="off")
    
    k4.metric("Ahorro (Intereses / Tiempo)", f"{ahorro_int:,.0f} €", txt_tiempo)
    
    st.markdown("---")

    tabs = st.tabs(["Evolución", "Amortización", "Patrimonio", "Riesgo"])

    with tabs[0]:
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            st.subheader("Euríbor Estimado")
            if modo_A == "FIJA":
                st.info("Hipoteca Fija: Sin cambios.")
            else:
                mat = np.array(eur_matrix)
                p10, p50, p90 = np.percentile(mat, [10, 50, 90], axis=0)
                x_ax = np.arange(1, len(p50)+1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_ax, y=p90, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x_ax, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,250,0.15)', name='Rango Probable'))
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

    with tabs[1]:
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.subheader("Intereses Acumulados")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=df_base_median_A['Intereses'].cumsum(), name='Sin Amortizar', line=dict(color='gray', dash='dash')))
            fig3.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Intereses'].cumsum(), name='Con Amortización', line=dict(color='#d9534f', width=3)))
            fig3.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig3, use_container_width=True)
        with c_a2:
            st.subheader("Saldo Pendiente")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=df_base_median_A['Saldo'], name='Saldo Base', line=dict(color='gray', dash='dash')))
            fig4.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'], fill='tozeroy', name='Saldo Real', line=dict(color='#5cb85c')))
            fig4.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig4, use_container_width=True)

    with tabs[2]:
        st.subheader("Patrimonio Neto")
        fig5 = go.Figure()
        g_base = df_base_median_A['Cuota'] + coste_mes_seguros_A + total_gastos
        ah_base = ahorro_inicial + (ingresos - g_base).cumsum()
        pat_base = ah_base + (precio_vivienda - df_base_median_A['Saldo'])
        
        fig5.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=pat_base, name='Base', line=dict(color='gray', dash='dot')))
        fig5.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Patrimonio'], name='Actual', line=dict(color='#6f42c1', width=3)))
        fig5.update_layout(template='plotly_white', height=400, hovermode="x unified")
        st.plotly_chart(fig5, use_container_width=True)

    with tabs[3]:
        if modo_A == "FIJA":
            st.success("Coste fijo. No hay riesgo de tipo de interés.")
        elif n_sims < 10:
            st.warning("Sube las simulaciones para ver el riesgo.")
        else:
            st.subheader("Distribución de Coste (Intereses)")
            p5, p50, p95 = np.percentile(kpis_int_A, [5, 50, 95])
            
            c_r1, c_r2 = st.columns([2, 1])
            with c_r1:
                fig_h = px.histogram(x=kpis_int_A, nbins=30, labels={'x': 'Total Intereses'}, color_discrete_sequence=['#8884d8'])
                fig_h.add_vline(x=p5, line_dash="dash", line_color="green", annotation_text="Mejor")
                fig_h.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text="Peor")
                fig_h.update_layout(template='plotly_white', height=400, showlegend=False)
                st.plotly_chart(fig_h, use_container_width=True)
            with c_r2:
                st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:15px; border-radius:5px;">
                <b>Mejor (P10):</b> {np.percentile(kpis_int_A, 10):,.0f}€<br>
                <b>Mediana:</b> {p50:,.0f}€<br>
                <b>Peor (P90):</b> {np.percentile(kpis_int_A, 90):,.0f}€<br>
                <hr>
                <b>Extremo (P95):</b> {p95:,.0f}€
                </div>
                """, unsafe_allow_html=True)
