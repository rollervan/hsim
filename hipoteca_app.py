import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN VISUAL (ESTILO WINVEST/FINTECH) ---
st.set_page_config(page_title="Simulador Hipotecario Estratégico", layout="wide", page_icon="🏠")

# CSS para dar aspecto profesional/bancario
st.markdown("""
    <style>
    .main { background-color: #FAFAFA; }
    .stMetric { background-color: #FFFFFF; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0E1117; font-family: 'Sans-serif'; }
    .highlight { color: #2E86C1; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE CÁLCULO FINANCIERO ---

def calcular_cuota_francesa(capital, tasa_anual, meses):
    """Calcula cuota mensual sistema francés."""
    if tasa_anual == 0: return capital / meses
    i = tasa_anual / 100 / 12
    return capital * (i * (1 + i)**meses) / ((1 + i)**meses - 1)

def simular_hipoteca(monto, plazo_anos, tipo, params):
    """
    Simula la vida de la hipoteca mes a mes.
    Maneja: Fija, Variable pura y Mixta.
    """
    meses_totales = plazo_anos * 12
    saldo = monto
    datos = []
    
    euribor_base = params.get('euribor', 2.5)
    
    # Desglose de parámetros
    periodo_fijo = params.get('periodo_fijo_anos', 0) * 12
    tasa_fija = params.get('tasa_fija', 0.0)
    diferencial = params.get('diferencial', 0.0)
    
    tasa_actual = 0.0
    cuota = 0.0
    
    for mes in range(1, meses_totales + 1):
        # Determinar Tasa
        es_periodo_fijo = False
        
        if tipo == "Fija":
            tasa_actual = tasa_fija
            es_periodo_fijo = True
        elif tipo == "Variable":
            tasa_actual = euribor_base + diferencial
        elif tipo == "Mixta":
            if mes <= periodo_fijo:
                tasa_actual = tasa_fija
                es_periodo_fijo = True
            else:
                tasa_actual = euribor_base + diferencial
        
        # Revisión de cuota (Anual o al cambio de tramo)
        # En sistema francés recalculamos si cambia el interés
        if mes == 1 or (not es_periodo_fijo and (mes - 1) % 12 == 0) or (mes == periodo_fijo + 1):
            cuota = calcular_cuota_francesa(saldo, tasa_actual, meses_totales - mes + 1)
            
        interes_pagado = saldo * (tasa_actual / 100 / 12)
        capital_amortizado = cuota - interes_pagado
        
        # Ajuste final
        if saldo - capital_amortizado < 0:
            capital_amortizado = saldo
            cuota = interes_pagado + capital_amortizado

        saldo -= capital_amortizado
        
        datos.append({
            "Mes": mes,
            "Año": (mes-1)//12 + 1,
            "Tasa %": round(tasa_actual, 2),
            "Cuota": round(cuota, 2),
            "Intereses": round(interes_pagado, 2),
            "Amortización": round(capital_amortizado, 2),
            "Saldo Pendiente": round(max(0, saldo), 2),
            "Tipo": tipo
        })
        
        if saldo <= 0: break
        
    return pd.DataFrame(datos)

# --- INTERFAZ PRINCIPAL ---

st.title("🏡 Simulador de Estrategia Hipotecaria")
st.markdown("Herramienta de análisis financiero para comparar **Mixta vs Variable vs Fija**.")

# 1. BARRA LATERAL: DATOS DE LA OPERACIÓN
with st.sidebar:
    st.header("1. Datos de la Operación")
    precio_vivienda = st.number_input("Precio de Compra (€)", value=300000, step=5000)
    ahorro_aportado = st.number_input("Aportación Inicial / Entrada (€)", value=60000, step=1000)
    plazo = st.slider("Plazo (Años)", 10, 40, 30)
    
    monto_prestamo = precio_vivienda - ahorro_aportado
    ltv = (monto_prestamo / precio_vivienda) * 100
    
    st.markdown("---")
    st.metric("Importe Hipoteca", f"{monto_prestamo:,.0f} €")
    if ltv > 80:
        st.error(f"⚠️ Financiación: {ltv:.1f}% (>80%). Requerirá negociación especial.")
    else:
        st.success(f"✅ Financiación: {ltv:.1f}% (Estándar)")

    st.markdown("---")
    st.header("2. Escenario de Mercado")
    euribor_simulado = st.slider("Euríbor Promedio Estimado (%)", 0.0, 6.0, 2.5, 0.1)
    st.caption("Define el Euríbor medio para los tramos variables.")

# 2. CONFIGURACIÓN DE PRODUCTOS
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configura tu Hipoteca Mixta")
    st.caption("El producto estrella actual.")
    mix_fijo_anos = st.number_input("Años Fijos (Mixta)", 1, 15, 5)
    mix_tasa_fija = st.number_input("Tasa Fija Inicial (%)", 0.0, 10.0, 2.25, step=0.05)
    mix_dif = st.number_input("Diferencial Variable (%)", 0.0, 5.0, 0.70, step=0.05)

    st.markdown("---")
    st.subheader("Comparativa (Opcional)")
    var_dif = st.number_input("Diferencial (Opción Variable Pura)", 0.0, 5.0, 0.89, step=0.05)
    fija_tasa = st.number_input("Tasa (Opción Fija Pura)", 0.0, 10.0, 2.95, step=0.05)

# CÁLCULO DE ESCENARIOS
params_mixta = {'periodo_fijo_anos': mix_fijo_anos, 'tasa_fija': mix_tasa_fija, 'diferencial': mix_dif, 'euribor': euribor_simulado}
df_mixta = simular_hipoteca(monto_prestamo, plazo, "Mixta", params_mixta)

params_variable = {'periodo_fijo_anos': 0, 'tasa_fija': 0, 'diferencial': var_dif, 'euribor': euribor_simulado}
df_variable = simular_hipoteca(monto_prestamo, plazo, "Variable", params_variable)

params_fija = {'periodo_fijo_anos': plazo, 'tasa_fija': fija_tasa, 'diferencial': 0, 'euribor': 0}
df_fija = simular_hipoteca(monto_prestamo, plazo, "Fija", params_fija)

# 3. RESULTADOS VISUALES
with col2:
    st.subheader("💡 Análisis de Cuota Mensual")
    
    # Crear gráfico comparativo
    fig = go.Figure()
    
    # Mixta
    fig.add_trace(go.Scatter(x=df_mixta['Año'], y=df_mixta['Cuota'], mode='lines', name='Mixta', line=dict(color='#2E86C1', width=4)))
    # Variable
    fig.add_trace(go.Scatter(x=df_variable['Año'], y=df_variable['Cuota'], mode='lines', name='Variable Pura', line=dict(color='#E74C3C', dash='dash')))
    # Fija
    fig.add_trace(go.Scatter(x=df_fija['Año'], y=df_fija['Cuota'], mode='lines', name='Fija Pura', line=dict(color='#27AE60', dash='dot')))
    
    fig.update_layout(title="Evolución de la Cuota Mensual", xaxis_title="Año", yaxis_title="Cuota (€)", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Métricas clave
    c_mix_ini = df_mixta['Cuota'].iloc[0]
    c_mix_var = df_mixta[df_mixta['Mes'] > (mix_fijo_anos*12)]['Cuota'].iloc[0] if len(df_mixta) > (mix_fijo_anos*12) else c_mix_ini
    
    st.info(f"**Análisis Mixta:** Pagarás **{c_mix_ini:,.2f}€** durante {mix_fijo_anos} años. Si el Euríbor está al {euribor_simulado}%, pasarás a pagar **{c_mix_var:,.2f}€**.")

# --- COMPARATIVA DE COSTES TOTALES ---
st.markdown("---")
st.subheader("💰 Coste Total del Préstamo (Intereses Pagados)")

total_mixta = df_mixta['Intereses'].sum()
total_variable = df_variable['Intereses'].sum()
total_fija = df_fija['Intereses'].sum()

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.metric("Total Intereses (Mixta)", f"{total_mixta:,.0f} €")
    st.progress(min(1.0, total_mixta / max(total_mixta, total_variable, total_fija)))

with col_res2:
    st.metric("Total Intereses (Variable)", f"{total_variable:,.0f} €", delta=f"{total_variable - total_mixta:,.0f} € vs Mixta", delta_color="inverse")
    st.progress(min(1.0, total_variable / max(total_mixta, total_variable, total_fija)))

with col_res3:
    st.metric("Total Intereses (Fija)", f"{total_fija:,.0f} €", delta=f"{total_fija - total_mixta:,.0f} € vs Mixta", delta_color="inverse")
    st.progress(min(1.0, total_fija / max(total_mixta, total_variable, total_fija)))

# --- DETALLE Y EXPORTACIÓN ---
with st.expander("Ver Tabla de Amortización Detallada (Hipoteca Mixta)"):
    st.dataframe(df_mixta[['Mes', 'Año', 'Tasa %', 'Cuota', 'Intereses', 'Amortización', 'Saldo Pendiente']].style.format("{:.2f}"))
    
    csv = df_mixta.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar Excel (.csv)", csv, "simulacion_winvest.csv", "text/csv")

# --- DISCLAIMER ---
st.caption("Nota: Simulación con fines informativos basada en sistema de amortización francés. No incluye gastos de compraventa (ITP, Notaría, Gestoría) ni vinculaciones (seguros de vida/hogar) que pueden bonificar el tipo de interés.")
