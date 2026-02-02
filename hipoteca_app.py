import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Simulador Hipotecario Pro", layout="wide")

# ==========================================
# 1. NÚCLEO MATEMÁTICO (Lógica Auditada v2.4)
# ==========================================
def calcular_hipoteca(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduccion):
    n_meses_total = anios * 12
    saldo_real = round(float(capital), 2)
    saldo_teorico = round(float(capital), 2)
    data = []
    mes_global = 1
    
    # Asegurar longitud de listas
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (anios - len(euribor_puntos))
    puntos_amort = list(amortizaciones) + [0] * (anios - len(amortizaciones))

    idx_var = 0 
    for anio in range(anios):
        if saldo_real <= 0: break

        # Tasa del año
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
        meses_restantes = n_meses_total - (mes_global - 1)
        base_calc = saldo_teorico if tipo_reduccion == 'PLAZO' else saldo_real
        if base_calc < saldo_real: base_calc = saldo_real

        if base_calc <= 0.01: cuota = 0
        else:
            if tasa_mensual > 0:
                cuota = base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1)
            else: cuota = base_calc / meses_restantes
        
        cuota = round(cuota, 2)

        for m in range(12):
            if saldo_real <= 0.009: break
            interes_m = round(saldo_real * tasa_mensual, 2)
            capital_m = round(cuota - interes_m, 2)
            if capital_m > saldo_real:
                capital_m = saldo_real
                cuota = round(capital_m + interes_m, 2)

            saldo_real = round(saldo_real - capital_m, 2)
            
            # Sincronización teórica
            int_teorico = round(saldo_teorico * tasa_mensual, 2)
            amort_teorica = round(cuota - int_teorico, 2)
            saldo_teorico = round(saldo_teorico - amort_teorica, 2)

            data.append({'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual, 'Cuota': cuota, 'Intereses': interes_m, 'Saldo': saldo_real, 'Amort_Extra': 0})
            
            if m == 11 and saldo_real > 0 and puntos_amort[anio] > 0:
                ejec = round(min(puntos_amort[anio], saldo_real), 2)
                saldo_real = round(saldo_real - ejec, 2)
                if tipo_reduccion == 'CUOTA': saldo_teorico = saldo_real
                data[-1]['Amort_Extra'] = ejec
            mes_global += 1

    return pd.DataFrame(data)

# ==========================================
# 2. INTERFAZ STREAMLIT
# ==========================================
st.title("🏦 Simulador Hipotecario Profesional")
st.markdown("---")

# Barra Lateral - Configuración Principal
with st.sidebar:
    st.header("⚙️ Configuración Préstamo")
    modo = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
    tipo_reduc = st.radio("Estrategia Amortización", ["PLAZO", "CUOTA"], help="PLAZO: Mantienes cuota y acabas antes. CUOTA: Bajas lo que pagas al mes.")
    
    capital = st.number_input("Capital Pendiente (€)", value=180000, step=1000)
    anios = st.number_input("Plazo Restante (Años)", value=25, min_value=1, max_value=40)
    pagado_previo = st.number_input("Ya pagado anteriormente (€)", value=0)
    
    st.markdown("---")
    st.header("📊 Condiciones Banco")
    tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.95, format="%.2f")
    anios_fijos = st.number_input("Años de tramo fijo", value=5) if modo == "MIXTA" else 0
    diferencial = st.number_input("Diferencial Variable (%)", value=0.55, format="%.2f")

# Cuerpo Principal - Sliders Dinámicos
col_eur, col_amort = st.columns(2)

with col_eur:
    st.subheader("📉 Previsión Euríbor")
    n_sliders_eur = anios if modo == "VARIABLE" else max(0, anios - anios_fijos)
    eur_list = []
    # Usamos un expander para no ocupar toda la pantalla
    with st.expander("Configurar Euríbor por año", expanded=True):
        cols = st.columns(4) # Distribuir sliders en columnas pequeñas
        for i in range(n_sliders_eur):
            with cols[i % 4]:
                val = st.slider(f"Año {anios-n_sliders_eur+i+1}", -1.0, 6.0, 2.5, key=f"eur_{i}", step=0.1)
                eur_list.append(val)

with col_amort:
    st.subheader("💰 Amortización Extra")
    amort_list = []
    with st.expander("Configurar Pagos Extra (€)", expanded=True):
        cols_a = st.columns(4)
        for i in range(anios):
            with cols_a[i % 4]:
                val_a = st.slider(f"Año {i+1}", 0, 20000, 0, key=f"am_{i}", step=500)
                amort_list.append(val_a)

# ==========================================
# 3. CÁLCULOS Y RESULTADOS
# ==========================================
df = calcular_hipoteca(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, eur_list, amort_list, tipo_reduc)
df_b = calcular_hipoteca(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, eur_list, [0]*anios, 'PLAZO')

# KPIs
int_act = df['Intereses'].sum()
ahorro_int = max(0, df_b['Intereses'].sum() - int_act)
ahorro_t = (anios * 12) - len(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Intereses Totales", f"{int_act:,.0f} €", delta=f"-{ahorro_int:,.0f} €", delta_color="normal")
m2.metric("Ahorro Intereses", f"{ahorro_int:,.0f} €")
m3.metric("Tiempo Ahorrado", f"{ahorro_t // 12}a {ahorro_t % 12}m")
m4.metric("Cuota Máxima", f"{df['Cuota'].max():,.2f} €")

# Gráficos
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.write("**Evolución de la Deuda**")
    st.area_chart(df.set_index('Mes')['Saldo'])

with c2:
    st.write("**Evolución de la Cuota**")
    st.line_chart(df.set_index('Mes')['Cuota'])

# Tabla Detallada
if st.checkbox("Ver cuadro de amortización completo"):
    st.dataframe(df)
