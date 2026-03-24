"""
Dashboard Interativo - Tech Challenge Fase 3
FIAP PosTech - Machine Learning Engineering

Analise de Atrasos de Voos nos EUA (2015)

Para executar:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURACAO DA PAGINA
# ============================================================
st.set_page_config(
    page_title="Dashboard - Atrasos de Voos EUA",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNCOES DE CARREGAMENTO
# ============================================================
@st.cache_data
def carregar_dados():
    """Carrega e processa os dados dos voos."""
    DATA_PATH = r'C:\Users\silva\workspace\FIAP\arquivos-fase-3'

    # Carregar dados
    df = pd.read_csv(f'{DATA_PATH}/flights.csv', low_memory=False)
    airlines = pd.read_csv(f'{DATA_PATH}/airlines.csv')
    airports = pd.read_csv(f'{DATA_PATH}/airports.csv')

    # Amostragem de 1 milhao de voos
    df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)

    # Remover cancelados e desviados
    df = df[df['CANCELLED'] == 0].copy()
    df = df[df['DIVERTED'] == 0].copy()

    # Tratar valores ausentes
    cols_atraso = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
                   'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
    df[cols_atraso] = df[cols_atraso].fillna(0)
    df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].fillna(df['DEPARTURE_DELAY'].median())
    df['ARRIVAL_DELAY'] = df['ARRIVAL_DELAY'].fillna(df['ARRIVAL_DELAY'].median())

    # Criar variaveis derivadas
    df['HORA_PARTIDA'] = df['SCHEDULED_DEPARTURE'] // 100
    df['ATRASOU'] = (df['DEPARTURE_DELAY'] >= 15).astype(int)

    # Merge com nomes das companhias
    df = df.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left', suffixes=('', '_CIA'))
    df.rename(columns={'AIRLINE_CIA': 'NOME_COMPANHIA'}, inplace=True)

    return df, airlines, airports

# ============================================================
# CARREGAR DADOS
# ============================================================
with st.spinner('Carregando dados...'):
    df, airlines, airports = carregar_dados()

# ============================================================
# SIDEBAR - FILTROS
# ============================================================
st.sidebar.title("Filtros")
st.sidebar.markdown("---")

# Filtro de Companhia Aerea
companhias = ['Todas'] + sorted(df['NOME_COMPANHIA'].dropna().unique().tolist())
companhia_selecionada = st.sidebar.selectbox("Companhia Aerea", companhias)

# Filtro de Mes
meses_nome = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Marco', 4: 'Abril',
              5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
              9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
meses = ['Todos'] + [meses_nome[i] for i in range(1, 13)]
mes_selecionado = st.sidebar.selectbox("Mes", meses)

# Filtro de Dia da Semana
dias_nome = {1: 'Segunda', 2: 'Terca', 3: 'Quarta', 4: 'Quinta',
             5: 'Sexta', 6: 'Sabado', 7: 'Domingo'}
dias = ['Todos'] + [dias_nome[i] for i in range(1, 8)]
dia_selecionado = st.sidebar.selectbox("Dia da Semana", dias)

# Filtro de Horario
hora_min, hora_max = st.sidebar.slider(
    "Horario de Partida",
    min_value=0, max_value=23,
    value=(0, 23)
)

# Aplicar filtros
df_filtrado = df.copy()

if companhia_selecionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['NOME_COMPANHIA'] == companhia_selecionada]

if mes_selecionado != 'Todos':
    mes_num = [k for k, v in meses_nome.items() if v == mes_selecionado][0]
    df_filtrado = df_filtrado[df_filtrado['MONTH'] == mes_num]

if dia_selecionado != 'Todos':
    dia_num = [k for k, v in dias_nome.items() if v == dia_selecionado][0]
    df_filtrado = df_filtrado[df_filtrado['DAY_OF_WEEK'] == dia_num]

df_filtrado = df_filtrado[(df_filtrado['HORA_PARTIDA'] >= hora_min) &
                          (df_filtrado['HORA_PARTIDA'] <= hora_max)]

# ============================================================
# HEADER
# ============================================================
st.title("✈️ Dashboard de Atrasos de Voos - EUA 2015")
st.markdown("**Tech Challenge Fase 3** | FIAP PosTech - Machine Learning Engineering")
st.markdown("---")

# ============================================================
# METRICAS PRINCIPAIS (KPIs)
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

total_voos = len(df_filtrado)
voos_atrasados = df_filtrado['ATRASOU'].sum()
pct_atrasos = (voos_atrasados / total_voos * 100) if total_voos > 0 else 0
atraso_medio = df_filtrado['DEPARTURE_DELAY'].mean()
atraso_mediano = df_filtrado['DEPARTURE_DELAY'].median()

col1.metric("Total de Voos", f"{total_voos:,}")
col2.metric("Voos Atrasados", f"{voos_atrasados:,}")
col3.metric("% Atrasos", f"{pct_atrasos:.1f}%")
col4.metric("Atraso Medio", f"{atraso_medio:.1f} min")
col5.metric("Atraso Mediano", f"{atraso_mediano:.1f} min")

st.markdown("---")

# ============================================================
# GRAFICOS - LINHA 1
# ============================================================
col_left, col_right = st.columns(2)

# Grafico 1: Atrasos por Companhia
with col_left:
    st.subheader("Atraso Medio por Companhia Aerea")

    atraso_cia = df_filtrado.groupby('NOME_COMPANHIA').agg(
        atraso_medio=('DEPARTURE_DELAY', 'mean'),
        total_voos=('DEPARTURE_DELAY', 'count')
    ).reset_index().sort_values('atraso_medio', ascending=True)

    fig1 = px.bar(
        atraso_cia,
        x='atraso_medio',
        y='NOME_COMPANHIA',
        orientation='h',
        color='atraso_medio',
        color_continuous_scale='RdYlGn_r',
        labels={'atraso_medio': 'Atraso Medio (min)', 'NOME_COMPANHIA': ''},
        hover_data=['total_voos']
    )
    fig1.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

# Grafico 2: Distribuicao de Atrasos
with col_right:
    st.subheader("Distribuicao dos Atrasos")

    df_hist = df_filtrado[df_filtrado['DEPARTURE_DELAY'].between(-30, 120)]

    fig2 = px.histogram(
        df_hist,
        x='DEPARTURE_DELAY',
        nbins=50,
        color_discrete_sequence=['steelblue'],
        labels={'DEPARTURE_DELAY': 'Atraso (minutos)', 'count': 'Frequencia'}
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="No horario")
    fig2.add_vline(x=15, line_dash="dash", line_color="red", annotation_text="Limite atraso (15 min)")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# GRAFICOS - LINHA 2
# ============================================================
col_left2, col_right2 = st.columns(2)

# Grafico 3: Heatmap Hora x Dia da Semana
with col_left2:
    st.subheader("Atraso por Hora e Dia da Semana")

    pivot = df_filtrado.groupby(['HORA_PARTIDA', 'DAY_OF_WEEK'])['DEPARTURE_DELAY'].mean().unstack()
    pivot.columns = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']

    fig3 = px.imshow(
        pivot.values,
        x=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom'],
        y=list(range(24)),
        color_continuous_scale='RdYlGn_r',
        labels={'x': 'Dia da Semana', 'y': 'Hora', 'color': 'Atraso (min)'},
        aspect='auto'
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

# Grafico 4: Atraso por Mes
with col_right2:
    st.subheader("Atraso Medio por Mes")

    atraso_mes = df_filtrado.groupby('MONTH').agg(
        atraso_medio=('DEPARTURE_DELAY', 'mean'),
        pct_atrasados=('ATRASOU', 'mean')
    ).reset_index()
    atraso_mes['MES_NOME'] = atraso_mes['MONTH'].map(meses_nome)

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])

    fig4.add_trace(
        go.Bar(x=atraso_mes['MES_NOME'], y=atraso_mes['atraso_medio'],
               name='Atraso Medio', marker_color='steelblue'),
        secondary_y=False
    )
    fig4.add_trace(
        go.Scatter(x=atraso_mes['MES_NOME'], y=atraso_mes['pct_atrasados']*100,
                   name='% Atrasados', marker_color='salmon', mode='lines+markers'),
        secondary_y=True
    )

    fig4.update_yaxes(title_text="Atraso Medio (min)", secondary_y=False)
    fig4.update_yaxes(title_text="% Voos Atrasados", secondary_y=True)
    fig4.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# GRAFICOS - LINHA 3
# ============================================================
st.subheader("Atraso por Hora do Dia - Efeito Cascata")

atraso_hora = df_filtrado.groupby('HORA_PARTIDA').agg(
    atraso_medio=('DEPARTURE_DELAY', 'mean'),
    total_voos=('DEPARTURE_DELAY', 'count'),
    pct_atrasados=('ATRASOU', 'mean')
).reset_index()

fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=atraso_hora['HORA_PARTIDA'],
    y=atraso_hora['atraso_medio'],
    mode='lines+markers',
    name='Atraso Medio',
    line=dict(color='steelblue', width=3),
    fill='tozeroy',
    fillcolor='rgba(70, 130, 180, 0.2)'
))

fig5.update_layout(
    height=350,
    xaxis_title='Hora de Partida',
    yaxis_title='Atraso Medio (min)',
    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
)

st.plotly_chart(fig5, use_container_width=True)

st.info("💡 **Insight:** Voos no inicio do dia tem menos atraso. O efeito cascata faz com que atrasos se acumulem ao longo do dia.")

# ============================================================
# MAPA GEOGRAFICO
# ============================================================
st.markdown("---")
st.subheader("Mapa de Atrasos por Aeroporto")

# Agregar dados por aeroporto
atraso_aeroporto = df_filtrado.groupby('ORIGIN_AIRPORT').agg(
    atraso_medio=('DEPARTURE_DELAY', 'mean'),
    total_voos=('DEPARTURE_DELAY', 'count'),
    pct_atrasados=('ATRASOU', 'mean')
).reset_index()

# Filtrar aeroportos com volume significativo
atraso_aeroporto = atraso_aeroporto[atraso_aeroporto['total_voos'] >= 500]

# Merge com coordenadas
atraso_aeroporto = atraso_aeroporto.merge(
    airports[['IATA_CODE', 'AIRPORT', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']],
    left_on='ORIGIN_AIRPORT',
    right_on='IATA_CODE',
    how='inner'
)

if len(atraso_aeroporto) > 0:
    fig_mapa = px.scatter_geo(
        atraso_aeroporto,
        lat='LATITUDE',
        lon='LONGITUDE',
        color='atraso_medio',
        size='total_voos',
        hover_name='AIRPORT',
        hover_data={
            'IATA_CODE': True,
            'CITY': True,
            'STATE': True,
            'atraso_medio': ':.1f',
            'total_voos': ':,',
            'pct_atrasados': ':.1%',
            'LATITUDE': False,
            'LONGITUDE': False
        },
        color_continuous_scale='RdYlGn_r',
        scope='usa',
        labels={'atraso_medio': 'Atraso Medio (min)', 'total_voos': 'Total de Voos'}
    )
    fig_mapa.update_layout(height=500)
    st.plotly_chart(fig_mapa, use_container_width=True)
else:
    st.warning("Nenhum aeroporto com dados suficientes para exibir no mapa.")

# ============================================================
# TOP AEROPORTOS
# ============================================================
st.markdown("---")
col_top1, col_top2 = st.columns(2)

with col_top1:
    st.subheader("🔴 Top 10 Aeroportos - Maior Atraso")
    top_atraso = atraso_aeroporto.nlargest(10, 'atraso_medio')[['IATA_CODE', 'AIRPORT', 'STATE', 'atraso_medio', 'total_voos']]
    top_atraso.columns = ['Codigo', 'Aeroporto', 'Estado', 'Atraso Medio (min)', 'Total Voos']
    st.dataframe(top_atraso, use_container_width=True, hide_index=True)

with col_top2:
    st.subheader("🟢 Top 10 Aeroportos - Menor Atraso")
    bottom_atraso = atraso_aeroporto.nsmallest(10, 'atraso_medio')[['IATA_CODE', 'AIRPORT', 'STATE', 'atraso_medio', 'total_voos']]
    bottom_atraso.columns = ['Codigo', 'Aeroporto', 'Estado', 'Atraso Medio (min)', 'Total Voos']
    st.dataframe(bottom_atraso, use_container_width=True, hide_index=True)

# ============================================================
# CAUSAS DE ATRASO
# ============================================================
st.markdown("---")
st.subheader("Causas de Atraso (Voos Atrasados)")

df_atrasados = df_filtrado[df_filtrado['ATRASOU'] == 1]

if len(df_atrasados) > 0:
    causas = {
        'Sistema Aereo': df_atrasados['AIR_SYSTEM_DELAY'].sum(),
        'Seguranca': df_atrasados['SECURITY_DELAY'].sum(),
        'Companhia Aerea': df_atrasados['AIRLINE_DELAY'].sum(),
        'Aeronave Anterior': df_atrasados['LATE_AIRCRAFT_DELAY'].sum(),
        'Clima': df_atrasados['WEATHER_DELAY'].sum()
    }

    causas_df = pd.DataFrame(list(causas.items()), columns=['Causa', 'Minutos'])
    causas_df = causas_df.sort_values('Minutos', ascending=False)

    fig_causas = px.pie(
        causas_df,
        values='Minutos',
        names='Causa',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4
    )
    fig_causas.update_layout(height=400)
    st.plotly_chart(fig_causas, use_container_width=True)
else:
    st.info("Nenhum voo atrasado com os filtros selecionados.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Tech Challenge Fase 3 - FIAP PosTech Machine Learning Engineering</p>
    <p>Dashboard desenvolvido com Streamlit e Plotly</p>
</div>
""", unsafe_allow_html=True)
