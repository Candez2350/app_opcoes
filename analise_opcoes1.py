import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np

# ================= CONFIGURAÇÃO =================
st.set_page_config(page_title="Radar Opções Master (Layout Pro)", page_icon="🦅", layout="wide")

# ================= LISTA IBrX 100 =================
IBXX_FULL_LIST = [
    "ABEV3", "ALOS3", "ALPA4", "ASAI3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4",
    "BBSE3", "BEEF3", "BPAC11", "BPAN4", "BRAP4", "BRAV3", "BRFS3", "BRKM5", "CAML3",
    "CASH3", "CCRO3", "CMIG4", "CMIN3", "COGN3", "CPFE3", "CPLE6", "CRFB3", "CSAN3",
    "CSNA3", "CVCB3", "CYRE3", "DXCO3", "ECOR3", "EGIE3", "ELET3", "ELET6", "EMBR3",
    "ENEV3", "ENGI11", "EQTL3", "EZTC3", "FLRY3", "GGBR4", "GOAU4", "GOLL4", "HAPV3",
    "HYPE3", "IGTI11", "IRBR3", "ITSA4", "ITUB4", "JBSS3", "JHSF3", "KLBN11", "LREN3",
    "LWSA3", "MGLU3", "MOVI3", "MRFG3", "MRVE3", "MULT3", "NTCO3", "PCAR3", "PETR3",
    "PETR4", "PETZ3", "PLPL3", "POMO4", "PRIO3", "PSSA3", "RADL3", "RAIL3", "RAIZ4",
    "RANI3", "RDOR3", "RECV3", "RENT3", "ROMI3", "SANB11", "SBSP3", "SLCE3", "SMTO3",
    "STBP3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "TRPL4", "UGPA3", "USIM5", "VALE3",
    "VBBR3", "VIVA3", "VIVT3", "WEGE3", "YDUQ3", "VAMO3", "BOVA11", "SMAL11"
]
IBXX_FULL_LIST.sort()

# ================= CÁLCULOS (CNPI) =================

@st.cache_data(ttl=600)
def obter_dados(ticker):
    if not ticker.endswith(".SA"): ticker += ".SA"
    try:
        # === MUDANÇA FUNDAMENTAL ===
        # Usamos a classe Ticker e o método .history
        # Isso evita os erros de formatação do yf.download recente
        acao = yf.Ticker(ticker)
        df = acao.history(period='6mo', interval='1d', auto_adjust=False)
        
        # O .history já retorna a tabela limpa, não precisa daquele loop de colunas
        
        # Apenas garantimos que temos as colunas certas (remove Dividends/Splits se vierem)
        cols_necessarias = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in cols_necessarias):
            return None
            
        df = df[cols_necessarias]

        # Filtro de segurança final (caso o Yahoo realmente mande dado corrompido)
        # Mas com o .history isso raramente acontece
        df = df[df['Close'] > 0]

        if len(df) > 50: return df
        return None
    except: return None

def calcular_indicadores(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Médias Setup
    df['EMA21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['SMA50'] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    
    # Osciladores
    df['MACD'] = ta.trend.MACD(close).macd()
    df['MACD_Signal'] = ta.trend.MACD(close).macd_signal()
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # --- CÁLCULO DE VOLATILIDADE HISTÓRICA (Substituto Robusto para IV) ---
    df['Log_Ret'] = np.log(close / close.shift(1))
    
    # Volatilidade Curta (20 dias) - Representa o momento ATUAL
    df['HV20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
    
    # Volatilidade Média (50 dias) - Representa a "Normalidade" do papel
    df['HV50'] = df['Log_Ret'].rolling(window=50).std() * np.sqrt(252) * 100

    return df

def criar_grafico_candle(df, ticker):
    # Cria a figura base
    fig = go.Figure()

    # 1. Adiciona os Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name=f'{ticker} Price',
        increasing_line_color='#26A69A', # Verde bonito
        decreasing_line_color='#EF5350'  # Vermelho bonito
    ))

    # 2. Adiciona as Médias do seu Setup (se elas existirem no DF)
    if 'EMA21' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21 (Rápida)', line=dict(color='cyan', width=1.5)))
    if 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50 (Lenta)', line=dict(color='yellow', width=1.5)))

    # 3. Ajustes de Layout (Visual Profissional)
    fig.update_layout(
        title=f"Gráfico Técnico: {ticker}",
        yaxis_title='Preço (R$)',
        template="plotly_dark",   # Tema escuro para combinar com o mercado
        xaxis_rangeslider_visible=False, # Remove a barra de rolagem inferior (ocupa muito espaço)
        height=600, # Altura do gráfico
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legenda no topo
    )
    return fig

def analisar_ativo(ticker, df):
    last = df.iloc[-1]
    
    score = 0
    direcao = "NEUTRO"
    setup_sugerido = "-"
    motivos = []
    
    stop_loss = 0.0
    alvo_gain = 0.0
    
    # 1. TENDÊNCIA
    # ALTA
    if (last['Close'] > last['EMA21']) and (last['EMA21'] > last['SMA50']):
        direcao = "ALTA"
        score += 2
        
        if last['MACD'] > last['MACD_Signal']: score += 1
        else: motivos.append("MACD Cruzado Venda")
        
        if last['RSI'] > 50: score += 1
        else: motivos.append("RSI Fraco")
        
        stop_loss = last['Close'] - (1.0 * last['ATR'])
        alvo_gain = last['Close'] + (1.3 * last['ATR'])

    # BAIXA
    elif (last['Close'] < last['EMA21']) and (last['EMA21'] < last['SMA50']):
        direcao = "BAIXA"
        score += 2
        
        if last['MACD'] < last['MACD_Signal']: score += 1
        else: motivos.append("MACD Cruzado Compra (Repique)")
        
        if last['RSI'] < 50: score += 1
        else: motivos.append("RSI Alto")
        
        stop_loss = last['Close'] + (1.0 * last['ATR'])
        alvo_gain = last['Close'] - (1.3 * last['ATR'])

    else:
        motivos.append("Sem Tendência Definida")

    # 2. STATUS DA VOLATILIDADE (O Pulo do Gato)
    # Compara a volatilidade de agora (HV20) com a média recente (HV50)
    vol_status = "NORMAL"
    if last['HV20'] < last['HV50'] * 0.9:
        vol_status = "📉 BAIXA (Barata)"
        cond_vol = "baixa"
    elif last['HV20'] > last['HV50'] * 1.2:
        vol_status = "📈 ALTA (Cara)"
        cond_vol = "alta"
    else:
        vol_status = "⚖️ MÉDIA"
        cond_vol = "media"

    # 3. ESTRATÉGIA FINAL
    if direcao != "NEUTRO":
        if last['ADX'] > 20: score += 1
        else: motivos.append("ADX Baixo (Lento)")
        
        if score >= 4:
            # Lógica Combinada: Força (ADX) + Preço da Volatilidade
            if last['ADX'] > 30 and cond_vol != "alta":
                setup_sugerido = "A SECO 🚀"
            elif cond_vol == "alta":
                setup_sugerido = "TRAVA (Proteção) 🛡️"
            else:
                setup_sugerido = "TRAVA ou SECO"
        else:
            setup_sugerido = "AGUARDAR"
            
    # % para o Alvo
    pct_alvo = 0.0
    if direcao == "ALTA": pct_alvo = ((alvo_gain - last['Close']) / last['Close']) * 100
    elif direcao == "BAIXA": pct_alvo = ((last['Close'] - alvo_gain) / last['Close']) * 100

    return {
        "Ativo": ticker,
        "Preço": f"R$ {last['Close']:.2f}",
        "Direção": direcao,
        "Score Num": score,
        "Score": f"{score}/5",
        "Setup": setup_sugerido,
        "Volatilidade": vol_status,
        "HV20": f"{last['HV20']:.1f}%",
        "Stop": f"R$ {stop_loss:.2f}",
        "Alvo": f"R$ {alvo_gain:.2f} ({abs(pct_alvo):.1f}%)",
        "Retorno_Pct": abs(pct_alvo), # <--- CAMPO NOVO PARA O GRÁFICO
        "Motivos": ", ".join(motivos)
    }

# ================= INTERFACE =================
st.title("⚡ Radar Opções: Preço & Volatilidade")
st.markdown("""
**Rastreamento Automático de Oportunidades (Tendência + Volatilidade)**
* **🎯 Direção:** Identifica tendências de Alta (Call) ou Baixa (Put) via Setup Gráfico.
* **📊 Volatilidade:** Analisa se o prêmio está caro ou barato comparando a HV20 vs HV50.
* **🛠️ Setup:** Sugere automaticamente se o ideal é operar **A Seco** (Explosão) ou com **Travas** (Proteção).
""")

# Sidebar
analisar_tudo = st.sidebar.checkbox("Analisar IBrX 100 Completo")
selecao = IBXX_FULL_LIST if analisar_tudo else st.sidebar.multiselect("Seleção:", IBXX_FULL_LIST, default=["PETR4", "VALE3", "PRIO3", "MGLU3", "BOVA11"])

if st.sidebar.button("🔍 Rodar Análise"):
    
    lista_alta = []
    lista_baixa = []
    lista_neutra = []
    
    barra = st.progress(0)
    
    for i, ticker in enumerate(selecao):
        df = obter_dados(ticker)
        if df is not None:
            df = calcular_indicadores(df)
            res = analisar_ativo(ticker, df)
            
            # Distribuição nas Listas
            if res['Score Num'] >= 4:
                # Remove chaves internas para exibição limpa
                exibir = {k: v for k, v in res.items() if k not in ['Direção', 'Score Num', 'Motivos']}
                # Adiciona Obs se for útil
                exibir['Obs'] = res['Motivos'] if res['Motivos'] else "Setup Limpo"
                
                if res['Direção'] == "ALTA":
                    lista_alta.append(exibir)
                elif res['Direção'] == "BAIXA":
                    lista_baixa.append(exibir)
            else:
                lista_neutra.append(res)
                
        barra.progress((i + 1) / len(selecao))
        
    barra.empty()
    
    # === LAYOUT 2 COLUNAS ===
    total_ativos = len(lista_alta) + len(lista_baixa) + len(lista_neutra)
    bullish_pct = int((len(lista_alta) / total_ativos) * 100) if total_ativos > 0 else 0
    bearish_pct = int((len(lista_baixa) / total_ativos) * 100) if total_ativos > 0 else 0

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("🐂 Ativos em Alta", len(lista_alta), f"{bullish_pct}% do radar")
    m2.metric("🐻 Ativos em Baixa", len(lista_baixa), f"-{bearish_pct}% do radar", delta_color="inverse")
    m3.metric("⚖️ Em Observação", len(lista_neutra))
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐂 CALL / ALTA")
        if lista_alta:
            st.dataframe(pd.DataFrame(lista_alta), use_container_width=True, hide_index=True)
        else:
            st.info("Sem oportunidades claras de Alta.")
            
    with col2:
        st.subheader("🐻 PUT / BAIXA")
        if lista_baixa:
            st.dataframe(pd.DataFrame(lista_baixa), use_container_width=True, hide_index=True)
        else:
            st.info("Sem oportunidades claras de Baixa.")
            
    # === ÁREA DE OBSERVAÇÃO (EXPANDER) ===
    st.divider()
    with st.expander(f"📋 Zona de Observação / Reprovados ({len(lista_neutra)})", expanded=False):
        if lista_neutra:
            df_n = pd.DataFrame(lista_neutra)
            # Ordena pelos melhores scores reprovados
            df_n = df_n.sort_values(by="Score Num", ascending=False)
            
            # Seleciona colunas relevantes
            cols_neutras = ["Ativo", "Preço", "Direção", "Score", "Volatilidade", "Motivos"]
            st.dataframe(df_n[cols_neutras], use_container_width=True, hide_index=True)
        else:
            st.write("Nenhum ativo na lista de observação.")

# ================= ÁREA DO GRÁFICO INTERATIVO =================
    st.divider()
    st.subheader("📈 Análise Gráfica Detalhada (Candles)")

    # Junta apenas os ativos que deram oportunidade (Alta + Baixa)
    oportunidades_para_grafico = [item['Ativo'] for item in lista_alta + lista_baixa]

    if oportunidades_para_grafico:
        # Cria um selectbox para o usuário escolher qual gráfico ver
        ativo_selecionado = st.selectbox("Selecione um ativo da lista para visualizar o gráfico:", oportunidades_para_grafico)
        
        if ativo_selecionado:
            with st.spinner(f"Carregando gráfico de {ativo_selecionado}..."):
                # Precisamos pegar os dados novamente para garantir que temos o histórico para o gráfico
                # (Poderíamos usar session_state para otimizar, mas assim é mais simples por enquanto)
                df_chart = obter_dados(ativo_selecionado)
                
                if df_chart is not None:
                    # Recalcula os indicadores para plotar as médias
                    df_chart = calcular_indicadores(df_chart)
                    
                    # Cria e exibe o gráfico
                    figura_plotly = criar_grafico_candle(df_chart, ativo_selecionado)
                    st.plotly_chart(figura_plotly, use_container_width=True)
                else:
                    st.error("Erro ao carregar dados para o gráfico.")
    else:
        st.info("Rode a análise e aguarde encontrar oportunidades para visualizar os gráficos.")
