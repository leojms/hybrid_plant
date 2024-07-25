import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import calendar
 
sns.set_style("whitegrid")


# Dicionário de meses em português
meses_pt = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}

#Variável  Global
Local = 'PINDAI'
altura = 100    #Goldwind/Enercon 100 metros e Gamesa 78 metros
ano = 2023       # Período de 2020 a 2023 # Verificar se o ano escolhido é bisexto

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Lê o CSV da irradiância solar do TMY

irrad_TMY = pd.read_csv( fr'C:\Users\rodrigo.miranda\Sistema FIEB\GTD- Projeto Chesf - General\03_INF\Estudo - 17 - Hibridização Eletrobrás\complementariedade\Dados\Irradiância_NREL\{Local}\tmy-2022.csv', 
                    nrows=8761, skiprows=2)

#Criar minha coluna datatime index
irrad_TMY['datetime'] = pd.to_datetime(irrad_TMY[['Year', 'Month', 'Day', 'Hour', 'Minute']])
irrad_TMY.set_index('datetime', inplace=True)

# Remover as colunas individuais, se necessário
irrad_TMY.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)

#Modifica o nome das colunas para corresponder com a leitura da biblioteca pvlib
irrad_TMY.columns = ['dhi', 'dni', 'ghi', 'temp_air', 'wind_speed', 'Surface_Albedo']
columns_order = ['temp_air','ghi','dni','dhi','wind_speed', 'Surface_Albedo']
irrad_TMY = irrad_TMY[columns_order]

#Modifica minha coluna index com as datas referentes 
irrad_TMY.index = pd.date_range(start='2023-01-01 00:00', end='2023-12-31 23:00', freq='h')

irrad_TMY.index = pd.to_datetime(irrad_TMY.index)

irrad_TMY.index.name = 'datetime'

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Lê o CSV da velocidade do vento 

vento = pd.read_csv( fr'C:\Users\rodrigo.miranda\Sistema FIEB\GTD- Projeto Chesf - General\03_INF\Estudo - 17 - Hibridização Eletrobrás\complementariedade\Dados\Vento_Nasa\{Local}\POWER_Point_Hourly_Wind.csv', 
                    nrows=35067, skiprows=9)

#Criar minha coluna datatime index
# Combine as colunas YEAR, MO, DY e HR em uma única coluna 'datetime'
vento['datetime'] = vento[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1)

# Converta a coluna 'datetime' para o tipo datetime
vento['datetime'] = pd.to_datetime(vento['datetime'], format='%Y-%m-%d-%H')

# Remover as colunas individuais, se necessário
vento.drop(['YEAR', 'MO', 'DY', 'HR'], axis=1, inplace=True)

# Configure 'datetime' como índice
vento.set_index('datetime', inplace=True)

# Filtre os dados para o ano de escolhido para o vento
vento_filt = vento[vento.index.year == ano]
#--------------------------------------------------------------

#--------------------------------------------------------------
#Correção da velocidade do vento -- Lei da Potência--

# Constantes para correção

a = 0.143                    #Coeficiente de atrito (adimensional) 0.143
v_r = vento_filt["WS50M"]    #Velecidade de vento de referência (m/s)
h = altura                   #Altura da nova velocidade do vento (m)
h_r = 50                     #Altura da velocidade do vento de referência (m)

#Calculo para correção da velocidade do vento para nova altura

vento_filt[f'WS{altura}M']= v_r*(h/h_r)**a

#--------------------------------------------------------------

#--------------------------------------------------------------
#Normalização dos dados de vento e GHI

# Calcular o valor máximo de GHI
max_GHI = irrad_TMY['ghi'].max()
# Calcular o valor máximo de vento
max_Vento = vento_filt[f'WS{altura}M'].max()

# Normalizar os valores de GHI e adicionar uma nova coluna
irrad_TMY['GHI_Normalizado'] = irrad_TMY['ghi'] / max_GHI
# Normalizar os valores de vento e adicionar uma nova coluna
vento_filt['Vento_Normalizado'] = vento_filt[f'WS{altura}M'] / max_Vento

#--------------------------------------------------------------

#--------------------------------------------------------------
# Função para calcular o dia típico de cada mês
def calcular_dia_tipico(df, coluna):
    # Adicionar colunas para ano, mês, dia e hora
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    
    # Agrupar por mês e hora e calcular a média
    dia_tipico = df.groupby(['month', 'hour'])[coluna].mean().unstack(level=0)
    
    return dia_tipico

# Calcular o dia típico para GHI normalizado e vento normalizado
dia_tipico_GHI = calcular_dia_tipico(irrad_TMY, 'GHI_Normalizado')
dia_tipico_vento = calcular_dia_tipico(vento_filt, 'Vento_Normalizado')



#print(dia_tipico_vento)
#print(dia_tipico_GHI)

# Calcular a correlação de Pearson entre GHI normalizado e vento normalizado para cada mês
correlacoes = {}
for mes in dia_tipico_GHI.columns:
    correlacoes[mes] = dia_tipico_GHI[mes].corr(dia_tipico_vento[mes])

# Imprimir as correlações
for mes, correlacao in correlacoes.items():
    print(f'Correlação de Pearson para o mês {mes}: {correlacao:.2f}')
# Plotar os resultados mês a mês
for mes in dia_tipico_GHI.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(dia_tipico_GHI.index, dia_tipico_GHI[mes], label='GHI Normalizado', color='orange')
    plt.plot(dia_tipico_vento.index, dia_tipico_vento[mes], label='Vento Normalizado', color='blue')
    plt.title(f'Dia Típico Mensal - {meses_pt[mes]} {ano}',fontweight='bold', fontsize=18)
    plt.xlabel('Hora do Dia', fontweight='bold',fontsize=18)
    plt.ylabel('Valor Normalizado', fontweight='bold', fontsize=18)
    plt.xticks(range(24))
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold',fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    
    plt.savefig(f'Dia_Tipico_Mensal_{meses_pt[mes]}_{ano}.png')
    plt.show()
