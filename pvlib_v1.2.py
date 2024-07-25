#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTAÇÃO DE BIBLIOTECAS

import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, SingleAxisTrackerMount
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates as mdates
import time
import seaborn as sns
import json

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINIÇÃO DO ESTILO DE PLOTAGEM

sns.set_style("whitegrid")
#sns.set_context("poster")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INICIAR A CONTAGEM DE TEMPO DA SIMULAÇÃO

start_time = time.time()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS

def tmy_data_treatment(tmy_name):
    # Lê o CSV e configura o índice como uma coluna de data e hora no formato fornecido
    tmy = pd.read_csv(f'{tmy_name}.csv', skiprows=2, nrows=8760, 
                      usecols=['Year','Month','Day', 'Hour', 'Minute','DHI', 'DNI', 'GHI', 'Temperature', 'Wind Speed'],)
    # Criar minha coluna datatime index
    tmy['datetime'] = pd.to_datetime(tmy[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    tmy.set_index('datetime', inplace=True)
    # Remover as colunas individuais, se necessário
    tmy.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)
    # Modifica o nome das colunas para corresponder com a leitura da biblioteca pvlib
    tmy.columns = ['dhi', 'dni', 'ghi', 'temp_air', 'wind_speed']
    columns_order = ['temp_air','ghi','dni','dhi','wind_speed']
    tmy = tmy[columns_order]
    # Modifica minha coluna index com as datas referentes
    tmy.index = pd.date_range(start='2023-01-01 00:30', end='2023-12-31 23:30', freq='h')
    tmy.to_csv('teste.csv', index=False)
    #tmy.to_csv('nrel_data_process.csv')
    return tmy

def curtailment(nome, tmy, pot_pv, pot_wind, must, export_csv=False):
    #Definir os dicionários para preenchimento com os dados de solar e vento, com e sem curtailment
    no_curtailment = dict()
    with_curtailment = dict()
    assoc = dict()
    #Associação do tempo ao dicionário
    no_curtailment["time"] = tmy.index
    with_curtailment["time"] = tmy.index
    assoc["time"] = tmy.index
    #Definição das potências das usinas solares
    N = np.arange(2, must+1, 2)
    # Loop para calcular o curtailment para cada potência
    for n in N:
        #Preenchimento dos valores de solar (respectivos para cada potência) nos dicionários
        no_curtailment[f"solar {n} MW"] = pot_pv*(n / 2)
        with_curtailment[f"solar {n} MW"] =  pot_pv*(n / 2)
        assoc[f"solar {n} MW + wind"] = []
        #Loop para percorrer a lista com os dados de geração solar
        for i, p in enumerate(with_curtailment[f"solar {n} MW"]):
            assoc_hour = with_curtailment[f"solar {n} MW"].iloc[i] + pot_wind["pot"][i]
            #Condicional para fazer o curtailment da solar em relação ao MUST de 27 MW
            if (assoc_hour) > must:
                with_curtailment[f"solar {n} MW"].iloc[i] = must - pot_wind["pot"][i]
                assoc_hour = with_curtailment[f"solar {n} MW"].iloc[i] + pot_wind["pot"][i]
            #Condicional para zerar valores negativos que surgem nos horários de ausência solar
            if with_curtailment[f"solar {n} MW"].iloc[i] < 0:
                no_curtailment[f"solar {n} MW"].iloc[i] = 0
                with_curtailment[f"solar {n} MW"].iloc[i] = 0
                assoc_hour = with_curtailment[f"solar {n} MW"].iloc[i] + pot_wind["pot"][i]
            assoc[f"solar {n} MW + wind"].append(assoc_hour)

    #Exportação dos dados para excel
    nc = pd.DataFrame.from_dict(no_curtailment)
    wc = pd.DataFrame.from_dict(with_curtailment)
    cga = pd.DataFrame.from_dict(assoc)

    # Condicional para exportar ou não para csv
    if export_csv == True:
        # Exportar os dados da planta híbrida para csv
        nc.to_csv(f'no_curtailment - {nome}.csv', index=False)
        pv_curtailment.to_csv(f'with_curtailment - {nome}.csv', index=False)
        cga.to_csv(f'cga_power - {nome}.csv', index=False)
    else:
        pass

    return nc, wc, cga

def finance(pv_month, wind_month, must, type_pv):
    # Definição de parâmetros financeiros
    tust = 6.702
    desc_tust = 0.5
    must_kw = must * 1000
    r = 0.08
    degradation_pv = 0.0045
    #degradation_eol = 0.0064
    anos = 20

    # Criação do range de potência
    N = np.arange(2, must+1, 2)

    # Criação do dicionário com os dados de EUST
    eust = {}

    # Loop para calcular os EUSTs correspondentes a todas as potências instaladas
    for n in N:
        # Criação de lista para a potência n, para ser alocada no dicionário
        eust[f"solar {n} MW"] = []
        for i, p in enumerate(pv_month[f"solar {n} MW"]):
            # Cálculo do EUST e append dos valores na lista
            eust_value = tust * must_kw * (1-(desc_tust * wind_month["pot"].iloc[i])/(pv_month[f"solar {n} MW"].iloc[i] + wind_month["pot"].iloc[i]))
            eust[f"solar {n} MW"].append(eust_value)

    # Converter dicionário em Dataframe
    eust_month = pd.DataFrame.from_dict(eust)
    # Somar valores mensais com intuito de obter anuais
    eust_month = eust_month.sum()
    # Criação do EUST referente à eólica
    eust_wind_month = 12 * tust * must_kw * (1 - desc_tust)

    # Criação do dataframe referente ao CAPEX
    capex = pd.DataFrame()
    # Ajuste do index do CAPEX para alinhar com o do EUST
    capex.index = eust_month.index
    # Atribuição do array de potências para o dataframe do CAPEX
    capex[''] = N

    # Criação do dataframe referente ao OPEX
    opex = pd.DataFrame()
    # Ajuste do index do OPEX para alinhar com o do EUST
    opex.index = eust_month.index
    # Atribuição do array de potências para o dataframe do OPEX
    opex[''] = N
    #opex[''] = opex[''] * 1000 * 35.16
    #opex[''] = opex[''] * 1000 * 50
    
    # Cálculo do CAPEX e O&M de acordo com a tecnologia implementada
    if type_pv == 'fixed':
        #formula do CFV -> CFV = 3.3441*(Pot*1.3)^(-0.035)
        # Cálculo do CAPEX de acordo com a curva desenvolvida
        capex[''] = ((3.3441 * (capex[''] * 1.3) ** (-0.035)) * (capex[''] * 1.3) * 10**6) * (1 - 0.1)
        # Cálculo do O&M para associar com o OPEX
        opex[''] = 0.0058 * capex['']
    elif type_pv == 'tracking':
        # Cálculo do CAPEX para o tracking, com o aumento de 20%    
        capex[''] = (((3.3441 * (capex[''] * 1.3) ** (-0.035)) * (capex[''] * 1.3) * 10**6) * (1 - 0.1)) * 1.20
        # Cálculo do O&M do tracking para associar com o OPEX
        opex[''] = 0.0062 * capex['']
    else:
        pass
    
    # Associação da lista de CAPEX a uma outra variável
    capex = capex['']
    # Criação de dicionários para OPEX, Energia e EUST da Central Geradora Associada (CGA)
    opex_mod_d = {}
    ei_d = {}
    ei_mod_d = {}
    eust_cga_d = {}

    # Loop para calcular variações anuais de opex e energia
    for ano in range(1, anos+1):
        # Cálculo da tarifa EUST para cada ano, com a variação anual
        eust_cga_d[f'Ano {ano}'] = (eust_month-eust_wind_month) * pow((1 + 0.034), ano)
        # Soma do O&M calculado com o EUST
        opex[f'Ano {ano}'] = opex[''] + eust_cga_d[f'Ano {ano}']
        # Cálculo do OPEX com a variação anual de juros
        opex_mod_d[f'Ano {ano}'] = (opex[f'Ano {ano}']) / ((1 + r) ** (ano - 1))
        # Cálculo da energia FV conforme a degração estimada
        ei_d[f'Ano {ano}'] = ((pv_month.sum()*1000) * ((1 - degradation_pv) ** (ano - 1))) #+ ((wind_month['pot'].sum()*1000) * ((1 - degradation_eol) ** (ano - 1)))
        # Cálculo da energia com a variação anual de juros
        ei_mod_d[f'Ano {ano}'] = (ei_d[f'Ano {ano}']) / ((1 + r) ** (ano - 1))

    # Excluir coluna anterior do OPEX
    del opex['']

    # Converter todos oos dicionários em DataFrames
    opex_mod = pd.DataFrame.from_dict(opex_mod_d)
    #ei = pd.DataFrame.from_dict(ei_d)
    ei_mod = pd.DataFrame.from_dict(ei_mod_d)
    eust_cga = pd.DataFrame.from_dict(eust_cga_d)

    # Somar os valores anuais de opex e energia
    opex_mod_sum = opex_mod.sum(axis=1)
    ei_mod_sum =  ei_mod.sum(axis=1)
    
    # Cálculo do LCOE com os valores de CAPEX, OPEX e energia
    lcoe_mwh = 1000 * (capex + opex_mod_sum) / (ei_mod_sum)
    # Excluir valores nulos
    lcoe_mwh = lcoe_mwh.dropna()
    # Lógica para organizar os dados de LCOE em ordem crescente
    num = lcoe_mwh.index.str.extract('(\d+)')[0].astype(int)
    lcoe_mwh = lcoe_mwh.iloc[num.argsort()]
    # Extração do valor de CAPEX referente ao menor LCOE
    capex_ideal = capex[lcoe_mwh.idxmin()]
    # Retornar LCOE, CAPEX, OPEX e energia
    return lcoe_mwh, capex_ideal, opex_mod, ei_mod

def resolve_warnings():
    # A biblioteca pvlib quando executada gera alguns warnings desnecessários de serem imprimidos
    # Dessa forma, a função bypassa todos os warnings gerados pelo módulo pvlib
    warnings.filterwarnings(action='ignore', module='pvfactors')
    warnings.filterwarnings("ignore", message="Original names contain .* duplicate.*")
    warnings.filterwarnings("ignore", message="Normalized names contain .* duplicate.*")

def grafico_irradiancia(nome, time, ghi, dni, dhi, lower_date, upper_date, show=False, savefig=False):
    # Função para gerar gráfico de irradiância
    # Criação da Figura
    plt.figure(figsize=(16,9))
    # Título
    plt.title(f"Irradiância - {nome}", fontsize=25, fontweight='bold')
    # Plot das 3 irradiancias
    plt.plot(time, ghi)
    plt.plot(time, dni)
    plt.plot(time, dhi)
    # Definição dos limite de data a ser plotada
    plt.xlim(pd.to_datetime(lower_date), pd.to_datetime(upper_date))
    #Definição das legendas e labels; ajuste de data
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M', tz='America/Sao_Paulo'))
    plt.ylabel('Irradiância (W/m²)', weight='bold', fontsize=22)
    plt.xticks(fontweight='bold', fontsize=20, rotation=45)
    plt.yticks(fontweight='bold', fontsize=20)
    plt.legend(['ghi', 'dni', 'dhi'], fontsize=20, loc='upper right')  # Adiciona legenda
    # Ajustar gráfico e plotar
    plt.tight_layout()
    if show == True: plt.show()
    else: pass
    if savefig == True: plt.savefig(f"Irradiância - {nome}.png")
    else: pass

def grafico_vento(nome, time, wind, lower_date, upper_date, show=False, savefig=False):
    # GRÁFICO VENTO
    plt.figure(figsize=(16,9))
    plt.title(f"Vento - {nome}")
    plt.plot(tmy.index, wind, label='Vento')
    plt.ylabel('Velocidade do Vento (m/s)', weight='bold')
    plt.xlim(pd.to_datetime(lower_date), pd.to_datetime(upper_date))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M', tz='America/Sao_Paulo'))
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend()  
    if show == True: plt.show()
    else: pass
    if savefig == True: plt.savefig(f"Velocidade do Vento - {nome}.png")
    else: pass

def grafico_tracking(nome, orientation, lower_date, upper_date, show=False, savefig=False):
    # Realizar o plot do tracking
    plt.figure()
    orientation.fillna(0).plot()
    plt.title(f'Orientação do Tracker - {nome}', fontsize=20, fontweight='bold')
    # Definição dos limite de data a ser plotada
    plt.xlim(pd.to_datetime(lower_date).tz_localize(timezone), pd.to_datetime(upper_date).tz_localize(timezone))
    #Definição das legendas e labels; ajuste de data
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M', tz='America/Sao_Paulo'))
    plt.xticks(fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.xlabel('Ângulo (°)', fontweight='bold', fontsize=15)
    plt.ylabel('h', fontweight='bold', fontsize=15)
    # Ajustar gráfico e plotar
    plt.tight_layout()
    if show == True: plt.show()
    else: pass
    if savefig == True: plt.savefig(f'Tracker - {nome}.png', dpi=1000, bbox_inches='tight')
    else: pass

def grafico_solar(nome, fixo, tracker, lower_date, upper_date, show=False, savefig=False):
    # Plot do gráfico de geração da planta híbrida
    plt.figure(figsize=(16,9))
    plt.title(f"Geração PV - {nome}")
    fixo.plot(figsize=(16,9), color='#2D2D96').set_ylabel('MWh')
    tracker.plot(figsize=(16,9), color='#00B050').set_ylabel('MWh')
    plt.legend(['Sistema Fixo', 'Sistema com Seguidor de um Eixo'])
    plt.xlim(lower_date, upper_date)
    if show == True: plt.show()
    else: pass
    if savefig == True: plt.savefig(f'Geração PV - {nome}.png', dpi=1000, bbox_inches='tight')
    else: pass

def grafico_cga(nome, tmy, pv, wind, cga, tec, pot_solar, must, lower_date, upper_date, show=False, savefig=False):
    pv.index = tmy.index   
    wind.index = tmy.index
    cga.index = tmy.index
    # Plot do gráfico de geração da planta híbrida com eixo fixo
    plt.figure(figsize=(16,9))
    plt.title(f"Geração - {nome} - {tec}", fontweight='bold', fontsize=25)
    pv[f'{pot_solar}'].plot(figsize=(16,9), linewidth=3, color='#FF7B07')
    wind['pot'].plot(figsize=(16,9), linewidth=3, color='#237AB5')
    cga[f'{pot_solar} + wind'].plot(figsize=(16,9), linewidth=3, color='#269D26')
    plt.legend(['Solar', 'Eólica', 'Sistema Associado'], fontsize=20)
    plt.xlim(pd.to_datetime(lower_date), pd.to_datetime(upper_date))
    plt.ylim(-1, must+9)
    plt.xticks(fontweight='bold', fontsize=20)
    plt.yticks(fontweight='bold', fontsize=20)
    plt.xlabel('Data (dia-mês hora)', fontweight='bold', fontsize=22)
    plt.ylabel('Energia (MWh)', fontweight='bold', fontsize=22)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M', tz='America/Sao_Paulo'))
    plt.hlines(y=must, xmin=pd.to_datetime(lower_date), xmax=pd.to_datetime(upper_date), linestyles='dashed', color='r', linewidth=3)  
    plt.text(pd.to_datetime(lower_date), must, 'MUST Contratado', color='r', ha='right', va='bottom', fontsize=20)
    if show == True: plt.show()
    else: pass
    if savefig == True: plt.savefig(f'geração cga {dados["nome"][i]}.png', dpi=1000, bbox_inches='tight')
    else: pass   

def grafico_lcoe(nome, lcoe_fixo, lcoe_tracking, N, show=False, savefig=False):
    # Ajuste da variável do LCOE para plotar
    plt.figure()
    plt.title(f"LCOE - {nome}", fontweight='bold', fontsize=25)
    lcoe_fixo.index = N
    lcoe_fixo.plot(figsize=(12.8, 9.6), color='#333399', linewidth=3) #figsize=(12, 6.75) #figsize=(16, 9)
    lcoe_tracking.index = N
    lcoe_tracking.plot(figsize=(12.8, 9.6), color='#00B050', linewidth=3) #figsize=(8, 4.5) #figsize=(16, 9)
    plt.xlabel('Potência da Usina Fotovoltaica (MW)', fontweight='bold', fontsize=22)
    plt.xticks(fontweight='bold', fontsize=20)
    plt.yticks(fontweight='bold', fontsize=20)
    plt.ylabel('LCOE (R$/MWh)', fontweight='bold', fontsize=22)
    plt.legend(['Sistema fixo', 'Sistema com seguidor de um eixo'], loc='best', fontsize=20)
    if show == True: plt.show()
    else: pass
    if savefig == True: plt.savefig(f'lcoe {dados["nome"][i]}.png', dpi=1000, bbox_inches='tight')
    else: pass   

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##############################################################      MAIN      ###############################################################################################
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Função para resolver os warnings
resolve_warnings()

# Lendo o arquivo JSON com as localidades e convertendo-o em um dicionário
with open("loc_parques.json", 'r', encoding='utf-8') as json_file:
    dados = json.load(json_file)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# VARIÁVEIS E LISTAS

# Dataframe onde serão armazenados os melhores cenários
usinas = pd.DataFrame()

# Fuso Horário
timezone = 'America/Bahia'

# Definição do Arranjo fotovoltaico
modules_per_string = 30
strings_per_inverter = 160

# Variável de controle
j = 0
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOOP 

# Condiciona a duração do loop à quantidade de parques
for i in range(len(dados['latitude'])):

    # Atribuição dos dados de localização ao objeto do PVLIB
    location = Location(latitude = dados['latitude'][i], 
                        longitude = dados['longitude'][i], 
                        tz= timezone, 
                        altitude = dados['altitude'][i], 
                        name= dados['nome'][i])
    
    # Ler o arquivo com os dados TMY reais
    tmy = pd.read_csv('{}_tratado.csv'.format(dados['tmy'][i]))
    # Ajuste do índice do arquivo TMY
    tmy.index = tmy['Unnamed: 0']
    tmy.index.name = None
    tmy.drop(columns=['Unnamed: 0'], inplace=True)
    tmy.index = pd.to_datetime(tmy.index)
    tmy.index = tmy.index.tz_localize(timezone)
    
    # Criação do modelo de temperatura dos módulos
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS ["sapm"]["open_rack_glass_glass"]

    # Importa biblioteca dos parametros eletricos dos modulos fotovoltaicos e inversores do SAM (System Advisor Model)
    cec_modules = pvlib.pvsystem.retrieve_sam(path = 'https://raw.githubusercontent.com/NREL/SAM/patch/deploy/libraries/CEC%20Modules.csv')
    sapm_inverters = pvlib.pvsystem.retrieve_sam(path = 'https://raw.githubusercontent.com/NREL/SAM/patch/deploy/libraries/CEC%20Inverters.csv')

    # Definir o modelo de módulo e inversor
    module = cec_modules['JA_Solar_JAM72D30_540_MB']
    inverter = sapm_inverters['Schneider_Electric_Solar_Inverters_USA__Inc___CS2000_NA__575V_']

    # Definição do valor do ângulo de inclinação do módulo, para a tecnologia fixa
    if abs(dados['latitude'][i]) < 10:
        angulo = 10
    else:
        angulo = abs(dados['latitude'][i])
    
    # Criar o array da usina solar
    system = PVSystem(name= f"Usina Solar {dados['nome'][i]}", surface_tilt = angulo, surface_azimuth = 0,
                    module_parameters=module, inverter_parameters= inverter,
                    temperature_model_parameters = temperature_model_parameters, 
                    modules_per_string = modules_per_string, strings_per_inverter = strings_per_inverter)

    # Criar o ModelChain, para realizar a simulação
    mc = ModelChain(system, location, 
                    aoi_model= 'no_loss', 
                    spectral_model='no_loss')

    # Criar o objeto de tracking
    mount = SingleAxisTrackerMount(axis_tilt=0, axis_azimuth=180, max_angle=60, backtrack=False)

    # Pegar a posição do sol
    sol_pos = location.get_solarposition(tmy.index)
    # Orientar o módulo de acordo com a direção do sol
    orientation = mount.get_orientation(solar_zenith = sol_pos['apparent_zenith'], solar_azimuth = sol_pos['azimuth'])


    # Criar o array da usina solar com tracking 
    array = Array(mount=mount, module_parameters=module, temperature_model_parameters=temperature_model_parameters,
              modules_per_string=modules_per_string, strings=strings_per_inverter)
    system_tracking = PVSystem(arrays=[array], inverter_parameters=inverter)

    # Criação do modelchain com tracking
    mc_tracking = ModelChain(system_tracking, location,
                            aoi_model= "no_loss",
                            spectral_model="no_loss"
                            )

    # Simular os modelos
    mc.run_model(tmy)
    mc_tracking.run_model(tmy)

    # Imprimir os resultados
    # Dividir por 1M para encontrar o valor equivalente em MW
    pot_pv = mc.results.ac/1000000
    pot_pv_tracking = mc_tracking.results.ac/1000000

    # Leitura e tratamento dos dados de vento
    wind = pd.read_csv(f"{dados['wind'][i]}.csv")
    temp_pre = pd.read_csv(f"{dados['tp'][i]}.csv")
    
    # Criação de dicionário com os dados de potência eólica
    pot_wind = dict()
    # Criação de colunas de tempo (data) e potência
    if dados['nome'][i] == 'Casa Nova A':
        pot_wind['time'] = wind['Data']
        pot_wind['pot no correction'] = wind["Geração"]
        pot_wind['pot'] = wind["Geração"]
        pot_wind['ad ratio'] = temp_pre['ad ratio']
    else:
        pot_wind['time'] = wind['Data']
        pot_wind['pot no correction'] = wind["Geração"]
        pot_wind['pot'] = wind["Geração"] * temp_pre['ad ratio']
        pot_wind['ad ratio'] = temp_pre['ad ratio']    
    
    # Transformação do dicionário em dataframe
    pot_wind = pd.DataFrame.from_dict(pot_wind)

    # Chamar a função que executa a lógica do curtailment
    # São exportados alguns dataframes com dados de geração de cada usina e da central geradora associada (cga)
    # Caso seja desejado exportar os resultados para csv, atribuir export_csv=True
    nc, pv_curtailment, cga = curtailment(nome=dados['nome'][i], tmy=tmy, pot_pv=pot_pv, pot_wind=pot_wind, must=dados['must'][i])
    nc_tracking, pv_curtailment_tracking, cga_tracking = curtailment(nome='{} (tracking)'.format(dados['nome'][i]), tmy=tmy, pot_pv=pot_pv_tracking, pot_wind=pot_wind, must=dados['must'][i])
    #print(f'Curtailment solar fixo: \n{100*(nc.iloc[:, 1:].sum()-pv_curtailment.iloc[:, 1:].sum())/nc.iloc[:, 1:].sum()}')
    #print(f'Curtailment solar tracking: \n{100*(nc_tracking.iloc[:, 1:].sum()-pv_curtailment_tracking.iloc[:, 1:].sum())/nc_tracking.iloc[:, 1:].sum()}')
    #print('Rendimento Tracking: \n{}'.format(pv_curtailment_tracking.iloc[:, 1:].sum()/pv_curtailment.iloc[:, 1:].sum()))

    # Setar a data como index do pv e wind
    pv_curtailment.set_index('time', inplace=True)
    pv_curtailment_tracking.set_index('time', inplace=True)
    pot_wind.set_index('time', inplace=True)

    # Ajustar os dados de pv e wind para mensal
    pv_month = pv_curtailment.groupby(pv_curtailment.index.month).sum()
    pv_tracking_month = pv_curtailment_tracking.groupby(pv_curtailment_tracking.index.month).sum()
    wind_month = pot_wind.groupby(pv_curtailment.index.month).sum()

    # Calcular o LCOE para a usina híbrida
    lcoe_mwh, capex_fixed, opex_fixed, energy_fixed = finance(pv_month=pv_month, wind_month=wind_month, must=dados['must'][i], type_pv='fixed')
    lcoe_mwh_tracking, capex_tracking, opex_tracking, energy_tracking = finance(pv_month=pv_tracking_month, wind_month=wind_month, must=dados['must'][i], type_pv='tracking')

    # Ajustar o N, para o valor máximo ser o do MUST
    N = np.arange(2, dados['must'][i]+1, 2)

    # Ajuste da variável do LCOE para plotar
    lcoe_mwh_plot = lcoe_mwh
    lcoe_mwh_tracking_plot = lcoe_mwh_tracking

    # Print dos melhores cenários de LCOE, considerando fixo e tracking
    print('Usina {} PV Fixo:  Usina {} | LCOE R${:.2f} | CAPEX R${:,.2f}'.format(dados['nome'][i], lcoe_mwh.idxmin(), lcoe_mwh.min(), capex_fixed))
    print('Usina {} PV com Tracking:  Usina {} | LCOE R${:.2f} | CAPEX R${:,.2f}'.format(dados['nome'][i], lcoe_mwh_tracking.idxmin(), lcoe_mwh_tracking.min(), capex_tracking))
    print('\n')

    # Junção dos dados para colocar no dataframe Usinas
    parques = {
        'Usina': dados['nome'][i],
        'Capacidade Instalada (MW)': lcoe_mwh.idxmin(),
        'LCOE (R$/MWh)': lcoe_mwh.min(),
        'CAPEX (R$)': capex_fixed,
        'Capacidade Instalada (MW) - Tracking': lcoe_mwh_tracking.idxmin(),
        'LCOE (R$/MWh) - Tracking': lcoe_mwh_tracking.min(),
        'CAPEX (R$) - Tracking': capex_tracking
    }
    parques_df = pd.DataFrame(parques, index=[0])

    # Anexe o DataFrame temporário ao DataFrame final
    usinas = pd.concat([usinas, parques_df], ignore_index=True)

    # Exportação de dados anuais para excel
    #opex_fixed.to_excel(f"{dados['nome'][i]} opex fixo.xlsx")
    #energy_fixed.to_excel(f"{dados['nome'][i]} energia fixo.xlsx")
    #opex_tracking.to_excel(f"{dados['nome'][i]} opex tracking.xlsx")
    #energy_tracking.to_excel(f"{dados['nome'][i]} energia tracking.xlsx")

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GRÁFICOS
    # Caso seja desejado, plotar a figura ou salvar, inserir os atributos show e savefig como iguais a True (e.g., show=True, savefig=True)

    # Obtenção do gráfico de irradiância
    grafico_irradiancia(nome=dados['nome'][i], time=tmy.index, ghi=tmy["ghi"], dni=tmy["dni"], dhi=tmy["dhi"], lower_date='2023-01-01', upper_date='2023-01-10')

    # Obtenção do gráfico de velocidade do vento
    grafico_vento(nome=dados['nome'][i], time=tmy.index, wind=wind['Wind Speed'], lower_date='2023-01-01', upper_date='2023-12-31')

    # Obtenção do gráfico de orientação do tracker
    grafico_tracking(nome=dados['nome'][i], orientation=orientation['tracker_theta'], lower_date='2023-01-01', upper_date='2023-01-02')
    
    # Obtenção do gráfico de geração solar comparando fixo com tracking de um eixo
    grafico_solar(nome=dados['nome'][i], fixo=pot_pv, tracker=pot_pv_tracking, lower_date='2023-01-01', upper_date='2023-01-02')

    # Obtenção do gráfico de geração associada, considerando a tecnologia fixa
    grafico_cga(nome=dados['nome'][i], tmy=tmy, pv=pv_curtailment, wind=pot_wind, cga=cga, tec="Fixo", 
                pot_solar=f'{lcoe_mwh.idxmin()}', must=dados['must'][i], lower_date='2023-08-23', upper_date='2023-08-27', 
                show=False, savefig=False)
    
    # Obtenção do gráfico de geração associada, considerando a tecnologia de seguidor solar de um eixo
    grafico_cga(nome=dados['nome'][i], tmy=tmy, pv=pv_curtailment_tracking, wind=pot_wind, cga=cga_tracking, tec="Seguidor Solar", 
                pot_solar=f'{lcoe_mwh.idxmin()}', must=dados['must'][i], lower_date='2023-08-23', upper_date='2023-08-27', show=False, savefig=False)
    
    # Obtenção do gráfico de LCOE, ccomparando as duas tecnologias (fixa e tracker)
    grafico_lcoe(nome=dados['nome'][i], lcoe_fixo=lcoe_mwh_plot, lcoe_tracking=lcoe_mwh_tracking_plot, N=N, show=False, savefig=False)


# Encerrar o contador de tempo da simulação e printar em tela
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTempo de execução do código: {execution_time:.2f} segundos")
usinas.to_excel('Usinas.xlsx', index_label=False)

##############################################################      FIM       ###############################################################################################