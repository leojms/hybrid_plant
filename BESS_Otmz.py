# IMPORTAÇÃO DE MÓDULOS EXTERNOS
###############################################################################
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# CONFIGURAÇÕES DO AMBIENTE DE SIMULAÇÃO
###############################################################################
gp.setParam('OutputFlag', 0)
gp.setParam('TimeLimit', 30.0)
gp.setParam('MIPGap', 1e-4)

'Função p/ importação de dados via csv'
###############################################################################
def getData(n):
     
    # Listagem dos artigos com dados de entrada
    fils = { 1: 'OtmzBESS_test24h.csv'    ,\
             2: None                    }
        
    # Lê os dados 
    with open(fils[n],'r') as file:
       plh = np.loadtxt(file, delimiter = ';')
    
    # Anota os dados de Velociade do Vento(V, m/s), Potencia (Pe, MW) e Coeficiente de Potência (Cp)  
    dat = {'hora': plh[:, 0] ,\
           'Solr': plh[:, 1] ,\
           'Wind': plh[:, 2] ,\
           'PLDh': plh[:, 3] ,\
           'En0': 0.0        }

    return dat

'Função p/ importação de dados via excel'
###############################################################################
def getDataFromExcel(n):
     
    # Listagem dos artigos com dados de entrada
    fils = { 1: 'Dataset Week.xlsx'     ,\
             2: 'Dataset Month.xlsx'    ,\
             3: 'Dataset YR.xlsx'       ,\
             4: 'Dataset 20YR.xlsx'     }
    
    # Lê os dados 
    plh = pd.read_excel(fils[n])
    
    # Anota os dados de Velociade do Vento(V, m/s), Potencia (Pe, MW) e Coeficiente de Potência (Cp)  
    dat = {'hora': pd.to_datetime(plh.iloc[:, 0]) ,\
           'Solr': plh.iloc[:, 1] ,\
           'Wind': plh.iloc[:, 2] ,\
           'PLDh': plh.iloc[:, 3] ,\
           'En0': 0.0        }
    comp = len(plh)

    return dat, comp

'Problema de teste'
###############################################################################
def testProb():
    
    #--------------------------------------------------------------------------
    # OTIMIZAÇÃO
   
    # Importa dados
    #gD = getData(1)
    gD, comp = getDataFromExcel(3)
    # Inicializa o modelo
    modl = gp.Model('Test')
    rgHr = range(comp)
    # Definição de parâmetros
    chMax = 2
    t = 50
    bSize = 4
    PsVar = 3
    must = 27
    SoCs = []
    # Definição dos preços da bateria
    # De acordo com o tamanho da bateria em horas (bSize)
    match bSize:
        case 1:
            a1 = 1538.32 #1538.32
            a2 = a1
        case 2:
            a1 = 1366.80 #1366.80
            a2 = a1
        case 4:
            a1 = 1248.88 #1248.88
            a2 = a1
        case _:
            a1 = 1538.32 #1282
            a2 = a1

    # Variáveis
    Ps   = modl.addVars(rgHr,vtype="C",name='Ps' ,lb= 0.00)                ## Potência total
    PbD  = modl.addVars(rgHr,vtype="C",name='PbD',lb= 0.00)                ## Potência de descarga do BESS
    PbC  = modl.addVars(rgHr,vtype="C",name='PbC',lb= 0.00)                ## Potência de carga do BESS
    STb  = modl.addVars(rgHr,vtype="B",name='STb')                         ## Estado do BESS (1/0 <-> descarga/carga)
    En  = modl.addVars(range(comp+1),vtype="C",name='En',lb=0.00,ub=t*bSize)     ## Nível de carga do BESS
    SzbMWh  = modl.addVar(vtype="C",name='SzbMWh',lb=0.00,ub=t*bSize)            ## Capacidade energética do BESS
    SzbMW  = modl.addVar(vtype="C",name='SzbMW',lb=0.00,ub=t)              ## Capacidade de potência do BESS
    modl.update()

    # Restrições
    for h in rgHr:
        
        ## Limite de injeção total
        modl.addConstr(Ps[h]  <= must, name='Must_'+str(h+1))
        
        ## Equação de balanço energético
        modl.addConstr(Ps[h]  == (PbD[h]-PbC[h])+gD['Solr'][h]+gD['Wind'][h], name='Bal_'+str(h+1))
       
        ## Exculsão mútua entre variáves para geração em carga/descarga do BESS
        modl.addConstr(PbD[h] <= STb[h]*1000, name='STchr_'+str(h+1))
        modl.addConstr(PbC[h] <= (1-STb[h])*1000, name='STdis_'+str(h+1))
        
        ## Dinâmicas de carga/descarga do BESS
        if (h==0):
            modl.addConstr(En[h] <= 0.8*SzbMWh, name='EnSizMaj_'+str(h+1))
            En[h] = 0.2*SzbMWh
        if (h<len(range(comp))):
            modl.addConstr(En[h+1] == En[h]-1*(PbD[h]-PbC[h]), name='dEn'+str(h+1))
            modl.addConstr(En[h] <= 0.8*SzbMWh, name='EnSizMaj_'+str(h+1))
            modl.addConstr(En[h+1] - En[h] <= SzbMW, name='dEn'+str(h+1))
            if (h!=0):
                #modl.addConstr(Ps[h] - Ps[h-1] <= PsVar, name='PsVar'+str(h+1))
                modl.addConstr(En[h] >= 0.2*SzbMWh, name='EnSizMin_'+str(h+1))

        ## Limitação de injeção e absorção de potência
        modl.addConstr(PbD[h] <= SzbMW, name='PbDsiz_'+str(h+1))
        modl.addConstr(PbC[h] <= SzbMW, name='PbCsiz_'+str(h+1))

    ## Relação entre capacidade de potência e capacidade energética
    modl.addConstr(SzbMW*bSize == SzbMWh, name='BsH')
        
    # Objetivo    
    objFun = 0
    for h in rgHr: objFun = objFun + Ps[h]*gD['PLDh'][h] - a1*(PbD[h]+PbC[h])
    objFun = objFun - a2*SzbMWh
    modl.setObjective(objFun)
    
    # Resolve (1 = min / -1 = max)
    modl.setAttr("ModelSense", -1)
    modl.params.nonConvex = 0 
    modl.optimize()
    modl.update()
    
    #--------------------------------------------------------------------------
    # IMPRESSÃO
    
    ## Prompt warning 
    print('\n>> RESULTADOS <<\n')
    print('OBJETIVO:\n',modl.getAttr('ObjVal'),'\n')     

    # Obtenção do SoC em %
    for h in rgHr:
        try:
            SoC = (modl.getVarByName('En['+str(h)+']').getAttr('X') / SzbMWh.X) * 100
            SoCs.append(SoC)
        except ZeroDivisionError:
            SoCs.append(0)
        
       
    ## Referência ao dicionário de dados da tabela 'd'
    
    auxDic = {'PLDh':[],'Ptot':[],'Pitr':[],'En':[],'Pbes':[], 'SoC':[]} 
    for h in rgHr:
        auxDic['PLDh'].append(gD['PLDh'][h])
        auxDic['SoC'].append(SoCs[h])
        auxDic['Ptot'].append(modl.getVarByName('Ps['+str(h)+']').getAttr('X'))
        auxDic['Pitr'].append(gD['Solr'][h]+gD['Wind'][h])
        auxDic['En'].append(modl.getVarByName('En['+str(h)+']').getAttr('X'))
        auxDic['Pbes'].append(modl.getVarByName('PbD['+str(h)+']').getAttr('X')\
                             -modl.getVarByName('PbC['+str(h)+']').getAttr('X'))
    
    # Converter o dicionário para dataframe
    df = pd.DataFrame.from_dict(auxDic)
    df = df.round(2)
    df.index = gD['hora']

    # Exportar para csv
    df.to_csv('BESS Otimizado.csv')
    
    # Printar a tabela com os dados otimizados e o tamanho do BESS
    print(df)
    print(f'BESS Otimizado: {SzbMW.X} MW / {SzbMWh.X} MWh')

    return modl, df

# Função para plotar os gráficos
def plotDoisEixos(data1, data2, titleData1, titleData2, titleGraph):
    fig, ax1 = plt.subplots()
    
    data1.plot(ax=ax1, label=titleData1)
    ax1.set_ylabel(titleData1, fontdict={'weight': 'bold', 'size': 15})
    ax1.tick_params(axis='y', labelsize=15)
    plt.setp(ax1.get_yticklabels(), fontweight='bold')
    
    ax2 = ax1.twinx()
    data2.plot(ax=ax2, color='r', label=titleData2)
    ax2.set_ylabel(titleData2, fontdict={'weight': 'bold', 'size': 15})
    ax2.tick_params(axis='y', labelsize=15)
    plt.setp(ax2.get_yticklabels(), fontweight='bold')
    
    ax1.legend(loc='upper left', prop={'size': 15})
    ax2.legend(loc='upper right', prop={'size': 15})
    
    plt.title(titleGraph, fontsize=20, fontweight='bold')
    
    # Formatando os xticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M', tz='America/Sao_Paulo'))
    plt.setp(ax1.get_xticklabels(), fontsize=15, fontweight='bold')
    
    # Adicionando o xlabel
    ax1.set_xlabel('Data', fontdict={'weight': 'bold', 'size': 15})
    
    fig.tight_layout()
    plt.show()

# Rodar a função de otimização
model, df = testProb()

# Gerar os gráficos
plotDoisEixos(df[['Ptot', 'Pitr']], df['PLDh'], 'Potência (MW)', 'PLD (R$)', 'Potência Total + Potência EOL-PV (MW) x PLD (R$)')
plotDoisEixos(df[['En']], df['PLDh'], 'SoC (MWh)', 'PLD (R$)', 'SoC (MWh) x PLD (R$)')
plotDoisEixos(df[['SoC']], df['PLDh'], 'SoC (%)', 'PLD (R$)', 'SoC (%) x PLD (R$)')
plotDoisEixos(df[['Pbes']], df['PLDh'], 'Potência BESS (MW)', 'PLD (R$)', 'Potência BESS (MW) x PLD (R$)')

# Mostrar os gráficos em tela
plt.show()