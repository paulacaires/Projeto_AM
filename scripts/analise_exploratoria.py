# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Paula Caires Silva
# RA: 792230
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria
import numpy as np
import pandas as pd 

# Para a visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns

def analise_descritiva(df, coluna):
    """
    Função que analisa os atributos numéricos sob a ótica da estatísitica descritiva

    Args:
        df: DataFrame com os dados
        coluna: Nome da coluna a ser analisada
    """
    tam_df = df.shape[0]

    qtd_nulos = df[coluna].isnull().sum()
    print(f"A coluna '{coluna}' possui {qtd_nulos} valores nulos ({qtd_nulos/tam_df*100:.2f}%).\n")

    qtd_negativos = (df[coluna] < 0).sum()
    print(f"A coluna '{coluna}' possui {qtd_negativos} valores negativos ({qtd_negativos/tam_df*100:.2f}%).\n")

    qtd_zeros = (df[coluna] == 0).sum()
    print(f"A coluna '{coluna}' possui {qtd_zeros} valores zero ({qtd_zeros/tam_df*100:.2f}%).\n")

    media = df[coluna].mean()
    print(f"Média da coluna '{coluna}': {media:.2f}\n")

    mediana = df[coluna].median()
    print(f"Mediana da coluna '{coluna}': {mediana:.2f}\n")
          
    moda = df[coluna].mode()[0]  # Retorna o primeiro valor da moda
    print(f"Moda da coluna '{coluna}': {moda:.2f}\n")
          
    desvio_padrao = df[coluna].std()
    print(f"Desvio padrão da coluna '{coluna}': {desvio_padrao:.2f}\n")
    
    variancia = df[coluna].var()
    print(f"Variância da coluna '{coluna}': {variancia:.2f}\n")
    
    quartis = df[coluna].quantile([0.25, 0.5, 0.75])
    print(f"Quartis da coluna '{coluna}': \n{quartis}\n")

def visualizacao(df, coluna):
    """
    Função que utiliza técnicas de visualização da distribuição dos dados numéricos

    Args:
        df: DataFrame com os dados
        coluna: Nome da coluna a ser analisada
    """

    # Histograma
    plt.figure(figsize = (8, 6))
    sns.histplot(data = df, x = coluna)
    plt.title(f'Distribuição da coluna {coluna}')
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y = df[coluna])
    plt.title(f'Boxplot da coluna {coluna}')
    plt.show()

def analise_descritiva_categorica(df, coluna):
    """
    Função que utiliza técnicas de estatísitica descritiva para dados categóricos

    Args:
        df: DataFrame com os dados
        coluna: Nome da coluna a ser analisada
    """
    quant_categorias = df[coluna].value_counts().size
    print(f"Existem {quant_categorias} tipos de categorias diferentes para a coluna {coluna}.\n")

    print(f"Categorias da coluna {coluna} e a quantidade: \n")
    print(df[coluna].value_counts())

    tam_df = df.shape[0]
    qtd_nulos = df[coluna].isnull().sum()
    print(f"\nA coluna '{coluna}' possui {qtd_nulos} valores nulos ({qtd_nulos/tam_df*100:.2f}%).\n")

def remove_objetos_incompletos(df):
    '''
    Remove objetos com sexo indefinido ('I') e que têm muitos atributos nulos.
    '''
    df_invalidos = df[df['SEXO'] == 'I']
    print(f'São {df_invalidos.shape[0]} objetos com o sexo inválido.\n')

    # Calcular o número de atributos nulos para cada linha
    df_invalidos = df_invalidos.copy() # Corrigir o warning
    df_invalidos.loc[:, 'qtd_nulos'] = df_invalidos.isnull().sum(axis=1)
    display(df_invalidos[['Id', 'qtd_nulos']])

    limite_nulls = int(df.shape[1] * 0.7)
    print(f'Existem {df.shape[1]} atributos no data frame. O limite é {limite_nulls} colunas nulas.')

    # Remover do dataframe apenas objetos com mais do que o limite de nulos.
    indices_remover = df_invalidos[df_invalidos['qtd_nulos'] >= limite_nulls].index
    print(f'Removendo {len(indices_remover)} objetos com mais ou igual do que {limite_nulls} colunas nulas.\n')

    # Remover os índices do dataframe original
    df = df.drop(indices_remover)
    
    return df

def normalizar_sexo(sexo):    
    if (isinstance(sexo, str)):
        sexo = sexo.lower()
    
        if sexo.startswith('f'):
            return 'F'
        
        if sexo.startswith('m'):
            return 'M'
    return 'I' # NaN

def normalizar_convenio(convenio):
    '''
    Função que normaliza o nome das funções do convênio
    '''
    
    mapeamento = {
        # sul america
        'sa': 'sul america',
        'sula': 'sul america',
        'sulam': 'sul america',
        's. america': 'sul america',
        'sul america': 'sul america',
        'sulamerica': 'sul america',
        'américa': 'sul america',
        'sul am': 'sul america',
        'america s.': 'sul america',
        's.am': 'sul america',
        's.america': 'sul america',
        'america': 'sul america',
        's.a.': 'sul america',
        's.a': 'sul america',
        'américa s.': 'sul america',
        'sul américa': 'sul america',
        'américa saude': 'sul america', 
        'sul amer.': 'sul america',
        's a': 'sul america',
        'america saude': 'sul america',
        'sulamérica': 'sul america',
        's.amer.': 'sul america',  
        's.amer': 'sul america',
        'amériaca s.': 'sul america',
        'sul-amer': 'sul america',
        'sulamer': 'sul america',
        'america  s.': 'sul america',
                
        # norclinicas
        'norcl': 'norclinicas',
        'norclin': 'norclinicas',
        'norclin.': 'norclinicas',
        'norclinica': 'norclinicas',
        'norc': 'norclinicas',
        'norclínicas': 'norclinicas',
        'norclínica': 'norclinicas',
        'norcli': 'norclinicas',
        'norc.': 'norclinicas',
        'norclín': 'norclinicas',
        'norcínicas': 'norclinicas',
        'norclinic': 'norclinicas',
        'norclinc': 'norclinicas',
        'norclínia': 'norclinicas',
        'norcl.': 'norclinicas',
        'norlcínicas': 'norclinicas',

        # santa helena
        'sta helena': 'santa helena',
        'st.helena': 'santa helena',
        'sant h': 'santa helena',
        's.helena': 'santa helena',
        'sta. helena': 'santa helena',
        'santa helen': 'santa helena',
        'sta helen': 'santa helena',
        'st helena': 'santa helena',
        'sta hel.': 'santa helena',
        'sta he': 'santa helena',
        's.helen': 'santa helena',
        'sant helen': 'santa helena',
        'shelena': 'santa helena',

        # santa clara
        'sta. clara': 'santa clara',
        'sta clara': 'santa clara',               

        # unimed recife
        'ur': 'unimed recife',
        'u.recife': 'unimed recife',
        'unimed foratleza': 'unimed recife',
        'unimed rec': 'unimed recife',
        'unimed r': 'unimed recife',
        'un.recife': 'unimed recife',
        'u. recife': 'unimed recife',
        'urecife': 'unimed recife',
        'unimed r.': 'unimed recife',
        'u.guararapes': 'unimed recife', # Guararapes é um bairro em Fortaleza
        'unim.recife': 'unimed recife',
        'unimed guararapes': 'unimed recife',
        'unimed re': 'unimed recife',
        'ur/maceió': 'unimed recife',
        'u. guara': 'unimed recife',
        'u.guarap': 'unimed recife',
        'u.guararape': 'unimed recife',
        'unimed gua': 'unimed recife',
        'u.guararap.': 'unimed recife',
                                        
        # saude recife
        'saúde recife': 'saude recife',
        's. recife': 'saude recife',
        'sauderecife': 'saude recife',
        's.recife': 'saude recife',
        's recife': 'saude recife',
        'saúde rec.': 'saude recife',
        'saúde rec': 'saude recife',       
        
        # aeronautica
        'aero': 'aeronautica',
        'aeronáutica': 'aeronautica',
        'aeron.': 'aeronautica',
        'aeron': 'aeronautica',

        # mediservice
        'medserv': 'mediservice',
        'mediserv.': 'mediservice',
        'mediserv': 'mediservice',
        'mediser': 'mediservice',
        'medial s.': 'mediservice',
        'mediservece': 'mediservice',
        'medise': 'mediservice',
        'medser': 'mediservice',
        'medese': 'mediservice',
        'medis': 'mediservice',
        'medserv.': 'mediservice',
        'medial  s.': 'mediservice',
        'media s.': 'mediservice',
        'medesv': 'mediservice',  
        'mediservic': 'mediservice',
        'mersev': 'mediservice',
        'mediservi': 'mediservice',   
        'medial s': 'mediservice',
        
        # real saude
        'real': 'real saude',
        'real s.': 'real saude',
        'real saúde': 'real saude',
        'real s': 'real saude',
        's. real': 'real saude',
        'reals.': 'real saude',
        'saude re': 'real saude',
        'real. s.': 'real saude',        

        # oab
        'oab saúde': 'oab',

        # blue life
        'blueli': 'blue life',
        'bl': 'blue life',
        'bluelife': 'blue life',
        'blul': 'blue life',

        # petrobras
        'petrob': 'petrobras',
        'petrob.': 'petrobras',
        'petrobrás': 'petrobras',
        'petrof': 'petrobras',
        'petrox': 'petrobras',
        'petroflex': 'petrobras',
        'petro': 'petrobras',
        
        # fechesf
        'faches': 'fachesf',
        'facjesf': 'fachesf',
        'fachaesf': 'fachesf',
        'funsef': 'fachesf',
        

        # particular
        'particl': 'particular',
        'p': 'particular',
        'partic': 'particular',
        'particulra': 'particular',

        # bradesco
        'saude bradesco': 'bradesco',
        's bradesco': 'bradesco',
        'bs': 'bradesco',
        's.bradesco': 'bradesco',
        'saude bardesco': 'bradesco',

        # correio
        'correios': 'correio',
        'coreios': 'correio',       
    }

    
    return mapeamento.get(convenio, convenio)  # Retorna o valor mapeado ou o valor original 

def idade_meses_ate_2_anos(idade_anos):
	"""
	Converte uma idade em anos (menor ou igual a 2) para meses.

	Args:
		idade_anos: Idade em anos (float).

	Returns:
		Idade em meses (int).
	"""
    
	if (int(idade_anos) == 2):
		return 24 # A criança tem 24 meses
	else:
		return round(idade_anos * 12)

def tabela_altura(idade, sexo):	
	"""
	Retorna a altura média estimada para uma pessoa, dada a idade e o sexo.

	Args:
		idade: Idade em anos (float).
		sexo: 'M' para masculino, 'F' para feminino (str).

	Returns:
		Altura média estimada em centímetros (float).
	"""
    
	altura_adultos = {'M': 170, 'F': 160}
	
	if (int(idade) > 18):
		return altura_adultos[sexo]
	
	# Altura média, em centímetros
	altura = {
		'M': {
			'até 2 anos': [49.9, 54.7, 58.4, 61.4, 63.9, 65.9, 67.6, 69.2, 70.6, 72, 73.3, 74.5, 75.7, 76.9, 78, 79.1, 80.2, 81.2, 82.3, 83.2, 84.2, 85.1, 86, 86.9, 87.8],
			'acima de 2 anos': [96.2, 103.4, 108.7, 117.5, 124.1, 130, 135.5, 140.3, 144.2, 149.6, 155, 162.7, 167.8, 171.6, 173.7, 174.5]
		},
		'F': {
			'até 2 anos': [49.1, 53.7, 57.1, 59.8, 62.1, 64, 65.7, 67.3, 68.7, 70.1, 71.5, 72.8, 74, 75.2, 76.4, 77.5, 78.6, 79.7, 80.7, 81.7, 82.7, 83.7, 84.6, 85.5, 86.4],
			'acima de 2 anos': [95.7, 103.2, 109.1, 115.9, 122.3, 128, 132.9, 138.6, 144.7, 151.9, 157.1, 159.6, 161.1, 162.2, 162.5, 162.5]
		}
	}
	
	idade_anos = int(idade)
	if (idade_anos <= 2):
		idade = idade_meses_ate_2_anos(idade)
		return (altura[sexo]['até 2 anos'][idade - 1])
	
	# Para pessoas acima de 2 anos	
	return altura[sexo]['acima de 2 anos'][idade_anos - 3]



def tabela_peso(idade, sexo):
	"""
	Retorna o peso médio estimado para uma pessoa, dada a idade e o sexo.

	Args:
		idade: Idade em anos (float).
		sexo: 'M' para masculino, 'F' para feminino (str).

	Returns:
		Peso médio estimado em quilos (float).
	"""
    
	peso_adultos = {'M': 70, 'F': 60}
	
	if (int(idade) > 18):
		return peso_adultos[sexo]
	
	pesos = {
		'M': {
			'até 2 anos': [3.3, 4.5, 5.6, 6.4, 7.0, 7.5, 7.9, 8.3, 8.6, 8.9, 9.2, 9.4, 9.6, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.8, 12, 12.2],
			'acima de 2 anos': [14.61, 16.51, 18,37, 21.91, 24.54, 27.26, 29.94, 32.61, 35.20, 38.28, 42.18, 48,81, 54.48, 58.83, 61.78, 63.05]
		},
		'F': {
			'até 2 anos': [3.2, 4.2, 5.1, 5.8, 6.4, 6.9, 7.3, 7.6, 7.9, 8.2, 8.7, 8.9, 9.2, 9.4, 9.6, 9.8, 10, 10.2, 10.4, 10.6, 10.9, 11.1, 11.3, 11.5],
			'acima de 2 anos': [14.42, 16.42, 18.37, 21.09, 23.68, 26.35, 28.94, 31.89, 35.74, 39.74, 44.95, 49.17, 51.48, 53.07, 54.02, 54.39]
		}
	}

	idade_anos = int(idade)
	if (idade_anos <= 2):
		idade = idade_meses_ate_2_anos(idade)
		return (pesos[sexo]['até 2 anos'][idade - 1])
	
	# Para pessoas acima de 2 anos	
	return pesos[sexo]['acima de 2 anos'][idade_anos - 3]

def define_medida(df, medida):
    '''
    Função que completa a coluna de pesos que são iguais a 0 ou null.

    Args:
        df: O dataframe
        medida: 'Peso' ou 'Altura'
    '''
    # Colunas com a medida igual a 0 ou null
    mask = (df[medida].isnull()) | (df[medida] == 0)

    if (medida.lower() == 'peso'):
        df.loc[mask, medida] = df.loc[mask].apply(
            lambda row: tabela_peso(row['IDADE'], row['SEXO']), axis=1
        )            
                 
    if (medida.lower() == 'altura'):
        df.loc[mask, medida] = df.loc[mask].apply(
            lambda row: tabela_altura(row['IDADE'], row['SEXO']), axis=1
        )            
        
    return df  


def define_valores_ausentes(df_data, df_classe, coluna, coluna_classificacao, coluna_id):
    '''
    Define valores para atributos com valores ausentes com
    base na média dos valores com a mesma classificação.

    Args:
        df_data: Dataframe com os dados
        df_classe: Dataframe com as classes
        coluna: Nome da coluna sobre a qual quero definir os valores ausentes (str)
        coluna_classificacao: Nome da coluna de df_classe que contém a classificação (str)
        coluna_id: Nome da em comum entre os dois datasets (str)
    '''
    
    # Classificações nulas como 'sem info'
    df_classe[coluna_classificacao] = df_classe[coluna_classificacao].fillna('sem info')
    
    df_merged = pd.merge(df_data, df_classe, on = coluna_id)
    
    # Calcular a média para cada valor da coluna de classificação
    group_means = df_merged.groupby(coluna_classificacao)[coluna].mean()
    
    df_merged[coluna] = df_merged[coluna].fillna(
        df_merged[coluna_classificacao].map(group_means)
    )

    return df_merged[df_data.columns]
    
def preencher_pressao_arterial_por_pulso(df, tipo):
    # Para cada valor único de pulso, calcular a moda da pressão
    moda_pressao = df.groupby('PULSOS')[tipo].apply(lambda x: x.mode()[0] if not x.mode().empty else None)

    # Preencher as pressões nulas com a moda correspondente
    for pulso in moda_pressao.index:
        df.loc[(df['PULSOS'] == pulso) & (df[tipo].isnull()), tipo] = moda_pressao[pulso]

    return df    
    