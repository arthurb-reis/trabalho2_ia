import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Função para ler o CSV e preparar os dados
def preparar_dados(csv_path):
    # Ler o CSV
    df = pd.read_csv(csv_path)
    
    # Definir a variável resposta (segunda coluna)
    y = df.iloc[:, 1]
    
    # Definir as variáveis explicativas (da terceira até a última coluna)
    X = df.iloc[:, 2:]
    
    return X, y

# Função para treinar e testar o modelo Random Forest
def treinar_e_testar_modelo(X, y):
    # Dividir os dados em treinamento (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criar o modelo Random Forest
    modelo = RandomForestClassifier(random_state=42)
    
    # Treinar o modelo
    modelo.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = modelo.predict(X_test)
    
    # Calcular a acurácia
    acurácia = accuracy_score(y_test, y_pred)
    
    return acurácia

# Caminho para o arquivo CSV
csv_path = 'breast+cancer+wisconsin+diagnostic/wdbc.data'

# Preparar os dados
X, y = preparar_dados(csv_path)

# Treinar e testar o modelo
acurácia = treinar_e_testar_modelo(X, y)

print(f'A acurácia do modelo é: {acurácia:.2f}')
