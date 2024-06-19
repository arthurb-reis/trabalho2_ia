import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para ler o CSV e preparar os dados
def preparar_dados(csv_path):
    # Ler o CSV
    df = pd.read_csv(csv_path, header=None)
    
    # Definir a variável resposta (segunda coluna)
    y = df.iloc[:, 1]
    
    # Converter a variável resposta para categórica (caso não esteja)
    y = pd.factorize(y)[0]
    
    # Definir as variáveis explicativas (da terceira até a última coluna)
    X = df.iloc[:, 2:]
    
    return X, y

# Função para treinar e testar a rede neural
def treinar_e_testar_rede_neural(X, y):
    # Dividir os dados em treinamento (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Definir a arquitetura da rede neural
    model = Sequential()
    model.add(Dense(6400, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(3200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar o modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Treinar o modelo
    model.fit(X_train, y_train, epochs=8, batch_size=10, verbose=1)
    
    # Avaliar o modelo no conjunto de teste
    perda, acuracia = model.evaluate(X_test, y_test, verbose=0)
    
    return perda, acuracia

# Caminho para o arquivo CSV
csv_path = 'breast+cancer+wisconsin+diagnostic/wdbc.data'

# Preparar os dados
X, y = preparar_dados(csv_path)

# Treinar e testar a rede neural
perda, acuracia = treinar_e_testar_rede_neural(X, y)

print(f'A acurácia da rede neural é: {acuracia:.2f} e a perda é: {perda:.2f}')
