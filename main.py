import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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

# Função para treinar e testar o modelo
def treinar_e_testar_modelo(X, y, modelo):
    # Dividir os dados em treinamento (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinar o modelo
    modelo.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = modelo.predict(X_test)
    
    # Calcular a acurácia, precisão e recall
    acurácia = accuracy_score(y_test, y_pred)
    precisão = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return acurácia, precisão, recall, tn, fp, fn, tp

# Caminho para o arquivo CSV
csv_path = 'breast+cancer+wisconsin+diagnostic/wdbc.data'

# Preparar os dados
X, y = preparar_dados(csv_path)

# Definir os modelos a serem testados
modelos = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=100, random_state=42)
}

# Treinar e testar cada modelo
for nome_modelo, modelo in modelos.items():
    acurácia, precisão, recall, tn, fp, fn, tp = treinar_e_testar_modelo(X, y, modelo)
    print(f'\nModelo: {nome_modelo}')
    print(f'Acurácia: {acurácia:.2f}')
    print(f'Precisão: {precisão:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Verdadeiros Negativos: {tn}')
    print(f'Falsos Positivos: {fp}')
    print(f'Falsos Negativos: {fn}')
    print(f'Verdadeiros Positivos: {tp}')
