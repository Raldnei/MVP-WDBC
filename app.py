import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classificador de Câncer de Mama", page_icon="🧬", layout="wide")

# Função para carregar os dados e treinar o modelo
def carregar_modelo():
    df = pd.read_csv('wdbc.data', header=None)  # Carrega o arquivo de dados
    df.columns = ['ID', 'Diagnosis',  # Dá nomes para as colunas
                  'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 
                  'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
                  'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 
                  'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']
    
    # Divide as colunas de dados em duas partes: as características (X) e o diagnóstico (y)
    X = df.drop(columns=['ID', 'Diagnosis'])  # Aqui estão as características que vamos usar para prever
    y = df['Diagnosis'].map({'M': 1, 'B': 0})  # Transformamos M (maligno) em 1 e B (benigno) em 0
    
    # Dividimos os dados em treino e teste (80% para treinar e 20% para testar)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Usamos um "scaler" para deixar os dados mais organizados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Organiza os dados de treino
    X_test = scaler.transform(X_test)  # Organiza os dados de teste

    # Criamos o modelo que vai aprender com os dados
    model = LogisticRegression(solver='liblinear')
    
    # Procuramos o melhor modelo, testando várias opções
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Testa diferentes valores de 'C'
        'penalty': ['l1', 'l2'],  # Testa diferentes tipos de penalidade
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)  # Treina o modelo com os dados de treino

    best_model = grid_search.best_estimator_  # Pega o melhor modelo
    return best_model, scaler, X_test, y_test, df  # Retorna o modelo, os dados de teste e o dataframe

# Chamamos a função para carregar o modelo e os dados
model, scaler, X_test, y_test, df = carregar_modelo()

# Lista de características que o modelo vai usar para fazer as previsões
features = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 
            'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
            'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 
            'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

# Aqui fazemos as previsões, calculamos a precisão e outras métricas importantes
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # Acurácia do modelo
cm = confusion_matrix(y_test, y_pred)  # Matriz de confusão

# Sensibilidade (como o modelo identifica tumores malignos)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])

# Especificidade (como o modelo identifica tumores benignos)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

# Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Agora, começamos a parte da interface gráfica com o Streamlit
st.title("Classificador de Câncer de Mama")  # Título do aplicativo
st.write("Preencha os valores das 30 características para prever se o tumor é maligno ou benigno.")  # Texto de introdução

# Criamos três botões para mostrar diferentes gráficos e informações
col1, col2, col3 = st.columns(3)  # Organiza os botões em três colunas

with col1:
    show_performance = st.button("Desempenho", use_container_width=True)  # Botão de desempenho
with col2:
    show_cm = st.button("Matriz de Confusão", use_container_width=True)  # Botão da matriz de confusão
with col3:
    show_bar = st.button("Diagnóstico", use_container_width=True)  # Botão para mostrar a distribuição de diagnóstico

# Se o usuário clicar em "Desempenho", mostramos um gráfico com as métricas de desempenho
if show_performance:
    fig, ax = plt.subplots(figsize=(3, 4))  # Reduzido para metade do tamanho anterior
    metricas = ['Acurácia', 'Sensibilidade', 'Especificidade', 'AUC']
    valores = [accuracy, sensitivity, specificity, auc]

    sns.barplot(x=metricas, y=valores, palette='viridis', ax=ax)  # Gráfico de barras

    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black',
                    xytext=(0, 15), textcoords='offset points')

    ax.set_title('Desempenho do Modelo LR')  # Título do gráfico
    ax.set_ylim(0.9, 1.01)  # Limita os valores do gráfico de 90% a 100%
    ax.set_ylabel('Valor')  # Rótulo do eixo Y

    plt.tight_layout()  # Ajusta o layout do gráfico
    st.pyplot(fig, use_container_width=False)  # Exibe o gráfico e não usa 100% da largura

# Se o usuário clicar em "Matriz de Confusão", mostramos um gráfico com a matriz de confusão
if show_cm:
    fig_cm, ax_cm = plt.subplots(figsize=(3, 2))  # Reduzido para metade do tamanho anterior
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], 
                yticklabels=['Benigno', 'Maligno'], ax=ax_cm)  # Gráfico de calor

    ax_cm.set_title('Matriz de Confusão')  # Título do gráfico
    st.pyplot(fig_cm, use_container_width=False)  # Exibe o gráfico

# Se o usuário clicar em "Diagnóstico", mostramos a distribuição dos diagnósticos
if show_bar:
    diagnosis_counts = df['Diagnosis'].value_counts()

    fig_bar, ax_bar = plt.subplots(figsize=(3, 2))  # Reduzido para metade do tamanho anterior
    bars = ax_bar.bar(diagnosis_counts.index, diagnosis_counts.values, color=['red', 'green'])

    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, yval + 2, 
                    f'{yval}', ha='center', va='bottom', fontsize=10, color='black')

    ax_bar.set_ylim(0, 400)  # Limita o eixo Y
    ax_bar.set_title('Distribuição de Diagnóstico (M - Maligno, B - Benigno)')  # Título
    ax_bar.set_xlabel('Diagnóstico')  # Rótulo do eixo X
    ax_bar.set_ylabel('Frequência')  # Rótulo do eixo Y
    st.pyplot(fig_bar, use_container_width=False)  # Exibe o gráfico

# Filtra os dados com base no diagnóstico escolhido pelo usuário
diagnostic_filter = st.selectbox("Escolha o diagnóstico", ['Todos', 'Benigno', 'Maligno'])

# Aplica o filtro nos dados
if diagnostic_filter == 'Benigno':
    filtered_df = df[df['Diagnosis'] == 'B']
elif diagnostic_filter == 'Maligno':
    filtered_df = df[df['Diagnosis'] == 'M']
else:
    filtered_df = df  # Se "Todos" for selecionado, não aplica filtro

# Exibe os dados filtrados e selecionados
columns_to_show = st.multiselect("Escolha as colunas para exibir", options=df.columns.tolist(), default=['ID', 'Diagnosis'])
columns_to_show = ['ID', 'Diagnosis'] + [col for col in columns_to_show if col not in ['ID', 'Diagnosis']]

# Se o usuário quiser ver os dados, mostra as primeiras 15 linhas
show_data = st.button("Buscar Dados do dataset", use_container_width=True)

if show_data:
    st.write(filtered_df[columns_to_show].head(10))  # Mostra as 15 primeiras linhas

# Permite que o usuário insira uma ID para preencher automaticamente os dados
id_selecionada = st.text_input("Digite a ID para preencher automaticamente os dados:")

# Se a ID for válida, mostramos os dados dessa ID
if id_selecionada:
    try:
        dados_id = df[df['ID'] == int(id_selecionada)].iloc[0]
        inputs = [dados_id[feature] for feature in features]
        st.write(f"Dados preenchidos para a ID {id_selecionada}:")
        st.write(dados_id)
    except IndexError:
        st.write("ID INVÁLIDO.")
    except ValueError:
        st.write("Por favor, insira uma ID válida.")
else:
    # Se a ID não for fornecida, usamos valores padrão para os dados
    inputs = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 
              0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 
              15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]

# Permite ao usuário inserir valores para as características
inputs_usuario = []
colunas = st.columns(4)  

for i, feature in enumerate(features):
    with colunas[i % 4]:
        value = st.number_input(f'{feature}', value=inputs[i], step=0.1, key=feature)
        inputs_usuario.append(value)

input_data = pd.DataFrame([inputs_usuario], columns=features)

# Fazemos as previsões com o modelo e mostramos o resultado
input_data_scaled = scaler.transform(input_data)

if st.button("Prever"):
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.write("O tumor é **Maligno** (classe M).")  # Se for 1, é maligno
    else:
        st.write("O tumor é **Benigno** (classe B).")  # Se for 0, é benigno
