# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modelo import carregar_modelo, calcular_metricas

st.set_page_config(page_title="Classificador de Câncer de Mama", page_icon="🧬", layout="wide")

# Carrega o modelo e os dados
model, scaler, X_test, y_test, df, X_train, y_train, train_indices, test_indices = carregar_modelo()

# Calcula as métricas
accuracy, sensitivity, specificity, auc, cm = calcular_metricas(model, X_test, y_test)

# Lista de características que o modelo vai usar para fazer as previsões
features = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 
            'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
            'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 
            'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

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
    fig, ax = plt.subplots(figsize=(8, 3))  # Tamanho do gráfico
    metricas = ['Acurácia', 'Sensibilidade', 'Especificidade', 'AUC']
    valores = [accuracy, sensitivity, specificity, auc]

    sns.barplot(x=metricas, y=valores, palette='viridis', ax=ax)  # Gráfico de barras

    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=8, color='black',  # Fonte menor
                    xytext=(0, 10), textcoords='offset points')  # Ajuste de distância do texto

    ax.set_title('Desempenho do Modelo LR')  # Título do gráfico
    ax.set_ylim(0.94, 1.01)  # Limita os valores do gráfico de 90% a 100%
    ax.set_ylabel('Valor')  # Rótulo do eixo Y

    plt.tight_layout()  # Ajusta o layout do gráfico
    st.pyplot(fig, use_container_width=False)  # Exibe o gráfico

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

# Seleção do diagnóstico
diagnostic_filter = st.selectbox("Escolha o diagnóstico", ['Todos', 'Benigno', 'Maligno'])

# Seleção das colunas a serem exibidas
columns_to_show = st.multiselect("Escolha as colunas para exibir", options=df.columns.tolist(), default=['ID', 'Diagnosis'])
columns_to_show = ['ID', 'Diagnosis'] + [col for col in columns_to_show if col not in ['ID', 'Diagnosis']]

# Botão para buscar dados do dataset
show_data = st.button("Buscar Dados do dataset", use_container_width=True)

if show_data:
    # Filtra os dados para incluir apenas os do conjunto de testes
    test_data = df.iloc[test_indices]  # Filtra os dados para incluir apenas os do conjunto de testes
    
    # Aplica o filtro nos dados de teste com base no diagnóstico escolhido
    if diagnostic_filter == 'Benigno':
        filtered_test_data = test_data[test_data['Diagnosis'] == 'B']
    elif diagnostic_filter == 'Maligno':
        filtered_test_data = test_data[test_data['Diagnosis'] == 'M']
    else:
        filtered_test_data = test_data  # Se "Todos" for selecionado, não aplica filtro

    # Exibe os dados de teste filtrados
    st.write(filtered_test_data[columns_to_show].head(10))  # Mostra as 10 primeiras linhas dos dados de teste

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