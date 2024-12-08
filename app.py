import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Função para carregar e treinar o modelo
def carregar_modelo():
    df = pd.read_csv('wdbc.data', header=None)
    df.columns = ['ID', 'Diagnosis', 
                  'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 
                  'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
                  'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 
                  'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']
    
    X = df.drop(columns=['ID', 'Diagnosis'])
    y = df['Diagnosis'].map({'M': 1, 'B': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(solver='liblinear')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model, scaler, X_test, y_test, df

model, scaler, X_test, y_test, df = carregar_modelo()

features = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 
            'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
            'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 
            'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

# Cálculos necessários antes da interface de Streamlit
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Interface do Streamlit
st.title("Classificador de Câncer de Mama")
st.write("Preencha os valores das 30 características para prever se o tumor é maligno ou benigno.")

# Organizando os botões de desempenho, matriz de confusão, distribuição de diagnóstico
col1, col2, col3 = st.columns(3)  # Organizando 3 colunas para os botões

with col1:
    show_performance = st.button("Desempenho", use_container_width=True)
with col2:
    show_cm = st.button("Matriz de Confusão", use_container_width=True)
with col3:
    show_bar = st.button("Diagnóstico", use_container_width=True)

# Exibir os gráficos logo após os botões
if show_performance:
    fig, ax = plt.subplots(figsize=(8, 5))
    metricas = ['Acurácia', 'Sensibilidade', 'Especificidade', 'AUC']
    valores = [accuracy, sensitivity, specificity, auc]

    sns.barplot(x=metricas, y=valores, palette='viridis', ax=ax)

    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black',
                    xytext=(0, 15), textcoords='offset points')

    ax.set_title('Desempenho do Modelo LR')
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel('Valor')

    plt.tight_layout()
    st.pyplot(fig)

if show_cm:
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], 
                yticklabels=['Benigno', 'Maligno'], ax=ax_cm)
    ax_cm.set_title('Matriz de Confusão')
    st.pyplot(fig_cm)

if show_bar:
    diagnosis_counts = df['Diagnosis'].value_counts()

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    bars = ax_bar.bar(diagnosis_counts.index, diagnosis_counts.values, color=['red', 'green'])

    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, yval + 2, 
                    f'{yval}', ha='center', va='bottom', fontsize=12, color='black')

    ax_bar.set_ylim(0, 400)
    ax_bar.set_title('Distribuição de Diagnóstico (M - Maligno, B - Benigno)')
    ax_bar.set_xlabel('Diagnóstico')
    ax_bar.set_ylabel('Frequência')
    st.pyplot(fig_bar)

# Filtros e dados
diagnostic_filter = st.selectbox("Escolha o diagnóstico", ['Todos', 'Benigno', 'Maligno'])

if diagnostic_filter != 'Todos':
    # Filtra o dataframe com base no diagnóstico escolhido
    filtered_df = df[df['Diagnosis'] == diagnostic_filter[0]]
else:
    filtered_df = df

# Permitir selecionar as colunas para exibir (com ID e Diagnosis sempre selecionados)
columns_to_show = st.multiselect(
    "Escolha as colunas para exibir", 
    options=df.columns.tolist(), 
    default=['ID', 'Diagnosis'],  # ID e Diagnosis sempre selecionados
    key="col_select"
)

# Garante que as colunas ID e Diagnosis estão sempre presentes na lista de colunas a serem exibidas
columns_to_show = ['ID', 'Diagnosis'] + [col for col in columns_to_show if col not in ['ID', 'Diagnosis']]

# Exibe os dados filtrados e com as colunas selecionadas
show_data = st.button("Buscar Dados do dataset", use_container_width=True)

if show_data:
    st.write(filtered_df[columns_to_show].head(15))

id_selecionada = st.text_input("Digite a ID para preencher automaticamente os dados:")

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
    inputs = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 
              0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 
              15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]

inputs_usuario = []
colunas = st.columns(4)  

for i, feature in enumerate(features):
    with colunas[i % 4]:
        value = st.number_input(f'{feature}', value=inputs[i], step=0.1, key=feature)
        inputs_usuario.append(value)

input_data = pd.DataFrame([inputs_usuario], columns=features)

input_data_scaled = scaler.transform(input_data)

if st.button("Prever"):
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.write("O tumor é **Maligno** (classe M).")
    else:
        st.write("O tumor é **Benigno** (classe B).")
