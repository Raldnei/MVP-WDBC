# Sistema Web de Detecção de Câncer de Mama

#### Este é um sistema web desenvolvido utilizando Streamlit para detectar câncer de mama com base no conjunto de dados Breast Cancer Wisconsin (Diagnostic). O sistema utiliza técnicas de aprendizado de máquina para classificar tumores como benignos ou malignos com alta precisão.

# Funcionalidades

### Análise de Dados: 
- Permite o upload de dados sobre tumores de mama e classifica-os em benignos ou malignos.
### Modelos de Classificação:
- O sistema utiliza diversos modelos de aprendizado de máquina, como Regressão Logística, Redes Neurais Artificiais, Máquinas de Vetores de Suporte e Florestas Aleatórias.
### Interface Simples: 
- A interface do usuário é fácil de usar, permitindo uma interação fluida.
### Resultados Imediatos:
- Visualize os resultados em tempo real após inserir os dados.-

# Tecnologias Utilizadas
### Backend/Frontend:
```bash
Streamlit (framework para criar aplicativos web interativos)
Python (para os modelos de aprendizado de máquina)
Scikit-learn (para treinamento dos modelos)
Pandas (para manipulação de dados)
Seaborn (para visualizações gráficas)
Matplotlib (para visualizações gráficas)
Numpy (para manipulação de arrays numéricos)
```

# Requisitos
Antes de rodar o projeto, garanta que as seguintes dependências estão instaladas:

```Python (recomendado versão 3.7 ou superior)
pip Streamlit 
````
# Instalação
1. Clonando o Repositório
Clone este repositório para a sua máquina local:

````bash

git clone https://github.com/Raldnei/MVP-WDBC.git
cd MVP-WDBC
````

2. Instalando as Dependências
Instale as dependências do projeto com o pip:

````bash

pip install -r requirements.txt
````
No arquivo requirements.txt, você deve incluir as dependências, por exemplo:
````
streamlit
scikit-learn
pandas
seaborn
matplotlib
numpy
````
3. Rodando o Sistema
Para rodar o sistema, execute o seguinte comando dentro da pasta do projeto:
````
streamlit run app.py
````
Isso abrirá o sistema no navegador, geralmente em http://localhost:8501.

# Modelos de Classificação
O sistema utiliza os seguintes modelos para classificar os tumores:

Regressão Logística (LR)
Redes Neurais Artificiais (ANN)
Máquinas de Vetores de Suporte (SVM)
Floresta Aleatória (RF)
Esses modelos foram treinados com o conjunto de dados Breast Cancer Wisconsin (Diagnostic).

# Exemplo de Uso
Entrada de Dados Manual:

Preencha os campos com os valores dos atributos (como raio, textura, etc.).
Clique em "Prever" para obter o diagnóstico do tumor.
# Upload de Arquivo CSV:

Selecione um arquivo CSV com os dados dos tumores.
O sistema processará o arquivo e exibirá o diagnóstico de cada tumor.


## Autores

| <img src="https://avatars.githubusercontent.com/Raldnei" width="50" height="50" style="border-radius: 50%;"> | [**Raldnei Miguel**](https://github.com/Raldnei)<br><small>Desenvolvedor</small> |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------|
| <img src="https://avatars.githubusercontent.com/Messias-Acacy" width="50" height="50" style="border-radius: 50%;"> | [**Messias Accacy**](https://github.com/Messias-Acacy)<br><small>Desenvolvedor</small> |

| <img src="https://media.licdn.com/dms/image/v2/C4D03AQFvGM-MhNmbLA/profile-displayphoto-shrink_100_100/profile-displayphoto-shrink_100_100/0/1516535088936?e=2147483647&v=beta&t=aeaVesMKGJCu5P5XFMYiEqsgNAK0jT7juJR2ESS9Png" width="50" height="50" style="border-radius: 50%;"> | [**Luciano Cabral**](https://br.linkedin.com/in/lucianocabral)<br><small>Orientador do projeto</small> |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------|

