# Tech Challenge - Fase 3: Machine Learning Engineering

## FIAP PosTech - Machine Learning Engineering

Pipeline completo de Machine Learning para previsao de atrasos de voos nos EUA utilizando o dataset de 2015.

## Arquitetura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dataset       │────▶│   EDA &         │────▶│   Feature       │
│   flights.csv   │     │   Limpeza       │     │   Engineering   │
│   (~5.8M voos)  │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
         ┌───────────────────────────────────────────────┴───────┐
         │                                                       │
         ▼                                                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Classificacao  │     │    Regressao    │     │  Clusterizacao  │
│  - Logistic Reg │     │  - SGDRegressor │     │  - K-Means      │
│  - Random Forest│     │  - Ridge/Lasso  │     │  - PCA          │
│  - Grad Boost   │     │  - Grad Boost   │     │  - Silhouette   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Avaliacao & Metricas  │
                    │   - F1, Precision, ROC  │
                    │   - R2, RMSE, MAE, MAPE │
                    │   - Learning Curves     │
                    └─────────────────────────┘
```

## Estrutura do Projeto

```
fase-3/
├── tech_challenge_fase3.ipynb   # Notebook principal com todo o pipeline
├── dashboard.py                 # Dashboard interativo (Streamlit)
├── requirements.txt             # Dependencias do projeto
├── README.md                    # Este arquivo
├── LICENSE                      # Licenca MIT
└── .gitignore                   # Arquivos ignorados pelo git
```

## Quick Start

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Dados

Baixe os dados do link fornecido no enunciado:
- `flights.csv` - Dataset principal (~5.8M voos)
- `airlines.csv` - Companhias aereas
- `airports.csv` - Aeroportos

### 3. Executar o Notebook

```bash
jupyter notebook tech_challenge_fase3.ipynb
```

### 4. Executar o Dashboard

```bash
streamlit run dashboard.py
```

## Modelos Implementados

### Classificacao (Prever se voo atrasa >= 15 min)

| Modelo | Tecnica |
|--------|---------|
| Logistic Regression | Regularizacao L1/L2/ElasticNet |
| Random Forest Classifier | Ensemble (Bagging) |
| Gradient Boosting Classifier | Ensemble (Boosting) |

### Regressao (Prever minutos de atraso)

| Modelo | Tecnica |
|--------|---------|
| SGDRegressor | Gradiente Descendente Estocastico |
| Ridge/Lasso | Regularizacao L1/L2 |
| Gradient Boosting Regressor | Ensemble |

### Clusterizacao (Agrupar aeroportos)

| Modelo | Tecnica |
|--------|---------|
| K-Means | Metodo do Cotovelo + Silhouette |
| PCA | Reducao de Dimensionalidade |

## Principais Descobertas

1. **Efeito Cascata**: Voos no inicio do dia tem menos atraso; atrasos acumulam ao longo do dia
2. **Sazonalidade**: Verao e inverno concentram maiores atrasos (clima e demanda)
3. **Hubs**: Aeroportos ATL, ORD, DFW concentram maiores volumes de atraso
4. **Clustering**: 4 perfis distintos de aeroportos identificados via K-Means

