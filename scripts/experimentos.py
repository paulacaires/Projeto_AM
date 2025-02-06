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

# Arquivo com todas as funcoes e codigos referentes aos experimentos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def train_and_evaluate(estimador, param_grid, X_train, y_train, X_test, y_test, X_val, y_val):
    '''
    Hyperparameter tuning: Adjust hiperparameters
    - GridSearchCV
    '''
    grid_search = GridSearchCV(
        estimator = estimador,
        param_grid = param_grid,
        cv = 5,               
        scoring = 'accuracy'  
    )
    
    treino = grid_search.fit(X_train, y_train)
  
    print("Melhores hiperparâmetros:", grid_search.best_params_)

    melhor_modelo = grid_search.best_estimator_
    
    # Avaliação no conjunto de teste
    y_pred = melhor_modelo.predict(X_test)

    # Métricas extras que quero avaliar (além do report)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metricas_extras = { 
        "confusion_matrix": conf_matrix,
        # Avalia o desempenho no conjunto de validação
        "score_val": melhor_modelo.score(X_val, y_val),
        # Avalia o desempenho final no conjunto de teste
        "score_test": melhor_modelo.score(X_test, y_test),
        "report_dict": classification_report(y_test, y_pred, output_dict = True)
    }

    report = classification_report(y_test, y_pred)

    print("Desempenho no conjunto de validação:", metricas_extras['score_val'])
    print("Desempenho no conjunto de teste:", metricas_extras['score_test'])
    
    return melhor_modelo, metricas_extras, report


