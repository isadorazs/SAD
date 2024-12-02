import numpy as np

# Normalização
def normalize_matrix(matrix):
    column_sums = matrix.sum(axis=0)
    return matrix / column_sums

# Cálculo de pesos
def calculate_weights(normalized_matrix):
    return normalized_matrix.mean(axis=1)

# Consistência
def calculate_consistency(matrix, weights, RI=1.12):
    n = matrix.shape[0]
    lambda_max = (matrix @ weights).sum() / weights.sum()
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI
    return CI, CR

# Processamento de alternativas
def process_alternatives(matrix, RI=0.58):
    normalized_matrix = normalize_matrix(matrix)
    weights = calculate_weights(normalized_matrix)
    CI, CR = calculate_consistency(matrix, weights, RI=RI)
    return weights, CR

# Disciplinas e Matrizes
disciplines = ["Estrutura de Dados", "Arquitetura de Computadores", "Engenharia de Software"]

criteria_matrix = np.array([
    [1, 1/2, 3],
    [2, 1, 4],
    [1/3, 1/4, 1]
])

difficulty_matrix = np.array([
    [1, 3, 4],
    [1/3, 1, 2],
    [1/4, 1/2, 1]
])

importance_matrix = np.array([
    [1, 1/3, 2],
    [3, 1, 4],
    [0.5, 1/4, 1]
])

interest_matrix = np.array([
    [1, 3, 0.5],
    [1/3, 1, 1/4],
    [2, 4, 1]
])

# Cálculos
criteria_normalized = normalize_matrix(criteria_matrix)
criteria_weights = calculate_weights(criteria_normalized)

difficulty_weights, _ = process_alternatives(difficulty_matrix)
importance_weights, _ = process_alternatives(importance_matrix)
interest_weights, _ = process_alternatives(interest_matrix)

# Matriz de decisão e pontuação
decision_matrix = np.array([
    difficulty_weights,
    importance_weights,
    interest_weights
])

final_scores = criteria_weights @ decision_matrix

# Melhor alternativa
best_choice_index = np.argmax(final_scores)
best_choice = disciplines[best_choice_index]

# Resultado
print(f"A disciplina recomendada para cursar é: {best_choice}")
