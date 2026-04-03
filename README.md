Neste projeto peguei um dataset para fazer ánalises e previsões com machine learning, no jupyter testei 2 modelos para prever números com base em números, o LienarRegression e o RandomForestRegressor, onde pude notar que
esse segundo se saiu melhor, tudo está registrado e explicado no arquivo ipynb.
Em seguida criei um main.py para uma FastApi, onde defini os valores médios com base no dataSet e usei o modelo para prever o preço da casa, integrei com um frontend simples onde o usuário fornece apenas o número de quartos 
da casa e o modelo prevê com base nos dados médios.
Para informaões como taxa de erros do modelo recomendo consultar o arquivo ipynb.
