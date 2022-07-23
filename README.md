# mlops-rio-de-janeiro
Projeto da matéria de MLOps da UFRN que tem como objetivo desenvolver uma pipeline reproduzível de Machine Learning.
Vídeos sobre o projeto:
[Week 9](https://www.youtube.com/watch?v=GRWEFZwzmSc&ab_channel=VictorVieira)

Alunos: 
-   EMANUEL COSTA BETCEL
-   VICTOR VIEIRA TARGINO

Nesse passo foram coletados dados do AIRBNB do Rio da Janeiro com o objetivo de desenvolver um modelo de ML para predição de preços de locação. Foi utilizado o mlflow para implementar árvores de decisão com a pipeline incorporando a geração de artefatos com treino e pré-processamento, além de hyper-parameter tuning.

Para rodar e exportar o modelo após ter o procediment descrito no repositório da [atividade anterior](https://github.com/victorvieirat/mlops-rio-de-janeiro):

    mlflow run . -P hydra_options="decision_tree_pipeline.decision_tree.max_depth=5 decision_tree_pipeline.export_artifact=model_export"

