# Análise de Dados de Serviços de Streaming

## Visão Geral do Projeto

Este repositório contém o trabalho prático da disciplina de Visualização de Dados do curso de Bacharelado em Ciência da Computação da Universidade Tecnológica Federal do Paraná (UTFPR). O projeto, desenvolvido por André Felipe Mireski, Danilo Balman Garcia, e Rafael Machado Wanner, tem como objetivo a análise e visualização de um conjunto de dados sobre filmes e séries disponíveis em três grandes plataformas de streaming: Netflix, Disney+, e Amazon Prime.

Conforme as diretrizes da Primeira Entrega (E1) do projeto, este trabalho inicial foca na especificação do conjunto de dados escolhido, suas características, e os pré-processamentos planejados para futuras análises e visualizações.

## Conjunto de Dados

O estudo utilizará três conjuntos de dados distintos, todos disponíveis na plataforma Kaggle. Os datasets agregam informações sobre o catálogo de filmes e séries de cada serviço de streaming:

* **Netflix:** Contendo 8807 instâncias.
* **Disney+:** Com 1450 instâncias.
* **Amazon Prime:** Com 9668 instâncias.

Ao todo, o projeto trabalhará com um conjunto de dados consolidado que atende aos requisitos mínimos de 100 instâncias e 8 atributos, estipulados pela disciplina.

### Características e Atributos

O conjunto de dados consolidado é classificado nas seguintes categorias: multivalorado, textual, topológica e temporal. Ele é composto por 12 atributos, sendo 10 textuais, 2 temporais, 1 espacial e 2 topológicos. Os atributos são:

* **show\_id** (textual): Identificador único do filme ou série.
* **type** (textual): Indica se a obra é um filme ou uma série.
* **title** (textual): Título da obra.
* **director** (textual+topológico): Diretores da obra.
* **cast** (textual+topológico): Elenco de atores.
* **country** (textual+espacial): País de produção da obra.
* **date\_added** (data+temporal): Data de adição da obra na plataforma.
* **release\_year** (inteiro+temporal): Ano de lançamento da obra.
* **rating** (textual): Classificação indicativa.
* **duration** (textual): Duração em minutos (para filmes) ou temporadas (para séries).
* **listed\_in** (textual): Gêneros da obra.
* **description** (textual): Breve descrição da obra.

## Pré-processamento dos Dados

Para a consolidação e preparação dos dados para análise, as seguintes etapas de pré-processamento serão realizadas:

* **Unificação dos Dados:** Os três conjuntos de dados serão mesclados em um único. Um novo atributo categórico, `streaming`, será criado para identificar a plataforma de origem de cada registro (NETFLIX, DISNEY+, PRIME).
* **Padronização da Classificação Indicativa:** Os valores do atributo `rating` serão convertidos para o padrão brasileiro de classificação indicativa. Por exemplo, "TV-14" será convertido para "16 anos".
* **Tratamento de Dados Agrupados:** Para facilitar análises específicas, dados agrupados como gêneros, países ou atores poderão ser separados em registros individuais.

## Questões de Pesquisa e Visualizações Planejadas

As visualizações a serem desenvolvidas neste projeto buscarão responder às seguintes perguntas:

* Quais são os gêneros de filme mais populares ao longo dos anos?
* Quais os gêneros mais populares entre as diferentes plataformas de streaming?
* Quais atores possuem mais filmes em seus catálogos ao longo dos anos?
* Qual é o público-alvo de cada plataforma?
* Quem são os diretores mais requisitados, tanto no geral quanto por plataforma?
* Como se parece um grafo de conexão entre os atores com maior número de filmes?

## Como Utilizar

Este repositório conterá um notebook em Python com todo o processo de consolidação, pré-processamento e geração das visualizações. Para executar o notebook, será necessário ter um ambiente Python com as bibliotecas de manipulação e visualização de dados, como Pandas, Matplotlib e Seaborn, instaladas.

1. TODO
2. TODO 
3. TODO

## Referências dos Conjuntos de Dados

* [Dataset Amazon Prime no Kaggle](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows)
* [Dataset Disney+ no Kaggle](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows)
* [Dataset Netflix no Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)