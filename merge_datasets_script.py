# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas",
# ]
# ///
import pandas as pd

# Define os caminhos dos arquivos de entrada e saída
amazon_file = "datasets/amazon_prime_titles.csv"
disney_file = "datasets/disney_plus_titles.csv"
netflix_file = "datasets/netflix_titles.csv"
output_file = "datasets/merged_streaming_titles.csv"

# Lê os datasets
amazon_df = pd.read_csv(amazon_file)
disney_df = pd.read_csv(disney_file)
netflix_df = pd.read_csv(netflix_file)

# Adiciona a coluna 'streaming' para identificar a origem dos dados
amazon_df['streaming'] = 'Amazon Prime'
disney_df['streaming'] = 'Disney Plus'
netflix_df['streaming'] = 'Netflix'

# Concatena os datasets
merged_df = pd.concat([amazon_df, disney_df, netflix_df], ignore_index=True)

# Salva o dataset final em um novo arquivo CSV
merged_df.to_csv(output_file, index=False)

print(f"Dataset final salvo em: {output_file}")