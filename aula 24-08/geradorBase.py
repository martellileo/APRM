import pandas as pd
import numpy as np

# Fixando a semente aleatória
np.random.seed(42)

classes = [
    'Bardo', 'Bruxo', 'Clérigo', 'Druida', 'Feiticeiro', 
    'Guerreiro', 'Ladino', 'Mago', 'Monge', 'Paladino', 'Patrulheiro'
]
racas = ['Humano', 'Elfo', 'Anão', 'Halfling', 'Tiefling', 'Draconato', 'Githyanki']

perfis = {
    'Guerreiro':   {'forca': (17, 2), 'destreza': (13, 2), 'constituicao': (16, 2), 'inteligencia': (9, 2),  'sabedoria': (10, 2), 'carisma': (10, 2)},
    'Paladino':    {'forca': (16, 2), 'destreza': (10, 2), 'constituicao': (15, 2), 'inteligencia': (9, 2),  'sabedoria': (11, 2), 'carisma': (16, 2)},
    'Mago':        {'forca': (8, 2),  'destreza': (14, 2), 'constituicao': (13, 2), 'inteligencia': (17, 2), 'sabedoria': (12, 2), 'carisma': (10, 2)},
    'Feiticeiro':  {'forca': (8, 2),  'destreza': (14, 2), 'constituicao': (14, 2), 'inteligencia': (10, 2), 'sabedoria': (10, 2), 'carisma': (17, 2)},
    'Bruxo':       {'forca': (9, 2),  'destreza': (13, 2), 'constituicao': (14, 2), 'inteligencia': (11, 2), 'sabedoria': (10, 2), 'carisma': (17, 2)},
    'Clérigo':     {'forca': (13, 2), 'destreza': (10, 2), 'constituicao': (14, 2), 'inteligencia': (10, 2), 'sabedoria': (17, 2), 'carisma': (12, 2)},
    'Druida':      {'forca': (10, 2), 'destreza': (12, 2), 'constituicao': (14, 2), 'inteligencia': (11, 2), 'sabedoria': (17, 2), 'carisma': (9, 2)},
    'Ladino':      {'forca': (9, 2),  'destreza': (17, 2), 'constituicao': (13, 2), 'inteligencia': (12, 2), 'sabedoria': (10, 2), 'carisma': (13, 2)},
    'Patrulheiro': {'forca': (11, 2), 'destreza': (16, 2), 'constituicao': (13, 2), 'inteligencia': (10, 2), 'sabedoria': (15, 2), 'carisma': (9, 2)},
    'Monge':       {'forca': (11, 2), 'destreza': (16, 2), 'constituicao': (13, 2), 'inteligencia': (9, 2),  'sabedoria': (16, 2), 'carisma': (9, 2)},
    'Bardo':       {'forca': (9, 2),  'destreza': (15, 2), 'constituicao': (12, 2), 'inteligencia': (11, 2), 'sabedoria': (10, 2), 'carisma': (17, 2)},
}

# Sorteando exatamente 3000 classes
classes_sorteadas = np.random.choice(classes, size=3000)

dados = []
for idx, classe in enumerate(classes_sorteadas, start=1):
    row = {
        'nome_personagem': f'Personagem_{idx}',
        'classe': classe,
        'raca': np.random.choice(racas),
        'nivel': int(np.clip(np.random.normal(6.5, 3), 1, 20))
    }
    for attr, (media, desvio) in perfis[classe].items():
        row[attr] = int(np.clip(np.random.normal(media, desvio), 3, 20))
    dados.append(row)

df = pd.DataFrame(dados)

# Inserindo dados sujos (NaNs, formatação incorreta e Outliers)
df.loc[np.random.choice(df.index, 60, replace=False), 'raca'] = np.nan
df.loc[np.random.choice(df.index, 40, replace=False), 'nivel'] = np.nan
df.loc[np.random.choice(df.index, 40, replace=False), 'forca'] = np.nan

racas_sujas = [' elfo ', 'HUMANO', '   anão', 'Tiefling  ', 'draconato']
indices_raca = np.random.choice(df.dropna(subset=['raca']).index, 100, replace=False)
for i in indices_raca:
    df.loc[i, 'raca'] = np.random.choice(racas_sujas)

df.loc[np.random.choice(df.index, 15, replace=False), 'destreza'] = np.random.choice([0, -5, 99, 150])
df.loc[np.random.choice(df.index, 10, replace=False), 'nivel'] = np.random.choice([0, 25, 100])

df.to_csv('bg3_personagens_distintos_sujo.csv', index=False)