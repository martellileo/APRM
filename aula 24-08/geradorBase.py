import numpy as np
import pandas as pd

# Define a semente aleatória solicitada para garantir a reprodutibilidade
np.random.seed(42)

n_registros = 3000

# Opções de nomes temáticos de Baldur's Gate 3
nomes_primeiros = [
    "Astarion", "Shadowheart", "Gale", "Wyll", "Karlach", "Lae'zel", "Halsin", "Jaheira", 
    "Minsc", "Minthara", "Viconia", "Sarevok", "Ketheric", "Gortash", "Orin", "Bhaal", 
    "Elminster", "Volothamp", "Alfira", "Barcus", "Bex", "Danis", "Cora", "Brem", 
    "Roah", "Abdirak", "Nere", "Dror", "Ragzlin", "Aradin", "Zevlor", "Dammon", "Mizora", 
    "Raphael", "Voss", "Omeluum", "Blurg", "Sazza", "Kagha", "Rath"
]

sobrenomes = [
    "Ansur", "Anchev", "Del'Vyre", "Dekarios", "Ravengard", "Cliffgate", "Hallowtide", 
    "Stoutbeard", "Brightwood", "Swiftfoot", "Ironfist", "Moonbreeze", "Silverfrond", 
    "Bloodaxe", "Firebrand", "Shadowdancer", "Spellweaver", "Stormwind", "Darkweaver",
    "Goldseeker", "Stonefoot", "Windrunner", "Starseeker", "Sunstrider", "Nightbreeze"
]

racas = ["Humano", "Elfo", "Meio-Elfo", "Anão", "Halfling", "Gnomo", "Tiefling", "Dragonborn", "Githyanki", "Orc"]
classes = ["Guerreiro", "Mago", "Clérigo", "Ladino", "Bardo", "Bruxo", "Paladino", "Patrulheiro", "Monge", "Druida", "Feiticeiro"]

# Sorteando os nomes limpos
primeiros_escolhidos = np.random.choice(nomes_primeiros, size=n_registros)
sobrenomes_escolhidos = np.random.choice(sobrenomes, size=n_registros)
nomes_gerados = [f"{p} {s}" for p, s in zip(primeiros_escolhidos, sobrenomes_escolhidos)]

# Atributos numéricos (proficiência removida)
nivel = np.random.randint(1, 13, size=n_registros)
forca = np.random.randint(8, 20, size=n_registros)
destreza = np.random.randint(8, 20, size=n_registros)
constituicao = np.random.randint(8, 20, size=n_registros)
sabedoria = np.random.randint(8, 20, size=n_registros)
inteligencia = np.random.randint(8, 20, size=n_registros)
carisma = np.random.randint(8, 20, size=n_registros)

raca_escolhida = np.random.choice(racas, size=n_registros)
classe_escolhida = np.random.choice(classes, size=n_registros)

# Construindo o DataFrame final
df_bg3_final = pd.DataFrame({
    'nome_personagem': nomes_gerados,
    'nivel': nivel,
    'raca': raca_escolhida,
    'classe': classe_escolhida,
    'forca': forca,
    'destreza': destreza,
    'constituicao': constituicao,
    'sabedoria': sabedoria,
    'inteligencia': inteligencia,
    'carisma': carisma
})

# Exportando para CSV
df_bg3_final.to_csv('bg3_personagens.csv', index=False)

print("Dataset 'bg3_personagens.csv' atualizado com sucesso!")