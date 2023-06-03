from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Inicializar un objeto Counter para contar las palabras
word_counts = Counter()

# Abre el archivo en modo lectura
with open('./spa-eng/spa.txt', 'r') as f:
    # Procesar el archivo línea por línea
    for line in f:
        # Dividir la línea en palabras
        words = line.split('\t')[1]
        words = words.split()

        # Actualizar el contador de palabras
        word_counts.update(words)

# Obtener las 10 palabras más comunes y sus conteos
most_common_words = word_counts.most_common(10)

# Descomprimir las palabras y los conteos en dos listas separadas
words, counts = zip(*most_common_words)

# Crear un histograma de las palabras más comunes
plt.bar(words, counts)
plt.savefig('HistogramTarget.png')