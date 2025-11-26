from nltk import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import random

# Przykładowy tekst
tekst = """
Sztuczna inteligencja zmienia świat. Uczenie maszynowe jest częścią sztucznej inteligencji.
Głębokie uczenie to poddziedzina uczenia maszynowego. Modele językowe wykorzystują uczenie głębokie.
"""

# Tokenizacja
tokens = word_tokenize(tekst.lower())

# 1. GENEROWANIE N-GRAMÓW
print("=== UNIGRAMY ===")
unigramy = list(ngrams(tokens, 1))
print(unigramy[:5])

print("\n=== BIGRAMY ===")
bigramy = list(ngrams(tokens, 2))
print(bigramy[:5])

print("\n=== TRIGRAMY ===")
trigramy = list(ngrams(tokens, 3))
print(trigramy[:5])

# 2. ZLICZANIE CZĘSTOŚCI N-GRAMÓW
print("\n=== NAJCZĘSTSZE BIGRAMY ===")
bigram_freq = Counter(bigramy)
for bigram, freq in bigram_freq.most_common(5):
    print(f"{bigram}: {freq}")


# 3. MODEL BIGRAM DO GENEROWANIA TEKSTU
class BigramModel:
    def __init__(self):
        self.model = defaultdict(list)

    def train(self, tokens):
        for w1, w2 in ngrams(tokens, 2):
            self.model[w1].append(w2)

    def generate(self, start_word, length=10):
        current_word = start_word
        result = [current_word]

        for _ in range(length - 1):
            if current_word not in self.model or not self.model[current_word]:
                break
            next_word = random.choice(self.model[current_word])
            result.append(next_word)
            current_word = next_word

        return ' '.join(result)


# Trenowanie modelu
bigram_model = BigramModel()
bigram_model.train(tokens)

print("\n=== GENEROWANIE TEKSTU ===")
print(bigram_model.generate('sztuczna', 8))


# 4. MODEL Z PRAWDOPODOBIEŃSTWAMI
class ProbabilisticBigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()

    def train(self, tokens):
        for w1, w2 in ngrams(tokens, 2):
            self.bigram_counts[w1][w2] += 1
            self.unigram_counts[w1] += 1

    def probability(self, w1, w2):
        """P(w2|w1) = Count(w1,w2) / Count(w1)"""
        if w1 not in self.bigram_counts:
            return 0
        return self.bigram_counts[w1][w2] / self.unigram_counts[w1]

    def most_likely_next(self, word, n=3):
        if word not in self.bigram_counts:
            return []
        return self.bigram_counts[word].most_common(n)


# Trenowanie modelu probabilistycznego
prob_model = ProbabilisticBigramModel()
prob_model.train(tokens)

print("\n=== PRAWDOPODOBIEŃSTWA ===")
print(f"P('maszynowe'|'uczenie'): {prob_model.probability('uczenie', 'maszynowe'):.2f}")
print(f"\nNajprawdopodobniejsze słowa po 'uczenie':")
for word, count in prob_model.most_likely_next('uczenie', 3):
    print(f"  {word}: {count} razy")


# 5. MODEL TRIGRAM
class TrigramModel:
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(list))

    def train(self, tokens):
        for w1, w2, w3 in ngrams(tokens, 3):
            self.model[w1][w2].append(w3)

    def generate(self, start_words, length=10):
        if len(start_words) != 2:
            raise ValueError("Potrzebne dwa początkowe słowa")

        w1, w2 = start_words
        result = [w1, w2]

        for _ in range(length - 2):
            if w1 not in self.model or w2 not in self.model[w1]:
                break
            w3 = random.choice(self.model[w1][w2])
            result.append(w3)
            w1, w2 = w2, w3

        return ' '.join(result)


trigram_model = TrigramModel()
trigram_model.train(tokens)

print("\n=== GENEROWANIE Z TRIGRAMÓW ===")
print(trigram_model.generate(['sztuczna', 'inteligencja'], 10))

# 6. PERPLEXITY - OCENA MODELU
import math


def calculate_perplexity(model, test_tokens):
    """Oblicza perplexity dla modelu bigramowego"""
    log_prob_sum = 0
    count = 0

    for w1, w2 in ngrams(test_tokens, 2):
        prob = model.probability(w1, w2)
        if prob > 0:
            log_prob_sum += math.log(prob)
            count += 1

    if count == 0:
        return float('inf')

    perplexity = math.exp(-log_prob_sum / count)
    return perplexity


print("\n=== PERPLEXITY ===")
print(f"Perplexity modelu: {calculate_perplexity(prob_model, tokens):.2f}")