from textblob.classifiers import NaiveBayesClassifier

print("\n\n=== CUSTOM NAIVE BAYES CLASSIFIER ===\n")

# Dane treningowe - recenzje produktów
train_data = [
    # Format: (text, label)
    ('Amazing product! Love it!', 'positive'),
    ('Best purchase ever. Highly recommend!', 'positive'),
    ('Excellent quality and fast shipping', 'positive'),
    ('Wonderful experience, will buy again', 'positive'),
    ('Great value for money', 'positive'),
    ('Perfect! Exactly what I needed', 'positive'),
    ('Outstanding product, very satisfied', 'positive'),
    ('Superb quality and service', 'positive'),

    ('Terrible quality, broke after one day', 'negative'),
    ('Waste of money. Very disappointed', 'negative'),
    ('Poor quality and bad customer service', 'negative'),
    ('Worst product ever. Do not buy!', 'negative'),
    ('Horrible experience. Total scam.', 'negative'),
    ('Completely useless. Regret buying it.', 'negative'),
    ('Awful quality. Not as described.', 'negative'),
    ('Disappointing. Not worth the price.', 'negative'),

    ('It\'s okay. Nothing special.', 'neutral'),
    ('Average product. Could be better.', 'neutral'),
    ('Decent quality for the price', 'neutral'),
    ('Not bad, not great. Just average.', 'neutral'),
    ('Acceptable. Met basic expectations.', 'neutral'),
]

# Trenowanie klasyfikatora
print("Trenowanie klasyfikatora...")
classifier = NaiveBayesClassifier(train_data)

# Test klasyfikatora
test_texts = [
    "This is absolutely fantastic! I love it!",
    "Really bad quality. Very unhappy with purchase.",
    "It's fine. Does the job.",
    "Exceptional quality! Best ever!",
    "Terrible. Broke immediately.",
    "Decent product, nothing extraordinary."
]

print("\n=== PREDYKCJE ===\n")
for text in test_texts:
    prediction = classifier.classify(text)
    prob_dist = classifier.prob_classify(text)

    print(f"Text: {text}")
    print(f"  Predicted: {prediction}")
    print(f"  P(positive): {prob_dist.prob('positive'):.3f}")
    print(f"  P(negative): {prob_dist.prob('negative'):.3f}")
    print(f"  P(neutral): {prob_dist.prob('neutral'):.3f}\n")

# Accuracy na danych testowych
test_data = [
    ('Great product, very happy!', 'positive'),
    ('Poor quality, not satisfied', 'negative'),
    ('Average, nothing special', 'neutral'),
    ('Excellent! Highly recommended!', 'positive'),
    ('Disappointing purchase', 'negative'),
]

accuracy = classifier.accuracy(test_data)
print(f"=== ACCURACY ===")
print(f"Dokładność: {accuracy:.2%}\n")

# Najważniejsze cechy (features)
print("=== TOP 10 INFORMATIVE FEATURES ===")
classifier.show_informative_features(10)