# OpenAI Cheat Sheet
OpenAI Cheat Sheet with the most needed stuff..




## Chat-GPT
- https://platform.openai.com/docs/api-reference/chat/create

<br><br>

### Node.js (https://www.npmjs.com/package/openai)
```
import { Configuration, OpenAIApi } from "openai"

const configuration = new Configuration({
  apiKey: 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
})

const openai = new OpenAIApi(configuration)
// const engines = await openai.listEngines()
// console.log(engines)

const content = `
What is 1+1?
`

const messages = [{"role": "user", content}]
const model = 'gpt-3.5-turbo'

const completion = await openai.createChatCompletion({
  model,
  messages,
})

const parsedResponse = JSON.parse(completion.data.choices[0].message.content)
console.log(JSON.stringify(parsedResponse, null,4))
```
















<br><br>
<br><br>
___
<br><br>
<br><br>


# Embedding



<details><summary>Click to expand..</summary>

## Was sind Embeddings?

Embeddings sind numerische Repräsentationen (Vektoren von Fließkommazahlen) von Text, die dessen semantische Bedeutung erfassen. Die Distanz zwischen zwei Vektoren misst ihre Ähnlichkeit:
*   **Kleine Distanz**: Hohe Ähnlichkeit / Verwandtschaft
*   **Große Distanz**: Geringe Ähnlichkeit / Verwandtschaft

OpenAI's Text-Embeddings messen die Verwandtschaft von Text-Strings.

### Anwendungsfälle:
*   **Suche**: Ergebnisse nach Relevanz zu einer Suchanfrage ordnen.
*   **Clustering**: Text-Strings nach Ähnlichkeit gruppieren.
*   **Empfehlungen**: Ähnliche Items empfehlen.
*   **Anomalieerkennung**: Ausreißer mit geringer Verwandtschaft identifizieren.
*   **Diversitätsmessung**: Ähnlichkeitsverteilungen analysieren.
*   **Klassifikation**: Texte anhand des ähnlichsten Labels klassifizieren.

## Neue Embedding-Modelle (v3)

OpenAI bietet neue, leistungsstärkere Embedding-Modelle:
*   `text-embedding-3-small`
*   `text-embedding-3-large`

**Vorteile**:
*   Geringere Kosten
*   Höhere mehrsprachige Performanz
*   Neuer Parameter (`dimensions`), um die Größe des Embeddings zu steuern.

**Abrechnung**: Basiert auf der Anzahl der Tokens im Input. Siehe [Pricing-Seite](https://openai.com/pricing).

## Wie erhält man Embeddings?

Sende den Text-String zusammen mit dem Modellnamen an den Embeddings API-Endpunkt.

### Beispiel: Embeddings erstellen (Python)

```python
import OpenAI
from openai import OpenAI # Nötig für neuere Versionen

client = OpenAI() # API-Key wird typischerweise über Umgebungsvariable OPENAI_API_KEY gelesen

embedding_response = client.embeddings.create(
  model="text-embedding-3-small",
  input="Dein Text-String hier",
  encoding_format="float" # oder "base64"
  # dimensions=256 # Optional: Reduziert die Dimension des Embeddings
)

embedding_vector = embedding_response.data[0].embedding
# print(embedding_vector)
```

### Beispiel-Antwort (JSON):

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        // ... weitere Zahlen
        -0.024047505110502243
      ]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```
Der Embedding-Vektor kann in einer Vektor-Datenbank gespeichert werden.

### Dimensionen von Embeddings:
*   `text-embedding-3-small`: Standardmäßig 1536 Dimensionen.
*   `text-embedding-3-large`: Standardmäßig 3072 Dimensionen.
*   Mit dem `dimensions`-Parameter kann die Länge des Embedding-Vektors reduziert werden, ohne dass die konzeptdarstellenden Eigenschaften verloren gehen (Trade-off zwischen Performance und Kosten/Ressourcen).

## Embedding-Modelle im Überblick

| Modell                   | ~ Seiten pro Dollar | Performance (MTEB) | Max. Input Tokens | Standard-Dimensionen |
| ------------------------ | ------------------- | -------------------- | ----------------- | -------------------- |
| `text-embedding-3-small` | 62.500              | 62.3%                | 8192              | 1536                 |
| `text-embedding-3-large` | 9.615               | 64.6%                | 8192              | 3072                 |
| `text-embedding-ada-002` | 12.500              | 61.0%                | 8192              | 1536                 |
*(Annahme: ~800 Tokens pro Seite)*

## Wichtige Anwendungsfälle & Techniken

### 1. Embeddings erstellen & speichern (Python mit Pandas)

```python
from openai import OpenAI
import pandas as pd # Annahme: df ist ein DataFrame mit einer Spalte 'combined_text'

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small", dimensions=None):
    text = text.replace("\n", " ")
    params = {"input": [text], "model": model}
    if dimensions:
        params["dimensions"] = dimensions
    return client.embeddings.create(**params).data[0].embedding

# Beispiel für eine Spalte 'combined' in einem DataFrame 'df'
# df['embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_csv('output/embedded_reviews.csv', index=False)

# Laden gespeicherter Embeddings (die als String gespeichert wurden)
# df = pd.read_csv('output/embedded_reviews.csv')
# df['embedding'] = df.embedding.apply(eval).apply(np.array) # np für numpy
```

### 2. Dimensionen reduzieren

*   **Empfohlen**: `dimensions`-Parameter beim API-Aufruf `embeddings.create()` nutzen.
    ```python
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="Text",
        dimensions=256 # Fordert ein Embedding mit 256 Dimensionen an
    )
    embedding = response.data[0].embedding
    ```
*   **Manuell (Fortgeschritten)**: Wenn Embeddings bereits generiert wurden, können sie gekürzt und L2-normalisiert werden.
    ```python
    import numpy as np

    def normalize_l2(x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)

    # Annahme: 'full_embedding' ist ein bereits generiertes Embedding
    # cut_dim_embedding = full_embedding[:256]
    # normalized_embedding = normalize_l2(cut_dim_embedding)
    ```
    Dies ermöglicht flexible Nutzung, z.B. wenn Vektor-Datenbanken eine maximale Dimension haben.

### 3. Textsuche (Ähnlichkeitssuche)

Dokumente anhand der Kosinus-Ähnlichkeit zwischen der Suchanfrage-Embedding und den Dokument-Embeddings finden.

```python
# from openai.embeddings_utils import cosine_similarity # Veraltet, selbst implementieren oder numpy nutzen
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # oder Exception
    return dot_product / (norm_vec1 * norm_vec2)

# Annahme: df hat eine Spalte 'embedding' mit den Embeddings der Dokumente
# def search_documents(df, query_text, n=3, model="text-embedding-3-small"):
#     query_embedding = get_embedding(query_text, model=model)
#     df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))
#     results = df.sort_values('similarities', ascending=False).head(n)
#     return results

# res = search_documents(df, 'köstliche Bohnen', n=3)
```

### 4. Code-Suche

Ähnlich wie Textsuche, aber auf Code-Snippets angewendet. Jede Funktion/Codeblock wird eingebettet. Natürliche Sprach-Query wird ebenfalls eingebettet und per Kosinus-Ähnlichkeit verglichen.

### 5. Empfehlungen (Recommendations)

Embeddings von Items (z.B. Artikelbeschreibungen) berechnen. Für ein Quell-Item die Items mit der höchsten Kosinus-Ähnlichkeit (kleinste Distanz) finden.

### 6. Datenvisualisierung (z.B. mit t-SNE)

Hochdimensionale Embeddings auf 2D reduzieren, um Cluster oder Beziehungen visuell darzustellen.

```python
# import pandas as pd
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import matplotlib

# Annahme: df['embedding'] enthält die Embeddings, df['Score'] die Bewertungen
# matrix = np.vstack(df.embedding.values) # Stapelt die Embedding-Arrays

# tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
# vis_dims = tsne.fit_transform(matrix)

# x = [v[0] for v in vis_dims]
# y = [v[1] for v in vis_dims]
# colors_map = {1:"red", 2:"darkorange", 3:"gold", 4:"turquoise", 5:"darkgreen"}
# color_indices = df.Score.map(colors_map)

# plt.scatter(x, y, c=color_indices, alpha=0.3)
# plt.title("Visualisierung mit t-SNE")
# plt.show()
```

### 7. Embeddings als Feature Encoder für ML-Algorithmen

Embeddings können als Input-Features für traditionelle ML-Modelle (Regression, Klassifikation) dienen, besonders wenn Freitext-Daten relevant sind.
*   **Regression**: Vorhersage eines numerischen Werts (z.B. Review-Score).
*   **Klassifikation**: Vorhersage einer Kategorie (z.B. Sentiment).

### 8. Zero-Shot Klassifikation

Klassifizierung ohne gelabelte Trainingsdaten.
1.  Embedde die Klassennamen oder kurze Beschreibungen der Klassen.
2.  Embedde den zu klassifizierenden Text.
3.  Vergleiche das Text-Embedding mit allen Klassen-Embeddings (Kosinus-Ähnlichkeit).
4.  Die Klasse mit der höchsten Ähnlichkeit wird vorhergesagt.

```python
# Beispiel: Sentiment-Klassifikation
# labels = ['negativ', 'positiv']
# label_embeddings = [get_embedding(label, model="text-embedding-3-small") for label in labels]

# def classify_sentiment(review_text, label_embeddings_list, model="text-embedding-3-small"):
#     review_embedding = get_embedding(review_text, model=model)
#     # similarity_negative = cosine_similarity(review_embedding, label_embeddings_list[0])
#     # similarity_positive = cosine_similarity(review_embedding, label_embeddings_list[1])
#     # return "positiv" if similarity_positive > similarity_negative else "negativ"

# prediction = classify_sentiment("Das Essen war fantastisch!", label_embeddings)
```

### 9. Clustering

Unüberwachtes Entdecken von Gruppen in Textdaten basierend auf der Ähnlichkeit der Embeddings (z.B. mit K-Means).

```python
# from sklearn.cluster import KMeans
# matrix = np.vstack(df.embedding.values)
# n_clusters = 4
# kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init='auto')
# kmeans.fit(matrix)
# df['Cluster'] = kmeans.labels_
```

## FAQ

### Q: Token-Anzahl vor dem Embedden bestimmen?
**A**: Nutze `tiktoken`. Für v3-Modelle (`text-embedding-3-small`, `text-embedding-3-large`) die Kodierung `cl100k_base` verwenden.

```python
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# print(num_tokens_from_string("tiktoken ist super!"))
```

### Q: K nächste Embedding-Vektoren schnell finden?
**A**: Vektor-Datenbanken verwenden (z.B. Pinecone, Weaviate, ChromaDB, FAISS).

### Q: Welche Distanzfunktion verwenden?
**A**: **Kosinus-Ähnlichkeit** wird empfohlen. OpenAI-Embeddings sind auf Länge 1 normalisiert, daher:
*   Kosinus-Ähnlichkeit kann schneller per Skalarprodukt berechnet werden.
*   Kosinus-Ähnlichkeit und Euklidische Distanz führen zu identischen Rankings.

### Q: Darf ich meine Embeddings online teilen?
**A**: Ja, Kunden besitzen Input und Output, inkl. Embeddings. Stelle sicher, dass der Inhalt keine Gesetze oder Nutzungsbedingungen verletzt.

### Q: Wissen v3 Embedding-Modelle über aktuelle Ereignisse Bescheid?
**A**: Nein, die Modelle (`text-embedding-3-large` und `text-embedding-3-small`) haben kein Wissen über Ereignisse nach **September 2021**. Dies ist meist weniger limitierend als bei Textgenerierungsmodellen, kann aber in Grenzfällen die Performance beeinflussen.


  
</details>





