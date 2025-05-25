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


# Embeddings
- https://platform.openai.com/docs/guides/embeddings



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

Sende den Text-String zusammen mit dem Modellnamen an den Embeddings API-Endpunkt. Stelle sicher, dass du das OpenAI SDK installiert hast (`npm install openai`).

### Beispiel: Embeddings erstellen (Node.js)

```javascript
import OpenAI from 'openai';

// API-Key wird typischerweise über die Umgebungsvariable OPENAI_API_KEY gelesen
const openai = new OpenAI();

async function createEmbedding() {
  try {
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: "Dein Text-String hier", // Kann auch ein Array von Strings sein
      encoding_format: "float", // oder "base64"
      // dimensions: 256, // Optional: Reduziert die Dimension des Embeddings
    });

    // Wenn 'input' ein einzelner String war, ist das Embedding in data[0]
    // Wenn 'input' ein Array von Strings war, ist 'data' ein Array von Embedding-Objekten
    const embeddingVector = embeddingResponse.data[0].embedding;
    // console.log(embeddingVector);
    // console.log("Dimension:", embeddingVector.length);
    // console.log("Tokens verbraucht:", embeddingResponse.usage.total_tokens);
    return embeddingVector;
  } catch (error) {
    console.error("Fehler beim Erstellen des Embeddings:", error);
  }
}

// Beispielaufruf:
// createEmbedding().then(vector => {
//   if (vector) {
//     // Mache etwas mit dem Vektor
//   }
// });
```

### Beispiel-Antwort (JSON-Struktur, die das SDK zurückgibt):

Das `embeddingResponse`-Objekt im Code oben hätte eine ähnliche Struktur wie dieses JSON:
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
    // Weitere Objekte, falls 'input' ein Array war
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```
Der Embedding-Vektor (ein Array von Zahlen) kann in einer Vektor-Datenbank gespeichert werden.

### Dimensionen von Embeddings:
*   `text-embedding-3-small`: Standardmäßig 1536 Dimensionen.
*   `text-embedding-3-large`: Standardmäßig 3072 Dimensionen.
*   Mit dem `dimensions`-Parameter im `create`-Aufruf kann die Länge des Embedding-Vektors reduziert werden, ohne dass die konzeptdarstellenden Eigenschaften verloren gehen (Trade-off zwischen Performance und Kosten/Ressourcen).

## Embedding-Modelle im Überblick

| Modell                   | ~ Seiten pro Dollar | Performance (MTEB) | Max. Input Tokens | Standard-Dimensionen |
| ------------------------ | ------------------- | -------------------- | ----------------- | -------------------- |
| `text-embedding-3-small` | 62.500              | 62.3%                | 8192              | 1536                 |
| `text-embedding-3-large` | 9.615               | 64.6%                | 8192              | 3072                 |
| `text-embedding-ada-002` | 12.500              | 61.0%                | 8192              | 1536                 |
*(Annahme: ~800 Tokens pro Seite)*

## Wichtige Anwendungsfälle & Techniken (Node.js)

### 1. Embeddings erstellen & (hypothetisch) speichern

```javascript
import OpenAI from 'openai';
// import fs from 'fs/promises'; // Zum Speichern in eine Datei (Beispiel)

const openai = new OpenAI();

async function getAndStoreEmbedding(text, model = "text-embedding-3-small", dimensions = undefined) {
  try {
    const cleanedText = text.replace(/\n/g, " "); // \n können Probleme verursachen oder die Semantik leicht verändern
    
    const params = {
      input: [cleanedText], // API erwartet ein Array von Strings
      model: model,
    };
    if (dimensions) {
      params.dimensions = dimensions;
    }

    const response = await openai.embeddings.create(params);
    const embedding = response.data[0].embedding;

    // Hypothetisches Speichern (z.B. als Teil eines größeren Objekts in einer JSON-Datei)
    // const dataToStore = { originalText: cleanedText, embedding: embedding };
    // await fs.writeFile('output/my_embedding.json', JSON.stringify(dataToStore, null, 2));
    // console.log('Embedding gespeichert.');

    return embedding;
  } catch (error) {
    console.error("Fehler in getAndStoreEmbedding:", error);
    return null;
  }
}

// Beispiel für mehrere Texte:
// async function processTexts(texts) {
//   const allEmbeddings = [];
//   for (const text of texts) {
//     const embedding = await getAndStoreEmbedding(text); // Oder nur getEmbedding
//     if (embedding) {
//       allEmbeddings.push({ text, embedding });
//     }
//   }
//   // Hier könntest du 'allEmbeddings' in einer Vektor-DB speichern
//   // console.log(JSON.stringify(allEmbeddings, null, 2));
//   return allEmbeddings;
// }

// processTexts(["Erste Regel.", "Zweite, etwas längere Regel."]);
```
*Hinweis zum Laden*: Gespeicherte Embeddings (z.B. aus JSON) wären einfach JavaScript Arrays.

### 2. Dimensionen reduzieren

*   **Empfohlen**: `dimensions`-Parameter beim API-Aufruf `openai.embeddings.create()` nutzen.
    ```javascript
    async function getReducedDimensionEmbedding() {
      try {
        const response = await openai.embeddings.create({
          model: "text-embedding-3-large", // oder text-embedding-3-small
          input: "Ein Text, dessen Embedding mit weniger Dimensionen erstellt werden soll.",
          dimensions: 256, // Gewünschte Anzahl an Dimensionen
        });
        const embedding = response.data[0].embedding;
        // console.log("Reduziertes Embedding:", embedding);
        // console.log("Neue Dimension:", embedding.length); // Sollte 256 sein
        return embedding;
      } catch (error) {
        console.error("Fehler beim Erstellen des reduzierten Embeddings:", error);
      }
    }
    // getReducedDimensionEmbedding();
    ```
*   **Manuell (Fortgeschritten)**: Wenn Embeddings bereits generiert wurden, können sie gekürzt und L2-normalisiert werden.
    ```javascript
    function normalizeL2(vector) {
      if (!Array.isArray(vector) || vector.length === 0) {
        return vector; // Oder Fehler werfen
      }

      let sumOfSquares = 0;
      for (const val of vector) {
        sumOfSquares += val * val;
      }
      const norm = Math.sqrt(sumOfSquares);

      if (norm === 0) {
        // Vektor besteht nur aus Nullen, kann nicht normalisiert werden (oder ist bereits "normal")
        return vector.slice(); // Kopie zurückgeben
      }

      return vector.map(val => val / norm);
    }

    // Annahme: 'fullEmbedding' ist ein bereits generiertes Array von Zahlen
    // const fullEmbedding = [0.1, 0.02, -0.5, /* ... Hunderte Zahlen ... */, 0.3];
    // const cutDimEmbedding = fullEmbedding.slice(0, 256); // Schneidet den Vektor auf die ersten 256 Dimensionen ab
    // const normalizedReducedEmbedding = normalizeL2(cutDimEmbedding);
    // console.log(normalizedReducedEmbedding);
    ```
    Dies ermöglicht flexible Nutzung, z.B. wenn Vektor-Datenbanken eine maximale Dimension haben.

### 3. Textsuche (Ähnlichkeitssuche)

Dokumente anhand der Kosinus-Ähnlichkeit zwischen der Suchanfrage-Embedding und den Dokument-Embeddings finden.

```javascript
function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) {
    // console.error("Vektoren müssen existieren, gleiche Länge haben und nicht leer sein.");
    return 0; // Oder Fehler werfen, oder NaN
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0; // Division durch Null vermeiden, keine Ähnlichkeit wenn ein Vektor Null ist
  }

  return dotProduct / (normA * normB);
}

// Annahme: 'documents' ist ein Array von Objekten,
// jedes mit einem Text und einem 'embedding'-Feld (Array von Zahlen)
// const documents = [
//   { id: 1, rule: "Regel A...", embedding: [/* ... */] },
//   { id: 2, rule: "Regel B...", embedding: [/* ... */] },
// ];

// async function searchDocuments(docs, queryText, topN = 3, model = "text-embedding-3-small") {
//   try {
//     const queryEmbedding = await getAndStoreEmbedding(queryText, model); // Oder eine generische getEmbedding Funktion
//     if (!queryEmbedding) return [];

//     const resultsWithSimilarity = docs.map(doc => ({
//       ...doc, // Behalte alle ursprünglichen Dokumentdaten
//       similarity: cosineSimilarity(doc.embedding, queryEmbedding)
//     }));

//     // Sortiere nach Ähnlichkeit (absteigend)
//     resultsWithSimilarity.sort((a, b) => b.similarity - a.similarity);

//     return resultsWithSimilarity.slice(0, topN);
//   } catch (error) {
//     console.error("Fehler bei der Dokumentsuche:", error);
//     return [];
//   }
// }

// (async () => {
//   // Vorab Embeddings für alle Dokumente erstellen und 'documents' füllen
//   const sampleDocs = [
//      { id: 1, rule: "React Komponente immer mit PascalCase benennen.", embedding: await getAndStoreEmbedding("React Komponente immer mit PascalCase benennen.") },
//      { id: 2, rule: "Styles in separaten CSS-Modulen halten.", embedding: await getAndStoreEmbedding("Styles in separaten CSS-Modulen halten.") },
//      { id: 3, rule: "Funktionskomponenten bevorzugen.", embedding: await getAndStoreEmbedding("Funktionskomponenten bevorzugen.") }
//   ];
//   const searchResults = await searchDocuments(sampleDocs, 'Wie style ich meine React Komponente?', 2);
//   console.log(searchResults);
// })();
```

### 4. Code-Suche

Ähnlich wie Textsuche, aber auf Code-Snippets angewendet. Jede Funktion/Codeblock wird eingebettet. Eine natürliche Sprach-Query wird ebenfalls eingebettet und per Kosinus-Ähnlichkeit verglichen. Der Prozess ist analog zur Textsuche.

### 5. Empfehlungen (Recommendations)

Embeddings von Items (z.B. Artikelbeschreibungen) berechnen. Für ein Quell-Item die Items mit der höchsten Kosinus-Ähnlichkeit (kleinste Distanz) finden. Wieder analog zur Textsuche.

### 6. Datenvisualisierung (z.B. mit t-SNE)

Hochdimensionale Embeddings auf 2D reduzieren, um Cluster oder Beziehungen visuell darzustellen.
In Node.js könnten hierfür Bibliotheken wie `tensorflow.js` (für t-SNE Implementierungen) und Plotting-Bibliotheken wie `chart.js` (im Browser) oder `node-chartjs` (serverseitig) verwendet werden. Der Prozess wäre:
1.  Alle Embeddings sammeln.
2.  Mit t-SNE auf 2 Dimensionen reduzieren.
3.  Die 2D-Punkte plotten.
Dies ist komplexer und sprengt den Rahmen eines einfachen Cheatsheet-Beispiels.

### 7. Embeddings als Feature Encoder für ML-Algorithmen

Embeddings können als Input-Features für traditionelle ML-Modelle (Regression, Klassifikation) dienen, besonders wenn Freitext-Daten relevant sind. In Node.js könnten ML-Bibliotheken wie `tensorflow.js` oder spezialisiertere Bibliotheken genutzt werden.
*   **Regression**: Vorhersage eines numerischen Werts.
*   **Klassifikation**: Vorhersage einer Kategorie.

### 8. Zero-Shot Klassifikation

Klassifizierung ohne gelabelte Trainingsdaten.
1.  Embedde die Klassennamen oder kurze Beschreibungen der Klassen.
2.  Embedde den zu klassifizierenden Text.
3.  Vergleiche das Text-Embedding mit allen Klassen-Embeddings (Kosinus-Ähnlichkeit).
4.  Die Klasse mit der höchsten Ähnlichkeit wird vorhergesagt.

```javascript
// async function zeroShotClassify(textToClassify, classLabels, model = "text-embedding-3-small") {
//   try {
//     const textEmbedding = await getAndStoreEmbedding(textToClassify, model);
//     if (!textEmbedding) return null;

//     let bestMatch = { label: null, similarity: -Infinity };

//     for (const label of classLabels) {
//       const labelEmbedding = await getAndStoreEmbedding(label, model);
//       if (labelEmbedding) {
//         const similarity = cosineSimilarity(textEmbedding, labelEmbedding);
//         if (similarity > bestMatch.similarity) {
//           bestMatch = { label, similarity };
//         }
//       }
//     }
//     return bestMatch.label;
//   } catch (error) {
//     console.error("Fehler bei der Zero-Shot Klassifikation:", error);
//     return null;
//   }
// }

// (async () => {
//   const labels = ['Technik', 'Sport', 'Wirtschaft'];
//   const articleText = "Das neue Smartphone hat eine verbesserte Kamera und schnelleren Prozessor.";
//   const predictedLabel = await zeroShotClassify(articleText, labels);
//   console.log(`Der Artikel "${articleText.substring(0,30)}..." wurde als "${predictedLabel}" klassifiziert.`);
// })();
```

### 9. Clustering

Unüberwachtes Entdecken von Gruppen in Textdaten basierend auf der Ähnlichkeit der Embeddings (z.B. mit K-Means). In Node.js könnten Bibliotheken wie `ml-kmeans` oder andere aus dem `ml.js`-Ökosystem verwendet werden.

## FAQ

### Q: Token-Anzahl vor dem Embedden bestimmen?
**A**: Nutze `tiktoken`. Für Node.js ist das Paket `@dqbd/tiktoken` (ein WASM-Port des offiziellen Python `tiktoken`) sehr verbreitet und genau. Installiere es mit `npm install @dqbd/tiktoken`.

```javascript
import { get_encoding, encoding_for_model } from "@dqbd/tiktoken";

// Für v3 Modelle wie text-embedding-3-small/-large: "cl100k_base"
// Für ältere wie text-embedding-ada-002: auch "cl100k_base" oder spezifischer "text-embedding-ada-002"
function numTokensFromString(string, modelName = "text-embedding-3-small") {
    let encoding;
    try {
        // Versuche, die spezifische Kodierung für das Modell zu bekommen
        // encoding_for_model wirft einen Fehler, wenn das Modell nicht bekannt ist.
        encoding = encoding_for_model(modelName);
    } catch (e) {
        // Fallback auf eine generische Kodierung, wenn das Modell nicht direkt unterstützt wird
        // console.warn(`Keine spezifische Kodierung für ${modelName} gefunden, verwende cl100k_base.`);
        encoding = get_encoding("cl100k_base");
    }
    
    const tokens = encoding.encode(string);
    encoding.free(); // Wichtig, um WASM-Speicher freizugeben
    return tokens.length;
}

// const text = "tiktoken ist super für Node.js!";
// console.log(`"${text}" hat ${numTokensFromString(text, "text-embedding-3-small")} Tokens (Modell: text-embedding-3-small).`);
// console.log(`"${text}" hat ${numTokensFromString(text, "gpt-4")} Tokens (Modell: gpt-4).`);
```

### Q: K nächste Embedding-Vektoren schnell finden?
**A**: Vektor-Datenbanken verwenden (z.B. Pinecone, Weaviate, ChromaDB, Qdrant, Milvus, Supabase pgvector). Viele haben Node.js Client-Bibliotheken.

### Q: Welche Distanzfunktion verwenden?
**A**: **Kosinus-Ähnlichkeit** wird empfohlen. OpenAI-Embeddings sind auf Länge 1 normalisiert, daher:
*   Kosinus-Ähnlichkeit kann schneller per Skalarprodukt berechnet werden (wenn beide Vektoren bereits normalisiert sind, ist das Skalarprodukt gleich der Kosinus-Ähnlichkeit).
*   Kosinus-Ähnlichkeit und Euklidische Distanz führen zu identischen Rankings bei normalisierten Vektoren.

### Q: Darf ich meine Embeddings online teilen?
**A**: Ja, Kunden besitzen Input und Output, inkl. Embeddings. Stelle sicher, dass der Inhalt keine Gesetze oder Nutzungsbedingungen verletzt.

### Q: Wissen v3 Embedding-Modelle über aktuelle Ereignisse Bescheid?
**A**: Nein, die Modelle (`text-embedding-3-large` und `text-embedding-3-small`) haben kein Wissen über Ereignisse nach **September 2021**. Dies ist meist weniger limitierend als bei Textgenerierungsmodellen, kann aber in Grenzfällen die Performance beeinflussen.
```

```




  
</details>





<br><br>
<br><br>

# Embeddings Dimension


<details><summary>Click to expand..</summary>

Stell dir vor, du möchtest ein komplexes Objekt oder eine Idee beschreiben.

*   **Wenige "Dimensionen"**: Du hast nur wenige Wörter oder Merkmale zur Verfügung. Die Beschreibung wird grob, aber leicht zu verstehen und schnell zu verarbeiten sein.
*   **Viele "Dimensionen"**: Du hast sehr viele Wörter oder Merkmale. Die Beschreibung kann extrem detailliert und nuanciert sein, aber sie wird länger, komplexer und aufwendiger zu verarbeiten.

Im Kontext von **Vector Embeddings**:

1.  **Was sind "Dimensionen"?**
    Ein Embedding ist ein Vektor, also eine Liste von Zahlen. Jede einzelne Zahl in dieser Liste ist eine "Dimension".
    *   Wenn ein Embedding-Modell wie `text-embedding-3-small` standardmäßig **1536 Dimensionen** hat, bedeutet das, dass jeder Text in eine Liste von 1536 Fließkommazahlen umgewandelt wird.
    *   `text-embedding-3-large` hat **3072 Dimensionen**, also eine Liste von 3072 Zahlen.

    Diese Zahlen sind nicht direkt von Menschen interpretierbar (z.B. "Dimension 5 steht für das Konzept 'Tier'"). Stattdessen hat das Modell gelernt, die semantische Bedeutung und die Beziehungen von Wörtern und Sätzen in diesem hochdimensionalen Raum so zu kodieren, dass ähnliche Texte nahe beieinander liegen. Jede Dimension trägt einen kleinen Teil zur Gesamtrepräsentation der Bedeutung bei.

2.  **Was bedeutet "Dimensionen reduzieren"?**
    Dimensionen reduzieren bedeutet, die **Anzahl der Zahlen in dieser Liste (dem Vektor) zu verringern**.
    Anstatt beispielsweise 1536 Zahlen zu verwenden, um einen Text darzustellen, könntest du dich entscheiden, nur 512 oder 256 Zahlen zu verwenden.

    OpenAI's neue Modelle (`text-embedding-3-small` und `-large`) sind so trainiert, dass sie dies auf eine "intelligente" Weise tun können. Wenn du beim Erstellen des Embeddings den Parameter `dimensions` angibst (z.B. `dimensions=256`), generiert das Modell direkt einen kürzeren Vektor, der immer noch versucht, die wichtigsten konzeptuellen Eigenschaften des Textes zu bewahren. Es ist nicht einfach nur ein Abschneiden der letzten Zahlen eines längeren Vektors (obwohl man das nachträglich auch tun könnte, es ist aber weniger optimal).

3.  **Was bringt es, die Dimensionen zu reduzieren?**

    Das Reduzieren der Dimensionen hat mehrere praktische Vorteile, die meistens mit **Effizienz und Kosten** zu tun haben:

    *   **Weniger Speicherplatz**:
        *   Kürzere Vektoren (weniger Zahlen) benötigen weniger Speicherplatz in deiner Datenbank (z.B. Vektor-Datenbank) oder wo auch immer du sie speicherst. Bei Millionen von Texten kann das einen erheblichen Unterschied machen.
        *   *Beispiel*: Ein Vektor mit 1536 Zahlen braucht sechsmal so viel Platz wie einer mit 256 Zahlen (unter der Annahme, dass jede Zahl gleich viel Platz braucht).

    *   **Schnellere Berechnungen**:
        *   Wenn du Ähnlichkeiten zwischen Embeddings berechnest (z.B. Kosinus-Ähnlichkeit), sind diese Berechnungen bei kürzeren Vektoren schneller, da weniger Zahlen miteinander verrechnet werden müssen.
        *   Das macht Suchen, Clustering und andere Operationen performanter.

    *   **Geringere Kosten**:
        *   Weniger Speicherplatz kann zu geringeren Speicherkosten führen.
        *   Schnellere Berechnungen können zu geringeren Rechenkosten führen (weniger CPU-/GPU-Zeit).
        *   Manche Vektor-Datenbanken könnten ihre Preise auch (teilweise) an der Dimensionalität oder dem Gesamtvolumen der Daten ausrichten.

    *   **Weniger Arbeitsspeicher (RAM) benötigt**:
        *   Das Laden und Verarbeiten von kürzeren Vektoren benötigt weniger RAM. Das ist wichtig für Anwendungen mit begrenzten Speicherressourcen.

    *   **Schnellere Datenübertragung**:
        *   Kleinere Embeddings können schneller über Netzwerke gesendet oder von Festplatten gelesen werden.

    *   **Kompatibilität**:
        *   Manche Systeme oder ältere Vektor-Datenbanken haben möglicherweise eine Obergrenze für die Anzahl der Dimensionen, die sie verarbeiten können. Das Reduzieren der Dimensionen kann die Nutzung solcher Systeme ermöglichen.

    **Der Trade-off**:
    Natürlich gibt es einen Kompromiss. Ein Vektor mit sehr wenigen Dimensionen kann möglicherweise nicht mehr alle Nuancen und feinen Bedeutungsunterschiede eines Textes so gut erfassen wie ein Vektor mit vielen Dimensionen. Die Kunst besteht darin, eine Dimensionalität zu wählen, die für den spezifischen Anwendungsfall einen guten Kompromiss zwischen Performance/Kosten und der Qualität der semantischen Repräsentation darstellt.
    OpenAI gibt an, dass ihre neuen Modelle selbst bei reduzierter Dimensionalität (z.B. `text-embedding-3-large` auf 256 Dimensionen reduziert) immer noch sehr gut abschneiden und ältere Modelle mit höheren Dimensionen übertreffen können.

Zusammenfassend: **Dimensionen sind die Anzahl der Zahlen, die einen Text repräsentieren. Das Reduzieren der Dimensionen macht die Embeddings kleiner und schneller zu verarbeiten, was Speicher, Zeit und Geld spart, potenziell aber mit einem geringen Verlust an semantischer Detailtiefe einhergehen kann.**

  
</details>


