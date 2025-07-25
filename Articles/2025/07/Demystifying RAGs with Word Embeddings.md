---
title: Demystifying RAGs with Word Embeddings
date: July 2025
tags: RAG, nlp, embeddings, semantics
---

# Demystifying RAGs with Word Embeddings

> [!TIP]
> Before diving deep into this article, I highly suggest you to check [Key Terms](#key-terms) and spend 2-3 minutes playing [Semantris](https://research.google.com/semantris/) by Google to get an intuitive feel for how words relate to each other semantically.

- [Demystifying RAGs with Word Embeddings](#demystifying-rags-with-word-embeddings)
  - [Key Terms](#key-terms)
  - [Words in a Coordinate System](#words-in-a-coordinate-system)
  - [Understanding Feature Space](#understanding-feature-space)
  - [Embedding Models (Translating Words to Numbers)](#embedding-models-translating-words-to-numbers)
  - [RAG Architecture with Word Embeddings](#rag-architecture-with-word-embeddings)
  - [Code Snippets](#code-snippets)
    - [Vector Database (pgvector) Snippets](#vector-database-pgvector-snippets)
    - [Create Embeddings](#create-embeddings)
    - [Save Embeddings](#save-embeddings)
    - [Retrieve Similar Documents](#retrieve-similar-documents)
  - [Limitations](#limitations)
  - [References](#references)

## Key Terms

- **Semantic:** relating to significance or meaning.  
   <small>mid 17th century: from French sémantique, from Greek sēmantikos 'significant', from sēmainein 'signify', from sēma 'sign'.</small>

- **Semantics (Study):** The study of linguistic meaning. It examines what meaning is, how words get their meaning, and how the meaning of a complex expression depends on its parts.

- **Cosine similarity:** Measures how similar two **vectors** are, regardless of their magnitude, by calculating the cosine of the angle between them. Values range from -1 (opposite) to 1 (identical).

- **Word Embedding:** A numerical representation of a word as a vector in multi-dimensional feature space. Each dimension represents an abstract feature, allowing mathematical operations between words.

- **Feature Space**: The mathematical space where embeddings live. If words were cities, feature space would be the map—each dimension (feature) is like a coordinate that helps position words based on their meaning. Words with similar meanings are neighbors in this space.

- **Vector Database**: A specialized database designed to store and search through embeddings (vectors). Unlike traditional databases that search for exact matches, vector databases find items based on similarity—like finding words that are "close" to each other in meaning.

## Words in a Coordinate System

Word embeddings are simply **numerical representations of words as vectors** in a multi-dimensional feature space. Think of it as assigning coordinates to words, where each dimension represents some abstract feature.

For example, in a [3D feature space](#understanding-feature-space):

- "cat" might be `[0.2, 0.8, 0.5]`
- "dog" might be `[0.3, 0.9, 0.4]`
- "car" might be `[0.9, 0.1, 0.2]`

Notice how "cat" and "dog" have similar coordinates (they're both pets), while "car" is far away. In practice, for example OpenAI's text-embedding-3-small, uses 1536 dimensions to capture much richer relationships.

## Understanding Feature Space

What is a feature space actually? Each dimension in feature space represents an abstract "feature".

Human brains are not familiar with conceptualizing dimensions after 3-4 dimensions. Let's begin simple and build up to understand how feature spaces work:

**1D Feature Space: Temperature**

With just one dimension, we can only capture one aspect. Words line up on a single axis.

![1D Feature Space](../../../Assets/demystify-rag/1d-feature-space-dark.png#gh-dark-mode-only)

![1D Feature Space](../../../Assets/demystify-rag/1d-feature-space-light.png#gh-light-mode-only)

**2D Feature Space: Temperature + Size**

Now we can represent two features! Notice how "iceberg" and "ice" share coldness but differ in size.

![2D Feature Space](../../../Assets/demystify-rag/2d-feature-space-dark.png#gh-dark-mode-only)

![2D Feature Space](../../../Assets/demystify-rag/2d-feature-space-light.png#gh-light-mode-only)

**3D Feature Space: Temperature + Size + Natural/Man-made**

Each dimension adds more expressive power.

![3D Feature Space](../../../Assets/demystify-rag/3d-feature-space-dark.png#gh-dark-mode-only)

![3D Feature Space](../../../Assets/demystify-rag/3d-feature-space-light.png#gh-light-mode-only)


**Code Representation for 3D Feature Space**

```javascript
// Features: [temperature, size, natural/man-made]
// Scale: -1 (cold/small/artificial) to 1 (hot/large/natural)

const embeddings = {
  coffee: [0.9, -0.8, -0.7], // [hot, small, man-made]
  tea: [0.8, -0.8, 0.3], // [hot, small, somewhat natural]
  juice: [-0.7, -0.5, 0.8], // [cold, small-med, natural]
  milk: [-0.3, -0.3, 0.9], // [cool, medium, natural]
  beer: [-0.6, -0.2, -0.3], // [cold, medium, processed]
  cola: [-0.8, -0.5, -0.9], // [cold, small-med, man-made]
  soup: [0.9, 0.7, -0.2], // [hot, large, somewhat processed]
  smoothie: [-0.5, 0.3, 0.4], // [cold, large, somewhat natural]
  espresso: [0.95, -0.95, -0.8], // [very hot, tiny, man-made]
};
```

## Embedding Models (Translating Words to Numbers)

An embedding model is like a specialized translator that converts words into numbers. Just as different human translators might interpret text slightly differently, different embedding models create different numerical representations.

```
"What is the weather today?" 
    ↓
[Embedding Model]
    ↓
[0.23, -0.67, 0.91, ..., 0.45] (Size of model's feature space)
```

**What to consider when choosing an Embedding Model?**

- **Context Window**: How much text can be processed at once?

- **Chunk Size**: is how much text you process into a single embedding. It's the "window" of text that becomes one vector.


**Popular Embedding Models**

| Model                  | Provider    | Dimensions | Context Window              |
| ---------------------- | ----------- | ---------- | --------------------------- |
| text-embedding-3-small | OpenAI      | 1,536      | 8,191 tokens (~6,000 words) |
| text-embedding-3-large | OpenAI      | 3,072      | 8,191 tokens (~6,000 words) |
| all-MiniLM-L6-v2       | Open Source | 384        | 256 tokens (~200 words)     |
| all-mpnet-base-v2      | Open Source | 768        | 384 tokens (~300 words)     |
| e5-large-v2            | Open Source | 1,024      | 512 tokens (~400 words)     |
| instructor-xl          | Open Source | 768        | 512 tokens (~400 words)     |
| voyage-2               | Voyage AI   | 1,024      | 4,096 tokens (~3,000 words) |
| embed-english-v3.0     | Cohere      | 1,024      | 512 tokens (~400 words)     |


## RAG Architecture with Word Embeddings

![RAG](../../../Assets/demystify-rag/simplyfied-rag-architecture-dark.png#gh-dark-mode-only)

![RAG](../../../Assets/demystify-rag/simplyfied-rag-architecture.png#gh-light-mode-only)


## Code Snippets

### Vector Database (pgvector) Snippets

Set up your PostgreSQL database with pgvector support. This creates a table to store your documents and their embeddings.

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(1536),  -- text-embedding-3-small dimensions
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Create Embeddings

Convert text into embeddings using OpenAI's API.

```typescript
import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * Converts text into embeddings (numerical vectors).
 * 
 * @param texts - Array of text strings to convert
 * @returns Array of number arrays (vectors)
 */
async function createEmbeddings(texts: string[]): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts,
  });

  return response.data.map(item => item.embedding);
}
```

### Save Embeddings

Store your documents and their embeddings in the database for later retrieval.

```typescript
import { Pool } from 'pg';

const db = new Pool({ connectionString: process.env.DATABASE_URL });

/**
 * Saves text and its embedding to the database.
 * The embedding is stored as a vector that pgvector can search through.
 * 
 * @param content - The text to save
 * @param embedding - The embedding vector for this text
 * @returns The ID of the saved document
 */
async function saveEmbedding(
  content: string, 
  embedding: number[]
): Promise<number> {
  const query = `
    INSERT INTO documents (content, embedding)
    VALUES ($1, $2)
    RETURNING id
  `;
  
  // Convert array to pgvector format
  const vectorString = `[${embedding.join(',')}]`;
  
  const result = await db.query(query, [content, vectorString]);
  return result.rows[0].id;
}
```

### Retrieve Similar Documents

Find documents similar to your search query using pgvector's built-in similarity search.

```typescript
/**
 * Finds the most similar documents to your search query.
 * pgvector handles the math - it finds vectors closest to your query vector.
 * 
 * @param query - What you're searching for
 * @param limit - How many results to return (default: 5)
 * @returns Array of similar documents with their similarity scores
 */
async function retrieveSimilarDocuments(
  query: string, 
  limit: number = 5
): Promise<Array<{ content: string; similarity: number }>> {
  // First, convert the search query into an embedding
  const [queryEmbedding] = await createEmbeddings([query]);
  
  // Then search for similar vectors in the database
  const searchQuery = `
    SELECT 
      content,
      1 - (embedding <=> $1::vector) as similarity
    FROM documents
    ORDER BY embedding <=> $1::vector
    LIMIT $2
  `;
  
  // Convert embedding to pgvector format
  const vectorString = `[${queryEmbedding.join(',')}]`;
  
  const result = await db.query(searchQuery, [vectorString, limit]);
  return result.rows;
}
```

## Limitations

Word Embeddings in RAG systems have some constraints:

1. **Context Window:** Embeddings capture general meaning but may miss specific nuances or context from longer passages. Think of it as a computer with limited RAM. It can only hold so much text in active memory.

2. **Semantic Ambiguity:** Similar embeddings don't always mean relevant content. "bank" (financial) and "bank" (river) might retrieve each other. Remember, words have a coordinate in the feature space. They can't be placed in two different coordinates.

3. **Chunking Challenges:** Breaking documents into chunks is like cutting a movie into scenes with your eyes closed. Sometimes you slice right through a conversation, leaving "I love..." in one chunk and "...chocolate cake" in another. The system might retrieve the confession of love but miss what they're actually loving.

## References

- [Semantic Etymology](https://www.etymonline.com/search?q=semantic)
- [Semantics (Study)](https://en.wikipedia.org/wiki/Semantics)
- [Cosine Similarity](https://www.wikiwand.com/en/articles/Cosine_similarity)
- [Word Embedding Tutorial - CMU](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
