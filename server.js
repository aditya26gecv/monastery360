// Step 1: Backend server for AI semantic search
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { OpenAI } = require("openai");
require("dotenv").config();

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Sample monastery data (replace with your full list)
const monasteries = [
  { name: "Rumtek Monastery", description: "Famous for spiritual events and cultural festivals.", type: "Monastery", embedding: null },
  { name: "Enchey Monastery", description: "Historic monastery known for Cham dances.", type: "Monastery", embedding: null },
  { name: "Pemayangtse Monastery", description: "One of the oldest monasteries in Sikkim.", type: "Monastery", embedding: null }
];

// Compute embeddings on server start
async function computeEmbeddings() {
  for (let item of monasteries) {
    if (!item.embedding) {
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: item.name + ". " + item.description
      });
      item.embedding = response.data[0].embedding;
    }
  }
}
computeEmbeddings();

// Cosine similarity function
function cosineSimilarity(a, b) {
  let sum = 0, sumA = 0, sumB = 0;
  for (let i = 0; i < a.length; i++) { sum += a[i]*b[i]; sumA += a[i]*a[i]; sumB += b[i]*b[i]; }
  return sum / (Math.sqrt(sumA) * Math.sqrt(sumB));
}

// Endpoint for search
app.post("/search", async (req, res) => {
  const query = req.body.query;
  if (!query) return res.json([]);

  const queryEmbeddingResp = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: query
  });
  const queryEmbedding = queryEmbeddingResp.data[0].embedding;

  const results = monasteries.map(item => ({
    ...item,
    score: cosineSimilarity(item.embedding, queryEmbedding)
  }));
  results.sort((a,b) => b.score - a.score);
  res.json(results.slice(0,5)); // top 5
});

app.listen(port, () => console.log(`AI Search server running at http://localhost:${port}`));
