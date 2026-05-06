require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const Anthropic = require('@anthropic-ai/sdk');
const axios   = require('axios');
const path    = require('path');

const app  = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// ✅ Serve frontend properly
app.use(express.static(path.join(__dirname, '../frontend')));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/candlecast.html'));
});

// ── COIN SYMBOL → ID MAP ─────────────────────────
const COIN_MAP = {
  btc: "bitcoin",
  eth: "ethereum",
  sol: "solana",
  ada: "cardano",
  xrp: "ripple",
  doge: "dogecoin",
  bnb: "binancecoin",
  matic: "polygon",
  dot: "polkadot",
  ltc: "litecoin",
  btc: "bitcoin",
  eth: "ethereum",
  bnb: "binancecoin",
  xrp: "ripple",
  ada: "cardano",
  doge: "dogecoin",
  dot: "polkadot",
  avax: "avalanche-2",
  link: "chainlink",
  pol: "matic-network",
  trx: "tron",
  ltc: "litecoin",
  bch: "bitcoin-cash",
  xlm: "stellar",
  atom: "cosmos",
  uni: "uniswap",
  etc: "ethereum-classic",
  fil: "filecoin",
  near: "near",
  inj: "injective-protocol",
  icp: "internet-computer",
  hbar: "hedera-hashgraph",
  vet: "vechain",
  aave: "aave",
  grt: "the-graph",
  sand: "the-sandbox",
  mana: "decentraland",
  shib: "shiba-inu",
  algo: "algorand"
};



// ── COINGECKO PRICE API ─────────────────────────
app.get('/api/price/:coin', async (req, res) => {
  try {
    const input = req.params.coin.toLowerCase();
    const coinId = COIN_MAP[input] || input;

    const response = await axios.get(
      `https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd`
    );

    if (!response.data[coinId]) {
      return res.status(404).json({ error: "Coin not found" });
    }

    res.json({
      coin: coinId,
      price: response.data[coinId].usd
    });

  } catch (err) {
    console.error("CoinGecko Error:", err.message);
    res.status(500).json({ error: "Failed to fetch price" });
  }
});

// ── COINGECKO PRICE API ─────────────────────────
// ── COINGECKO CHART API ─────────────────────────
app.get('/api/chart/:coin', async (req, res) => {
  try {
    const input = req.params.coin.toLowerCase();
    const coinId = COIN_MAP[input] || input;

    const response = await axios.get(
      `https://api.coingecko.com/api/v3/coins/${coinId}/market_chart`,
      {
        params: {
          vs_currency: 'usd',
          days: 7
        }
      }
    );

    res.json(response.data);

  } catch (err) {
    console.error("Chart API Error:", err.message);
    res.status(500).json({ error: "Failed to fetch chart data" });
  }
});

// ── GET ALL COINS (CoinGecko) ─────────────────────
app.get('/api/coins', async (req, res) => {
  try {
    const response = await axios.get(
      'https://api.coingecko.com/api/v3/coins/list'
    );

    res.json(response.data);

  } catch (err) {
    console.error("Coin list error:", err.message);
    res.status(500).json({ error: "Failed to fetch coins" });
  }
});

// ── CLAUDE CLIENT ─────────────────────────
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// ── ML MODEL ENDPOINT ─────────────────────
app.post('/api/ml-predict', async (req, res) => {
  try {
    const response = await axios.post(
      'http://127.0.0.1:5000/predict',
      req.body
    );

    res.json(response.data);

  } catch (err) {
    console.error("ML Error:", err.message);
    res.status(500).json({ error: "ML model failed" });
  }
});

// ── CLAUDE REPORT ─────────────────────────
app.post('/api/report', async (req, res) => {
  const { prompt } = req.body;

  try {
    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-5',
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = message.content
      .filter(b => b.type === 'text')
      .map(b => b.text)
      .join('');

    res.json({ text });

  } catch (err) {
    console.error('Claude error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── START SERVER ─────────────────────────
app.listen(port, () => {
  console.log(`🕯 CandleCast running at http://localhost:${port}`);
});
