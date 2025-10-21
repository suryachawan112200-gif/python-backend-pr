
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { coin, entryPrice, targets, userStoploss, marketStoplosses } = req.body;

  // Validate input
  if (!coin) {
    return res.status(400).json({ error: "Coin symbol is required" });
  }

  try {
    // Bybit V5 API endpoint for spot ticker (public, no API key needed)
    const apiUrl = `https://api.bybit.com/v5/market/tickers?category=spot&symbol=${coin}`;
    const apiResponse = await fetch(apiUrl);
    
    if (!apiResponse.ok) {
      throw new Error(`Bybit API error: ${apiResponse.status}`);
    }

    const data = await apiResponse.json();
    
    if (data.retCode !== 0) {
      throw new Error(data.retMsg || "API response error");
    }

    const ticker = data.result.list[0]; // First item in list
    if (!ticker) {
      throw new Error(`No ticker data for symbol: ${coin}`);
    }

    const currentPrice = parseFloat(ticker.lastPrice); // Extract last price

    // Optional: Check if price hits TGT or SL (you can do this in frontend too)
    let tgtHit = false;
    let slHit = false;

    if (targets && targets.length > 0) {
      tgtHit = targets.some(target => currentPrice >= target);
    }

    if (userStoploss) {
      slHit = currentPrice <= userStoploss;
    }

    if (marketStoplosses && marketStoplosses.length > 0) {
      slHit = slHit || marketStoplosses.some(sl => currentPrice <= sl);
    }

    res.status(200).json({ 
      currentPrice,
      tgtHit,
      slHit,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error("Bybit API fetch error:", error);
    res.status(500).json({ error: "Failed to fetch current price: " + error.message });
  }
}
