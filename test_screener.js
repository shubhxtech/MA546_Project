const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;
const html = fs.readFileSync('index.html', 'utf8');
const dom = new JSDOM(html, { runScripts: "dangerously" });
const window = dom.window;

// Mock fetch
window.fetch = async (url) => {
  return {
    json: async () => {
      return {
        "min_score": 45, 
        "loaded": true, 
        "stocks": {
            "HDFCBANK.NS": {"name": "HDFC Bank Limited", "sector": "Financial Services", "pe": 17.27, "pb": 2.1, "roe": 0.138, "score": 78.8}
        }
      };
    }
  };
};

(async () => {
  try {
      await window.fetchFundamentals();
      console.log("Success! Table display:", window.document.getElementById('screenerTable').style.display);
      console.log("Table innerHTML:", window.document.getElementById('screenerBody').innerHTML);
  } catch (e) {
      console.log("Error:", e);
  }
})();
