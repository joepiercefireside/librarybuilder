const FirecrawlApp = require('@mendable/firecrawl-js');
  const app = new FirecrawlApp({ apiKey: process.env.FIRECRAWL_API_KEY });

  async function crawl(url) {
      try {
          const results = await app.crawlUrl(url, {
              actions: [
                  { type: 'click', selector: '.collapsible-header' },
                  { type: 'click', selector: '.pii-tag.country-tag' },
                  { type: 'wait', value: 5000 }
              ]
          });
          console.log(JSON.stringify(results));
      } catch (error) {
          console.error(error);
      }
  }

  crawl(process.argv[2]);