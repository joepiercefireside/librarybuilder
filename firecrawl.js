import('@mendable/firecrawl-js').then(({ FirecrawlApp }) => {
    console.log('Firecrawl module imported successfully');
    const app = new FirecrawlApp({ apiKey: process.env.FIRECRAWL_API_KEY });
    console.log('FirecrawlApp initialized with API key');

    async function crawl(url) {
        try {
            console.log(`Starting crawl for URL: ${url}`);
            const results = await app.crawlUrl(url, {
                actions: [
                    { type: 'click', selector: '.collapsible-header' },
                    { type: 'click', selector: '.pii-tag.country-tag' },
                    { type: 'wait', value: 5000 }
                ]
            });
            console.log('Crawl completed, results:', JSON.stringify(results));
            console.log(JSON.stringify(results));
        } catch (error) {
            console.error('Crawl error:', error.message, error.stack);
        }
    }

    const url = process.argv[2];
    if (!url) {
        console.error('No URL provided');
        process.exit(1);
    }
    crawl(url);
}).catch(error => {
    console.error('Failed to import Firecrawl:', error.message, error.stack);
});