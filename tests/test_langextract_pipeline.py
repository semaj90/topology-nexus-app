from pathlib import Path

from scraper.langextract_pipeline import DocumentationCrawler, TopicConfig


def test_documentation_crawler_generates_records(tmp_path):
    base = Path('examples/docs').resolve()
    topics = [
        TopicConfig(topic='TypeScript', urls=[base.joinpath('typescript_handbook_intro.html').as_uri()])
    ]

    crawler = DocumentationCrawler()
    records = crawler.crawl_topics(topics, tmp_path)

    assert records, 'Expected at least one record from local documentation crawl'
    json_path = tmp_path / 'documentation_corpus.json'
    jsonl_path = tmp_path / 'documentation_corpus.jsonl'
    assert json_path.exists()
    assert jsonl_path.exists()
