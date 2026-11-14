from core.dedupe import TitleIndex, canonicalize_url, title_similarity, title_tokens


def test_canonicalize_strips_tracking_params():
    url = "https://Example.com/story/?utm_source=rss&utm_medium=feed&id=7&fbclid=abc#comments"
    assert canonicalize_url(url) == "https://example.com/story?id=7"


def test_canonicalize_keeps_meaningful_query_and_order():
    assert canonicalize_url("https://example.com/a?b=2&a=1") == "https://example.com/a?b=2&a=1"


def test_canonicalize_normalises_host_port_and_slash():
    assert canonicalize_url("HTTPS://EXAMPLE.com:443/path/") == "https://example.com/path"
    assert canonicalize_url("http://example.com:8080") == "http://example.com:8080/"
    assert canonicalize_url("https://example.com/a") == canonicalize_url("https://example.com/a/?utm_campaign=x")


def test_title_similarity_is_jaccard():
    a, b = title_tokens("Red Bull unveils new floor"), title_tokens("Red Bull unveils new wing")
    assert title_similarity(a, b) == 4 / 6
    assert title_similarity(a, a) == 1.0
    assert title_similarity(a, frozenset()) == 0.0


def test_near_duplicate_titles_are_caught():
    index = TitleIndex(["Verstappen wins the Dutch Grand Prix after late safety car"])
    assert index.is_duplicate("Verstappen wins Dutch Grand Prix after a late safety car!")
    assert index.is_duplicate("VERSTAPPEN WINS THE DUTCH GRAND PRIX AFTER LATE SAFETY CAR")
    assert not index.is_duplicate("Norris wins the Dutch Grand Prix after Verstappen retires")


def test_short_titles_need_exact_match():
    index = TitleIndex(["Weekly open thread"])
    assert index.is_duplicate("Weekly Open Thread")
    assert not index.is_duplicate("Weekly open questions")


def test_index_learns_new_titles():
    index = TitleIndex()
    title = "A compiler pass that finds memory bugs fuzzers miss"
    assert not index.is_duplicate(title)
    index.add(title)
    assert index.is_duplicate(title)
