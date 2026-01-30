from src.categories import clear_cache, get_category_list, is_valid_category, normalize_category


def setup_function():
    clear_cache()


def test_get_category_list():
    cats = get_category_list()
    assert "Person" in cats
    assert "Medical" in cats
    assert "Unknown" in cats


def test_is_valid_category():
    assert is_valid_category("Person")
    assert is_valid_category("Medical")
    assert not is_valid_category("Nonsense")
    assert not is_valid_category("")


def test_normalize_direct_match():
    assert normalize_category("Person") == "Person"
    assert normalize_category("Medical") == "Medical"


def test_normalize_case_insensitive():
    assert normalize_category("person") == "Person"
    assert normalize_category("MEDICAL") == "Medical"


def test_normalize_variations():
    assert normalize_category("name") == "Person"
    assert normalize_category("people") == "Person"
    assert normalize_category("organization") == "Place"
    assert normalize_category("health") == "Medical"
    assert normalize_category("legal") == "Technical"
    assert normalize_category("other") == "Unknown"


def test_normalize_empty():
    assert normalize_category("") == "Unknown"
    assert normalize_category("gibberish_xyz") == "Unknown"
