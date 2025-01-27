from t5_generate_recipe_script import (
    validate_action_ingredient_contextual,
    validate_action_ingredient_spacy,
    validate_word_groupings
)

def assert_equals(actual, expected):
    """
    Checks if the actual value is equal to the expected value.
    Prints a success message if equal; raises an AssertionError if not.

    Parameters:
      expected: The expected value to compare against.
      actual: The actual value obtained.

    Raises:
      AssertionError: If the actual value is not equal to the expected value.
    """
    if actual != expected:
        raise AssertionError(f"Assertion failed: Expected {expected}, but got {actual}.")


def test_validate_action_ingredient_contextual():
    result = validate_action_ingredient_contextual("melt", "chicken")
    assert_equals(result, False)

    result = validate_action_ingredient_contextual("print", "chicken")
    assert_equals(result, False)

    result = validate_action_ingredient_contextual("stir", "chicken")
    assert_equals(result, False)

    result = validate_action_ingredient_contextual("fry", "ice")
    assert_equals(result, False)

    result = validate_action_ingredient_contextual("fry", "chicken")
    assert_equals(result, True)

    result = validate_action_ingredient_contextual("melt", "ice")
    assert_equals(result, True)

    result = validate_action_ingredient_contextual("melt", "butter")
    assert_equals(result, True)

    result = validate_action_ingredient_contextual("cook", "chicken")
    assert_equals(result, True)

    print("test_validate_action_ingredient_contextual Passed!")

def test_validate_action_ingredient_spacy():
    result = validate_action_ingredient_spacy("melt", "chicken")
    assert_equals(result, False)

    result = validate_action_ingredient_spacy("print", "chicken")
    assert_equals(result, False)

    result = validate_action_ingredient_spacy("fry", "ice")
    assert_equals(result, False)

    result = validate_action_ingredient_spacy("fry", "chicken")
    assert_equals(result, True)

    result = validate_action_ingredient_spacy("cook", "chicken")
    assert_equals(result, True)

    print("test_validate_action_ingredient_spacy Passed!")

def test_validate_word_groupings():
    recipe = "melt butter and fry chicken"
    ingredients = ["butter", "chicken"]
    
    count, invalid_groupings = validate_word_groupings(recipe, ingredients, n=2)
    assert_equals(count, 0)
    assert_equals(invalid_groupings, [])

    recipe = "melt butter and stir ice"
    count, invalid_groupings = validate_word_groupings(recipe, ingredients, n=2)
    assert_equals(count, 0)
    assert_equals(invalid_groupings, [])

    recipe = "chop carrots and fry potatoes"
    ingredients = ["carrots", "potatoes"]
    count, invalid_groupings = validate_word_groupings(recipe, ingredients, n=3)
    assert_equals(count, 0)
    assert_equals(invalid_groupings, [])

    print("test_validate_word_groupings Passed!")




test_validate_action_ingredient_contextual()
test_validate_action_ingredient_spacy()
test_validate_word_groupings()
    
