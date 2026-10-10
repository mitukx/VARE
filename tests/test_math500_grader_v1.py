from scripts.math500_grader_v1 import exact_match, has_valid_final_box, parse_prediction


def test_requires_one_balanced_final_box():
    assert exact_match(r"Work: \boxed{1}. Check: \boxed{\frac{2}{4}}", r"\frac{1}{2}")
    assert not has_valid_final_box("The answer is 2.")
    assert not has_valid_final_box(r"The answer is \boxed{2")
    assert parse_prediction(r"Earlier \boxed{2}, final \boxed{3}").raw == "3"


def test_exact_symbolic_equivalence_and_negative_control():
    assert exact_match(r"\boxed{\frac{2}{4}}", r"\frac{1}{2}")
    assert exact_match(r"\boxed{\sqrt{12}/2}", r"\sqrt{3}")
    assert not exact_match(r"\boxed{0}", r"\infty")
    assert not exact_match(r"\boxed{2.56}", r"2.557")


def test_units_vectors_sets_and_text_answers():
    assert exact_match(r"\boxed{\pi/2}", r"90^\circ")
    assert exact_match(
        r"\boxed{(-1/3, 2/3, 5/3)}",
        r"\begin{pmatrix}-1/3\\2/3\\5/3\end{pmatrix}",
    )
    assert exact_match(r"\boxed{\{1+\sqrt{5},1-\sqrt{5},-2\}}", r"\{1\pm\sqrt{5},-2\}")
    assert exact_match(r"\boxed{Evelyn}", r"\text{Evelyn}")
    assert not exact_match(r"\boxed{Evelyn}", r"\text{Navin}")


def test_one_matching_box_cannot_hide_a_wrong_final_box():
    assert not exact_match(r"First \boxed{42}; final \boxed{41}", "42")


def test_malformed_final_box_cannot_fall_back_to_earlier_correct_box():
    completion = r"Scratch: \boxed{42}; final: \boxed{41"
    assert not has_valid_final_box(completion)
    assert not exact_match(completion, "42")


def test_empty_final_box_cannot_fall_back_to_earlier_correct_box():
    completion = r"Scratch: \boxed{42}; final: \boxed{}"
    assert not has_valid_final_box(completion)
    assert not exact_match(completion, "42")
