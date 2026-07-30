import pytest
from pydantic import ValidationError

from src.schemas.base import BaseSchema, Float, round_probability


class TestRoundProbability:
    def test_rounds_to_two_decimals(self) -> None:
        assert round_probability(3.14159) == 3.14
        assert round_probability(2.005) == 2.00  # round to even

    def test_rounds_float_str(self) -> None:
        assert round_probability("3.14159") == 3.14

    def test_rounds_int(self) -> None:
        assert round_probability(3) == 3.0

    def test_passes_through_non_numeric(self) -> None:
        v = "hello"
        assert round_probability(v) is v

    def test_passes_through_none(self) -> None:
        assert round_probability(None) is None


class TestFloatAnnotation:
    class Model(BaseSchema):
        score: Float

    def test_valid_float_is_rounded(self) -> None:
        m = self.Model(score=3.14159)
        assert m.score == 3.14

    def test_invalid_float_raises(self) -> None:
        with pytest.raises(ValidationError):
            self.Model(score="not-a-number")


class TestBaseSchemaConfig:
    class Sample(BaseSchema):
        first_name: str
        last_name: str
        age: int

    def test_frozen_prevents_mutation(self) -> None:
        # pydantic resolves camelCase aliases at runtime
        m = self.Sample(firstName="John", lastName="Doe", age=30)  # type: ignore
        with pytest.raises(ValidationError):
            # testing that frozen model raises on mutation
            m.first_name = "Jane"  # type: ignore

    def test_populate_by_name_allows_snake_case_input(self) -> None:
        m = self.Sample(first_name="John", last_name="Doe", age=30)
        assert m.first_name == "John"
        assert m.last_name == "Doe"

    def test_alias_generates_camel_case(self) -> None:
        m = self.Sample(first_name="John", last_name="Doe", age=30)
        data = m.model_dump(by_alias=True)
        assert "firstName" in data
        assert "lastName" in data
        assert "age" in data

    def test_str_strip_whitespace(self) -> None:
        # pydantic resolves camelCase aliases at runtime
        m = self.Sample(firstName="  John  ", lastName="  Doe  ", age=30)  # type: ignore
        assert m.first_name == "John"
        assert m.last_name == "Doe"

    def test_extra_fields_allowed(self) -> None:
        # pydantic resolves camelCase aliases at runtime; extra fields are allowed
        m = self.Sample(firstName="John", lastName="Doe", age=30, extra="allowed")  # type: ignore
        assert m.first_name == "John"
        assert not hasattr(m, "extra")

    def test_validate_assignment_prevents_invalid_change(self) -> None:
        # pydantic resolves camelCase aliases at runtime
        m = self.Sample(firstName="John", lastName="Doe", age=30)  # type: ignore
        with pytest.raises(ValidationError):
            # testing that validate_assignment rejects invalid type
            m.age = "not-an-int"  # type: ignore


class TestUseEnumValues:
    def test_serializes_enum_as_value(self) -> None:
        from enum import StrEnum

        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        class Model(BaseSchema):
            color: Color

        m = Model(color=Color.RED)
        data = m.model_dump()
        assert data["color"] == "red"
