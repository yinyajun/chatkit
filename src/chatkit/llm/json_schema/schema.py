from dataclasses import dataclass
from typing import Union, get_args, get_origin


class ModelMeta(type):
    def __new__(cls, name, bases, namespace):
        fields = {}
        annotations = namespace.get("__annotations__", {})
        for key, value in annotations.items():
            field = namespace.get(key)
            if isinstance(field, Field):
                field._type = value
                fields[key] = field
            else:
                fields[key] = Field(_type=value)
        namespace["_fields"] = fields
        return super().__new__(cls, name, bases, namespace)


@dataclass
class Field:
    _type: type = None
    description: str = None
    pattern: str = None
    enum: list[str] = None
    max_items: int = None
    min_items: int = None
    alias: str = None

    def json_schema(self, extra: dict) -> dict:
        if self._type == str:
            res = self.string_json_schema(extra)
        elif self._type == int:
            res = self.int_json_schema(extra)
        elif self._type == float:
            res = self.float_json_schema(extra)
        elif get_origin(self._type) == list:
            res = self.array_json_schema(extra)
        else:
            raise ValueError(f"{self._type} not supported")

        return res

    def string_json_schema(self, extra: dict) -> dict:
        assert self._type == str
        res = {}

        if self.description is not None:
            assert isinstance(self.description, str)
            res["description"] = self.description
        if self.pattern is not None:
            res["pattern"] = self.pattern
        if self.enum is not None:
            assert isinstance(self.enum, list)
            res["enum"] = self.enum

        res["type"] = "string"
        return res

    def array_json_schema(self, extra: dict) -> dict:
        _type = get_origin(self._type)
        assert _type == list
        res = {}
        if self.description is not None:
            res["description"] = self.description

        element_type = get_args(self._type)[0]
        if get_origin(element_type) == Union:
            items = []
            for sub_element_type in get_args(element_type):
                if issubclass(sub_element_type, Model):
                    items.append(sub_element_type.json_schema())
                else:
                    items.append(Field(_type=sub_element_type).json_schema(extra))

            key_any_of = "anyOf"
            if extra and "key_any_of" in extra:
                key_any_of = extra.get("key_any_of")
            res["items"] = {key_any_of: items, "type": "object"}

        elif issubclass(element_type, Model):
            res["items"] = element_type.json_schema()
        else:
            res["items"] = Field(_type=element_type).json_schema(extra)

        if self.max_items is not None:
            res["maxItems"] = self.max_items
        if self.min_items is not None:
            res["minItems"] = self.min_items

        res["type"] = "array"
        return res

    def int_json_schema(self, extra: dict):
        assert self._type == int
        res = {}
        if self.description is not None:
            assert isinstance(self.description, str)
            res["description"] = self.description
        res["type"] = "integer"

        return res

    def float_json_schema(self, extra: dict):
        assert self._type == float
        res = {}
        if self.description is not None:
            assert isinstance(self.description, str)
            res["description"] = self.description
        res["type"] = "number"
        return res


class Model(metaclass=ModelMeta):
    @classmethod
    def json_schema(cls, extra: dict = None) -> dict:
        """Usage:

        1. extra={"key_any_of": "oneOf"} for anthropic
        2. extra={"key_any_of": "oneOf"} for bedrock boto3
        3. extra={"additionalProperties": False} for openai
        """
        properties = {}
        required = []
        for field_name, field in cls._fields.items():
            if field.alias is not None:
                field_name = field.alias

            required.append(field_name)
            properties[field_name] = field.json_schema(extra)
        res = {"properties": properties, "required": required, "type": "object"}

        if extra and "additionalProperties" in extra:
            res["additionalProperties"] = extra.get("additionalProperties")

        return res
