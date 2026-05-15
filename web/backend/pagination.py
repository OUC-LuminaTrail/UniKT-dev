from collections.abc import Sequence
from typing import Generic, TypeVar

from fastapi_pagination.bases import AbstractPage, AbstractParams, RawParams
from pydantic import Field

T = TypeVar("T")


class Params(AbstractParams):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    def to_raw_params(self) -> RawParams:
        return RawParams(
            limit=self.page_size,
            offset=(self.page - 1) * self.page_size,
        )


class Page(AbstractPage[T], Generic[T]):
    items: Sequence[T]
    total: int = Field(default=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1)

    __params_type__ = Params

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        total: int,
        params: AbstractParams,
    ) -> "Page[T]":
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
        )
