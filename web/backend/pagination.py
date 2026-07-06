"""Pagination parameter and response page models.

Defines ``Params`` (paginated request parameters) and ``Page`` (generic
paginated response wrapper) using fastapi-pagination, providing a consistent
pagination interface for the task listing endpoints.
"""

from collections.abc import Sequence
from typing import Generic, TypeVar

from fastapi_pagination.bases import AbstractPage, AbstractParams, RawParams
from pydantic import BaseModel, Field

T = TypeVar("T")


class Params(BaseModel, AbstractParams):
    """Pagination query parameters.

    Attributes:
        page: Current page number, starting from 1.
        page_size: Number of items per page, between 1 and 100.
    """

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    def to_raw_params(self) -> RawParams:
        """Convert to raw limit/offset parameters.

        Returns:
            A RawParams instance with limit and offset computed from page and page_size.
        """
        return RawParams(
            limit=self.page_size,
            offset=(self.page - 1) * self.page_size,
        )


class Page(AbstractPage[T], Generic[T]):
    """Generic paginated response for API endpoints.

    Attributes:
        items: The sequence of items for the current page.
        total: Total number of items across all pages.
        page: Current page number.
        page_size: Number of items per page.
    """

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
        """Create a Page instance from items, total count, and pagination params.

        Args:
            items: The items for the current page.
            total: Total number of items across all pages.
            params: Pagination parameters (must have ``page`` and ``page_size``).

        Returns:
            A new Page instance populated with the provided data.
        """
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
        )
