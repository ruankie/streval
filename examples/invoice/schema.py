from pydantic import BaseModel
from typing import List


class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str
    vendor_name: str
    total_amount: float
    currency: str
    line_items: List[LineItem]
